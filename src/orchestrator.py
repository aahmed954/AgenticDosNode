"""Main orchestrator that integrates all components."""

from typing import Dict, Any, Optional, List, Union, AsyncGenerator
from dataclasses import dataclass
import asyncio
import time
import json
from enum import Enum

from langchain.tools import BaseTool
from langchain_core.language_models import BaseLLM
from redis.asyncio import Redis as AsyncRedis

from .config import settings, TaskComplexity
from .models.router import ModelRouter, TaskProfile
from .agents.react_agent import ReActAgent, ToolExecutor
from .agents.langgraph_orchestrator import LangGraphOrchestrator, WorkflowConfig
from .rag.advanced_rag import AdvancedRAG, RAGConfig, RetrievalStrategy
from .optimization.cache_manager import PromptCache, BatchOptimizer
from .monitoring.metrics import MetricsCollector
from .utils.logging import get_logger, configure_logging

logger = get_logger(__name__)


class ExecutionMode(str, Enum):
    """Execution modes for the orchestrator."""
    REACT = "react"
    LANGGRAPH = "langgraph"
    DIRECT = "direct"
    AUTO = "auto"


@dataclass
class OrchestrationRequest:
    """Request for orchestration."""

    task: str
    mode: ExecutionMode = ExecutionMode.AUTO
    context: Optional[Dict[str, Any]] = None
    tools: Optional[List[BaseTool]] = None
    rag_query: Optional[str] = None
    stream: bool = False
    use_cache: bool = True
    priority: int = 0
    budget_limit: Optional[float] = None


@dataclass
class OrchestrationResponse:
    """Response from orchestration."""

    success: bool
    result: Any
    execution_time: float
    model_used: str
    total_cost: float
    cached: bool
    metadata: Dict[str, Any]
    traces: Optional[List[Dict[str, Any]]] = None


class AgenticOrchestrator:
    """
    Main orchestrator that coordinates all components.

    This is the primary interface for the agentic orchestration framework,
    providing:
    - Intelligent routing between execution modes
    - Multi-model coordination
    - RAG integration
    - Caching and optimization
    - Comprehensive monitoring
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        vector_store: Optional[Any] = None,
        tools: Optional[List[BaseTool]] = None,
        enable_monitoring: bool = True
    ):
        # Initialize configuration
        configure_logging(settings.monitoring.log_level)

        # Core components
        self.model_router = ModelRouter()
        self.metrics_collector = MetricsCollector() if enable_monitoring else None

        # Initialize Redis if available
        self.redis_client = None
        if redis_url:
            asyncio.create_task(self._init_redis(redis_url))

        # Caching
        self.prompt_cache = PromptCache(
            redis_client=self.redis_client,
            semantic_threshold=settings.cache.semantic_cache_threshold
        )
        self.batch_optimizer = BatchOptimizer(
            max_batch_size=settings.performance.batch_size,
            batch_timeout_ms=settings.performance.batch_timeout_ms
        )

        # Agents
        self.react_agent = None
        self.langgraph_orchestrator = None
        self.tools = tools or []

        # RAG system
        self.rag_system = None
        if vector_store:
            self.rag_system = AdvancedRAG(
                vector_store=vector_store,
                config=RAGConfig()
            )

        # Statistics
        self.total_requests = 0
        self.total_cost = 0.0

        logger.info("Agentic Orchestrator initialized")

    async def _init_redis(self, redis_url: str):
        """Initialize Redis connection."""
        try:
            self.redis_client = await AsyncRedis.from_url(redis_url)
            self.prompt_cache.redis_client = self.redis_client
            logger.info("Redis connection established")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")

    async def execute(
        self,
        request: OrchestrationRequest
    ) -> Union[OrchestrationResponse, AsyncGenerator[Dict[str, Any], None]]:
        """
        Execute an orchestration request.

        This is the main entry point for task execution.
        """

        start_time = time.time()
        self.total_requests += 1

        # Check cache if enabled
        if request.use_cache:
            cache_key = self._generate_cache_key(request)
            cached_result = await self.prompt_cache.get(cache_key)

            if cached_result:
                logger.info("Cache hit for request")
                return OrchestrationResponse(
                    success=True,
                    result=cached_result,
                    execution_time=time.time() - start_time,
                    model_used="cached",
                    total_cost=0.0,
                    cached=True,
                    metadata={"cache_key": cache_key}
                )

        # Determine execution mode
        execution_mode = await self._determine_execution_mode(request)
        logger.info(f"Selected execution mode: {execution_mode}")

        # Execute based on mode
        if request.stream:
            return self._execute_streaming(request, execution_mode, start_time)
        else:
            return await self._execute_complete(request, execution_mode, start_time)

    async def _execute_complete(
        self,
        request: OrchestrationRequest,
        mode: ExecutionMode,
        start_time: float
    ) -> OrchestrationResponse:
        """Execute request to completion."""

        try:
            # RAG retrieval if needed
            rag_context = None
            if request.rag_query and self.rag_system:
                rag_result = await self.rag_system.retrieve(
                    query=request.rag_query,
                    strategy=RetrievalStrategy.HYBRID
                )
                rag_context = {
                    "documents": [doc.page_content for doc in rag_result.documents[:5]],
                    "scores": rag_result.scores[:5]
                }
                logger.info(f"Retrieved {len(rag_result.documents)} documents")

            # Merge RAG context with request context
            if rag_context:
                request.context = {**(request.context or {}), "rag": rag_context}

            # Execute based on mode
            if mode == ExecutionMode.REACT:
                result = await self._execute_react(request)
            elif mode == ExecutionMode.LANGGRAPH:
                result = await self._execute_langgraph(request)
            else:
                result = await self._execute_direct(request)

            execution_time = time.time() - start_time

            # Calculate cost
            total_cost = self._calculate_cost(result, execution_time)
            self.total_cost += total_cost

            # Cache result if successful
            if request.use_cache and result.get("success"):
                cache_key = self._generate_cache_key(request)
                await self.prompt_cache.set(
                    key=cache_key,
                    value=result.get("answer", result),
                    cost=total_cost
                )

            # Record metrics
            if self.metrics_collector:
                self.metrics_collector.record_request(
                    model=result.get("model", "unknown"),
                    operation=mode.value,
                    duration=execution_time,
                    success=result.get("success", True),
                    cost=total_cost
                )

            return OrchestrationResponse(
                success=result.get("success", True),
                result=result.get("answer", result),
                execution_time=execution_time,
                model_used=result.get("model", "unknown"),
                total_cost=total_cost,
                cached=False,
                metadata=result.get("metadata", {}),
                traces=result.get("traces")
            )

        except Exception as e:
            logger.error(f"Execution error: {str(e)}")

            # Record error metrics
            if self.metrics_collector:
                self.metrics_collector.record_request(
                    model="error",
                    operation=mode.value,
                    duration=time.time() - start_time,
                    success=False,
                    cost=0.0
                )

            return OrchestrationResponse(
                success=False,
                result=None,
                execution_time=time.time() - start_time,
                model_used="error",
                total_cost=0.0,
                cached=False,
                metadata={"error": str(e)}
            )

    async def _execute_streaming(
        self,
        request: OrchestrationRequest,
        mode: ExecutionMode,
        start_time: float
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute request with streaming results."""

        # Initial metadata
        yield {
            "type": "start",
            "mode": mode.value,
            "timestamp": start_time
        }

        try:
            # RAG retrieval if needed
            if request.rag_query and self.rag_system:
                rag_result = await self.rag_system.retrieve(
                    query=request.rag_query,
                    strategy=RetrievalStrategy.HYBRID
                )
                yield {
                    "type": "rag",
                    "documents": len(rag_result.documents),
                    "retrieval_time": rag_result.retrieval_time
                }

            # Stream based on mode
            if mode == ExecutionMode.REACT:
                async for chunk in self._stream_react(request):
                    yield chunk
            else:
                # For non-streaming modes, execute and yield result
                result = await self._execute_complete(request, mode, start_time)
                yield {
                    "type": "complete",
                    "result": result.result,
                    "metadata": result.metadata
                }

        except Exception as e:
            yield {
                "type": "error",
                "error": str(e),
                "timestamp": time.time()
            }

    async def _determine_execution_mode(self, request: OrchestrationRequest) -> ExecutionMode:
        """Determine the best execution mode for the request."""

        if request.mode != ExecutionMode.AUTO:
            return request.mode

        # Analyze task complexity
        task_profile = TaskProfile(
            prompt_tokens=len(request.task) // 4,
            expected_output_tokens=500,
            requires_tools=bool(request.tools),
            requires_reasoning="explain" in request.task.lower() or "analyze" in request.task.lower()
        )

        complexity = self.model_router.analyze_task_complexity(task_profile)

        # Select mode based on complexity
        if complexity in [TaskComplexity.COMPLEX, TaskComplexity.CRITICAL]:
            return ExecutionMode.LANGGRAPH
        elif task_profile.requires_tools:
            return ExecutionMode.REACT
        else:
            return ExecutionMode.DIRECT

    async def _execute_react(self, request: OrchestrationRequest) -> Dict[str, Any]:
        """Execute using ReAct agent."""

        if not self.react_agent:
            tool_executor = SimpleToolExecutor(request.tools or self.tools)
            self.react_agent = ReActAgent(
                model_router=self.model_router,
                tool_executor=tool_executor
            )

        return await self.react_agent.run(
            task=request.task,
            context=request.context,
            stream=False
        )

    async def _stream_react(self, request: OrchestrationRequest) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream ReAct agent execution."""

        if not self.react_agent:
            tool_executor = SimpleToolExecutor(request.tools or self.tools)
            self.react_agent = ReActAgent(
                model_router=self.model_router,
                tool_executor=tool_executor
            )

        async for chunk in self.react_agent.run(
            task=request.task,
            context=request.context,
            stream=True
        ):
            yield chunk

    async def _execute_langgraph(self, request: OrchestrationRequest) -> Dict[str, Any]:
        """Execute using LangGraph orchestrator."""

        if not self.langgraph_orchestrator:
            self.langgraph_orchestrator = LangGraphOrchestrator(
                model_router=self.model_router,
                tools=request.tools or self.tools,
                config=WorkflowConfig()
            )

        return await self.langgraph_orchestrator.run(
            task=request.task,
            context=request.context
        )

    async def _execute_direct(self, request: OrchestrationRequest) -> Dict[str, Any]:
        """Execute directly with selected model."""

        # Create task profile
        task_profile = TaskProfile(
            prompt_tokens=len(request.task) // 4,
            expected_output_tokens=500,
            budget_constraint=request.budget_limit
        )

        # Select model
        model_id, routing_metadata = self.model_router.select_model(task_profile)

        # Here you would call the actual model API
        # For now, return placeholder
        return {
            "success": True,
            "answer": f"Direct execution result for: {request.task}",
            "model": model_id,
            "metadata": routing_metadata
        }

    def _generate_cache_key(self, request: OrchestrationRequest) -> str:
        """Generate cache key for request."""

        # Create deterministic key from request
        key_parts = [
            request.task,
            str(request.mode.value),
            json.dumps(request.context or {}, sort_keys=True)
        ]

        key_string = "|".join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()

    def _calculate_cost(self, result: Dict[str, Any], execution_time: float) -> float:
        """Calculate execution cost."""

        # Placeholder calculation
        # In production, calculate based on actual token usage
        base_cost = 0.001  # Base cost per request
        time_cost = execution_time * 0.0001  # Time-based cost

        return base_cost + time_cost

    async def get_statistics(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""

        stats = {
            "total_requests": self.total_requests,
            "total_cost": self.total_cost,
            "cache_stats": self.prompt_cache.get_stats() if self.prompt_cache else {},
            "routing_stats": self.model_router.get_routing_stats(),
        }

        if self.metrics_collector:
            stats["performance_report"] = self.metrics_collector.get_performance_report()

        return stats

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all components."""

        health = {
            "status": "healthy",
            "components": {}
        }

        # Check Redis
        if self.redis_client:
            try:
                await self.redis_client.ping()
                health["components"]["redis"] = "healthy"
            except:
                health["components"]["redis"] = "unhealthy"
                health["status"] = "degraded"

        # Check RAG system
        if self.rag_system:
            health["components"]["rag"] = "healthy"

        # Check metrics
        if self.metrics_collector:
            alerts = self.metrics_collector.check_alerts()
            health["components"]["metrics"] = "healthy"
            if alerts:
                health["alerts"] = alerts
                health["status"] = "warning" if health["status"] == "healthy" else health["status"]

        return health


class SimpleToolExecutor(ToolExecutor):
    """Simple implementation of tool executor."""

    def __init__(self, tools: List[BaseTool]):
        self.tools = {tool.name: tool for tool in tools}

    async def execute(self, tool_name: str, tool_input: Dict[str, Any]) -> Any:
        """Execute a tool."""
        if tool_name not in self.tools:
            return f"Tool {tool_name} not found"

        tool = self.tools[tool_name]
        return await tool.ainvoke(tool_input)

    def get_available_tools(self) -> List[BaseTool]:
        """Get available tools."""
        return list(self.tools.values())