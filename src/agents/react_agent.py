"""ReAct agent implementation with tool use and self-reflection."""

from typing import Dict, Any, Optional, List, Callable, Union, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import json
import time
from abc import ABC, abstractmethod

from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.callbacks.base import AsyncCallbackHandler
from langchain.tools import BaseTool, StructuredTool
from langchain_core.language_models import BaseLLM
from pydantic import BaseModel, Field as PydanticField

from ..models.router import ModelRouter, TaskProfile
from ..utils.logging import get_logger

logger = get_logger(__name__)


class ReActStep(str, Enum):
    """Steps in the ReAct reasoning process."""
    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    REFLECTION = "reflection"
    ANSWER = "answer"


@dataclass
class ReActTrace:
    """Trace of a ReAct execution."""

    step_type: ReActStep
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReflectionResult:
    """Result of agent self-reflection."""

    quality_score: float  # 0.0 to 1.0
    issues: List[str]
    improvements: List[str]
    should_retry: bool
    suggested_changes: Dict[str, Any]


class ToolExecutor(ABC):
    """Abstract base class for tool execution."""

    @abstractmethod
    async def execute(self, tool_name: str, tool_input: Dict[str, Any]) -> Any:
        """Execute a tool with given input."""
        pass

    @abstractmethod
    def get_available_tools(self) -> List[BaseTool]:
        """Get list of available tools."""
        pass


class ReActAgent:
    """
    ReAct agent with extended thinking, tool use, and self-reflection.

    Implements the Reasoning and Acting (ReAct) pattern with:
    - Multi-step reasoning
    - Tool execution
    - Self-reflection and error correction
    - Quality scoring
    """

    def __init__(
        self,
        model_router: ModelRouter,
        tool_executor: ToolExecutor,
        max_iterations: int = 10,
        max_reflection_attempts: int = 3,
        enable_extended_thinking: bool = True,
        memory_window: int = 10
    ):
        self.model_router = model_router
        self.tool_executor = tool_executor
        self.max_iterations = max_iterations
        self.max_reflection_attempts = max_reflection_attempts
        self.enable_extended_thinking = enable_extended_thinking
        self.memory = ConversationBufferWindowMemory(k=memory_window)
        self.execution_traces: List[ReActTrace] = []

    async def run(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        stream: bool = False
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """
        Execute the ReAct agent on a task.

        Args:
            task: The task to execute
            context: Additional context for the task
            stream: Whether to stream intermediate results

        Returns:
            Final result or async generator of intermediate results
        """

        if stream:
            return self._run_streaming(task, context)
        else:
            return await self._run_complete(task, context)

    async def _run_complete(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Run agent to completion."""

        start_time = time.time()
        self.execution_traces = []

        # Analyze task and select model
        task_profile = self._analyze_task(task, context)
        model_id, routing_metadata = self.model_router.select_model(task_profile)

        logger.info(f"Starting ReAct execution with model: {model_id}")

        # Initialize conversation
        messages = self._build_initial_messages(task, context)

        # Main execution loop
        iteration = 0
        reflection_attempts = 0
        final_answer = None

        while iteration < self.max_iterations:
            iteration += 1

            try:
                # Step 1: Generate thought
                thought = await self._generate_thought(
                    messages, model_id, task_profile
                )
                self._add_trace(ReActStep.THOUGHT, thought)

                # Check if thought contains final answer
                if self._is_final_answer(thought):
                    final_answer = self._extract_final_answer(thought)
                    self._add_trace(ReActStep.ANSWER, final_answer)
                    break

                # Step 2: Generate action
                action = await self._generate_action(
                    messages, thought, model_id, task_profile
                )
                self._add_trace(ReActStep.ACTION, json.dumps(action))

                # Step 3: Execute action
                observation = await self._execute_action(action)
                self._add_trace(ReActStep.OBSERVATION, str(observation))

                # Update messages with thought-action-observation
                messages.append(AIMessage(content=f"Thought: {thought}"))
                messages.append(AIMessage(content=f"Action: {json.dumps(action)}"))
                messages.append(HumanMessage(content=f"Observation: {observation}"))

                # Step 4: Periodic reflection
                if iteration % 3 == 0 and self._should_reflect():
                    reflection = await self._reflect_on_progress(
                        messages, model_id, task_profile
                    )
                    self._add_trace(ReActStep.REFLECTION, json.dumps(reflection.__dict__))

                    if reflection.should_retry and reflection_attempts < self.max_reflection_attempts:
                        reflection_attempts += 1
                        # Apply suggested changes
                        messages = self._apply_reflection_changes(
                            messages, reflection
                        )

            except Exception as e:
                logger.error(f"Error in iteration {iteration}: {str(e)}")
                # Add error to context for next iteration
                messages.append(HumanMessage(
                    content=f"Error occurred: {str(e)}. Please adjust your approach."
                ))

        # If no final answer after max iterations, generate one
        if final_answer is None:
            final_answer = await self._generate_final_answer(
                messages, model_id, task_profile
            )
            self._add_trace(ReActStep.ANSWER, final_answer)

        # Calculate execution metrics
        execution_time = time.time() - start_time
        quality_score = await self._evaluate_quality(final_answer, task)

        # Update router metrics
        self.model_router.update_metrics(
            model_id=model_id,
            latency=execution_time,
            success=final_answer is not None,
            cost=self._calculate_cost(task_profile, model_id),
            quality_score=quality_score
        )

        return {
            "answer": final_answer,
            "model": model_id,
            "iterations": iteration,
            "execution_time": execution_time,
            "quality_score": quality_score,
            "traces": [
                {
                    "type": t.step_type.value,
                    "content": t.content,
                    "timestamp": t.timestamp
                }
                for t in self.execution_traces
            ],
            "routing_metadata": routing_metadata
        }

    async def _run_streaming(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Run agent with streaming intermediate results."""

        start_time = time.time()
        self.execution_traces = []

        # Analyze task and select model
        task_profile = self._analyze_task(task, context)
        model_id, routing_metadata = self.model_router.select_model(task_profile)

        # Yield initial metadata
        yield {
            "type": "metadata",
            "model": model_id,
            "routing": routing_metadata
        }

        # Initialize conversation
        messages = self._build_initial_messages(task, context)
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1

            # Generate and stream thought
            thought = await self._generate_thought(messages, model_id, task_profile)
            self._add_trace(ReActStep.THOUGHT, thought)
            yield {
                "type": "thought",
                "iteration": iteration,
                "content": thought
            }

            # Check for final answer
            if self._is_final_answer(thought):
                final_answer = self._extract_final_answer(thought)
                self._add_trace(ReActStep.ANSWER, final_answer)
                yield {
                    "type": "answer",
                    "content": final_answer,
                    "execution_time": time.time() - start_time
                }
                return

            # Generate and stream action
            action = await self._generate_action(messages, thought, model_id, task_profile)
            self._add_trace(ReActStep.ACTION, json.dumps(action))
            yield {
                "type": "action",
                "iteration": iteration,
                "content": action
            }

            # Execute and stream observation
            observation = await self._execute_action(action)
            self._add_trace(ReActStep.OBSERVATION, str(observation))
            yield {
                "type": "observation",
                "iteration": iteration,
                "content": str(observation)
            }

            # Update messages
            messages.append(AIMessage(content=f"Thought: {thought}"))
            messages.append(AIMessage(content=f"Action: {json.dumps(action)}"))
            messages.append(HumanMessage(content=f"Observation: {observation}"))

    def _analyze_task(self, task: str, context: Optional[Dict[str, Any]]) -> TaskProfile:
        """Analyze task to create task profile."""

        # Estimate token counts (rough approximation)
        prompt_tokens = len(task) // 4  # Rough token estimate
        if context:
            prompt_tokens += len(json.dumps(context)) // 4

        # Detect requirements
        requires_tools = any(
            keyword in task.lower()
            for keyword in ["search", "calculate", "fetch", "query", "execute"]
        )
        requires_reasoning = any(
            keyword in task.lower()
            for keyword in ["explain", "analyze", "compare", "evaluate", "reason"]
        )

        return TaskProfile(
            prompt_tokens=prompt_tokens,
            expected_output_tokens=500,  # Default estimate
            requires_tools=requires_tools,
            requires_reasoning=requires_reasoning,
            requires_vision=False,
            requires_realtime=False
        )

    def _build_initial_messages(
        self,
        task: str,
        context: Optional[Dict[str, Any]]
    ) -> List[BaseMessage]:
        """Build initial messages for the conversation."""

        system_prompt = """You are a ReAct agent that solves problems through reasoning and acting.

For each step, you should:
1. THOUGHT: Analyze the current situation and plan your next action
2. ACTION: Specify a tool to use with its parameters
3. OBSERVATION: Receive the result of your action

Available tools:
{tools}

When you have enough information to answer the task, start your thought with "FINAL ANSWER:" followed by your response.

Format your actions as JSON:
{{"tool": "tool_name", "parameters": {{"param1": "value1"}}}}

Think step by step and be thorough in your reasoning."""

        tools_description = "\n".join([
            f"- {tool.name}: {tool.description}"
            for tool in self.tool_executor.get_available_tools()
        ])

        messages = [
            SystemMessage(content=system_prompt.format(tools=tools_description)),
            HumanMessage(content=f"Task: {task}")
        ]

        if context:
            messages.append(HumanMessage(
                content=f"Additional context: {json.dumps(context, indent=2)}"
            ))

        return messages

    async def _generate_thought(
        self,
        messages: List[BaseMessage],
        model_id: str,
        task_profile: TaskProfile
    ) -> str:
        """Generate a reasoning thought."""

        # This would integrate with actual LLM
        # For now, returning placeholder
        prompt = "Based on the conversation so far, what is your next thought? " \
                "If you have enough information, start with 'FINAL ANSWER:'"

        # In production, this would call the actual model
        # through LangChain or direct API
        return f"Analyzing the task requirements and available information..."

    async def _generate_action(
        self,
        messages: List[BaseMessage],
        thought: str,
        model_id: str,
        task_profile: TaskProfile
    ) -> Dict[str, Any]:
        """Generate an action based on thought."""

        # This would integrate with actual LLM
        # For now, returning placeholder
        return {
            "tool": "search",
            "parameters": {"query": "relevant information"}
        }

    async def _execute_action(self, action: Dict[str, Any]) -> Any:
        """Execute the specified action."""

        tool_name = action.get("tool")
        parameters = action.get("parameters", {})

        try:
            result = await self.tool_executor.execute(tool_name, parameters)
            return result
        except Exception as e:
            logger.error(f"Tool execution failed: {str(e)}")
            return f"Error executing tool: {str(e)}"

    async def _reflect_on_progress(
        self,
        messages: List[BaseMessage],
        model_id: str,
        task_profile: TaskProfile
    ) -> ReflectionResult:
        """Reflect on progress and suggest improvements."""

        # Analyze execution traces
        thought_count = sum(1 for t in self.execution_traces if t.step_type == ReActStep.THOUGHT)
        action_count = sum(1 for t in self.execution_traces if t.step_type == ReActStep.ACTION)

        # Simple heuristic-based reflection
        issues = []
        improvements = []

        if action_count > thought_count * 2:
            issues.append("Too many actions without sufficient reasoning")
            improvements.append("Spend more time analyzing before acting")

        if len(self.execution_traces) > 15:
            issues.append("Taking too long to reach conclusion")
            improvements.append("Focus on most relevant information")

        quality_score = 0.7  # Placeholder
        should_retry = len(issues) > 2

        return ReflectionResult(
            quality_score=quality_score,
            issues=issues,
            improvements=improvements,
            should_retry=should_retry,
            suggested_changes={"focus": "efficiency"}
        )

    def _apply_reflection_changes(
        self,
        messages: List[BaseMessage],
        reflection: ReflectionResult
    ) -> List[BaseMessage]:
        """Apply changes suggested by reflection."""

        # Add reflection guidance to messages
        reflection_message = f"Based on reflection:\n"
        reflection_message += f"Issues identified: {', '.join(reflection.issues)}\n"
        reflection_message += f"Suggested improvements: {', '.join(reflection.improvements)}"

        messages.append(SystemMessage(content=reflection_message))
        return messages

    async def _generate_final_answer(
        self,
        messages: List[BaseMessage],
        model_id: str,
        task_profile: TaskProfile
    ) -> str:
        """Generate final answer after max iterations."""

        # This would integrate with actual LLM
        return "Based on the analysis, here is the final answer..."

    async def _evaluate_quality(self, answer: str, task: str) -> float:
        """Evaluate quality of the final answer."""

        # Simple heuristic evaluation
        # In production, this could use another LLM for evaluation
        if answer and len(answer) > 50:
            return 0.8
        return 0.5

    def _calculate_cost(self, task_profile: TaskProfile, model_id: str) -> float:
        """Calculate cost of execution."""

        # This would calculate actual token usage and cost
        return 0.01  # Placeholder

    def _is_final_answer(self, thought: str) -> bool:
        """Check if thought contains final answer."""
        return "FINAL ANSWER:" in thought

    def _extract_final_answer(self, thought: str) -> str:
        """Extract final answer from thought."""
        if "FINAL ANSWER:" in thought:
            return thought.split("FINAL ANSWER:")[1].strip()
        return thought

    def _should_reflect(self) -> bool:
        """Determine if reflection is needed."""
        # Reflect if many steps without progress
        return len(self.execution_traces) > 5

    def _add_trace(self, step_type: ReActStep, content: str):
        """Add trace to execution history."""
        self.execution_traces.append(ReActTrace(
            step_type=step_type,
            content=content
        ))