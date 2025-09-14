"""FastAPI server for the Agentic Orchestrator."""

from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import asyncio
import json
import time
from contextlib import asynccontextmanager

from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from .orchestrator import (
    AgenticOrchestrator,
    OrchestrationRequest,
    ExecutionMode
)
from .config import settings
from .utils.logging import get_logger, configure_logging

logger = get_logger(__name__)


# Request/Response models
class APIRequest(BaseModel):
    """API request model."""

    task: str = Field(..., description="The task to execute")
    mode: str = Field(default="auto", description="Execution mode: auto, react, langgraph, direct")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")
    rag_query: Optional[str] = Field(default=None, description="RAG retrieval query")
    stream: bool = Field(default=False, description="Enable streaming response")
    use_cache: bool = Field(default=True, description="Use caching")
    priority: int = Field(default=0, description="Request priority (0-10)")
    budget_limit: Optional[float] = Field(default=None, description="Maximum cost budget")


class APIResponse(BaseModel):
    """API response model."""

    success: bool
    result: Optional[Any]
    execution_time: float
    model_used: str
    total_cost: float
    cached: bool
    metadata: Dict[str, Any]
    request_id: Optional[str] = None


# Global orchestrator instance
orchestrator: Optional[AgenticOrchestrator] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global orchestrator

    # Startup
    configure_logging(settings.monitoring.log_level)
    logger.info("Starting Agentic Orchestrator API")

    # Initialize orchestrator
    orchestrator = AgenticOrchestrator(
        redis_url=settings.cache.redis_url,
        enable_monitoring=True
    )

    # Start background tasks
    asyncio.create_task(periodic_health_check())
    asyncio.create_task(metrics_reporter())

    yield

    # Shutdown
    logger.info("Shutting down Agentic Orchestrator API")


# Create FastAPI app
app = FastAPI(
    title="Agentic Orchestrator API",
    description="Production-ready multi-model agentic orchestration framework",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint
@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")

    health = await orchestrator.health_check()
    status_code = 200 if health["status"] == "healthy" else 503

    return health


# Metrics endpoint
@app.get("/metrics", response_class=Response)
async def metrics():
    """Prometheus metrics endpoint."""
    if not orchestrator or not orchestrator.metrics_collector:
        raise HTTPException(status_code=503, detail="Metrics not available")

    metrics_data = orchestrator.metrics_collector.get_prometheus_metrics()
    return Response(content=metrics_data, media_type=CONTENT_TYPE_LATEST)


# Statistics endpoint
@app.get("/stats")
async def get_statistics() -> Dict[str, Any]:
    """Get orchestrator statistics."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")

    return await orchestrator.get_statistics()


# Main execution endpoint
@app.post("/execute", response_model=APIResponse)
async def execute_task(
    request: APIRequest,
    background_tasks: BackgroundTasks
) -> APIResponse:
    """Execute an orchestration task."""

    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")

    request_id = f"req_{int(time.time() * 1000)}"
    logger.info(f"Received request {request_id}: {request.task[:100]}...")

    try:
        # Convert API request to orchestration request
        orchestration_request = OrchestrationRequest(
            task=request.task,
            mode=ExecutionMode(request.mode.lower()),
            context=request.context,
            rag_query=request.rag_query,
            stream=False,  # Streaming handled separately
            use_cache=request.use_cache,
            priority=request.priority,
            budget_limit=request.budget_limit
        )

        # Execute request
        response = await orchestrator.execute(orchestration_request)

        # Log completion
        logger.info(
            f"Request {request_id} completed",
            request_id=request_id,
            success=response.success,
            model=response.model_used,
            cost=response.total_cost,
            time=response.execution_time
        )

        return APIResponse(
            success=response.success,
            result=response.result,
            execution_time=response.execution_time,
            model_used=response.model_used,
            total_cost=response.total_cost,
            cached=response.cached,
            metadata=response.metadata,
            request_id=request_id
        )

    except Exception as e:
        logger.error(f"Request {request_id} failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Streaming endpoint
@app.post("/execute/stream")
async def execute_task_stream(request: APIRequest):
    """Execute task with streaming response."""

    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")

    async def generate():
        try:
            orchestration_request = OrchestrationRequest(
                task=request.task,
                mode=ExecutionMode(request.mode.lower()),
                context=request.context,
                rag_query=request.rag_query,
                stream=True,
                use_cache=request.use_cache,
                priority=request.priority,
                budget_limit=request.budget_limit
            )

            async for chunk in await orchestrator.execute(orchestration_request):
                yield f"data: {json.dumps(chunk)}\n\n"

        except Exception as e:
            error_chunk = {"type": "error", "error": str(e)}
            yield f"data: {json.dumps(error_chunk)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# WebSocket endpoint for real-time interaction
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time interaction."""

    await websocket.accept()
    logger.info("WebSocket connection established")

    try:
        while True:
            # Receive request
            data = await websocket.receive_json()

            # Process request
            request = APIRequest(**data)
            orchestration_request = OrchestrationRequest(
                task=request.task,
                mode=ExecutionMode(request.mode.lower()),
                context=request.context,
                rag_query=request.rag_query,
                stream=True,
                use_cache=request.use_cache,
                priority=request.priority,
                budget_limit=request.budget_limit
            )

            # Stream results
            async for chunk in await orchestrator.execute(orchestration_request):
                await websocket.send_json(chunk)

                # Check if this is the final chunk
                if chunk.get("type") == "answer" or chunk.get("type") == "complete":
                    break

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        await websocket.send_json({"type": "error", "error": str(e)})
        await websocket.close()


# Batch execution endpoint
@app.post("/execute/batch")
async def execute_batch(requests: List[APIRequest]) -> List[APIResponse]:
    """Execute multiple tasks in batch."""

    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")

    async def process_request(req: APIRequest) -> APIResponse:
        orchestration_request = OrchestrationRequest(
            task=req.task,
            mode=ExecutionMode(req.mode.lower()),
            context=req.context,
            rag_query=req.rag_query,
            stream=False,
            use_cache=req.use_cache,
            priority=req.priority,
            budget_limit=req.budget_limit
        )

        response = await orchestrator.execute(orchestration_request)

        return APIResponse(
            success=response.success,
            result=response.result,
            execution_time=response.execution_time,
            model_used=response.model_used,
            total_cost=response.total_cost,
            cached=response.cached,
            metadata=response.metadata
        )

    # Process all requests in parallel
    responses = await asyncio.gather(
        *[process_request(req) for req in requests],
        return_exceptions=True
    )

    # Handle any exceptions
    final_responses = []
    for i, response in enumerate(responses):
        if isinstance(response, Exception):
            final_responses.append(APIResponse(
                success=False,
                result=None,
                execution_time=0.0,
                model_used="error",
                total_cost=0.0,
                cached=False,
                metadata={"error": str(response)}
            ))
        else:
            final_responses.append(response)

    return final_responses


# Cache management endpoints
@app.post("/cache/clear")
async def clear_cache(pattern: Optional[str] = None) -> Dict[str, str]:
    """Clear cache entries."""

    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")

    await orchestrator.prompt_cache.invalidate(pattern)
    return {"status": "success", "message": f"Cache cleared for pattern: {pattern or 'all'}"}


@app.get("/cache/stats")
async def cache_stats() -> Dict[str, Any]:
    """Get cache statistics."""

    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")

    return orchestrator.prompt_cache.get_stats()


# Model routing endpoints
@app.get("/models")
async def list_models() -> Dict[str, Any]:
    """List available models and their specifications."""

    return {
        "models": settings.model.model_specs,
        "routing_rules": {
            k.value: v for k, v in settings.routing_rules.items()
        }
    }


@app.get("/models/stats")
async def model_stats() -> Dict[str, Any]:
    """Get model usage statistics."""

    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")

    return orchestrator.model_router.get_routing_stats()


# Background tasks
async def periodic_health_check():
    """Periodic health check task."""
    while True:
        await asyncio.sleep(60)  # Check every minute
        if orchestrator:
            health = await orchestrator.health_check()
            if health["status"] != "healthy":
                logger.warning(f"Health check failed: {health}")


async def metrics_reporter():
    """Periodic metrics reporting."""
    while True:
        await asyncio.sleep(300)  # Report every 5 minutes
        if orchestrator:
            stats = await orchestrator.get_statistics()
            logger.info(
                "Metrics report",
                total_requests=stats["total_requests"],
                total_cost=stats["total_cost"],
                cache_hit_rate=stats["cache_stats"].get("hit_rate", 0)
            )


# Error handlers
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return {
        "success": False,
        "error": str(exc),
        "type": type(exc).__name__
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                },
            },
            "root": {
                "level": "INFO",
                "handlers": ["default"],
            },
        }
    )