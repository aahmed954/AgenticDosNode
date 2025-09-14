"""Main proxy server implementation with OpenAI-compatible API endpoints."""

import asyncio
import json
import time
import uuid
from typing import Dict, Any, Optional, AsyncGenerator
import httpx
import uvicorn
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GzipMiddleware
from contextlib import asynccontextmanager
import logging

from .proxy_config import config, ProxyConfig
from .models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    ErrorResponse,
    HealthStatus,
    ProxyStats
)
from .transformers import RequestTransformer, ResponseTransformer, ModelMapper
from .rate_limiter import AdaptiveRateLimiter
from .auth import ProxyAuth, AuthMiddleware
from .monitoring import ProxyMonitor, RequestMetrics, get_monitor


# Configure logging
logging.basicConfig(
    level=getattr(logging, config.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ClaudeProxyServer:
    """Claude Code Proxy Server with OpenAI compatibility."""

    def __init__(self, config: ProxyConfig):
        self.config = config
        self.rate_limiter = AdaptiveRateLimiter(
            requests_per_minute=config.rate_limit_requests_per_minute,
            tokens_per_minute=config.rate_limit_tokens_per_minute,
            concurrent_requests=config.rate_limit_concurrent_requests
        )
        self.auth_handler = ProxyAuth(
            auth_method=config.auth_method,
            api_keys=config.api_keys,
            bearer_token=config.bearer_token,
            jwt_secret=getattr(config, 'jwt_secret', None)
        )

        # HTTP client for Claude API
        self.http_client = None
        self.start_time = time.time()

        # Initialize monitoring
        self.monitor = ProxyMonitor(config)

        # Statistics
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_response_time": 0.0,
            "current_active_requests": 0
        }

    async def startup(self):
        """Initialize the proxy server."""
        logger.info("Starting Claude Code Proxy Server")

        # Initialize HTTP client with proper configuration
        self.http_client = httpx.AsyncClient(
            base_url=self.config.claude_api_base,
            timeout=httpx.Timeout(self.config.timeout_seconds),
            headers={
                "anthropic-version": self.config.claude_version,
                "x-api-key": self.config.claude_api_key,
                "content-type": "application/json"
            },
            limits=httpx.Limits(
                max_connections=20,
                max_keepalive_connections=10
            )
        )

        logger.info(f"Proxy server initialized on {self.config.host}:{self.config.port}")

    async def shutdown(self):
        """Cleanup resources."""
        logger.info("Shutting down Claude Code Proxy Server")
        if self.http_client:
            await self.http_client.aclose()

    async def chat_completions(self, request: ChatCompletionRequest, identity: str) -> Any:
        """Handle OpenAI-compatible chat completions."""
        request_start = time.time()
        self.stats["current_active_requests"] += 1

        try:
            # Map model name
            claude_model = ModelMapper.map_openai_to_claude(
                request.model, self.config.openai_model_mapping
            )

            # Validate model support
            if claude_model not in self.config.supported_models:
                raise HTTPException(
                    status_code=400,
                    detail=f"Model '{request.model}' is not supported"
                )

            # Estimate token count for rate limiting
            estimated_tokens = self._estimate_tokens(request)

            # Check rate limits
            rate_allowed, rate_reason = await self.rate_limiter.check_rate_limit(
                api_key=identity,
                estimated_tokens=estimated_tokens
            )

            if not rate_allowed:
                self.stats["failed_requests"] += 1
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded: {rate_reason}"
                )

            # Acquire concurrency slot
            await self.rate_limiter.acquire_concurrency_slot()

            try:
                # Transform request to Claude format
                claude_request = RequestTransformer.openai_to_claude(request)
                claude_request.model = claude_model

                if request.stream:
                    return await self._handle_streaming_request(
                        claude_request, request, request_start
                    )
                else:
                    return await self._handle_non_streaming_request(
                        claude_request, request, request_start
                    )

            finally:
                await self.rate_limiter.release_concurrency_slot()

        except HTTPException:
            raise
        except Exception as e:
            self.stats["failed_requests"] += 1
            logger.error(f"Unexpected error in chat_completions: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            self.stats["current_active_requests"] -= 1

    async def _handle_non_streaming_request(
        self,
        claude_request,
        original_request: ChatCompletionRequest,
        request_start: float
    ) -> ChatCompletionResponse:
        """Handle non-streaming chat completion request."""
        try:
            # Make request to Claude API
            response = await self.http_client.post(
                "/v1/messages",
                json=claude_request.dict(exclude_none=True)
            )

            response_time = time.time() - request_start
            self.stats["total_response_time"] += response_time

            if response.status_code != 200:
                error_detail = await self._handle_claude_error(response)
                self.stats["failed_requests"] += 1
                await self.rate_limiter.record_response(False, response_time)
                raise HTTPException(
                    status_code=response.status_code,
                    detail=error_detail
                )

            # Parse Claude response
            claude_response_data = response.json()

            # Transform to OpenAI format
            openai_response = ResponseTransformer.claude_to_openai(
                claude_response_data,
                original_request
            )

            # Record metrics
            await self._record_request_metrics(
                original_request, openai_response, response_time, True
            )

            self.stats["successful_requests"] += 1
            self.stats["total_requests"] += 1
            await self.rate_limiter.record_response(True, response_time)

            return openai_response

        except HTTPException:
            raise
        except Exception as e:
            self.stats["failed_requests"] += 1
            logger.error(f"Error in non-streaming request: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def _handle_streaming_request(
        self,
        claude_request,
        original_request: ChatCompletionRequest,
        request_start: float
    ) -> StreamingResponse:
        """Handle streaming chat completion request."""
        claude_request.stream = True

        async def stream_generator() -> AsyncGenerator[str, None]:
            request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

            try:
                async with self.http_client.stream(
                    "POST",
                    "/v1/messages",
                    json=claude_request.dict(exclude_none=True)
                ) as response:

                    if response.status_code != 200:
                        error_detail = await self._handle_claude_error(response)
                        self.stats["failed_requests"] += 1
                        error_response = {
                            "error": {
                                "message": error_detail,
                                "type": "api_error",
                                "code": str(response.status_code)
                            }
                        }
                        yield f"data: {json.dumps(error_response)}\n\n"
                        yield "data: [DONE]\n\n"
                        return

                    # Process streaming chunks
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data = line[6:]  # Remove "data: " prefix

                            if data.strip() == "[DONE]":
                                yield "data: [DONE]\n\n"
                                break

                            try:
                                chunk_data = json.loads(data)

                                # Transform chunk to OpenAI format
                                openai_chunk = ResponseTransformer.claude_stream_to_openai(
                                    chunk_data, original_request, request_id
                                )

                                if openai_chunk:
                                    yield f"data: {openai_chunk.json()}\n\n"

                            except json.JSONDecodeError:
                                logger.warning(f"Failed to parse streaming chunk: {data}")
                                continue

                    response_time = time.time() - request_start
                    self.stats["successful_requests"] += 1
                    self.stats["total_requests"] += 1
                    self.stats["total_response_time"] += response_time
                    await self.rate_limiter.record_response(True, response_time)

            except Exception as e:
                self.stats["failed_requests"] += 1
                logger.error(f"Error in streaming request: {e}")
                error_response = {
                    "error": {
                        "message": str(e),
                        "type": "api_error"
                    }
                }
                yield f"data: {json.dumps(error_response)}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            stream_generator(),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache"}
        )

    async def _handle_claude_error(self, response: httpx.Response) -> str:
        """Handle errors from Claude API."""
        try:
            error_data = response.json()
            if isinstance(error_data, dict) and "error" in error_data:
                return error_data["error"].get("message", "Unknown error")
            return str(error_data)
        except:
            return f"HTTP {response.status_code}: {response.text}"

    def _estimate_tokens(self, request: ChatCompletionRequest) -> int:
        """Estimate token count for rate limiting."""
        # Simple estimation based on character count
        total_chars = 0
        for message in request.messages:
            if isinstance(message.content, str):
                total_chars += len(message.content)
            elif isinstance(message.content, list):
                for item in message.content:
                    if isinstance(item, dict) and "text" in item:
                        total_chars += len(item["text"])

        # Rough estimation: ~4 chars per token
        return max(total_chars // 4, 10)

    async def _record_request_metrics(
        self,
        request: ChatCompletionRequest,
        response,
        response_time: float,
        success: bool
    ):
        """Record request metrics and log details."""
        # Create metrics object
        metrics = RequestMetrics(
            timestamp=datetime.now(),
            endpoint="/v1/chat/completions",
            method="POST",
            status_code=200 if success else 500,
            response_time=response_time,
            model=request.model,
            tokens_used=getattr(response, 'usage', {}).get('total_tokens') if hasattr(response, 'usage') else None,
            api_key_hash=None,  # Would need to get from request context
            error_type=None if success else "processing_error"
        )

        # Record metrics
        await self.monitor.record_request_metrics(metrics)

        # Update active request count
        self.monitor.metrics.set_active_requests(self.stats["current_active_requests"])

    async def health_check(self):
        """Get comprehensive health status."""
        return await self.monitor.get_health_status()

    async def get_metrics(self):
        """Get Prometheus metrics."""
        return await self.monitor.get_metrics_response()

    async def get_stats(self) -> ProxyStats:
        """Get proxy statistics."""
        rate_limiter_stats = await self.rate_limiter.get_stats()
        uptime = time.time() - self.start_time

        avg_response_time = (
            self.stats["total_response_time"] / max(self.stats["successful_requests"], 1)
        )

        return ProxyStats(
            total_requests=self.stats["total_requests"],
            successful_requests=self.stats["successful_requests"],
            failed_requests=self.stats["failed_requests"],
            average_response_time=avg_response_time,
            current_active_requests=self.stats["current_active_requests"],
            rate_limit_hits=rate_limiter_stats.get("rate_limited_requests", 0),
            uptime_seconds=int(uptime)
        )


# Global proxy server instance
proxy_server = ClaudeProxyServer(config)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    await proxy_server.startup()
    yield
    await proxy_server.shutdown()


# Create FastAPI application
app = FastAPI(
    title="Claude Code Proxy",
    description="OpenAI-compatible proxy for Claude API",
    version="1.0.0",
    lifespan=lifespan
)

# Add middleware
if config.enable_cors:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

app.add_middleware(GzipMiddleware, minimum_size=1000)

# Add authentication middleware
if config.auth_method != "none":
    app.add_middleware(AuthMiddleware, proxy_auth=proxy_server.auth_handler)


async def get_authenticated_identity(request: Request) -> str:
    """Dependency to get authenticated identity."""
    return getattr(request.state, "authenticated_identity", "anonymous")


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    identity: str = Depends(get_authenticated_identity)
):
    """OpenAI-compatible chat completions endpoint."""
    return await proxy_server.chat_completions(request, identity)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return await proxy_server.health_check()


@app.get("/stats")
async def get_stats():
    """Get proxy statistics."""
    return await proxy_server.get_stats()


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return await proxy_server.get_metrics()


@app.get("/v1/models")
async def list_models():
    """List supported models."""
    return {
        "object": "list",
        "data": [
            {
                "id": model,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "anthropic"
            }
            for model in config.supported_models
        ]
    }


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Claude Code Proxy API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health": "/health",
        "stats": "/stats"
    }


def run_server():
    """Run the proxy server."""
    uvicorn.run(
        "proxy_server:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
        access_log=True,
        log_level=config.log_level.lower()
    )


if __name__ == "__main__":
    run_server()