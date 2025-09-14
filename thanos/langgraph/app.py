import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import httpx
import redis
import json
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

# Metrics
REQUEST_COUNT = Counter('langgraph_requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('langgraph_request_duration_seconds', 'Request duration')

app = FastAPI(title="LangGraph Thanos API", version="1.0.0")

# Configuration
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
QDRANT_URL = os.getenv("QDRANT_URL", "http://100.96.197.84:6333")
EMBEDDING_SERVICE_URL = os.getenv("EMBEDDING_SERVICE_URL", "http://localhost:8001")
REDIS_URL = os.getenv("REDIS_URL", "redis://:thanos123!@redis:6379")

# Initialize Redis
try:
    redis_client = redis.from_url(REDIS_URL)
    redis_client.ping()
except Exception as e:
    print(f"Redis connection failed: {e}")
    redis_client = None

class ChatRequest(BaseModel):
    message: str
    model: Optional[str] = "phi-3.5-mini"
    max_tokens: Optional[int] = 1000
    temperature: Optional[float] = 0.7

class EmbeddingRequest(BaseModel):
    text: str
    model: Optional[str] = "sentence-transformers/all-MiniLM-L6-v2"

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    REQUEST_COUNT.labels(method="GET", endpoint="/health").inc()

    status = {"status": "healthy", "services": {}}

    # Check vLLM
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{VLLM_BASE_URL}/models", timeout=5.0)
            status["services"]["vllm"] = "healthy" if response.status_code == 200 else "unhealthy"
    except:
        status["services"]["vllm"] = "unhealthy"

    # Check Qdrant
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{QDRANT_URL}/", timeout=5.0)
            status["services"]["qdrant"] = "healthy" if response.status_code == 200 else "unhealthy"
    except:
        status["services"]["qdrant"] = "unhealthy"

    # Check Redis
    if redis_client:
        try:
            redis_client.ping()
            status["services"]["redis"] = "healthy"
        except:
            status["services"]["redis"] = "unhealthy"
    else:
        status["services"]["redis"] = "unhealthy"

    return status

@app.post("/chat")
async def chat_completion(request: ChatRequest):
    """Chat completion using vLLM"""
    REQUEST_COUNT.labels(method="POST", endpoint="/chat").inc()

    with REQUEST_DURATION.time():
        try:
            payload = {
                "model": request.model,
                "messages": [{"role": "user", "content": request.message}],
                "max_tokens": request.max_tokens,
                "temperature": request.temperature
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{VLLM_BASE_URL}/chat/completions",
                    json=payload,
                    timeout=60.0
                )

                if response.status_code != 200:
                    raise HTTPException(status_code=response.status_code, detail=response.text)

                result = response.json()

                # Cache result if Redis is available
                if redis_client:
                    cache_key = f"chat:{hash(request.message)}:{request.model}"
                    redis_client.setex(cache_key, 3600, json.dumps(result))

                return result

        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="vLLM service timeout")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/embed")
async def get_embedding(request: EmbeddingRequest):
    """Generate text embeddings"""
    REQUEST_COUNT.labels(method="POST", endpoint="/embed").inc()

    with REQUEST_DURATION.time():
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{EMBEDDING_SERVICE_URL}/embed",
                    json={"inputs": request.text},
                    timeout=30.0
                )

                if response.status_code != 200:
                    raise HTTPException(status_code=response.status_code, detail=response.text)

                return response.json()

        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Embedding service timeout")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/qdrant/collections")
async def list_collections():
    """List Qdrant collections"""
    REQUEST_COUNT.labels(method="GET", endpoint="/qdrant/collections").inc()

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{QDRANT_URL}/collections", timeout=10.0)

            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=response.text)

            return response.json()

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Qdrant service timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "LangGraph Thanos API",
        "version": "1.0.0",
        "node": "thanos",
        "gpu_enabled": True,
        "endpoints": {
            "health": "/health",
            "chat": "/chat",
            "embed": "/embed",
            "collections": "/qdrant/collections",
            "metrics": "/metrics"
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        log_level="info"
    )