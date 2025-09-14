"""Configuration management for the orchestration framework."""

from typing import Optional, Dict, Any, List
from pydantic import Field, validator
from pydantic_settings import BaseSettings
from enum import Enum
import os


class ModelProvider(str, Enum):
    """Supported model providers."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OPENROUTER = "openrouter"
    VLLM = "vllm"
    LOCAL = "local"


class TaskComplexity(str, Enum):
    """Task complexity levels for routing."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    CRITICAL = "critical"


class ModelConfig(BaseSettings):
    """Model-specific configuration."""

    # API Keys
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openrouter_api_key: Optional[str] = Field(default=None, env="OPENROUTER_API_KEY")

    # Anthropic Proxy
    anthropic_proxy_url: Optional[str] = Field(default=None, env="ANTHROPIC_PROXY_URL")
    anthropic_proxy_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_PROXY_API_KEY")

    # vLLM Configuration
    vllm_server_url: str = Field(default="http://localhost:8000", env="VLLM_SERVER_URL")

    # Model Routing
    default_model: str = Field(default="claude-sonnet-4", env="DEFAULT_MODEL")
    complex_task_model: str = Field(default="claude-opus-4-1", env="COMPLEX_TASK_MODEL")
    simple_task_model: str = Field(default="gpt-4o-mini", env="SIMPLE_TASK_MODEL")
    local_model: str = Field(default="llama-3.1-8b", env="LOCAL_MODEL")

    # Model Specifications
    model_specs: Dict[str, Dict[str, Any]] = Field(default_factory=lambda: {
        "claude-opus-4-1": {
            "provider": ModelProvider.ANTHROPIC,
            "context_window": 1000000,
            "max_output": 8192,
            "cost_per_1k_input": 0.015,
            "cost_per_1k_output": 0.075,
            "supports_tools": True,
            "supports_vision": True,
            "supports_thinking": True,
        },
        "claude-sonnet-4": {
            "provider": ModelProvider.ANTHROPIC,
            "context_window": 200000,
            "max_output": 8192,
            "cost_per_1k_input": 0.003,
            "cost_per_1k_output": 0.015,
            "supports_tools": True,
            "supports_vision": True,
            "supports_thinking": False,
        },
        "claude-haiku-3": {
            "provider": ModelProvider.ANTHROPIC,
            "context_window": 200000,
            "max_output": 4096,
            "cost_per_1k_input": 0.00025,
            "cost_per_1k_output": 0.00125,
            "supports_tools": True,
            "supports_vision": False,
            "supports_thinking": False,
        },
        "gpt-4o": {
            "provider": ModelProvider.OPENAI,
            "context_window": 128000,
            "max_output": 16384,
            "cost_per_1k_input": 0.0025,
            "cost_per_1k_output": 0.01,
            "supports_tools": True,
            "supports_vision": True,
            "supports_thinking": False,
        },
        "gpt-4o-mini": {
            "provider": ModelProvider.OPENAI,
            "context_window": 128000,
            "max_output": 16384,
            "cost_per_1k_input": 0.00015,
            "cost_per_1k_output": 0.0006,
            "supports_tools": True,
            "supports_vision": True,
            "supports_thinking": False,
        },
        "o1-preview": {
            "provider": ModelProvider.OPENAI,
            "context_window": 128000,
            "max_output": 32768,
            "cost_per_1k_input": 0.015,
            "cost_per_1k_output": 0.06,
            "supports_tools": False,
            "supports_vision": False,
            "supports_thinking": True,
        },
        "llama-3.1-70b": {
            "provider": ModelProvider.OPENROUTER,
            "context_window": 131072,
            "max_output": 4096,
            "cost_per_1k_input": 0.00059,
            "cost_per_1k_output": 0.00079,
            "supports_tools": True,
            "supports_vision": False,
            "supports_thinking": False,
        },
        "llama-3.1-8b": {
            "provider": ModelProvider.VLLM,
            "context_window": 131072,
            "max_output": 4096,
            "cost_per_1k_input": 0.0,  # Local model
            "cost_per_1k_output": 0.0,
            "supports_tools": True,
            "supports_vision": False,
            "supports_thinking": False,
        },
    })

    class Config:
        env_file = ".env"
        extra = "allow"


class VectorDBConfig(BaseSettings):
    """Vector database configuration."""

    # Qdrant
    qdrant_url: str = Field(default="http://localhost:6333", env="QDRANT_URL")
    qdrant_api_key: Optional[str] = Field(default=None, env="QDRANT_API_KEY")

    # Chroma
    chroma_host: str = Field(default="localhost", env="CHROMA_HOST")
    chroma_port: int = Field(default=8000, env="CHROMA_PORT")

    # Embedding Settings
    embedding_model: str = Field(default="text-embedding-3-large", env="EMBEDDING_MODEL")
    embedding_dimension: int = Field(default=3072, env="EMBEDDING_DIMENSION")
    chunk_size: int = Field(default=1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="CHUNK_OVERLAP")

    class Config:
        env_file = ".env"


class CacheConfig(BaseSettings):
    """Caching configuration."""

    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    enable_prompt_cache: bool = Field(default=True, env="ENABLE_PROMPT_CACHE")
    cache_ttl_seconds: int = Field(default=3600, env="CACHE_TTL_SECONDS")
    semantic_cache_threshold: float = Field(default=0.95, env="SEMANTIC_CACHE_THRESHOLD")

    class Config:
        env_file = ".env"


class PerformanceConfig(BaseSettings):
    """Performance and cost configuration."""

    # Concurrency
    max_concurrent_requests: int = Field(default=10, env="MAX_CONCURRENT_REQUESTS")
    request_timeout_seconds: int = Field(default=300, env="REQUEST_TIMEOUT_SECONDS")

    # Cost Management
    max_request_cost: float = Field(default=1.0, env="MAX_REQUEST_COST")
    daily_budget_limit: float = Field(default=100.0, env="DAILY_BUDGET_LIMIT")

    # Batching
    batch_size: int = Field(default=10, env="BATCH_SIZE")
    batch_timeout_ms: int = Field(default=100, env="BATCH_TIMEOUT_MS")

    # Retry
    max_retries: int = Field(default=3, env="MAX_RETRIES")
    retry_delay_seconds: float = Field(default=1.0, env="RETRY_DELAY_SECONDS")

    class Config:
        env_file = ".env"


class MonitoringConfig(BaseSettings):
    """Monitoring and observability configuration."""

    otel_endpoint: Optional[str] = Field(default=None, env="OTEL_EXPORTER_OTLP_ENDPOINT")
    prometheus_port: int = Field(default=8080, env="PROMETHEUS_PORT")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    enable_tracing: bool = Field(default=True, env="ENABLE_TRACING")
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")

    class Config:
        env_file = ".env"


class Settings(BaseSettings):
    """Main application settings."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    vector_db: VectorDBConfig = Field(default_factory=VectorDBConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)

    # Task Routing Rules
    routing_rules: Dict[TaskComplexity, List[str]] = Field(default_factory=lambda: {
        TaskComplexity.SIMPLE: ["gpt-4o-mini", "claude-haiku-3", "llama-3.1-8b"],
        TaskComplexity.MODERATE: ["claude-sonnet-4", "gpt-4o", "llama-3.1-70b"],
        TaskComplexity.COMPLEX: ["claude-opus-4-1", "gpt-4o", "o1-preview"],
        TaskComplexity.CRITICAL: ["claude-opus-4-1", "o1-preview"],
    })

    class Config:
        env_file = ".env"
        extra = "allow"


# Global settings instance
settings = Settings()