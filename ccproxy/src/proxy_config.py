"""Configuration management for the Claude Code proxy."""

from typing import Optional, Dict, Any, List
from pydantic import Field, validator
from pydantic_settings import BaseSettings
from enum import Enum
import os


class AuthMethod(str, Enum):
    """Supported authentication methods."""
    API_KEY = "api_key"
    BEARER_TOKEN = "bearer_token"
    OAUTH2 = "oauth2"
    NONE = "none"


class ProxyConfig(BaseSettings):
    """Claude Code Proxy Configuration."""

    # Server Configuration
    host: str = Field(default="0.0.0.0", env="CCPROXY_HOST")
    port: int = Field(default=8000, env="CCPROXY_PORT")
    debug: bool = Field(default=False, env="CCPROXY_DEBUG")

    # Authentication
    auth_method: AuthMethod = Field(default=AuthMethod.API_KEY, env="CCPROXY_AUTH_METHOD")
    api_keys: List[str] = Field(default_factory=list, env="CCPROXY_API_KEYS")
    bearer_token: Optional[str] = Field(default=None, env="CCPROXY_BEARER_TOKEN")

    # Claude API Configuration
    claude_api_key: str = Field(..., env="ANTHROPIC_API_KEY")
    claude_api_base: str = Field(default="https://api.anthropic.com", env="ANTHROPIC_API_BASE")
    claude_version: str = Field(default="2023-06-01", env="ANTHROPIC_VERSION")

    # Rate Limiting
    rate_limit_requests_per_minute: int = Field(default=60, env="CCPROXY_RATE_LIMIT_RPM")
    rate_limit_tokens_per_minute: int = Field(default=100000, env="CCPROXY_RATE_LIMIT_TPM")
    rate_limit_concurrent_requests: int = Field(default=10, env="CCPROXY_CONCURRENT_REQUESTS")

    # Model Configuration
    default_model: str = Field(default="claude-3-5-sonnet-20241022", env="CCPROXY_DEFAULT_MODEL")
    supported_models: List[str] = Field(default_factory=lambda: [
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307"
    ], env="CCPROXY_SUPPORTED_MODELS")

    # OpenAI Compatibility Mapping
    openai_model_mapping: Dict[str, str] = Field(default_factory=lambda: {
        "gpt-4": "claude-3-5-sonnet-20241022",
        "gpt-4-turbo": "claude-3-5-sonnet-20241022",
        "gpt-4o": "claude-3-5-sonnet-20241022",
        "gpt-4o-mini": "claude-3-5-haiku-20241022",
        "gpt-3.5-turbo": "claude-3-5-haiku-20241022",
        "claude-sonnet-4": "claude-3-5-sonnet-20241022",
        "claude-opus-4-1": "claude-3-opus-20240229",
        "claude-haiku-3": "claude-3-5-haiku-20241022"
    })

    # Request/Response Configuration
    max_tokens: int = Field(default=4096, env="CCPROXY_MAX_TOKENS")
    temperature: float = Field(default=0.7, env="CCPROXY_TEMPERATURE")
    timeout_seconds: int = Field(default=300, env="CCPROXY_TIMEOUT")

    # Logging and Monitoring
    log_level: str = Field(default="INFO", env="CCPROXY_LOG_LEVEL")
    log_requests: bool = Field(default=True, env="CCPROXY_LOG_REQUESTS")
    log_responses: bool = Field(default=False, env="CCPROXY_LOG_RESPONSES")  # Security: off by default
    enable_metrics: bool = Field(default=True, env="CCPROXY_ENABLE_METRICS")
    metrics_port: int = Field(default=8001, env="CCPROXY_METRICS_PORT")

    # Security
    enable_cors: bool = Field(default=True, env="CCPROXY_ENABLE_CORS")
    cors_origins: List[str] = Field(default_factory=lambda: ["*"], env="CCPROXY_CORS_ORIGINS")
    max_request_size: int = Field(default=1024*1024, env="CCPROXY_MAX_REQUEST_SIZE")  # 1MB

    # Health Checks
    health_check_interval: int = Field(default=30, env="CCPROXY_HEALTH_INTERVAL")

    @validator('api_keys', pre=True)
    def parse_api_keys(cls, v):
        if isinstance(v, str):
            return [key.strip() for key in v.split(",") if key.strip()]
        return v or []

    @validator('cors_origins', pre=True)
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    @validator('supported_models', pre=True)
    def parse_supported_models(cls, v):
        if isinstance(v, str):
            return [model.strip() for model in v.split(",") if model.strip()]
        return v

    class Config:
        env_file = ".env"
        extra = "allow"


# Global configuration instance
config = ProxyConfig()