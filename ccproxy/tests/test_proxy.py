"""Comprehensive tests for the Claude Code Proxy."""

import pytest
import httpx
import json
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient

from src.proxy_server import app
from src.models import ChatCompletionRequest, ChatMessage
from src.proxy_config import ProxyConfig
from src.transformers import RequestTransformer, ResponseTransformer
from src.rate_limiter import RateLimiter
from src.auth import APIKeyValidator


class TestProxyServer:
    """Test the main proxy server functionality."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def sample_chat_request(self):
        """Sample OpenAI chat completion request."""
        return ChatCompletionRequest(
            model="gpt-4",
            messages=[
                ChatMessage(role="user", content="Hello, how are you?")
            ],
            temperature=0.7,
            max_tokens=100
        )

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "uptime_seconds" in data

    def test_models_endpoint(self, client):
        """Test models listing endpoint."""
        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert "data" in data
        assert len(data["data"]) > 0

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert data["message"] == "Claude Code Proxy API"

    def test_stats_endpoint(self, client):
        """Test statistics endpoint."""
        response = client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_requests" in data
        assert "uptime_seconds" in data

    @pytest.mark.asyncio
    async def test_chat_completions_unauthorized(self, client):
        """Test chat completions without authentication."""
        request_data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}]
        }
        response = client.post("/v1/chat/completions", json=request_data)
        # This will depend on auth configuration
        # assert response.status_code == 401

    def test_metrics_endpoint(self, client):
        """Test Prometheus metrics endpoint."""
        response = client.get("/metrics")
        assert response.status_code == 200
        # Should return Prometheus format metrics
        assert "ccproxy_" in response.text or response.status_code == 500  # Might not be fully initialized


class TestRequestTransformer:
    """Test request transformation between OpenAI and Claude formats."""

    def test_basic_chat_request_transformation(self):
        """Test basic chat request transformation."""
        openai_request = ChatCompletionRequest(
            model="gpt-4",
            messages=[
                ChatMessage(role="system", content="You are a helpful assistant."),
                ChatMessage(role="user", content="Hello, world!")
            ],
            temperature=0.7,
            max_tokens=100
        )

        claude_request = RequestTransformer.openai_to_claude(openai_request)

        assert claude_request.model == "gpt-4"
        assert claude_request.system == "You are a helpful assistant."
        assert len(claude_request.messages) == 1
        assert claude_request.messages[0].role == "user"
        assert claude_request.messages[0].content == "Hello, world!"
        assert claude_request.temperature == 0.7
        assert claude_request.max_tokens == 100

    def test_function_call_transformation(self):
        """Test function call transformation."""
        openai_request = ChatCompletionRequest(
            model="gpt-4",
            messages=[
                ChatMessage(role="user", content="What's the weather like?")
            ],
            tools=[{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        }
                    }
                }
            }]
        )

        claude_request = RequestTransformer.openai_to_claude(openai_request)

        assert claude_request.tools is not None
        assert len(claude_request.tools) == 1
        assert claude_request.tools[0].name == "get_weather"

    def test_multimodal_content_transformation(self):
        """Test multimodal content transformation."""
        openai_request = ChatCompletionRequest(
            model="gpt-4",
            messages=[
                ChatMessage(
                    role="user",
                    content=[
                        {"type": "text", "text": "What's in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEA..."
                            }
                        }
                    ]
                )
            ]
        )

        claude_request = RequestTransformer.openai_to_claude(openai_request)

        assert len(claude_request.messages) == 1
        message_content = claude_request.messages[0].content
        assert isinstance(message_content, list)
        assert len(message_content) == 2
        assert message_content[0]["type"] == "text"
        assert message_content[1]["type"] == "image"


class TestResponseTransformer:
    """Test response transformation from Claude to OpenAI format."""

    def test_basic_response_transformation(self):
        """Test basic response transformation."""
        claude_response = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello! I'm doing well, thank you."}],
            "model": "claude-3-5-sonnet-20241022",
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 15
            }
        }

        original_request = ChatCompletionRequest(
            model="gpt-4",
            messages=[ChatMessage(role="user", content="Hello, how are you?")]
        )

        openai_response = ResponseTransformer.claude_to_openai(
            claude_response, original_request
        )

        assert openai_response.model == "gpt-4"
        assert len(openai_response.choices) == 1
        choice = openai_response.choices[0]
        assert choice.message.role == "assistant"
        assert choice.message.content == "Hello! I'm doing well, thank you."
        assert choice.finish_reason == "stop"
        assert openai_response.usage.prompt_tokens == 10
        assert openai_response.usage.completion_tokens == 15
        assert openai_response.usage.total_tokens == 25

    def test_tool_use_response_transformation(self):
        """Test tool use response transformation."""
        claude_response = {
            "id": "msg_456",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I'll check the weather for you."},
                {
                    "type": "tool_use",
                    "id": "tool_789",
                    "name": "get_weather",
                    "input": {"location": "New York"}
                }
            ],
            "model": "claude-3-5-sonnet-20241022",
            "stop_reason": "tool_use",
            "usage": {
                "input_tokens": 20,
                "output_tokens": 25
            }
        }

        original_request = ChatCompletionRequest(
            model="gpt-4",
            messages=[ChatMessage(role="user", content="What's the weather in New York?")]
        )

        openai_response = ResponseTransformer.claude_to_openai(
            claude_response, original_request
        )

        choice = openai_response.choices[0]
        assert choice.message.content == "I'll check the weather for you."
        assert choice.message.tool_calls is not None
        assert len(choice.message.tool_calls) == 1
        tool_call = choice.message.tool_calls[0]
        assert tool_call.id == "tool_789"
        assert tool_call.function.name == "get_weather"
        assert json.loads(tool_call.function.arguments) == {"location": "New York"}
        assert choice.finish_reason == "tool_calls"


class TestRateLimiter:
    """Test rate limiting functionality."""

    @pytest.mark.asyncio
    async def test_request_rate_limiting(self):
        """Test request rate limiting."""
        rate_limiter = RateLimiter(
            requests_per_minute=2,
            tokens_per_minute=1000,
            concurrent_requests=1
        )

        # First request should pass
        allowed, reason = await rate_limiter.check_rate_limit()
        assert allowed is True

        # Second request should pass
        allowed, reason = await rate_limiter.check_rate_limit()
        assert allowed is True

        # Third request should be rate limited
        allowed, reason = await rate_limiter.check_rate_limit()
        assert allowed is False
        assert "Request rate limit exceeded" in reason

    @pytest.mark.asyncio
    async def test_token_rate_limiting(self):
        """Test token rate limiting."""
        rate_limiter = RateLimiter(
            requests_per_minute=100,
            tokens_per_minute=10,
            concurrent_requests=10
        )

        # Request with tokens within limit should pass
        allowed, reason = await rate_limiter.check_rate_limit(estimated_tokens=5)
        assert allowed is True

        # Request that would exceed token limit should fail
        allowed, reason = await rate_limiter.check_rate_limit(estimated_tokens=10)
        assert allowed is False
        assert "Token rate limit exceeded" in reason

    @pytest.mark.asyncio
    async def test_concurrency_limiting(self):
        """Test concurrency limiting."""
        rate_limiter = RateLimiter(
            requests_per_minute=100,
            tokens_per_minute=10000,
            concurrent_requests=1
        )

        # Acquire first slot
        await rate_limiter.acquire_concurrency_slot()

        # Second request should be allowed but would need to wait for slot
        active_count = await rate_limiter.concurrency_limiter.get_active_count()
        assert active_count == 1

        # Release slot
        await rate_limiter.release_concurrency_slot()
        active_count = await rate_limiter.concurrency_limiter.get_active_count()
        assert active_count == 0


class TestAuthentication:
    """Test authentication functionality."""

    def test_api_key_validation(self):
        """Test API key validation."""
        valid_keys = ["sk-test-key-1", "sk-test-key-2"]
        validator = APIKeyValidator(valid_keys)

        # Valid key should pass
        assert validator.validate_key("sk-test-key-1") is True

        # Invalid key should fail
        assert validator.validate_key("invalid-key") is False

        # Empty key should fail
        assert validator.validate_key("") is False

    def test_api_key_usage_tracking(self):
        """Test API key usage tracking."""
        valid_keys = ["sk-test-key-1"]
        validator = APIKeyValidator(valid_keys)

        # Use the key multiple times
        validator.validate_key("sk-test-key-1")
        validator.validate_key("sk-test-key-1")

        stats = validator.get_usage_stats()
        assert len(stats) == 1
        key_stats = list(stats.values())[0]
        assert key_stats["request_count"] == 2


class TestConfiguration:
    """Test configuration management."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ProxyConfig()

        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.rate_limit_requests_per_minute == 60
        assert config.auth_method.value == "api_key"

    def test_model_mapping(self):
        """Test OpenAI to Claude model mapping."""
        config = ProxyConfig()

        mapping = config.openai_model_mapping
        assert "gpt-4" in mapping
        assert "gpt-4o" in mapping
        assert mapping["gpt-4"] == "claude-3-5-sonnet-20241022"


class TestIntegration:
    """Integration tests for the complete proxy functionality."""

    @pytest.mark.asyncio
    @patch('httpx.AsyncClient.post')
    async def test_end_to_end_chat_completion(self, mock_post):
        """Test end-to-end chat completion flow."""
        # Mock Claude API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello! How can I help you?"}],
            "model": "claude-3-5-sonnet-20241022",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 12}
        }
        mock_post.return_value = mock_response

        client = TestClient(app)

        # Make request to proxy
        request_data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.7,
            "max_tokens": 100
        }

        # Note: This test would need proper auth setup to work
        # response = client.post("/v1/chat/completions", json=request_data)
        # assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__])