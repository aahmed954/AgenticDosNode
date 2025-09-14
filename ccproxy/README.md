# Claude Code Proxy

A production-ready, security-hardened proxy server that provides OpenAI-compatible API access to Claude models. Designed for enterprise deployment with comprehensive monitoring, rate limiting, and security features.

## 🚀 Features

### Core Functionality
- **OpenAI API Compatibility**: Drop-in replacement for OpenAI API calls
- **Streaming Support**: Real-time streaming responses
- **Model Mapping**: Automatic translation between OpenAI and Claude model names
- **Function Calling**: Full support for tools and function calls
- **Multi-modal**: Support for text and image inputs

### Security & Authentication
- **Multiple Auth Methods**: API keys, Bearer tokens, JWT/OAuth2
- **Rate Limiting**: Advanced rate limiting with token counting
- **Request Validation**: Input sanitization and validation
- **Security Headers**: Comprehensive security headers
- **ModSecurity Integration**: WAF protection against AI-specific attacks

### Monitoring & Observability
- **Prometheus Metrics**: Detailed metrics for monitoring
- **Structured Logging**: JSON-formatted logs with request tracking
- **Health Checks**: Comprehensive health monitoring
- **Error Tracking**: Detailed error reporting and alerting
- **Usage Analytics**: API usage statistics and trends

### Production Ready
- **Docker Containerization**: Secure, hardened containers
- **Security Hardening**: AppArmor, Seccomp, capability restrictions
- **High Availability**: Load balancing and failover support
- **Configuration Management**: Environment-based configuration
- **CI/CD Integration**: Automated deployment and testing

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Authentication](#authentication)
- [Deployment](#deployment)
- [API Reference](#api-reference)
- [Monitoring](#monitoring)
- [Security](#security)
- [Development](#development)
- [Troubleshooting](#troubleshooting)

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- Valid Anthropic API key
- Python 3.11+ (for development)

### Basic Setup

1. **Clone and Configure**
   ```bash
   cd /home/starlord/AgenticDosNode/ccproxy
   cp .env.example .env
   # Edit .env with your configuration
   ```

2. **Set Required Environment Variables**
   ```bash
   # In .env file
   ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
   CCPROXY_API_KEYS=sk-ccproxy-your-proxy-key-1,sk-ccproxy-your-proxy-key-2
   ```

3. **Deploy with Docker**
   ```bash
   ./deploy.sh deploy
   ```

4. **Test the Proxy**
   ```bash
   curl -X POST http://localhost:8000/v1/chat/completions \
     -H "Authorization: Bearer sk-ccproxy-your-proxy-key-1" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "gpt-4",
       "messages": [{"role": "user", "content": "Hello, world!"}],
       "max_tokens": 100
     }'
   ```

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `ANTHROPIC_API_KEY` | Your Anthropic API key | - | Yes |
| `CCPROXY_API_KEYS` | Comma-separated proxy API keys | - | Yes |
| `CCPROXY_HOST` | Server bind address | `0.0.0.0` | No |
| `CCPROXY_PORT` | Server port | `8000` | No |
| `CCPROXY_AUTH_METHOD` | Authentication method | `api_key` | No |
| `CCPROXY_RATE_LIMIT_RPM` | Requests per minute | `60` | No |
| `CCPROXY_RATE_LIMIT_TPM` | Tokens per minute | `100000` | No |
| `CCPROXY_CONCURRENT_REQUESTS` | Max concurrent requests | `10` | No |

### Model Mapping

The proxy automatically maps OpenAI model names to Claude models:

```json
{
  "gpt-4": "claude-3-5-sonnet-20241022",
  "gpt-4-turbo": "claude-3-5-sonnet-20241022",
  "gpt-4o": "claude-3-5-sonnet-20241022",
  "gpt-4o-mini": "claude-3-5-haiku-20241022",
  "gpt-3.5-turbo": "claude-3-5-haiku-20241022"
}
```

## 🔐 Authentication

### API Key Authentication (Default)

```bash
curl -H "Authorization: Bearer your-api-key" \
     -H "X-API-Key: your-api-key" \  # Alternative header
     http://localhost:8000/v1/chat/completions
```

### Bearer Token Authentication

```bash
# Set in environment
CCPROXY_AUTH_METHOD=bearer_token
CCPROXY_BEARER_TOKEN=your-static-token

# Use in requests
curl -H "Authorization: Bearer your-static-token" \
     http://localhost:8000/v1/chat/completions
```

### JWT Authentication

```bash
# Set in environment
CCPROXY_AUTH_METHOD=oauth2
CCPROXY_JWT_SECRET=your-jwt-secret

# Use JWT tokens in requests
curl -H "Authorization: Bearer eyJ0eXAiOiJKV1Q..." \
     http://localhost:8000/v1/chat/completions
```

## 🚀 Deployment

### Standalone Deployment

```bash
# Basic deployment
docker-compose up -d

# With monitoring
docker-compose -f docker-compose.yml -f docker-compose.monitoring.yml up -d
```

### Integration with Existing Infrastructure

The proxy integrates with the AgenticDosNode security infrastructure:

```bash
# Deploy with security integration
docker-compose -f docker-compose.secure.yml -f ccproxy-integration.yml up -d
```

### Production Deployment Checklist

- [ ] Set strong API keys and rotate regularly
- [ ] Configure rate limits appropriate for your use case
- [ ] Enable TLS/SSL termination at load balancer
- [ ] Set up log aggregation and monitoring
- [ ] Configure backup and disaster recovery
- [ ] Test failover scenarios
- [ ] Set up alerting for critical metrics

## 📚 API Reference

### Chat Completions

**POST** `/v1/chat/completions`

Compatible with OpenAI's chat completions API:

```json
{
  "model": "gpt-4",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "temperature": 0.7,
  "max_tokens": 150,
  "stream": false
}
```

### Function Calling

```json
{
  "model": "gpt-4",
  "messages": [
    {"role": "user", "content": "What's the weather like in New York?"}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string"}
          }
        }
      }
    }
  ]
}
```

### Streaming Responses

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer your-key" \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4", "messages": [{"role": "user", "content": "Tell me a story"}], "stream": true}'
```

### Health Check

**GET** `/health`

```json
{
  "status": "healthy",
  "uptime_seconds": 3600,
  "checks": {
    "claude_api": true,
    "rate_limiter": true,
    "auth": true
  },
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### Statistics

**GET** `/stats`

```json
{
  "total_requests": 1000,
  "successful_requests": 980,
  "failed_requests": 20,
  "average_response_time": 1.5,
  "current_active_requests": 3,
  "rate_limit_hits": 5,
  "uptime_seconds": 86400
}
```

### Metrics

**GET** `/metrics`

Prometheus-compatible metrics in text format.

## 📊 Monitoring

### Prometheus Metrics

Key metrics exposed:

- `ccproxy_requests_total` - Total requests by method/endpoint/status
- `ccproxy_request_duration_seconds` - Request duration histogram
- `ccproxy_tokens_total` - Total tokens processed
- `ccproxy_active_requests` - Current active requests
- `ccproxy_rate_limit_hits_total` - Rate limit violations
- `ccproxy_claude_api_requests_total` - Claude API calls
- `ccproxy_errors_total` - Error counts by type

### Grafana Dashboard

Example queries:

```promql
# Request rate
rate(ccproxy_requests_total[5m])

# Error rate
rate(ccproxy_errors_total[5m]) / rate(ccproxy_requests_total[5m])

# Average response time
rate(ccproxy_request_duration_seconds_sum[5m]) / rate(ccproxy_request_duration_seconds_count[5m])

# Token usage
rate(ccproxy_tokens_total[5m])
```

### Alerting Rules

```yaml
groups:
- name: ccproxy
  rules:
  - alert: HighErrorRate
    expr: rate(ccproxy_errors_total[5m]) / rate(ccproxy_requests_total[5m]) > 0.1
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "High error rate detected"

  - alert: HighResponseTime
    expr: rate(ccproxy_request_duration_seconds_sum[5m]) / rate(ccproxy_request_duration_seconds_count[5m]) > 10
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High average response time"
```

## 🔒 Security

### Security Features

1. **Container Security**
   - Non-root user execution
   - Read-only filesystem
   - Capability restrictions
   - Seccomp and AppArmor profiles

2. **Network Security**
   - WAF with ModSecurity
   - Rate limiting and DDoS protection
   - TLS termination
   - Network isolation

3. **Application Security**
   - Input validation and sanitization
   - Authentication and authorization
   - Request/response filtering
   - Security headers

4. **AI-Specific Security**
   - Prompt injection detection
   - Content filtering
   - Usage monitoring
   - Abuse detection

### Security Configuration

```yaml
# docker-compose.security.yml
security_opt:
  - no-new-privileges:true
  - seccomp:seccomp-ccproxy.json
  - apparmor:ccproxy-profile
cap_drop:
  - ALL
cap_add:
  - NET_BIND_SERVICE
read_only: true
```

### ModSecurity Rules

Custom rules for AI proxy protection:

- Prompt injection detection
- SQL injection protection
- Content size limits
- Rate limiting
- Response filtering

## 🛠️ Development

### Setup Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-asyncio pytest-cov black isort mypy

# Run tests
pytest tests/

# Code formatting
black src/
isort src/

# Type checking
mypy src/
```

### Running Locally

```bash
# Set environment variables
export ANTHROPIC_API_KEY=your-key
export CCPROXY_API_KEYS=test-key

# Run server
python -m src.proxy_server

# Or with uvicorn
uvicorn src.proxy_server:app --reload
```

### Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test
pytest tests/test_proxy.py::TestProxyServer::test_health_endpoint -v
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Write tests for new functionality
4. Ensure all tests pass
5. Format code with black and isort
6. Submit a pull request

## 🐛 Troubleshooting

### Common Issues

**Health Check Fails**
```bash
# Check container logs
docker-compose logs ccproxy

# Check Claude API connectivity
curl -H "x-api-key: your-key" https://api.anthropic.com/v1/messages
```

**Rate Limiting Issues**
```bash
# Check current rates in logs
docker-compose logs ccproxy | grep "rate_limit"

# Adjust limits in environment
CCPROXY_RATE_LIMIT_RPM=120
```

**Authentication Errors**
```bash
# Verify API key format
echo $CCPROXY_API_KEYS

# Test authentication
curl -H "Authorization: Bearer your-key" http://localhost:8000/health
```

### Debugging

Enable debug mode:
```bash
CCPROXY_DEBUG=true
CCPROXY_LOG_LEVEL=DEBUG
```

View detailed logs:
```bash
docker-compose logs -f ccproxy
```

### Performance Tuning

- Adjust rate limits based on usage patterns
- Monitor token consumption and costs
- Scale horizontally with load balancer
- Use Redis for distributed rate limiting
- Optimize container resource limits

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Support

For support and questions:

1. Check the [Troubleshooting](#troubleshooting) section
2. Review logs and metrics
3. Open an issue with detailed information
4. Include environment configuration (sanitized)

## 🔗 Related Projects

- [Claude API Documentation](https://docs.anthropic.com/claude/reference/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [AgenticDosNode Orchestrator](../README.md)

---

**⚠️ Important Security Notice**

This proxy provides access to AI models and should be deployed securely:

- Use strong, unique API keys
- Enable rate limiting appropriate for your use case
- Monitor for unusual usage patterns
- Keep the proxy updated with security patches
- Follow security best practices for container deployment
- Respect Anthropic's usage policies and terms of service

**📈 Usage Analytics**

The proxy collects usage metrics for monitoring and optimization:

- Request counts and response times
- Error rates and types
- Token usage and costs
- Authentication events
- Rate limiting events

All data is used solely for operational purposes and follows privacy best practices.