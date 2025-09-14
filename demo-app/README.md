# AgenticDosNode Demo Application

## Overview

This demo application showcases the complete agentic AI stack capabilities including multi-model routing, RAG integration, cost optimization, and automated workflows.

## Features Demonstrated

### 🤖 AI Capabilities
- Multi-model routing (Claude Opus/Sonnet, GPT-4, OpenRouter, local vLLM)
- Advanced RAG with semantic search
- ReAct agent patterns with tool usage
- Self-reflection and quality improvement loops
- Extended thinking for complex reasoning

### 🏗️ Infrastructure
- Private Tailscale mesh network
- Docker containerized services
- Vector database integration (Qdrant)
- Cost optimization and monitoring
- Security hardening and sandboxing

### 🔄 Automation
- n8n workflow orchestration
- GitHub integration (PR reviews, issue analysis)
- Real-time monitoring and alerting
- Automated backup and disaster recovery
- CI/CD pipeline integration

## Quick Start

### 1. Bootstrap Deployment
```bash
# Run the comprehensive bootstrap script
cd /home/starlord/AgenticDosNode
./scripts/bootstrap-complete.sh
```

### 2. Validate Installation
```bash
# Run end-to-end validation tests
./scripts/validate-deployment.sh
```

### 3. Access Demo Interface
- **Main Dashboard**: http://localhost:3000
- **n8n Automation**: http://localhost:5678
- **Monitoring**: http://localhost:9090
- **Vector Database**: http://localhost:6333

## Demo Scenarios

### Scenario 1: Multi-Model AI Chat
Test intelligent routing between different AI models based on query complexity.

### Scenario 2: Document Analysis with RAG
Upload documents, create embeddings, and perform semantic search with AI analysis.

### Scenario 3: Code Review Automation
Create a GitHub PR to trigger automated AI code review with suggestions.

### Scenario 4: Cost Optimization
Monitor real-time costs and observe automatic model switching for cost savings.

### Scenario 5: Security Testing
Test sandboxed code execution and security monitoring systems.

## Architecture Validation

The demo validates:
- ✅ Multi-node communication (thanos ↔ oracle1)
- ✅ Secure networking via Tailscale
- ✅ AI model integration and routing
- ✅ Vector database operations
- ✅ Automation workflow execution
- ✅ Cost tracking and optimization
- ✅ Security controls and monitoring
- ✅ Backup and recovery procedures

## Troubleshooting

See `docs/troubleshooting.md` for common issues and solutions.

## Production Readiness

This demo represents a production-ready deployment suitable for:
- Research and development environments
- Small team AI operations
- Prototype and MVP development
- Learning and experimentation

For enterprise deployment, see `docs/enterprise-guide.md`.