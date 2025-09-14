# Multi-Node Agentic AI Architecture

## System Overview

This architecture implements a distributed AI system across two nodes:
- **thanos**: GPU-accelerated inference and heavy compute workloads
- **oracle1**: Cloud-based orchestration, lightweight services, and failover

## Network Architecture

### Tailscale Mesh Network
```
┌─────────────────────────────────────────────────────────────┐
│                    Tailscale Mesh (100.x.x.x/24)           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────────┐      ┌──────────────────────┐   │
│  │  thanos (100.64.1.1) │◄────►│ oracle1 (100.64.1.2) │   │
│  │  On-Premise GPU Node │      │   Cloud CPU Node      │   │
│  └──────────────────────┘      └──────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Service Distribution

#### thanos (GPU Node) - Heavy Compute
- vLLM Server (GPU-accelerated local inference)
- Qdrant Vector Database (primary)
- Model inference containers
- Embedding generation services
- Computer vision/OCR services
- Whisper/TTS services

#### oracle1 (CPU Node) - Orchestration & Control
- n8n Automation Platform
- LangGraph Agent Orchestration
- API Gateway (Kong/Traefik)
- Redis (caching & queues)
- Qdrant Vector Database (replica)
- Monitoring stack (Prometheus/Grafana)
- Claude proxy services

## Detailed Service Architecture

### 1. Core AI Services

#### Model Routing Layer
```yaml
Service: AI Router
Location: oracle1
Purpose: Intelligent model routing based on cost/performance
Components:
  - Primary: Claude API (complex reasoning)
  - Secondary: OpenRouter (multi-model access)
  - Tertiary: vLLM local models (cost optimization)
  - Fallback: Smaller cloud models
```

#### Vector Database Cluster
```yaml
Service: Qdrant Cluster
Primary: thanos (GPU acceleration for indexing)
Replica: oracle1 (read-only failover)
Replication: Async over Tailscale
Backup: S3-compatible storage
```

### 2. Agent Orchestration Framework

#### LangGraph Architecture
```
┌─────────────────────────────────────────────────────┐
│                   oracle1                           │
├─────────────────────────────────────────────────────┤
│  ┌───────────────────────────────────────────┐     │
│  │         LangGraph Supervisor              │     │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  │     │
│  │  │ Planner │  │ Router  │  │ Monitor │  │     │
│  │  └─────────┘  └─────────┘  └─────────┘  │     │
│  └───────────────────────────────────────────┘     │
│                      │                              │
│  ┌───────────────────┼───────────────────────┐     │
│  │            Agent Workers                  │     │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  │     │
│  │  │Research │  │ Coder   │  │Analyzer │  │     │
│  │  └─────────┘  └─────────┘  └─────────┘  │     │
│  └───────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│                   thanos                            │
├─────────────────────────────────────────────────────┤
│  ┌───────────────────────────────────────────┐     │
│  │         Compute-Intensive Tasks           │     │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  │     │
│  │  │Embedder │  │ vLLM    │  │ Vision  │  │     │
│  │  └─────────┘  └─────────┘  └─────────┘  │     │
│  └───────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────┘
```

### 3. Security Architecture

#### Network Security
- Tailscale ACLs for service-to-service communication
- mTLS between all services
- API Gateway with rate limiting and authentication
- Sandboxed execution environments (gVisor/Firecracker)

#### Data Security
- Encryption at rest (LUKS for volumes)
- Encryption in transit (WireGuard via Tailscale)
- Secrets management (Vault or sealed-secrets)
- RBAC for all services

## Service Placement Strategy

### thanos Services (GPU-Optimized)
| Service | Port | Resource | Purpose |
|---------|------|----------|---------|
| vLLM Server | 8000 | GPU | Local LLM inference |
| Qdrant Primary | 6333 | GPU/RAM | Vector search & indexing |
| Embedding Service | 8001 | GPU | Generate embeddings |
| Whisper API | 8002 | GPU | Speech-to-text |
| ComfyUI | 8003 | GPU | Image generation |
| Code Interpreter | 8004 | CPU | Sandboxed code execution |

### oracle1 Services (CPU-Optimized)
| Service | Port | Resource | Purpose |
|---------|------|----------|---------|
| n8n | 5678 | CPU | Workflow automation |
| LangGraph API | 8080 | CPU | Agent orchestration |
| Kong Gateway | 8000 | CPU | API management |
| Redis | 6379 | RAM | Caching & queues |
| Qdrant Replica | 6334 | RAM | Vector DB failover |
| Prometheus | 9090 | CPU | Metrics collection |
| Grafana | 3000 | CPU | Monitoring dashboards |
| Claude Proxy | 8081 | CPU | API proxy & caching |

## Data Flow Patterns

### Request Flow
```
User Request → oracle1 (Gateway) → LangGraph Supervisor
    ↓
Route Decision:
    ├─ Complex: Claude API
    ├─ Simple: vLLM (thanos)
    ├─ Search: Qdrant + RAG
    └─ Multi-step: Agent Chain
```

### RAG Pipeline
```
Document → thanos (Embedding) → Qdrant (Store)
Query → oracle1 (Router) → Qdrant (Search) → LLM (Generate)
```

## Failover Strategy

### Primary Failures
1. **thanos offline**:
   - Route GPU tasks to cloud providers
   - Use oracle1 Qdrant replica for vector search
   - Increase Claude API usage temporarily

2. **oracle1 offline**:
   - thanos continues local inference
   - Direct API access to services
   - Manual workflow execution

### Service-Level Redundancy
- Database: Primary-replica with automatic failover
- Inference: Multi-model fallback chain
- Orchestration: Stateless workers with queue persistence
- Monitoring: External health checks via UptimeRobot

## Cost Optimization

### Model Routing Rules
```python
routing_rules = {
    "simple_queries": "vllm_local",      # Free, on thanos
    "code_generation": "claude_sonnet",   # Balanced cost/quality
    "complex_reasoning": "claude_opus",   # Premium quality
    "embeddings": "local_bge",           # Free, GPU-accelerated
    "fallback": "openrouter_mixtral"     # Cost-effective backup
}
```

### Resource Optimization
- Cache LLM responses in Redis (30-day TTL)
- Batch embedding generation
- Schedule heavy workloads during off-peak
- Use spot instances for non-critical tasks

## Monitoring & Observability

### Metrics Collection
```yaml
Prometheus Targets:
  - thanos:
    - node_exporter: system metrics
    - nvidia_exporter: GPU metrics
    - vllm_metrics: inference stats
  - oracle1:
    - node_exporter: system metrics
    - langgraph_metrics: agent performance
    - n8n_metrics: workflow execution
```

### Alerting Rules
- GPU temperature > 85°C
- Memory usage > 90%
- API error rate > 5%
- Vector DB replication lag > 5 minutes
- Model inference latency > 10s

## Deployment Configuration

### Prerequisites
1. Both nodes running Docker and Docker Compose
2. Tailscale installed and configured
3. SSL certificates for public endpoints
4. S3-compatible storage for backups

### Environment Variables
```bash
# Common
TAILSCALE_AUTHKEY=tskey-xxx
CLAUDE_API_KEY=sk-ant-xxx
OPENROUTER_API_KEY=sk-or-xxx

# thanos specific
CUDA_VISIBLE_DEVICES=0
VLLM_MODEL_PATH=/models
QDRANT_STORAGE=/data/qdrant

# oracle1 specific
N8N_WEBHOOK_URL=https://oracle1.yourdomain.com
LANGGRAPH_API_KEY=generated-key
REDIS_PASSWORD=secure-password
```

## Implementation Phases

### Phase 1: Core Infrastructure (Week 1)
- Set up Tailscale mesh network
- Deploy Qdrant on both nodes
- Configure basic monitoring

### Phase 2: Inference Layer (Week 2)
- Deploy vLLM on thanos
- Set up Claude proxy on oracle1
- Implement model routing logic

### Phase 3: Orchestration (Week 3)
- Deploy LangGraph framework
- Configure n8n workflows
- Implement agent patterns

### Phase 4: Optimization (Week 4)
- Fine-tune caching strategies
- Implement cost tracking
- Performance optimization

## Security Hardening Checklist

- [ ] Enable Tailscale ACLs
- [ ] Configure service mesh with mTLS
- [ ] Implement API rate limiting
- [ ] Set up WAF rules
- [ ] Enable audit logging
- [ ] Configure backup encryption
- [ ] Implement secret rotation
- [ ] Set up intrusion detection
- [ ] Configure DDoS protection
- [ ] Regular security scanning

## Operational Procedures

### Daily Operations
- Monitor resource usage
- Check replication status
- Review error logs
- Verify backup completion

### Weekly Maintenance
- Update container images
- Rotate API keys
- Review cost reports
- Performance tuning

### Monthly Tasks
- Security audit
- Disaster recovery test
- Capacity planning review
- Architecture optimization

## Next Steps

1. Review and approve architecture
2. Provision infrastructure
3. Begin Phase 1 implementation
4. Set up CI/CD pipelines
5. Document operational runbooks