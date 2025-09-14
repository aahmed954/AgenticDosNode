# Deployment Guide for Multi-Node Agentic AI Architecture

## Prerequisites

### Hardware Requirements
- **thanos**: Ubuntu 24.04, NVIDIA RTX 3080, 64GB RAM, 500GB+ SSD
- **oracle1**: Oracle Ampere A1, 4 OCPU, 24GB RAM, 200GB+ boot volume

### Software Requirements
- Docker Engine 24.0+
- Docker Compose 2.20+
- NVIDIA Container Toolkit (thanos only)
- Git
- Tailscale CLI

### Required API Keys
```bash
# Create .env file on both nodes
CLAUDE_API_KEY=sk-ant-xxx
OPENROUTER_API_KEY=sk-or-xxx
TAILSCALE_AUTHKEY=tskey-xxx
```

## Phase 1: Infrastructure Setup

### Step 1: Prepare Both Nodes

#### On thanos:
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker

# Verify GPU access
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi

# Clone repository
git clone https://github.com/yourusername/AgenticDosNode.git
cd AgenticDosNode
```

#### On oracle1:
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER

# Clone repository
git clone https://github.com/yourusername/AgenticDosNode.git
cd AgenticDosNode
```

### Step 2: Setup Tailscale Network

Run on both nodes:
```bash
# Install Tailscale
cd tailscale
chmod +x setup.sh
sudo TAILSCALE_AUTHKEY=tskey-xxx ./setup.sh

# Verify connection
tailscale status
tailscale ping <other-node>
```

### Step 3: Configure Environment Variables

Create `.env` files on both nodes:

#### thanos/.env:
```bash
# Model Configuration
VLLM_MODEL=mistralai/Mistral-7B-Instruct-v0.2
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5

# API Keys
CLAUDE_API_KEY=sk-ant-xxx
OPENROUTER_API_KEY=sk-or-xxx

# Security
CODE_SERVER_PASSWORD=changeme
BACKUP_PASSWORD=changeme

# Paths
MODEL_PATH=/data/models
QDRANT_PATH=/data/qdrant
```

#### oracle1/.env:
```bash
# Database
POSTGRES_USER=agentic
POSTGRES_PASSWORD=changeme
POSTGRES_DB=agentic_db

# Redis
REDIS_PASSWORD=changeme

# API Keys
CLAUDE_API_KEY=sk-ant-xxx
OPENROUTER_API_KEY=sk-or-xxx
LANGGRAPH_API_KEY=changeme

# n8n
N8N_HOST=oracle1.tail.net
N8N_ENCRYPTION_KEY=changeme

# Monitoring
GRAFANA_USER=admin
GRAFANA_PASSWORD=changeme

# Backup
AWS_ACCESS_KEY_ID=xxx
AWS_SECRET_ACCESS_KEY=xxx
BACKUP_REPO=s3:s3.amazonaws.com/bucket/oracle1
```

## Phase 2: Core Services Deployment

### Step 1: Deploy thanos Services

```bash
cd /path/to/AgenticDosNode/thanos

# Start core services
docker-compose up -d qdrant vllm embedding-service

# Wait for services to be healthy
docker-compose ps

# Start remaining services
docker-compose up -d

# Check logs
docker-compose logs -f vllm
```

### Step 2: Deploy oracle1 Services

```bash
cd /path/to/AgenticDosNode/oracle1

# Initialize database
docker-compose up -d postgres
sleep 10

# Run database migrations
docker-compose exec postgres psql -U agentic -d agentic_db -f /docker-entrypoint-initdb.d/init.sql

# Start remaining services
docker-compose up -d

# Check service health
docker-compose ps
```

### Step 3: Configure Qdrant Replication

```bash
# On oracle1, configure replication from thanos
curl -X POST 'http://100.64.1.2:6333/cluster/peer' \
  -H 'Content-Type: application/json' \
  -d '{
    "uri": "http://100.64.1.1:6334",
    "replication_factor": 2
  }'

# Verify replication status
curl 'http://100.64.1.2:6333/cluster/status'
```

## Phase 3: Agent Framework Setup

### Step 1: Deploy LangGraph Application

```bash
# On oracle1
cd oracle1/langgraph

# Build custom image
docker build -t langgraph-api:latest .

# Deploy
docker-compose up -d langgraph

# Test API
curl -H "X-API-Key: changeme" http://100.64.1.2:8080/health
```

### Step 2: Configure n8n Workflows

1. Access n8n UI: https://oracle1.tail.net:5678
2. Create credentials:
   - Claude API
   - OpenRouter API
   - vLLM endpoint
3. Import workflow templates from `n8n/workflows/`

### Step 3: Setup Kong API Gateway

```bash
# Configure services
curl -i -X POST http://100.64.1.2:8001/services/ \
  --data "name=vllm" \
  --data "url=http://100.64.1.1:8000"

curl -i -X POST http://100.64.1.2:8001/services/ \
  --data "name=langgraph" \
  --data "url=http://100.64.1.2:8080"

# Add routes
curl -i -X POST http://100.64.1.2:8001/services/vllm/routes \
  --data "paths[]=/vllm" \
  --data "name=vllm-route"

curl -i -X POST http://100.64.1.2:8001/services/langgraph/routes \
  --data "paths[]=/agents" \
  --data "name=langgraph-route"

# Enable rate limiting
curl -i -X POST http://100.64.1.2:8001/services/vllm/plugins \
  --data "name=rate-limiting" \
  --data "config.minute=60" \
  --data "config.policy=local"
```

## Phase 4: Monitoring & Security

### Step 1: Configure Prometheus

```bash
# Copy prometheus config
cp prometheus/prometheus.yml oracle1/prometheus/

# Restart Prometheus
cd oracle1
docker-compose restart prometheus

# Verify targets
curl http://100.64.1.2:9090/api/v1/targets
```

### Step 2: Import Grafana Dashboards

1. Access Grafana: http://oracle1.tail.net:3000
2. Login with admin/changeme
3. Add Prometheus data source: http://prometheus:9090
4. Import dashboards from `grafana/dashboards/`

### Step 3: Security Hardening

```bash
# Enable firewall on both nodes
sudo ufw enable
sudo ufw allow 22/tcp
sudo ufw allow 41641/udp  # Tailscale

# On thanos
sudo ufw allow from 100.64.0.0/16  # Allow Tailscale network

# On oracle1
sudo ufw allow 443/tcp  # HTTPS
sudo ufw allow from 100.64.0.0/16  # Allow Tailscale network

# Setup fail2ban
sudo apt install fail2ban -y
sudo systemctl enable fail2ban
sudo systemctl start fail2ban
```

## Phase 5: Testing & Validation

### Test 1: Network Connectivity
```bash
# From oracle1
curl http://100.64.1.1:8000/health  # vLLM
curl http://100.64.1.1:6333/health  # Qdrant

# From thanos
curl http://100.64.1.2:8080/health  # LangGraph
curl http://100.64.1.2:5678/healthz # n8n
```

### Test 2: Model Inference
```bash
# Test vLLM
curl -X POST http://100.64.1.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistralai/Mistral-7B-Instruct-v0.2",
    "prompt": "Hello, how are you?",
    "max_tokens": 50
  }'

# Test Claude proxy
curl -X POST http://100.64.1.2:8081/v1/messages \
  -H "X-API-Key: changeme" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-sonnet",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

### Test 3: Vector Search
```bash
# Create collection
curl -X PUT http://100.64.1.1:6333/collections/test \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": {
      "size": 1536,
      "distance": "Cosine"
    }
  }'

# Insert vector
curl -X PUT http://100.64.1.1:6333/collections/test/points \
  -H "Content-Type: application/json" \
  -d '{
    "points": [{
      "id": 1,
      "vector": [0.1, 0.2, ...],
      "payload": {"text": "test"}
    }]
  }'
```

### Test 4: Agent Workflow
```bash
# Test LangGraph agent
curl -X POST http://100.64.1.2:8080/agents/invoke \
  -H "X-API-Key: changeme" \
  -H "Content-Type: application/json" \
  -d '{
    "agent": "research",
    "input": "What is the weather today?",
    "config": {
      "model": "claude-3-sonnet"
    }
  }'
```

## Backup & Recovery

### Automated Backups
```bash
# Setup daily backup cron
crontab -e

# Add entries
0 2 * * * cd /path/to/thanos && docker-compose run --rm restic
0 3 * * * cd /path/to/oracle1 && docker-compose run --rm restic
```

### Manual Backup
```bash
# Backup thanos
docker-compose run --rm restic backup /data --tag manual

# Backup oracle1
docker-compose run --rm restic backup /data --tag manual
```

### Recovery Procedure
```bash
# List snapshots
docker-compose run --rm restic snapshots

# Restore specific snapshot
docker-compose run --rm restic restore <snapshot-id> --target /restore
```

## Troubleshooting

### Common Issues

#### GPU Not Available
```bash
# Check NVIDIA drivers
nvidia-smi

# Restart Docker daemon
sudo systemctl restart docker

# Check container GPU access
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

#### Service Connection Issues
```bash
# Check Tailscale status
tailscale status

# Test connectivity
tailscale ping <node>

# Check firewall
sudo ufw status verbose

# Check Docker networks
docker network ls
docker network inspect <network>
```

#### High Memory Usage
```bash
# Check memory usage
docker stats

# Limit container memory
docker-compose down
# Edit docker-compose.yml to add memory limits
docker-compose up -d
```

#### Database Connection Failed
```bash
# Check PostgreSQL status
docker-compose logs postgres

# Connect manually
docker-compose exec postgres psql -U agentic

# Reset database
docker-compose down -v postgres
docker-compose up -d postgres
```

## Maintenance Procedures

### Weekly Tasks
```bash
# Update containers
docker-compose pull
docker-compose up -d

# Clean unused resources
docker system prune -af

# Check logs for errors
docker-compose logs --tail=100 | grep ERROR

# Verify backups
docker-compose run --rm restic check
```

### Monthly Tasks
```bash
# Rotate API keys
# Update .env files with new keys
docker-compose restart

# Security updates
sudo apt update && sudo apt upgrade -y

# Performance review
# Check Grafana dashboards
# Optimize based on metrics
```

## Performance Optimization

### GPU Optimization
```bash
# Monitor GPU usage
nvidia-smi dmon -s u

# Adjust batch sizes in vLLM
# Edit docker-compose.yml
MAX_BATCH_SIZE=32
GPU_MEMORY_UTILIZATION=0.9
```

### Database Optimization
```bash
# Vacuum PostgreSQL
docker-compose exec postgres vacuumdb -U agentic -d agentic_db -z

# Optimize Redis
docker-compose exec redis redis-cli
> CONFIG SET maxmemory-policy allkeys-lru
> BGSAVE
```

### Network Optimization
```bash
# Enable BBR congestion control
echo "net.core.default_qdisc=fq" | sudo tee -a /etc/sysctl.conf
echo "net.ipv4.tcp_congestion_control=bbr" | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

## Scaling Considerations

### Adding More GPU Nodes
1. Provision new GPU node
2. Install Docker and NVIDIA toolkit
3. Join Tailscale network
4. Deploy vLLM and embedding services
5. Update load balancer configuration

### Adding More CPU Nodes
1. Provision new CPU node
2. Install Docker
3. Join Tailscale network
4. Deploy agent workers
5. Update orchestrator configuration

## Support & Resources

- Architecture Documentation: `/architecture.md`
- Security Configuration: `/security/security-config.yml`
- Monitoring Dashboards: http://oracle1.tail.net:3000
- API Documentation: http://oracle1.tail.net:8000/docs
- Logs: `docker-compose logs -f <service>`

## Next Steps

1. Configure SSL certificates for public endpoints
2. Setup external monitoring (UptimeRobot, Pingdom)
3. Implement CI/CD pipeline for deployments
4. Configure alerting rules in Prometheus
5. Document custom workflows and agents
6. Performance baseline and optimization