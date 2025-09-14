# AgenticDosNode Conflict Detection and Remediation Guide

## Overview

This guide provides comprehensive documentation for detecting, analyzing, and resolving resource conflicts before deploying AgenticDosNode on dedicated Ubuntu machines. The system includes automated detection, cleanup procedures, and validation tools to ensure a clean deployment environment.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture Overview](#architecture-overview)
3. [Conflict Categories](#conflict-categories)
4. [Detection Process](#detection-process)
5. [Cleanup Procedures](#cleanup-procedures)
6. [Validation Process](#validation-process)
7. [Troubleshooting](#troubleshooting)
8. [Best Practices](#best-practices)

## Quick Start

### Step 1: Run Conflict Detection

```bash
# Basic detection (non-root, limited checks)
./resource-conflict-detector.sh

# Full detection (recommended)
sudo ./resource-conflict-detector.sh
```

**Output Files:**
- `conflict-analysis-report-YYYYMMDD-HHMMSS.json` - Detailed JSON report
- `remediate-conflicts-YYYYMMDD-HHMMSS.sh` - Generated remediation script
- `/var/log/agenticdos-conflict-detection.log` - Detailed log file

### Step 2: Review and Remediate

```bash
# Review the generated report
cat conflict-analysis-report-*.json | jq '.summary'

# Run interactive cleanup
sudo ./automated-cleanup-procedures.sh

# Or run specific cleanup operations
sudo ./automated-cleanup-procedures.sh --dry-run  # Preview changes
sudo ./automated-cleanup-procedures.sh --non-interactive  # Automated mode
```

### Step 3: Validate System State

```bash
# Validate system readiness
sudo ./post-cleanup-validator.sh

# Check readiness score
cat validation-report-*.json | jq '.summary'
```

## Architecture Overview

The conflict detection system consists of three main components:

### 1. Resource Conflict Detector (`resource-conflict-detector.sh`)

**Purpose:** Comprehensive system analysis to identify potential conflicts

**Key Features:**
- Port availability scanning
- Service conflict detection
- Docker ecosystem analysis
- GPU resource checking
- Network configuration review
- Python/AI environment detection
- Disk and memory analysis

**Output:**
- JSON report with categorized conflicts
- Severity ratings (critical, high, medium, low)
- Automated remediation script generation

### 2. Automated Cleanup Procedures (`automated-cleanup-procedures.sh`)

**Purpose:** Safe, reversible cleanup of conflicting resources

**Key Features:**
- Interactive menu system
- Dry-run mode for preview
- Automatic backup creation
- Selective preservation of AI/ML data
- Rollback capability
- Non-interactive mode for automation

**Cleanup Categories:**
- Service cleanup (web servers, databases, ML services)
- Docker cleanup (containers, images, volumes, networks)
- Python environment cleanup
- GPU resource management
- Network configuration reset
- Port liberation
- System optimization

### 3. Post-Cleanup Validator (`post-cleanup-validator.sh`)

**Purpose:** Verify system readiness after cleanup

**Validation Checks:**
- Port availability verification
- Service status confirmation
- Docker environment health
- GPU availability and configuration
- Network setup validation
- System resource adequacy
- Python environment check
- Security configuration review

**Output:**
- Readiness score (0-100%)
- Detailed pass/fail/warning report
- Next steps recommendations

## Conflict Categories

### 1. Service Conflicts

**Detection:**
- Web servers on ports 80, 443, 3000
- Database services on standard ports
- AI/ML services (Jupyter, MLflow, Ollama)
- Container orchestration (K8s, Swarm)

**Common Conflicts:**
```yaml
Web Servers:
  - nginx: ports 80, 443
  - apache2: ports 80, 443
  - caddy: ports 80, 443, 2019

Databases:
  - postgresql: port 5432
  - mysql/mariadb: port 3306
  - mongodb: port 27017
  - redis: port 6379

ML Services:
  - jupyter: port 8888
  - mlflow: port 5000
  - tensorboard: port 6006
  - ollama: port 11434
```

### 2. Docker Ecosystem Conflicts

**Detection:**
- AI-related Docker images
- GPU-enabled containers
- Network subnet conflicts (100.64.x.x)
- Volume space usage
- Running containers on required ports

**Resolution Strategy:**
```bash
# Stop containers gracefully
docker stop $(docker ps -q)

# Clean up with preservation
docker system prune -a --volumes --filter "label!=preserve"

# Remove conflicting networks
docker network prune -f
```

### 3. Python/AI Environment Conflicts

**Detection:**
- Conda/Miniconda installations
- Virtual environments with AI packages
- System-wide AI package installations
- Hugging Face cache directories

**Common Locations:**
```
/opt/conda
/opt/miniconda3
$HOME/anaconda3
$HOME/.cache/huggingface
$HOME/.local/lib/python*/site-packages
```

### 4. GPU Resource Conflicts

**Detection:**
- NVIDIA/AMD GPU driver status
- Running GPU processes
- CUDA version conflicts
- Docker GPU runtime configuration
- Mining software detection

**Critical Checks:**
```bash
# GPU memory usage
nvidia-smi --query-gpu=memory.used --format=csv

# GPU processes
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

# CUDA versions
ls -la /usr/local/cuda*
```

### 5. Network and Firewall Conflicts

**Detection:**
- VPN services (Tailscale, Zerotier, WireGuard)
- Firewall rules blocking required ports
- IP forwarding configuration
- NAT/masquerading rules
- Network interface conflicts

**Tailscale Requirements:**
```bash
# Required port
41641/udp

# Subnet range
100.64.0.0/10

# IP forwarding
echo 1 > /proc/sys/net/ipv4/ip_forward
```

## Detection Process

### Phase 1: Initial System Scan

```bash
# Check system information
uname -a
lsb_release -a
free -h
df -h
```

### Phase 2: Service Analysis

```bash
# Active services
systemctl list-units --state=running

# Port usage
ss -tulnp

# Process analysis
ps aux --sort=-%mem | head -20
```

### Phase 3: Deep Inspection

```bash
# Docker inspection
docker system df
docker network ls
docker ps -a

# GPU inspection
nvidia-smi
lspci | grep -i nvidia

# Python packages
pip list | grep -E "torch|tensorflow|transformers"
```

## Cleanup Procedures

### Safe Cleanup Workflow

1. **Create Backups**
   ```bash
   # Automatic backup creation
   BACKUP_DIR="/var/backups/agenticdos-cleanup-$(date +%Y%m%d-%H%M%S)"
   ```

2. **Stop Services Gracefully**
   ```bash
   # Stop with timeout
   systemctl stop service-name --timeout=30
   ```

3. **Preserve Important Data**
   ```bash
   # AI models preservation
   --preserve-data flag maintains:
   - Ollama models
   - Hugging Face cache
   - Docker volumes with ML data
   ```

4. **Clean Incrementally**
   ```bash
   # Progressive cleanup
   1. Stop services
   2. Remove containers
   3. Clean images
   4. Prune volumes
   5. Reset network
   ```

### Rollback Procedure

```bash
# List available backups
ls -la /var/backups/agenticdos-cleanup-*

# Restore from backup
cd /
tar -xzf /var/backups/agenticdos-cleanup-*/backup-name.tar.gz
```

## Validation Process

### Critical Validation Points

1. **Port Availability**
   - All required ports must be free
   - No binding conflicts

2. **Docker Health**
   - Docker daemon running
   - Compose installed
   - GPU runtime configured (if applicable)

3. **Network Configuration**
   - Tailscale installed and authenticated
   - No conflicting VPN services
   - Firewall rules appropriate

4. **System Resources**
   - Minimum 8GB RAM available
   - 50GB+ disk space
   - CPU not under heavy load

### Readiness Scoring

```
Score Interpretation:
- 90-100%: Fully ready for deployment
- 70-89%: Minor issues, can proceed with caution
- 50-69%: Significant issues, resolve before deployment
- <50%: Major conflicts, extensive cleanup required
```

## Troubleshooting

### Common Issues and Solutions

#### Issue: Ports still in use after cleanup

```bash
# Force kill processes on port
lsof -ti:PORT | xargs kill -9

# Check for systemd socket activation
systemctl list-sockets | grep PORT
```

#### Issue: Docker won't start after cleanup

```bash
# Reset Docker to factory defaults
sudo systemctl stop docker
sudo rm -rf /var/lib/docker
sudo systemctl start docker
```

#### Issue: GPU not available to Docker

```bash
# Reinstall nvidia-docker2
sudo apt-get remove --purge nvidia-docker2
sudo apt-get install nvidia-docker2
sudo systemctl restart docker
```

#### Issue: Tailscale conflicts with existing VPN

```bash
# Disable conflicting VPN
sudo systemctl disable --now zerotier-one
sudo systemctl disable --now openvpn

# Reset network namespace
sudo ip netns delete tailscale 2>/dev/null || true
```

## Best Practices

### Pre-Deployment Checklist

- [ ] Run detection script with sudo
- [ ] Review all HIGH and CRITICAL conflicts
- [ ] Backup critical data before cleanup
- [ ] Use dry-run mode first
- [ ] Validate after each cleanup phase
- [ ] Document any manual interventions
- [ ] Test network connectivity
- [ ] Verify GPU accessibility
- [ ] Check disk space after cleanup
- [ ] Review security settings

### Recommended Deployment Order

1. **Clean System State**
   ```bash
   sudo ./resource-conflict-detector.sh
   sudo ./automated-cleanup-procedures.sh
   sudo ./post-cleanup-validator.sh
   ```

2. **Configure Network**
   ```bash
   # Install and configure Tailscale
   curl -fsSL https://tailscale.com/install.sh | sh
   sudo tailscale up
   ```

3. **Prepare Docker**
   ```bash
   # Ensure Docker is ready
   sudo docker run hello-world
   sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
   ```

4. **Deploy AgenticDosNode**
   ```bash
   # Based on node type
   ./setup-thanos.sh  # For GPU node
   ./setup-oracle1.sh  # For CPU node
   ```

### Maintenance Recommendations

**Daily:**
- Monitor port usage
- Check Docker disk usage
- Review system logs

**Weekly:**
- Run validation script
- Clean Docker cache
- Update conflict detection scripts

**Monthly:**
- Full system audit
- Backup configuration
- Review and update firewall rules

## Advanced Configuration

### Custom Port Mappings

Edit the detection script to add custom ports:

```bash
# In resource-conflict-detector.sh
declare -A CUSTOM_PORTS=(
    ["8000"]="My Custom Service"
    ["9000"]="Another Service"
)
```

### Preservation Rules

Configure what to preserve during cleanup:

```bash
# Set environment variables
export PRESERVE_DATA=true
export PRESERVE_MODELS=true
export PRESERVE_VOLUMES=true
```

### Automated Remediation

For CI/CD integration:

```bash
# Non-interactive cleanup
./automated-cleanup-procedures.sh \
    --non-interactive \
    --no-preserve \
    --log-level=debug
```

## Security Considerations

### Sensitive Data Protection

- Backups are created with restricted permissions (600)
- Logs sanitize sensitive information
- API keys and tokens are never logged

### Network Security

- Firewall rules are preserved during cleanup
- SSH access is maintained throughout
- VPN credentials are backed up securely

### Container Security

- Privileged containers are identified
- Root-running containers are flagged
- Capability requirements are documented

## Support and Contributions

### Getting Help

1. Check the troubleshooting section
2. Review logs in `/var/log/agenticdos-*.log`
3. Run validation with verbose mode
4. Consult the AgenticDosNode documentation

### Contributing

Improvements to the detection and cleanup system are welcome:

1. Test thoroughly in a safe environment
2. Document new detection patterns
3. Maintain backward compatibility
4. Include rollback procedures
5. Update this documentation

## Appendix

### Required Ports Reference

#### Thanos Node (GPU)
| Port | Service | Protocol | Description |
|------|---------|----------|-------------|
| 8000 | vLLM | TCP | LLM inference server |
| 6333 | Qdrant | TCP | Vector database primary |
| 8001 | Embeddings | TCP | Embedding generation |
| 8002 | Whisper | TCP | Speech-to-text |
| 8003 | ComfyUI | TCP | Image generation |
| 8004 | Code Interpreter | TCP | Sandboxed execution |
| 11434 | Ollama | TCP | Local model API |

#### Oracle1 Node (CPU)
| Port | Service | Protocol | Description |
|------|---------|----------|-------------|
| 5678 | n8n | TCP | Workflow automation |
| 8080 | LangGraph | TCP | Agent orchestration |
| 6379 | Redis | TCP | Cache and queues |
| 6334 | Qdrant | TCP | Vector database replica |
| 9090 | Prometheus | TCP | Metrics collection |
| 3000 | Grafana | TCP | Monitoring dashboards |
| 8081 | Claude Proxy | TCP | API proxy service |

### Environment Variables

```bash
# Detection settings
export DRY_RUN=false
export INTERACTIVE=true
export PRESERVE_DATA=true

# Cleanup settings
export BACKUP_DIR=/var/backups/agenticdos
export LOG_LEVEL=info
export CLEANUP_TIMEOUT=300

# Validation settings
export STRICT_MODE=false
export MIN_RAM_GB=8
export MIN_DISK_GB=50
```

### Log File Locations

```
/var/log/agenticdos-conflict-detection.log  # Detection logs
/var/log/agenticdos-cleanup.log            # Cleanup logs
/var/log/agenticdos-validation.log         # Validation logs
./conflict-analysis-report-*.json          # Detection reports
./validation-report-*.json                 # Validation reports
/var/backups/agenticdos-cleanup-*/         # Backup directories
```

---

*Last Updated: 2025*
*Version: 1.0.0*
*AgenticDosNode Conflict Detection System*