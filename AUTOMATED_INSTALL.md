# AgenticDosNode Automated Installation Guide

## Repository: https://github.com/aahmed954/AgenticDosNode

This guide provides automated installation commands for dedicated machine deployment.

## 🚀 Quick Automated Installation

### For Thanos (GPU Machine - RTX 3080, 64GB RAM)
```bash
# One-command automated setup
curl -sSL https://raw.githubusercontent.com/aahmed954/AgenticDosNode/main/scripts/full-setup.sh | sudo bash -s GPU auto

# Alternative: Clone and run locally
git clone https://github.com/aahmed954/AgenticDosNode.git /opt/AgenticDosNode
cd /opt/AgenticDosNode
sudo ./scripts/full-setup.sh GPU auto
```

### For Oracle1 (CPU Machine - Ampere, 24GB RAM)
```bash
# One-command automated setup
curl -sSL https://raw.githubusercontent.com/aahmed954/AgenticDosNode/main/scripts/full-setup.sh | sudo bash -s CPU auto

# Alternative: Clone and run locally
git clone https://github.com/aahmed954/AgenticDosNode.git /opt/AgenticDosNode
cd /opt/AgenticDosNode
sudo ./scripts/full-setup.sh CPU auto
```

## 📋 What the Automated Install Does

### Phase 1: System Preparation (5-10 minutes)
- ✅ Detects and stops all conflicting services
- ✅ Cleans up existing AI/ML installations
- ✅ Removes conflicting Docker containers and images
- ✅ Frees up all required ports
- ✅ Creates encrypted backups of important data

### Phase 2: Security Hardening (5 minutes)
- ✅ Configures UFW firewall with AI-specific rules
- ✅ Hardens SSH configuration
- ✅ Sets up fail2ban for intrusion detection
- ✅ Configures automatic security updates
- ✅ Creates dedicated AgenticDosNode user

### Phase 3: Performance Optimization (10 minutes)
- ✅ Optimizes CPU governors and memory settings
- ✅ Configures GPU settings (thanos only)
- ✅ Tunes kernel parameters for AI workloads
- ✅ Optimizes Docker for high-performance AI
- ✅ Sets up database performance tuning

### Phase 4: Tailscale Network Setup (5 minutes)
- ✅ Installs and configures Tailscale
- ✅ Sets up private mesh networking
- ✅ Configures ACLs for secure communication
- ✅ Tests inter-node connectivity

### Phase 5: AgenticDosNode Deployment (15-20 minutes)
- ✅ Deploys all Docker services
- ✅ Configures Claude proxy and OpenRouter integration
- ✅ Sets up vector database with replication
- ✅ Deploys n8n automation workflows
- ✅ Configures monitoring and alerting

### Phase 6: Validation & Testing (5 minutes)
- ✅ Runs comprehensive health checks
- ✅ Tests AI model routing
- ✅ Validates RAG pipeline
- ✅ Confirms cost optimization systems
- ✅ Generates deployment report

## 🔧 Prerequisites

### Both Machines
- Ubuntu 24.04+ (fresh installation recommended)
- Sudo access
- Internet connectivity
- At least 20GB free disk space

### Thanos (GPU Machine)
- NVIDIA RTX 3080 with latest drivers
- 64GB RAM
- CUDA-capable GPU

### Oracle1 (CPU Machine)
- 4+ CPU cores
- 24GB RAM
- Good network connectivity

## 🎛️ Environment Variables (Optional)

Create `/opt/AgenticDosNode/.env` before installation to customize:

```bash
# API Keys (can be set after installation)
ANTHROPIC_API_KEY=your_claude_api_key_here
OPENROUTER_API_KEY=your_openrouter_key_here

# Tailscale (will prompt if not set)
TAILSCALE_AUTH_KEY=your_tailscale_auth_key

# Custom Configuration
AGENTICNODE_DOMAIN=your-domain.com
ADMIN_EMAIL=admin@your-domain.com

# Database Passwords (auto-generated if not set)
POSTGRES_PASSWORD=your_secure_password
REDIS_PASSWORD=your_secure_password
```

## 📊 Installation Progress Tracking

The automated installer provides real-time progress:

```
🚀 AgenticDosNode Automated Installation
Repository: https://github.com/aahmed954/AgenticDosNode

[1/6] System Preparation          ████████████████████ 100%
[2/6] Security Hardening          ████████████████████ 100%
[3/6] Performance Optimization    ██████████████████░░  90%
[4/6] Tailscale Network Setup     ░░░░░░░░░░░░░░░░░░░░   0%
[5/6] AgenticDosNode Deployment   ░░░░░░░░░░░░░░░░░░░░   0%
[6/6] Validation & Testing        ░░░░░░░░░░░░░░░░░░░░   0%

Current: Configuring GPU performance settings...
Estimated time remaining: 12 minutes
```

## 🔍 Post-Installation Access

After successful installation:

### Service URLs
- **Demo Application**: http://localhost:3000
- **n8n Automation**: http://localhost:5678
- **Monitoring Dashboard**: http://localhost:9090
- **Vector Database**: http://localhost:6333
- **API Orchestrator**: http://localhost:8000

### Tailscale Mesh Access
- **Thanos**: http://thanos.your-tailnet.ts.net:3000
- **Oracle1**: http://oracle1.your-tailnet.ts.net:3000

### Log Files
- **Installation Log**: `/opt/AgenticDosNode/logs/installation.log`
- **Service Logs**: `/opt/AgenticDosNode/logs/`
- **Error Logs**: `/opt/AgenticDosNode/logs/errors.log`

## 🛠️ Troubleshooting

### Installation Fails
```bash
# Check installation log
tail -f /opt/AgenticDosNode/logs/installation.log

# Retry with debug output
sudo bash -x /opt/AgenticDosNode/scripts/full-setup.sh GPU auto

# Manual rollback if needed
sudo /opt/AgenticDosNode/scripts/rollback.sh
```

### Service Issues
```bash
# Check service status
sudo /opt/AgenticDosNode/scripts/validate-deployment.sh

# Restart services
sudo docker-compose -f /opt/AgenticDosNode/docker-compose.yml restart

# View service logs
sudo docker-compose -f /opt/AgenticDosNode/docker-compose.yml logs -f
```

### Network Issues
```bash
# Check Tailscale status
sudo tailscale status

# Test connectivity
sudo /opt/AgenticDosNode/scripts/test-connectivity.sh

# Reset network configuration
sudo /opt/AgenticDosNode/scripts/reset-network.sh
```

## ⚡ Quick Commands

### Check Installation Status
```bash
sudo /opt/AgenticDosNode/scripts/status.sh
```

### View Real-time Logs
```bash
sudo /opt/AgenticDosNode/scripts/logs.sh --follow
```

### Update System
```bash
sudo /opt/AgenticDosNode/scripts/update.sh
```

### Backup Configuration
```bash
sudo /opt/AgenticDosNode/scripts/backup.sh
```

## 🎯 Next Steps After Installation

1. **Configure API Keys**: Add your Anthropic and OpenRouter API keys
2. **Test Demo App**: Visit http://localhost:3000 to test functionality
3. **Set up Workflows**: Configure n8n automation at http://localhost:5678
4. **Monitor Costs**: Check real-time cost tracking in the dashboard
5. **Scale Usage**: Start with simple queries and scale up to complex workflows

## 📞 Support

- **Documentation**: Check `/opt/AgenticDosNode/docs/`
- **Issues**: https://github.com/aahmed954/AgenticDosNode/issues
- **Logs**: `/opt/AgenticDosNode/logs/` for detailed troubleshooting

---

**Ready to deploy your enterprise-grade agentic AI stack in under 45 minutes!** 🚀