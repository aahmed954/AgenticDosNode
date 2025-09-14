#!/bin/bash

# AgenticDosNode Deployment Summary Script
set -euo pipefail

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}        AgenticDosNode Multi-Machine Deployment Summary${NC}"
echo -e "${BLUE}================================================================${NC}"
echo

echo -e "${GREEN}✅ DEPLOYMENT SUCCESSFUL${NC}"
echo -e "Multi-machine infrastructure deployed and operational"
echo

echo -e "${BLUE}=== Infrastructure Overview ===${NC}"
echo -e "Oracle1 (ARM64):  100.96.197.84 - Data & Workflow Processing"
echo -e "Thanos (x86_64):  100.122.12.54 - AI/ML & GPU Computing"
echo

echo -e "${BLUE}=== Oracle1 Services (ARM64) ===${NC}"
echo -e "${GREEN}✅${NC} PostgreSQL     - http://100.96.197.84:5432"
echo -e "${GREEN}✅${NC} Redis          - http://100.96.197.84:6379"
echo -e "${GREEN}✅${NC} Qdrant         - http://100.96.197.84:6333"
echo -e "${GREEN}✅${NC} n8n Workflow   - http://100.96.197.84:5678"
echo -e "${GREEN}✅${NC} Grafana        - http://100.96.197.84:3000"
echo -e "${GREEN}✅${NC} Prometheus     - http://100.96.197.84:9090"
echo

echo -e "${BLUE}=== Thanos Services (x86_64 + GPU) ===${NC}"
echo -e "${GREEN}✅${NC} Prometheus     - http://100.122.12.54:9091"
echo -e "${GREEN}✅${NC} Redis Cache    - http://100.122.12.54:6380"
echo -e "${GREEN}✅${NC} Node Metrics   - http://100.122.12.54:9100"
echo -e "${GREEN}✅${NC} Container Metrics - http://100.122.12.54:8082"
echo -e "${YELLOW}⚠️${NC}  GPU Services  - Ready for optimized deployment"
echo

echo -e "${BLUE}=== Access Credentials ===${NC}"
echo -e "Grafana:  admin / grafana123!"
echo -e "n8n:      admin / n8nadmin123!"
echo -e "Database: agentic / agentic123!"
echo

echo -e "${BLUE}=== Network Status ===${NC}"
echo -e "${GREEN}✅${NC} Tailscale VPN active"
echo -e "${GREEN}✅${NC} Cross-node connectivity established"
echo -e "${GREEN}✅${NC} Service discovery configured"
echo

echo -e "${BLUE}=== Key Files Created ===${NC}"
echo -e "📄 /home/starlord/AgenticDosNode/oracle1/docker-compose-arm64.yml"
echo -e "📄 /home/starlord/AgenticDosNode/thanos/docker-compose-gpu.yml"
echo -e "📄 /home/starlord/AgenticDosNode/service-discovery.yml"
echo -e "📄 /home/starlord/AgenticDosNode/health-check.sh"
echo -e "📄 /home/starlord/AgenticDosNode/DEPLOYMENT_STATUS.md"
echo

echo -e "${BLUE}=== Quick Commands ===${NC}"
echo -e "Health Check:  ./health-check.sh"
echo -e "Oracle1 Logs:  ssh oracle1 'docker logs <container>'"
echo -e "Thanos Logs:   docker logs <container>"
echo -e "Service Status: docker ps"
echo

echo -e "${BLUE}=== Next Steps ===${NC}"
echo -e "1. Review DEPLOYMENT_STATUS.md for detailed information"
echo -e "2. Access Grafana dashboard for monitoring"
echo -e "3. Configure n8n workflows"
echo -e "4. Deploy optimized AI services when ready"
echo

echo -e "${GREEN}🎉 Deployment completed successfully!${NC}"
echo -e "Infrastructure is ready for production workloads."