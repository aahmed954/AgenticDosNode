#!/bin/bash

# AgenticDosNode Multi-Node Health Check Script
# Monitors all services across Oracle1 and Thanos

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Node IPs
ORACLE1_IP="100.96.197.84"
THANOS_IP="100.122.12.54"

# Function to check HTTP endpoint
check_http() {
    local url="$1"
    local service_name="$2"
    local timeout="${3:-5}"

    if curl -s --max-time "$timeout" "$url" > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} $service_name: $url"
        return 0
    else
        echo -e "${RED}✗${NC} $service_name: $url"
        return 1
    fi
}

# Function to check TCP port
check_port() {
    local host="$1"
    local port="$2"
    local service_name="$3"
    local timeout="${4:-5}"

    if timeout "$timeout" bash -c "echo >/dev/tcp/$host/$port" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} $service_name: $host:$port"
        return 0
    else
        echo -e "${RED}✗${NC} $service_name: $host:$port"
        return 1
    fi
}

# Function to check Docker container
check_container() {
    local node="$1"
    local container_name="$2"

    if [ "$node" = "local" ]; then
        status=$(docker ps --filter "name=$container_name" --format "{{.Status}}" 2>/dev/null || echo "not found")
    else
        status=$(ssh "$node" "docker ps --filter 'name=$container_name' --format '{{.Status}}'" 2>/dev/null || echo "not found")
    fi

    if [[ "$status" =~ ^Up.*healthy\) ]] || [[ "$status" =~ ^Up[[:space:]] ]]; then
        echo -e "${GREEN}✓${NC} $container_name: $status"
        return 0
    else
        echo -e "${RED}✗${NC} $container_name: $status"
        return 1
    fi
}

# Function to get GPU info
check_gpu() {
    echo -e "${BLUE}=== GPU Status ===${NC}"

    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits
    else
        echo -e "${YELLOW}⚠${NC} nvidia-smi not available"
    fi
    echo
}

# Function to check Tailscale connectivity
check_tailscale() {
    echo -e "${BLUE}=== Tailscale Connectivity ===${NC}"

    if ping -c 1 "$ORACLE1_IP" > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} Oracle1 reachable: $ORACLE1_IP"
    else
        echo -e "${RED}✗${NC} Oracle1 unreachable: $ORACLE1_IP"
    fi

    if ping -c 1 "$THANOS_IP" > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} Thanos reachable: $THANOS_IP"
    else
        echo -e "${RED}✗${NC} Thanos unreachable: $THANOS_IP"
    fi
    echo
}

echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}           AgenticDosNode Multi-Node Health Check${NC}"
echo -e "${BLUE}================================================================${NC}"
echo

check_tailscale
check_gpu

echo -e "${BLUE}=== Oracle1 Services (ARM64) ===${NC}"
echo -e "${YELLOW}Database & Workflow Processing Node${NC}"
echo

# Oracle1 Database Services
check_port "$ORACLE1_IP" 5432 "PostgreSQL"
check_port "$ORACLE1_IP" 6379 "Redis"
check_container "oracle1" "oracle1-postgres"
check_container "oracle1" "oracle1-redis"

echo

# Oracle1 Application Services
check_port "$ORACLE1_IP" 6333 "Qdrant Vector DB"
check_http "http://$ORACLE1_IP:6333/" "Qdrant API"
check_http "http://$ORACLE1_IP:5678/" "n8n Workflow"
check_container "oracle1" "oracle1-qdrant"
check_container "oracle1" "oracle1-n8n"

echo

# Oracle1 Monitoring Services
check_http "http://$ORACLE1_IP:9090/-/healthy" "Prometheus"
check_http "http://$ORACLE1_IP:3000/api/health" "Grafana"
check_http "http://$ORACLE1_IP:9100/metrics" "Node Exporter"
check_http "http://$ORACLE1_IP:9256/metrics" "Process Exporter"
check_http "http://$ORACLE1_IP:8081/metrics" "cAdvisor"
check_container "oracle1" "oracle1-prometheus"
check_container "oracle1" "oracle1-grafana"

echo
echo -e "${BLUE}=== Thanos Services (x86_64 + GPU) ===${NC}"
echo -e "${YELLOW}AI/ML Processing Node${NC}"
echo

# Thanos Infrastructure Services
check_port "$THANOS_IP" 6380 "Redis Cache"
check_container "local" "thanos-redis"

echo

# Thanos AI Services
check_http "http://$THANOS_IP:8000/health" "vLLM Inference"
check_http "http://$THANOS_IP:8001/health" "Embedding Service"
check_http "http://$THANOS_IP:8002/health" "LangGraph API"
check_http "http://$THANOS_IP:8188/" "ComfyUI"
check_container "local" "thanos-vllm"
check_container "local" "thanos-embeddings"
check_container "local" "thanos-langgraph"
check_container "local" "thanos-comfyui"

echo

# Thanos Monitoring Services
check_http "http://$THANOS_IP:9091/-/healthy" "Prometheus"
check_http "http://$THANOS_IP:9100/metrics" "Node Exporter"
check_http "http://$THANOS_IP:8082/metrics" "cAdvisor"
check_http "http://$THANOS_IP:9445/metrics" "GPU Exporter"
check_container "local" "thanos-prometheus"
check_container "local" "thanos-node-exporter"
check_container "local" "thanos-cadvisor"
check_container "local" "thanos-gpu-exporter"

echo
echo -e "${BLUE}=== Cross-Node Connectivity ===${NC}"

# Test cross-node service access
check_http "http://$ORACLE1_IP:9090/api/v1/query?query=up" "Oracle1 Prometheus API"
check_http "http://$THANOS_IP:9091/api/v1/query?query=up" "Thanos Prometheus API"

echo
echo -e "${BLUE}=== Service Status Summary ===${NC}"

# Count healthy vs unhealthy services
oracle1_services=$(ssh oracle1 "docker ps --format '{{.Names}}' | wc -l" 2>/dev/null || echo "0")
thanos_services=$(docker ps --format '{{.Names}}' | grep -c "thanos-" || echo "0")

echo -e "Oracle1 Running Services: ${GREEN}$oracle1_services${NC}"
echo -e "Thanos Running Services: ${GREEN}$thanos_services${NC}"

echo
echo -e "${BLUE}Health check completed at $(date)${NC}"