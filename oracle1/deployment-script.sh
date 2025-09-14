#!/bin/bash
# Oracle1 AgenticDosNode Deployment Script
# Fixes ARM64 compatibility and deployment conflicts

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Oracle1 AgenticDosNode Deployment Script${NC}"
echo -e "${BLUE}============================================${NC}"

# Check if running on ARM64
ARCH=$(uname -m)
echo -e "${BLUE}System Architecture: ${ARCH}${NC}"

if [ "$ARCH" != "x86_64" ] && [ "$ARCH" != "aarch64" ] && [ "$ARCH" != "arm64" ]; then
    echo -e "${RED}❌ Unsupported architecture: ${ARCH}${NC}"
    exit 1
fi

# Set working directory
cd "$(dirname "$0")"
ORACLE1_DIR=$(pwd)

echo -e "${BLUE}Working directory: ${ORACLE1_DIR}${NC}"

# Function to check if port is in use
check_port() {
    local port=$1
    if netstat -tuln | grep -q ":${port} "; then
        echo -e "${YELLOW}⚠️  Port ${port} is in use${NC}"
        return 1
    fi
    return 0
}

# Function to stop conflicting services
stop_conflicting_services() {
    echo -e "${YELLOW}🔍 Checking for port conflicts...${NC}"

    local ports=(5432 6379 8000 8001 5678 6333 9090 3000)
    local conflicts=false

    for port in "${ports[@]}"; do
        if ! check_port "$port"; then
            conflicts=true
            echo -e "${YELLOW}Port ${port} conflict detected${NC}"
        fi
    done

    if [ "$conflicts" = true ]; then
        echo -e "${YELLOW}🛑 Stopping existing containers to resolve conflicts...${NC}"
        docker compose down --remove-orphans 2>/dev/null || true
        sleep 5
    fi
}

# Function to validate environment
validate_environment() {
    echo -e "${BLUE}🔍 Validating environment...${NC}"

    # Check Docker
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}❌ Docker is not installed${NC}"
        exit 1
    fi

    # Check Docker Compose
    if ! docker compose version &> /dev/null; then
        echo -e "${RED}❌ Docker Compose is not available${NC}"
        exit 1
    fi

    # Check .env file
    if [ ! -f ".env" ]; then
        echo -e "${YELLOW}⚠️  .env file not found, using defaults${NC}"
    fi

    # Create required directories
    mkdir -p init-scripts prometheus process-exporter tailscale

    echo -e "${GREEN}✅ Environment validation complete${NC}"
}

# Function to pull ARM64 images
pull_images() {
    echo -e "${BLUE}📥 Pulling ARM64-compatible Docker images...${NC}"

    local images=(
        "postgres:16-alpine"
        "redis:7-alpine"
        "n8nio/n8n:latest"
        "kong:latest"
        "qdrant/qdrant:latest"
        "prom/prometheus:latest"
        "grafana/grafana:latest"
        "prom/node-exporter:latest"
        "ncabatoff/process-exporter:latest"
        "gcr.io/cadvisor/cadvisor:latest"
        "tailscale/tailscale:latest"
    )

    for image in "${images[@]}"; do
        echo -e "${BLUE}Pulling ${image}...${NC}"
        if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
            docker pull --platform linux/arm64 "$image" || echo -e "${YELLOW}⚠️  Failed to pull $image${NC}"
        else
            docker pull "$image" || echo -e "${YELLOW}⚠️  Failed to pull $image${NC}"
        fi
    done

    echo -e "${GREEN}✅ Image pulling complete${NC}"
}

# Function to initialize Kong database
init_kong_database() {
    echo -e "${BLUE}🗄️  Initializing Kong database...${NC}"

    # Wait for PostgreSQL to be ready
    echo -e "${BLUE}Waiting for PostgreSQL...${NC}"
    docker compose -f docker-compose-fixed.yml up -d postgres

    sleep 30

    # Run Kong migrations
    echo -e "${BLUE}Running Kong migrations...${NC}"
    docker compose -f docker-compose-fixed.yml run --rm kong kong migrations bootstrap || \
    docker compose -f docker-compose-fixed.yml run --rm kong kong migrations up || \
    echo -e "${YELLOW}⚠️  Kong migrations may have already been run${NC}"

    echo -e "${GREEN}✅ Kong database initialization complete${NC}"
}

# Function to deploy services
deploy_services() {
    echo -e "${BLUE}🚀 Deploying Oracle1 services...${NC}"

    # Use the fixed compose file
    local compose_file="docker-compose-fixed.yml"

    if [ ! -f "$compose_file" ]; then
        echo -e "${RED}❌ Fixed compose file not found: $compose_file${NC}"
        exit 1
    fi

    # Deploy in stages for proper dependency resolution
    echo -e "${BLUE}Stage 1: Core infrastructure (PostgreSQL, Redis)${NC}"
    docker compose -f "$compose_file" up -d postgres redis
    sleep 20

    echo -e "${BLUE}Stage 2: Vector database (Qdrant)${NC}"
    docker compose -f "$compose_file" up -d qdrant-replica
    sleep 10

    echo -e "${BLUE}Stage 3: API Gateway (Kong)${NC}"
    docker compose -f "$compose_file" up -d kong
    sleep 15

    echo -e "${BLUE}Stage 4: Application services (n8n)${NC}"
    docker compose -f "$compose_file" up -d n8n
    sleep 10

    echo -e "${BLUE}Stage 5: Monitoring (Prometheus, Grafana)${NC}"
    docker compose -f "$compose_file" up -d prometheus grafana
    sleep 10

    echo -e "${BLUE}Stage 6: System monitoring (Node Exporter, cAdvisor, Process Exporter)${NC}"
    docker compose -f "$compose_file" up -d node-exporter cadvisor process-exporter
    sleep 5

    echo -e "${BLUE}Stage 7: Network (Tailscale - if configured)${NC}"
    if [ -n "${TAILSCALE_AUTHKEY}" ]; then
        docker compose -f "$compose_file" up -d tailscale
    else
        echo -e "${YELLOW}⚠️  Tailscale not configured (TAILSCALE_AUTHKEY not set)${NC}"
    fi

    echo -e "${GREEN}✅ Service deployment complete${NC}"
}

# Function to check service health
check_health() {
    echo -e "${BLUE}🏥 Checking service health...${NC}"

    local services=(
        "postgres:5432"
        "redis:6379"
        "kong:8001"
        "n8n:5678"
        "qdrant-replica:6333"
        "prometheus:9090"
        "grafana:3000"
    )

    for service in "${services[@]}"; do
        local name=$(echo "$service" | cut -d: -f1)
        local port=$(echo "$service" | cut -d: -f2)

        echo -n -e "${BLUE}Checking $name ($port)... ${NC}"
        if docker compose -f docker-compose-fixed.yml ps "$name" | grep -q "Up"; then
            echo -e "${GREEN}✅ Running${NC}"
        else
            echo -e "${RED}❌ Not running${NC}"
        fi
    done
}

# Function to display access information
show_access_info() {
    local node_ip=${NODE_IP:-100.96.197.84}

    echo -e "${GREEN}🎉 Oracle1 AgenticDosNode Deployment Complete!${NC}"
    echo -e "${GREEN}================================================${NC}"
    echo -e "${BLUE}Access URLs:${NC}"
    echo -e "🔧 n8n Automation:     http://${node_ip}:5678"
    echo -e "🛡️  Kong Admin:         http://${node_ip}:8001"
    echo -e "🛡️  Kong Proxy:         http://${node_ip}:8000"
    echo -e "📊 Prometheus:         http://${node_ip}:9090"
    echo -e "📈 Grafana:            http://${node_ip}:3000"
    echo -e "🔍 Qdrant:             http://${node_ip}:6333"
    echo -e "📊 Node Exporter:      http://${node_ip}:9100"
    echo -e "📊 cAdvisor:           http://${node_ip}:8082"
    echo -e "📊 Process Exporter:   http://${node_ip}:9256"
    echo ""
    echo -e "${YELLOW}Default Credentials:${NC}"
    echo -e "📊 Grafana: admin / grafana123!"
    echo -e "🔧 n8n: admin / n8nadmin123!"
    echo -e "🗄️  PostgreSQL: agentic / agentic123!"
    echo -e "🗄️  Redis: redis123!"
}

# Main deployment flow
main() {
    validate_environment
    stop_conflicting_services
    pull_images
    init_kong_database
    deploy_services
    sleep 30  # Allow services to fully start
    check_health
    show_access_info
}

# Run main function
main "$@"