#!/bin/bash

# ARM64 Optimized Startup Script for Oracle Ampere CPU
# This script configures and starts the Docker Compose stack with ARM64 optimizations

set -e

echo "======================================"
echo "Oracle ARM64 Stack Startup Script"
echo "======================================"
echo ""

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check if running on ARM64
ARCH=$(uname -m)
if [[ "$ARCH" != "aarch64" ]] && [[ "$ARCH" != "arm64" ]]; then
    echo -e "${RED}Warning: This script is optimized for ARM64 architecture.${NC}"
    echo -e "Current architecture: $ARCH"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Function to check prerequisites
check_prerequisites() {
    echo "Checking prerequisites..."

    # Check Docker
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Docker is not installed!${NC}"
        exit 1
    fi

    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        if ! docker compose version &> /dev/null; then
            echo -e "${RED}Docker Compose is not installed!${NC}"
            exit 1
        fi
        COMPOSE_CMD="docker compose"
    else
        COMPOSE_CMD="docker-compose"
    fi

    # Check available memory
    TOTAL_MEM=$(grep MemTotal /proc/meminfo | awk '{print $2}')
    TOTAL_MEM_GB=$((TOTAL_MEM / 1024 / 1024))

    if [ $TOTAL_MEM_GB -lt 16 ]; then
        echo -e "${YELLOW}Warning: System has ${TOTAL_MEM_GB}GB RAM. Recommended: 24GB+${NC}"
    else
        echo -e "${GREEN}✓ Memory: ${TOTAL_MEM_GB}GB${NC}"
    fi

    # Check available CPUs
    CPU_COUNT=$(nproc)
    if [ $CPU_COUNT -lt 4 ]; then
        echo -e "${YELLOW}Warning: System has ${CPU_COUNT} CPUs. Recommended: 4+${NC}"
    else
        echo -e "${GREEN}✓ CPUs: ${CPU_COUNT}${NC}"
    fi
}

# Function to setup environment
setup_environment() {
    echo ""
    echo "Setting up environment..."

    # Copy ARM64 optimized env file if .env doesn't exist
    if [ ! -f .env ]; then
        if [ -f .env.arm64 ]; then
            cp .env.arm64 .env
            echo -e "${GREEN}✓ Created .env from ARM64 template${NC}"
        else
            echo -e "${YELLOW}Warning: No .env file found. Using defaults.${NC}"
        fi
    fi

    # Create necessary directories
    mkdir -p kong/plugins kong/ssl
    mkdir -p grafana/provisioning/dashboards grafana/provisioning/datasources
    mkdir -p prometheus
    mkdir -p init tailscale
    echo -e "${GREEN}✓ Created necessary directories${NC}"

    # Create Kong SSL certificates if they don't exist
    if [ ! -f kong/ssl/kong.crt ]; then
        echo "Generating self-signed SSL certificates for Kong..."
        openssl req -x509 -nodes -newkey rsa:2048 \
            -keyout kong/ssl/kong.key \
            -out kong/ssl/kong.crt \
            -days 365 \
            -subj "/CN=kong.local" 2>/dev/null
        echo -e "${GREEN}✓ Generated SSL certificates${NC}"
    fi
}

# Function to initialize Kong database
init_kong_db() {
    echo ""
    echo "Initializing Kong database..."

    # Start only PostgreSQL first
    $COMPOSE_CMD up -d postgres

    # Wait for PostgreSQL to be ready
    echo "Waiting for PostgreSQL to be ready..."
    sleep 10

    # Run Kong migrations
    docker run --rm \
        --platform linux/arm64 \
        --network oracle1_agentic_net \
        -e KONG_DATABASE=postgres \
        -e KONG_PG_HOST=postgres \
        -e KONG_PG_DATABASE=kong \
        -e KONG_PG_USER=agentic \
        -e KONG_PG_PASSWORD=changeme \
        kong:3.7-ubuntu kong migrations bootstrap || true

    echo -e "${GREEN}✓ Kong database initialized${NC}"
}

# Function to optimize system for ARM64
optimize_system() {
    echo ""
    echo "Applying ARM64 system optimizations..."

    # These require root privileges
    if [ "$EUID" -eq 0 ]; then
        # Set swappiness
        sysctl -w vm.swappiness=10 2>/dev/null || true

        # Increase file descriptors
        sysctl -w fs.file-max=100000 2>/dev/null || true

        # Network optimizations
        sysctl -w net.core.somaxconn=32768 2>/dev/null || true
        sysctl -w net.ipv4.tcp_max_syn_backlog=8096 2>/dev/null || true
        sysctl -w net.core.netdev_max_backlog=5000 2>/dev/null || true

        echo -e "${GREEN}✓ System optimizations applied${NC}"
    else
        echo -e "${YELLOW}Skipping system optimizations (requires root)${NC}"
    fi
}

# Function to start the stack
start_stack() {
    echo ""
    echo "Starting Docker Compose stack..."

    # Pull latest images
    echo "Pulling ARM64 compatible images..."
    $COMPOSE_CMD pull --ignore-pull-failures || true

    # Start all services
    $COMPOSE_CMD up -d

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Stack started successfully!${NC}"
    else
        echo -e "${RED}Failed to start stack${NC}"
        exit 1
    fi
}

# Function to show status
show_status() {
    echo ""
    echo "======================================"
    echo "Service Status:"
    echo "======================================"
    $COMPOSE_CMD ps

    echo ""
    echo "======================================"
    echo "Service URLs:"
    echo "======================================"
    echo "PostgreSQL:     postgres://localhost:5432"
    echo "Redis:          redis://localhost:6379"
    echo "n8n:            https://localhost:5678"
    echo "LangGraph API:  http://localhost:8080"
    echo "Kong Gateway:   http://localhost:8000"
    echo "Kong Admin:     http://localhost:8001"
    echo "Claude Proxy:   http://localhost:8081"
    echo "Qdrant:         http://localhost:6333"
    echo "Prometheus:     http://localhost:9090"
    echo "Grafana:        http://localhost:3000"
    echo "======================================"

    echo ""
    echo -e "${GREEN}Stack is ready for use!${NC}"
    echo ""
    echo "To view logs: $COMPOSE_CMD logs -f [service-name]"
    echo "To stop:      $COMPOSE_CMD down"
    echo "To restart:   $COMPOSE_CMD restart [service-name]"
}

# Main execution
main() {
    check_prerequisites
    setup_environment
    optimize_system
    init_kong_db
    start_stack
    show_status
}

# Run main function
main "$@"