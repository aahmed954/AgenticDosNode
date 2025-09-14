#!/bin/bash

# ARM64 Docker Image Compatibility Verification Script
# For Oracle Cloud Infrastructure Ampere CPU instances

set -e

echo "=================================================="
echo "ARM64 Docker Image Compatibility Verification"
echo "=================================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if image supports ARM64
check_arm64_support() {
    local image=$1
    local name=$2

    echo -n "Checking $name ($image)... "

    # Try to inspect the manifest
    if docker manifest inspect "$image" 2>/dev/null | grep -q "linux/arm64"; then
        echo -e "${GREEN}✓ ARM64 supported${NC}"
        return 0
    else
        # Try alternative check with docker pull dry-run
        if docker pull --platform=linux/arm64 "$image" --dry-run 2>&1 | grep -q "Downloaded"; then
            echo -e "${GREEN}✓ ARM64 supported${NC}"
            return 0
        else
            echo -e "${RED}✗ ARM64 NOT supported${NC}"
            return 1
        fi
    fi
}

# Function to get alternative image if needed
suggest_alternative() {
    local image=$1
    case "$image" in
        "kong:3.4-alpine")
            echo -e "${YELLOW}  Alternative: Use kong:3.7-ubuntu or kong:latest${NC}"
            ;;
        "restic/restic:latest")
            echo -e "${YELLOW}  Alternative: Use instrumentisto/restic:latest${NC}"
            ;;
        *)
            echo -e "${YELLOW}  Check Docker Hub for ARM64 compatible versions${NC}"
            ;;
    esac
}

echo "System Architecture Information:"
echo "--------------------------------"
uname -m
echo ""
docker version --format 'Docker Version: {{.Server.Version}}'
echo ""

echo "Checking Docker Images for ARM64 Support:"
echo "-----------------------------------------"

# List of images to check
declare -a images=(
    "postgres:15-alpine:PostgreSQL"
    "redis:7-alpine:Redis"
    "n8nio/n8n:latest:n8n Automation"
    "kong:3.7-ubuntu:Kong API Gateway"
    "qdrant/qdrant:latest:Qdrant Vector DB"
    "prom/prometheus:latest:Prometheus"
    "grafana/grafana:latest:Grafana"
    "prom/node-exporter:latest:Node Exporter"
    "tailscale/tailscale:latest:Tailscale"
    "instrumentisto/restic:latest:Restic Backup"
)

failed_count=0
success_count=0

for item in "${images[@]}"; do
    IFS=':' read -r image tag name <<< "$item"
    full_image="${image}:${tag}"

    if check_arm64_support "$full_image" "$name"; then
        ((success_count++))
    else
        ((failed_count++))
        suggest_alternative "$full_image"
    fi
done

echo ""
echo "=================================================="
echo "Verification Summary:"
echo "  Successful: $success_count"
echo "  Failed: $failed_count"
echo ""

if [ $failed_count -eq 0 ]; then
    echo -e "${GREEN}✓ All images support ARM64!${NC}"
    echo "You can proceed with deployment on Oracle Ampere CPU."
else
    echo -e "${YELLOW}⚠ Some images need alternatives for ARM64.${NC}"
    echo "Please update docker-compose.yml with suggested alternatives."
fi

echo ""
echo "Oracle Ampere CPU Optimization Tips:"
echo "------------------------------------"
echo "1. Use A1.Flex instances with at least 4 OCPUs"
echo "2. Allocate 24GB+ RAM for this stack"
echo "3. Use block volumes for database storage"
echo "4. Enable hugepages for PostgreSQL: echo 'vm.nr_hugepages=512' >> /etc/sysctl.conf"
echo "5. Set CPU governor to performance: cpupower frequency-set -g performance"
echo "6. Use the .env.arm64 file for optimized settings"
echo ""
echo "To apply ARM64 optimizations:"
echo "  cp .env.arm64 .env"
echo "  docker-compose --env-file .env up -d"
echo "=================================================="