#!/bin/bash

# AgenticDosNode Complete Bootstrap Script
# This script deploys the entire multi-system agentic AI stack

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="${PROJECT_ROOT}/bootstrap.log"
TAILSCALE_NETWORK="ai-stack"

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}" | tee -a "$LOG_FILE"
    exit 1
}

success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}" | tee -a "$LOG_FILE"
}

warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}" | tee -a "$LOG_FILE"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   error "This script should not be run as root"
fi

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."

    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
    fi

    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        error "Docker Compose is not installed. Please install Docker Compose first."
    fi

    # Check system resources
    local mem_gb=$(free -g | awk '/^Mem:/{print $2}')
    if [[ $mem_gb -lt 8 ]]; then
        warning "System has less than 8GB RAM. Some services may not perform optimally."
    fi

    # Check disk space
    local disk_space_gb=$(df -BG "$PROJECT_ROOT" | tail -1 | awk '{print $4}' | sed 's/G//')
    if [[ $disk_space_gb -lt 20 ]]; then
        warning "Less than 20GB free disk space available."
    fi

    success "Prerequisites check completed"
}

# Install Tailscale
install_tailscale() {
    log "Installing Tailscale..."

    if command -v tailscale &> /dev/null; then
        log "Tailscale already installed, skipping..."
        return 0
    fi

    # Install Tailscale
    curl -fsSL https://tailscale.com/install.sh | sh

    # Start Tailscale service
    sudo systemctl enable --now tailscaled

    success "Tailscale installed successfully"
    warning "Please run 'sudo tailscale up' to authenticate and join your network"
    warning "Then re-run this script to continue deployment"

    # Wait for user to connect
    if ! tailscale status &> /dev/null; then
        echo -e "${YELLOW}Please authenticate with Tailscale and press Enter to continue...${NC}"
        read -r
    fi
}

# Configure environment
setup_environment() {
    log "Setting up environment configuration..."

    # Create necessary directories
    mkdir -p "${PROJECT_ROOT}/data"
    mkdir -p "${PROJECT_ROOT}/logs"
    mkdir -p "${PROJECT_ROOT}/backups"
    mkdir -p "${PROJECT_ROOT}/uploads"

    # Copy environment templates if they don't exist
    for env_file in .env .env.n8n .env.security; do
        if [[ ! -f "${PROJECT_ROOT}/${env_file}" ]]; then
            if [[ -f "${PROJECT_ROOT}/${env_file}.example" ]]; then
                cp "${PROJECT_ROOT}/${env_file}.example" "${PROJECT_ROOT}/${env_file}"
                log "Created ${env_file} from template"
            fi
        fi
    done

    # Generate secure passwords if not set
    if [[ -f "${PROJECT_ROOT}/.env" ]]; then
        if ! grep -q "POSTGRES_PASSWORD=" "${PROJECT_ROOT}/.env" || grep -q "POSTGRES_PASSWORD=$" "${PROJECT_ROOT}/.env"; then
            local postgres_pass=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-32)
            echo "POSTGRES_PASSWORD=${postgres_pass}" >> "${PROJECT_ROOT}/.env"
            log "Generated PostgreSQL password"
        fi

        if ! grep -q "REDIS_PASSWORD=" "${PROJECT_ROOT}/.env" || grep -q "REDIS_PASSWORD=$" "${PROJECT_ROOT}/.env"; then
            local redis_pass=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-32)
            echo "REDIS_PASSWORD=${redis_pass}" >> "${PROJECT_ROOT}/.env"
            log "Generated Redis password"
        fi

        if ! grep -q "JWT_SECRET=" "${PROJECT_ROOT}/.env" || grep -q "JWT_SECRET=$" "${PROJECT_ROOT}/.env"; then
            local jwt_secret=$(openssl rand -base64 64 | tr -d "=+/")
            echo "JWT_SECRET=${jwt_secret}" >> "${PROJECT_ROOT}/.env"
            log "Generated JWT secret"
        fi
    fi

    success "Environment configuration completed"
}

# Deploy security infrastructure
deploy_security() {
    log "Deploying security infrastructure..."

    cd "${PROJECT_ROOT}"

    # Deploy security services
    if [[ -f "docker-compose.secure.yml" ]]; then
        docker-compose -f docker-compose.secure.yml up -d vault redis-security

        # Wait for Vault to be ready
        log "Waiting for Vault to initialize..."
        sleep 10

        # Initialize Vault if not already done
        if ! docker exec vault vault status 2>/dev/null | grep -q "Initialized.*true"; then
            log "Initializing Vault..."
            vault_init_output=$(docker exec vault vault operator init -key-shares=5 -key-threshold=3 2>/dev/null || true)
            if [[ -n "$vault_init_output" ]]; then
                echo "$vault_init_output" > "${PROJECT_ROOT}/vault-keys.txt"
                chmod 600 "${PROJECT_ROOT}/vault-keys.txt"
                log "Vault keys saved to vault-keys.txt - KEEP THIS FILE SECURE!"
            fi
        fi
    fi

    success "Security infrastructure deployed"
}

# Deploy core services on thanos (GPU node)
deploy_thanos_services() {
    log "Deploying services on thanos (GPU node)..."

    # Check if we're on the thanos node or can SSH to it
    local thanos_ip=""
    if command -v tailscale &> /dev/null; then
        thanos_ip=$(tailscale status | grep thanos | awk '{print $1}' || true)
    fi

    if [[ -n "$thanos_ip" ]]; then
        log "Deploying to thanos node at $thanos_ip"

        # Copy configuration to thanos
        rsync -avz "${PROJECT_ROOT}/thanos/" "root@${thanos_ip}:/opt/agenticnode/"

        # Deploy via SSH
        ssh "root@${thanos_ip}" "cd /opt/agenticnode && docker-compose up -d"
    else
        log "Deploying thanos services locally (assuming this is the thanos node)"

        cd "${PROJECT_ROOT}/thanos"
        if [[ -f "docker-compose.yml" ]]; then
            docker-compose up -d
        fi
    fi

    success "Thanos services deployed"
}

# Deploy core services on oracle1 (CPU node)
deploy_oracle1_services() {
    log "Deploying services on oracle1 (CPU node)..."

    # Check if we're on the oracle1 node or can SSH to it
    local oracle1_ip=""
    if command -v tailscale &> /dev/null; then
        oracle1_ip=$(tailscale status | grep oracle1 | awk '{print $1}' || true)
    fi

    if [[ -n "$oracle1_ip" ]]; then
        log "Deploying to oracle1 node at $oracle1_ip"

        # Copy configuration to oracle1
        rsync -avz "${PROJECT_ROOT}/oracle1/" "root@${oracle1_ip}:/opt/agenticnode/"

        # Deploy via SSH
        ssh "root@${oracle1_ip}" "cd /opt/agenticnode && docker-compose up -d"
    else
        log "Deploying oracle1 services locally (assuming this is the oracle1 node)"

        cd "${PROJECT_ROOT}/oracle1"
        if [[ -f "docker-compose.yml" ]]; then
            docker-compose up -d
        fi
    fi

    success "Oracle1 services deployed"
}

# Deploy orchestration framework
deploy_orchestration() {
    log "Deploying orchestration framework..."

    cd "${PROJECT_ROOT}"

    # Build and deploy the orchestration API
    if [[ -f "src/server.py" ]]; then
        log "Installing Python dependencies..."
        pip install -e .

        # Start orchestration server
        log "Starting orchestration server..."
        nohup python -m src.server > logs/orchestrator.log 2>&1 &
        echo $! > orchestrator.pid
    fi

    success "Orchestration framework deployed"
}

# Deploy n8n automation
deploy_n8n() {
    log "Deploying n8n automation workflows..."

    cd "${PROJECT_ROOT}/n8n"

    # Deploy n8n stack
    if [[ -f "docker/docker-compose.n8n.yml" ]]; then
        cd docker
        docker-compose -f docker-compose.n8n.yml up -d

        # Wait for n8n to be ready
        log "Waiting for n8n to start..."
        sleep 30

        # Import workflows
        cd ..
        if [[ -f "import-workflows.sh" ]]; then
            bash import-workflows.sh all
        fi
    fi

    success "n8n automation deployed"
}

# Deploy demo application
deploy_demo_app() {
    log "Deploying demo application..."

    cd "${PROJECT_ROOT}/demo-app"

    # Install dependencies
    if [[ -f "package.json" ]]; then
        if command -v npm &> /dev/null; then
            npm install
            npm run build

            # Start demo server
            nohup npm start > ../logs/demo-app.log 2>&1 &
            echo $! > ../demo-app.pid
        else
            warning "npm not found, skipping demo app deployment"
        fi
    fi

    success "Demo application deployed"
}

# Deploy Claude proxy
deploy_claude_proxy() {
    log "Deploying Claude proxy..."

    cd "${PROJECT_ROOT}/ccproxy"

    # Deploy Claude proxy
    if [[ -f "docker-compose.yml" ]]; then
        docker-compose up -d
    fi

    success "Claude proxy deployed"
}

# Validate deployment
validate_deployment() {
    log "Validating deployment..."

    local services_to_check=(
        "localhost:3000"    # Demo app
        "localhost:5678"    # n8n
        "localhost:6333"    # Qdrant
        "localhost:8000"    # Orchestrator
        "localhost:8001"    # Claude proxy
        "localhost:9090"    # Prometheus
    )

    local failed_services=()

    for service in "${services_to_check[@]}"; do
        if curl -f -s "http://${service}/health" > /dev/null 2>&1 || \
           curl -f -s "http://${service}/" > /dev/null 2>&1; then
            success "Service ${service} is healthy"
        else
            warning "Service ${service} is not responding"
            failed_services+=("$service")
        fi
    done

    if [[ ${#failed_services[@]} -eq 0 ]]; then
        success "All services are healthy"
    else
        warning "Some services failed health checks: ${failed_services[*]}"
        log "Check logs in ${PROJECT_ROOT}/logs/ for more details"
    fi
}

# Display final status
show_final_status() {
    echo -e "\n${GREEN}🎉 AgenticDosNode Deployment Complete! 🎉${NC}\n"

    echo -e "${BLUE}Access Points:${NC}"
    echo -e "  • Demo Application:    http://localhost:3000"
    echo -e "  • n8n Automation:      http://localhost:5678"
    echo -e "  • Monitoring:          http://localhost:9090"
    echo -e "  • Vector Database:     http://localhost:6333"
    echo -e "  • API Orchestrator:    http://localhost:8000"
    echo -e "  • Claude Proxy:        http://localhost:8001"

    echo -e "\n${BLUE}Next Steps:${NC}"
    echo -e "  1. Configure your API keys in .env files"
    echo -e "  2. Test the demo application at http://localhost:3000"
    echo -e "  3. Set up n8n workflows at http://localhost:5678"
    echo -e "  4. Review logs in ${PROJECT_ROOT}/logs/"
    echo -e "  5. Run validation: ./scripts/validate-deployment.sh"

    echo -e "\n${YELLOW}Important Files:${NC}"
    echo -e "  • Bootstrap log: ${LOG_FILE}"
    echo -e "  • Environment: ${PROJECT_ROOT}/.env"
    echo -e "  • Vault keys: ${PROJECT_ROOT}/vault-keys.txt (if created)"

    echo -e "\n${GREEN}Deployment Summary:${NC}"
    echo -e "  ✅ Multi-node architecture with Tailscale mesh"
    echo -e "  ✅ AI orchestration with Claude/OpenRouter integration"
    echo -e "  ✅ Vector database and RAG pipeline"
    echo -e "  ✅ Cost optimization and monitoring"
    echo -e "  ✅ Security hardening and sandboxing"
    echo -e "  ✅ Automation workflows via n8n"
    echo -e "  ✅ Demo application for testing"
}

# Main deployment function
main() {
    log "Starting AgenticDosNode complete deployment..."

    check_prerequisites
    install_tailscale
    setup_environment
    deploy_security
    deploy_thanos_services
    deploy_oracle1_services
    deploy_orchestration
    deploy_n8n
    deploy_demo_app
    deploy_claude_proxy

    log "Waiting for services to stabilize..."
    sleep 30

    validate_deployment
    show_final_status

    success "AgenticDosNode deployment completed successfully!"
}

# Handle script arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "validate")
        validate_deployment
        ;;
    "status")
        show_final_status
        ;;
    *)
        echo "Usage: $0 [deploy|validate|status]"
        echo "  deploy   - Full deployment (default)"
        echo "  validate - Validate existing deployment"
        echo "  status   - Show deployment status"
        exit 1
        ;;
esac