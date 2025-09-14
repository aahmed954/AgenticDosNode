#!/bin/bash

# N8N Automation Stack Deployment Script
# This script sets up the complete n8n automation environment for the agentic AI stack

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/docker/.env"
COMPOSE_FILE="${SCRIPT_DIR}/docker/docker-compose.n8n.yml"

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."

    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
    fi

    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed. Please install Docker Compose first."
    fi

    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        error "Docker daemon is not running. Please start Docker first."
    fi

    success "Prerequisites check passed"
}

# Setup environment file
setup_environment() {
    log "Setting up environment configuration..."

    if [[ ! -f "${ENV_FILE}" ]]; then
        if [[ -f "${SCRIPT_DIR}/docker/.env.n8n" ]]; then
            cp "${SCRIPT_DIR}/docker/.env.n8n" "${ENV_FILE}"
            warning "Environment file created from template. Please edit ${ENV_FILE} with your actual credentials."
        else
            error "Environment template file not found. Please ensure .env.n8n exists in the docker directory."
        fi
    else
        log "Environment file already exists: ${ENV_FILE}"
    fi
}

# Validate environment variables
validate_environment() {
    log "Validating environment configuration..."

    # Source the environment file
    set -a
    source "${ENV_FILE}"
    set +a

    # Check critical variables
    local missing_vars=()

    # Required for basic operation
    [[ -z "${N8N_BASIC_AUTH_USER:-}" ]] && missing_vars+=("N8N_BASIC_AUTH_USER")
    [[ -z "${N8N_BASIC_AUTH_PASSWORD:-}" ]] && missing_vars+=("N8N_BASIC_AUTH_PASSWORD")
    [[ -z "${N8N_POSTGRES_PASSWORD:-}" ]] && missing_vars+=("N8N_POSTGRES_PASSWORD")
    [[ -z "${REDIS_PASSWORD:-}" ]] && missing_vars+=("REDIS_PASSWORD")

    # Optional but recommended
    [[ -z "${CLAUDE_API_KEY:-}" ]] && warning "CLAUDE_API_KEY not set - AI workflows will not function"
    [[ -z "${GITHUB_TOKEN:-}" ]] && warning "GITHUB_TOKEN not set - GitHub integrations will not work"
    [[ -z "${SLACK_WEBHOOK_URL:-}" ]] && warning "SLACK_WEBHOOK_URL not set - Slack notifications disabled"

    if [[ ${#missing_vars[@]} -ne 0 ]]; then
        error "Missing required environment variables: ${missing_vars[*]}"
    fi

    success "Environment validation completed"
}

# Create required directories
create_directories() {
    log "Creating required directories..."

    local directories=(
        "data/n8n"
        "data/postgres"
        "data/redis"
        "data/qdrant"
        "logs"
        "backups"
    )

    for dir in "${directories[@]}"; do
        mkdir -p "${SCRIPT_DIR}/${dir}"
        log "Created directory: ${dir}"
    done

    success "Directories created successfully"
}

# Setup database schema
setup_database_schema() {
    log "Setting up database schema..."

    # Wait for database to be ready
    local max_attempts=30
    local attempt=0

    while [[ $attempt -lt $max_attempts ]]; do
        if docker-compose -f "${COMPOSE_FILE}" --env-file "${ENV_FILE}" exec -T postgres-n8n pg_isready -U n8n -d n8n &> /dev/null; then
            break
        fi

        ((attempt++))
        log "Waiting for database to be ready... (attempt $attempt/$max_attempts)"
        sleep 2
    done

    if [[ $attempt -eq $max_attempts ]]; then
        error "Database did not become ready within expected time"
    fi

    # Apply schema from config file
    if [[ -f "${SCRIPT_DIR}/config/postgres-init.sql" ]]; then
        docker-compose -f "${COMPOSE_FILE}" --env-file "${ENV_FILE}" exec -T postgres-n8n psql -U n8n -d n8n < "${SCRIPT_DIR}/config/postgres-init.sql"
        success "Database schema applied successfully"
    else
        warning "Database schema file not found, skipping schema setup"
    fi
}

# Pull Docker images
pull_images() {
    log "Pulling Docker images..."
    docker-compose -f "${COMPOSE_FILE}" --env-file "${ENV_FILE}" pull
    success "Docker images pulled successfully"
}

# Start services
start_services() {
    log "Starting n8n automation stack..."
    docker-compose -f "${COMPOSE_FILE}" --env-file "${ENV_FILE}" up -d
    success "Services started successfully"
}

# Check service health
check_service_health() {
    log "Checking service health..."

    local services=("n8n" "postgres-n8n" "redis-n8n" "qdrant-n8n")
    local max_attempts=20

    for service in "${services[@]}"; do
        local attempt=0
        log "Checking health of service: $service"

        while [[ $attempt -lt $max_attempts ]]; do
            if docker-compose -f "${COMPOSE_FILE}" ps "$service" | grep -q "Up"; then
                success "$service is running"
                break
            fi

            ((attempt++))
            log "Waiting for $service to be healthy... (attempt $attempt/$max_attempts)"
            sleep 3
        done

        if [[ $attempt -eq $max_attempts ]]; then
            warning "$service did not become healthy within expected time"
        fi
    done
}

# Display access information
display_access_info() {
    log "Deployment completed!"

    echo ""
    echo "==================================="
    echo "🚀 N8N AUTOMATION STACK READY"
    echo "==================================="
    echo ""
    echo "📊 Access Information:"
    echo "   • n8n Interface: http://localhost:5678"
    echo "   • Username: ${N8N_BASIC_AUTH_USER:-admin}"
    echo "   • Password: ${N8N_BASIC_AUTH_PASSWORD:-[check .env file]}"
    echo ""
    echo "🔧 Service Endpoints:"
    echo "   • Webhook Relay: http://localhost:8080"
    echo "   • Qdrant Vector DB: http://localhost:6333"
    echo "   • PostgreSQL: localhost:5432"
    echo ""
    echo "📋 Next Steps:"
    echo "   1. Access n8n at http://localhost:5678"
    echo "   2. Import workflows from the /workflows directory"
    echo "   3. Configure credentials for external services"
    echo "   4. Review and customize workflow schedules"
    echo "   5. Test workflows manually before enabling automation"
    echo ""
    echo "📖 Documentation: See N8N_SETUP_GUIDE.md for detailed instructions"
    echo ""
}

# Cleanup function
cleanup() {
    if [[ $? -ne 0 ]]; then
        error "Deployment failed. Check the logs above for details."
        echo ""
        echo "🛠️  Troubleshooting:"
        echo "   • Check Docker daemon is running"
        echo "   • Verify environment variables in ${ENV_FILE}"
        echo "   • Ensure ports 5678, 6333, 8080 are available"
        echo "   • Check system resources (RAM, disk space)"
        echo ""
        echo "📋 Cleanup Command:"
        echo "   docker-compose -f \"${COMPOSE_FILE}\" --env-file \"${ENV_FILE}\" down -v"
    fi
}

# Main deployment function
main() {
    echo ""
    echo "🤖 N8N Automation Stack Deployment"
    echo "===================================="
    echo ""

    # Set up cleanup trap
    trap cleanup EXIT

    # Run deployment steps
    check_prerequisites
    setup_environment
    validate_environment
    create_directories
    pull_images
    start_services
    sleep 10  # Give services time to start
    setup_database_schema
    check_service_health
    display_access_info

    # Remove trap on successful completion
    trap - EXIT
}

# Handle command line arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "stop")
        log "Stopping n8n automation stack..."
        docker-compose -f "${COMPOSE_FILE}" --env-file "${ENV_FILE}" down
        success "Services stopped"
        ;;
    "restart")
        log "Restarting n8n automation stack..."
        docker-compose -f "${COMPOSE_FILE}" --env-file "${ENV_FILE}" restart
        success "Services restarted"
        ;;
    "logs")
        docker-compose -f "${COMPOSE_FILE}" --env-file "${ENV_FILE}" logs -f
        ;;
    "status")
        docker-compose -f "${COMPOSE_FILE}" --env-file "${ENV_FILE}" ps
        ;;
    "cleanup")
        warning "This will remove all containers, volumes, and data!"
        read -p "Are you sure? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            docker-compose -f "${COMPOSE_FILE}" --env-file "${ENV_FILE}" down -v --remove-orphans
            success "Cleanup completed"
        else
            log "Cleanup cancelled"
        fi
        ;;
    *)
        echo "Usage: $0 {deploy|stop|restart|logs|status|cleanup}"
        echo ""
        echo "Commands:"
        echo "  deploy   - Deploy the complete n8n automation stack (default)"
        echo "  stop     - Stop all services"
        echo "  restart  - Restart all services"
        echo "  logs     - Show service logs"
        echo "  status   - Show service status"
        echo "  cleanup  - Remove all containers and data (destructive!)"
        exit 1
        ;;
esac