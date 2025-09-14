#!/bin/bash

# Claude Code Proxy Deployment Script
# This script deploys the proxy with security hardening and monitoring

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="ccproxy"
ENVIRONMENT="${ENVIRONMENT:-production}"
LOG_FILE="/var/log/${PROJECT_NAME}/deploy.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    local level=$1
    shift
    local message="$@"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    case $level in
        "INFO")
            echo -e "${GREEN}[INFO]${NC} ${message}"
            ;;
        "WARN")
            echo -e "${YELLOW}[WARN]${NC} ${message}"
            ;;
        "ERROR")
            echo -e "${RED}[ERROR]${NC} ${message}"
            ;;
    esac

    # Log to file if directory exists
    if [[ -d "$(dirname "$LOG_FILE")" ]]; then
        echo "[$timestamp] [$level] $message" >> "$LOG_FILE"
    fi
}

# Check prerequisites
check_prerequisites() {
    log "INFO" "Checking prerequisites..."

    # Check Docker
    if ! command -v docker &> /dev/null; then
        log "ERROR" "Docker is not installed"
        exit 1
    fi

    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log "ERROR" "Docker Compose is not installed"
        exit 1
    fi

    # Check environment file
    if [[ ! -f "$SCRIPT_DIR/.env" ]]; then
        log "WARN" "No .env file found. Using .env.example as template"
        if [[ -f "$SCRIPT_DIR/.env.example" ]]; then
            cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env"
            log "INFO" "Created .env from .env.example. Please configure it before deployment."
            exit 1
        fi
    fi

    log "INFO" "Prerequisites check passed"
}

# Validate configuration
validate_config() {
    log "INFO" "Validating configuration..."

    # Check required environment variables
    required_vars=(
        "ANTHROPIC_API_KEY"
        "CCPROXY_API_KEYS"
    )

    source "$SCRIPT_DIR/.env"

    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            log "ERROR" "Required environment variable $var is not set"
            exit 1
        fi
    done

    # Validate API key format
    if [[ ! "$ANTHROPIC_API_KEY" =~ ^sk-ant- ]]; then
        log "WARN" "Anthropic API key format seems invalid"
    fi

    log "INFO" "Configuration validation passed"
}

# Setup security
setup_security() {
    log "INFO" "Setting up security measures..."

    # Create necessary directories
    sudo mkdir -p /var/log/$PROJECT_NAME
    sudo mkdir -p /opt/${PROJECT_NAME}-data

    # Set permissions
    sudo chmod 750 /var/log/$PROJECT_NAME
    sudo chmod 750 /opt/${PROJECT_NAME}-data

    # Create apparmor profile (if apparmor is available)
    if command -v aa-status &> /dev/null; then
        log "INFO" "AppArmor detected, creating security profile..."
        # This would create an AppArmor profile for the container
        # For now, we'll use Docker's built-in security features
    fi

    log "INFO" "Security setup completed"
}

# Build and deploy
deploy() {
    log "INFO" "Starting deployment..."

    cd "$SCRIPT_DIR"

    # Build the image
    log "INFO" "Building Docker image..."
    if docker-compose build; then
        log "INFO" "Docker image built successfully"
    else
        log "ERROR" "Failed to build Docker image"
        exit 1
    fi

    # Run security scan (if available)
    if command -v docker scan &> /dev/null; then
        log "INFO" "Running security scan..."
        docker scan ${PROJECT_NAME}:latest || log "WARN" "Security scan completed with warnings"
    fi

    # Stop existing containers
    log "INFO" "Stopping existing containers..."
    docker-compose down || log "WARN" "No existing containers to stop"

    # Start services
    log "INFO" "Starting services..."
    if docker-compose up -d; then
        log "INFO" "Services started successfully"
    else
        log "ERROR" "Failed to start services"
        exit 1
    fi

    # Wait for services to be ready
    log "INFO" "Waiting for services to be ready..."
    sleep 10

    # Health check
    local health_endpoint="http://localhost:8000/health"
    local max_attempts=30
    local attempt=1

    while [[ $attempt -le $max_attempts ]]; do
        if curl -f -s "$health_endpoint" > /dev/null; then
            log "INFO" "Health check passed"
            break
        else
            log "INFO" "Health check attempt $attempt/$max_attempts failed, retrying..."
            sleep 2
            ((attempt++))
        fi
    done

    if [[ $attempt -gt $max_attempts ]]; then
        log "ERROR" "Health check failed after $max_attempts attempts"
        docker-compose logs
        exit 1
    fi

    log "INFO" "Deployment completed successfully"
}

# Show deployment status
show_status() {
    log "INFO" "Deployment Status:"
    echo

    # Container status
    echo "=== Container Status ==="
    docker-compose ps
    echo

    # Health status
    echo "=== Health Status ==="
    curl -s http://localhost:8000/health | jq . || echo "Health endpoint not available"
    echo

    # Stats
    echo "=== Proxy Stats ==="
    curl -s http://localhost:8000/stats | jq . || echo "Stats endpoint not available"
    echo

    # Show logs
    echo "=== Recent Logs ==="
    docker-compose logs --tail=20
}

# Cleanup function
cleanup() {
    log "INFO" "Performing cleanup..."
    docker-compose down
    docker system prune -f
    log "INFO" "Cleanup completed"
}

# Main execution
main() {
    case "${1:-deploy}" in
        "deploy")
            check_prerequisites
            validate_config
            setup_security
            deploy
            show_status
            ;;
        "status")
            show_status
            ;;
        "cleanup")
            cleanup
            ;;
        "restart")
            log "INFO" "Restarting services..."
            docker-compose restart
            sleep 5
            show_status
            ;;
        "logs")
            docker-compose logs -f
            ;;
        *)
            echo "Usage: $0 {deploy|status|cleanup|restart|logs}"
            echo
            echo "Commands:"
            echo "  deploy   - Deploy the proxy (default)"
            echo "  status   - Show deployment status"
            echo "  cleanup  - Stop and cleanup containers"
            echo "  restart  - Restart services"
            echo "  logs     - Show logs"
            exit 1
            ;;
    esac
}

# Trap to ensure cleanup on exit
trap cleanup EXIT

main "$@"