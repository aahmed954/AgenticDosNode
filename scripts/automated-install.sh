#!/bin/bash

# AgenticDosNode Automated Installation Script
# Repository: https://github.com/aahmed954/AgenticDosNode
# Usage: curl -sSL https://raw.githubusercontent.com/aahmed954/AgenticDosNode/main/scripts/automated-install.sh | sudo bash -s [GPU|CPU] [auto|interactive]

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
REPO_URL="https://github.com/aahmed954/AgenticDosNode.git"
INSTALL_DIR="/opt/AgenticDosNode"
LOG_FILE="/tmp/agenticnode-install.log"
MACHINE_TYPE="${1:-}"
INSTALL_MODE="${2:-auto}"

# Progress tracking
TOTAL_STEPS=6
CURRENT_STEP=0

# Logging functions
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}" | tee -a "$LOG_FILE"
    exit 1
}

warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}" | tee -a "$LOG_FILE"
}

info() {
    echo -e "${CYAN}[INFO] $1${NC}" | tee -a "$LOG_FILE"
}

# Progress bar function
show_progress() {
    local step=$1
    local total=$2
    local message="$3"
    local percent=$((step * 100 / total))
    local filled=$((percent / 5))
    local empty=$((20 - filled))

    printf "\r${PURPLE}[%d/%d]${NC} " "$step" "$total"
    printf "${GREEN}"
    printf "%*s" $filled | tr ' ' '█'
    printf "${NC}"
    printf "%*s" $empty | tr ' ' '░'
    printf " %d%% - %s" "$percent" "$message"

    if [[ $step -eq $total ]]; then
        echo ""
    fi
}

# Banner function
show_banner() {
    clear
    echo -e "${CYAN}"
    cat << "EOF"
    ╔══════════════════════════════════════════════════════════════╗
    ║                    AgenticDosNode                           ║
    ║              Automated Installation System                  ║
    ║                                                            ║
    ║    🤖 Multi-System Agentic AI Stack Deployment 🤖          ║
    ╚══════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
    echo -e "${BLUE}Repository:${NC} https://github.com/aahmed954/AgenticDosNode"
    echo -e "${BLUE}Machine Type:${NC} $MACHINE_TYPE"
    echo -e "${BLUE}Install Mode:${NC} $INSTALL_MODE"
    echo -e "${BLUE}Install Directory:${NC} $INSTALL_DIR"
    echo ""
}

# Validation function
validate_inputs() {
    if [[ -z "$MACHINE_TYPE" ]]; then
        error "Machine type not specified. Usage: $0 [GPU|CPU] [auto|interactive]"
    fi

    if [[ "$MACHINE_TYPE" != "GPU" && "$MACHINE_TYPE" != "CPU" ]]; then
        error "Invalid machine type. Must be 'GPU' or 'CPU'"
    fi

    if [[ "$INSTALL_MODE" != "auto" && "$INSTALL_MODE" != "interactive" ]]; then
        error "Invalid install mode. Must be 'auto' or 'interactive'"
    fi

    # Check if running as root
    if [[ $EUID -ne 0 ]]; then
        error "This script must be run as root (use sudo)"
    fi

    # Check Ubuntu version
    if ! grep -q "Ubuntu" /etc/os-release; then
        error "This script requires Ubuntu. Detected: $(cat /etc/os-release | grep PRETTY_NAME | cut -d'"' -f2)"
    fi

    local ubuntu_version=$(lsb_release -sr 2>/dev/null || echo "0")
    if (( $(echo "$ubuntu_version < 20.04" | bc -l) )); then
        error "Ubuntu 20.04 or newer required. Detected: $ubuntu_version"
    fi

    success "Input validation passed"
}

# Prerequisites check
check_prerequisites() {
    ((CURRENT_STEP++))
    show_progress $CURRENT_STEP $TOTAL_STEPS "Checking prerequisites..."

    log "Checking system prerequisites..."

    # Check available disk space (need at least 20GB)
    local available_space_gb=$(df / | tail -1 | awk '{print int($4/1024/1024)}')
    if [[ $available_space_gb -lt 20 ]]; then
        error "Insufficient disk space. Need 20GB, available: ${available_space_gb}GB"
    fi

    # Check memory (need at least 8GB for GPU, 4GB for CPU)
    local memory_gb=$(free -g | awk '/^Mem:/{print $2}')
    local min_memory=$([[ "$MACHINE_TYPE" == "GPU" ]] && echo 8 || echo 4)
    if [[ $memory_gb -lt $min_memory ]]; then
        warning "Low memory detected. Recommended: ${min_memory}GB, available: ${memory_gb}GB"
    fi

    # Check for NVIDIA GPU on GPU machine
    if [[ "$MACHINE_TYPE" == "GPU" ]]; then
        if ! command -v nvidia-smi &> /dev/null; then
            warning "NVIDIA drivers not detected. GPU functionality may be limited."
        else
            local gpu_info=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
            info "Detected GPU: $gpu_info"
        fi
    fi

    # Install basic dependencies
    log "Installing basic dependencies..."
    apt-get update -qq
    apt-get install -y git curl wget jq bc htop net-tools lsof

    success "Prerequisites check completed"
}

# Clone repository
clone_repository() {
    ((CURRENT_STEP++))
    show_progress $CURRENT_STEP $TOTAL_STEPS "Cloning AgenticDosNode repository..."

    log "Cloning repository from $REPO_URL"

    # Remove existing directory if it exists
    if [[ -d "$INSTALL_DIR" ]]; then
        warning "Existing installation found at $INSTALL_DIR"
        if [[ "$INSTALL_MODE" == "interactive" ]]; then
            echo -e "${YELLOW}Remove existing installation? [y/N]${NC}"
            read -r response
            if [[ "$response" =~ ^[Yy]$ ]]; then
                rm -rf "$INSTALL_DIR"
            else
                error "Installation cancelled by user"
            fi
        else
            log "Auto mode: removing existing installation"
            rm -rf "$INSTALL_DIR"
        fi
    fi

    # Clone repository
    git clone "$REPO_URL" "$INSTALL_DIR" 2>&1 | tee -a "$LOG_FILE"

    if [[ ! -d "$INSTALL_DIR" ]]; then
        error "Failed to clone repository"
    fi

    # Set permissions
    chown -R root:root "$INSTALL_DIR"
    chmod +x "$INSTALL_DIR"/scripts/*.sh

    success "Repository cloned successfully"
}

# System preparation and cleanup
system_preparation() {
    ((CURRENT_STEP++))
    show_progress $CURRENT_STEP $TOTAL_STEPS "Preparing system and cleaning conflicts..."

    log "Running system preparation and conflict cleanup..."

    cd "$INSTALL_DIR"

    # Run conflict detection
    if [[ -f "resource-conflict-detector.sh" ]]; then
        log "Detecting resource conflicts..."
        bash resource-conflict-detector.sh --auto-report 2>&1 | tee -a "$LOG_FILE"
    fi

    # Run automated cleanup
    if [[ -f "automated-cleanup-procedures.sh" ]]; then
        log "Running automated cleanup procedures..."
        if [[ "$INSTALL_MODE" == "auto" ]]; then
            bash automated-cleanup-procedures.sh --non-interactive 2>&1 | tee -a "$LOG_FILE"
        else
            bash automated-cleanup-procedures.sh 2>&1 | tee -a "$LOG_FILE"
        fi
    fi

    # Machine-specific cleanup
    if [[ "$MACHINE_TYPE" == "GPU" && -f "cleanup-thanos.sh" ]]; then
        log "Running GPU-specific cleanup..."
        bash cleanup-thanos.sh --auto 2>&1 | tee -a "$LOG_FILE"
    elif [[ "$MACHINE_TYPE" == "CPU" && -f "cleanup-oracle1.sh" ]]; then
        log "Running CPU-specific cleanup..."
        bash cleanup-oracle1.sh --auto 2>&1 | tee -a "$LOG_FILE"
    fi

    success "System preparation completed"
}

# Performance optimization
performance_optimization() {
    ((CURRENT_STEP++))
    show_progress $CURRENT_STEP $TOTAL_STEPS "Applying performance optimizations..."

    log "Applying performance optimizations for $MACHINE_TYPE machine..."

    cd "$INSTALL_DIR"

    # Run optimization suite
    if [[ -d "optimization" ]]; then
        log "Running performance optimization suite..."
        cd optimization

        if [[ "$INSTALL_MODE" == "auto" ]]; then
            bash run-optimization.sh --apply-all --no-benchmark 2>&1 | tee -a "$LOG_FILE"
        else
            bash run-optimization.sh 2>&1 | tee -a "$LOG_FILE"
        fi

        cd ..
    fi

    success "Performance optimization completed"
}

# Tailscale setup
tailscale_setup() {
    ((CURRENT_STEP++))
    show_progress $CURRENT_STEP $TOTAL_STEPS "Setting up Tailscale mesh network..."

    log "Installing and configuring Tailscale..."

    # Install Tailscale
    if ! command -v tailscale &> /dev/null; then
        log "Installing Tailscale..."
        curl -fsSL https://tailscale.com/install.sh | sh 2>&1 | tee -a "$LOG_FILE"
        systemctl enable --now tailscaled
    else
        log "Tailscale already installed"
    fi

    # Check if already authenticated
    if ! tailscale status &> /dev/null; then
        warning "Tailscale not authenticated"
        if [[ "$INSTALL_MODE" == "interactive" ]]; then
            echo -e "${YELLOW}Please authenticate Tailscale by running:${NC}"
            echo -e "${CYAN}sudo tailscale up${NC}"
            echo -e "${YELLOW}Press Enter when complete...${NC}"
            read -r
        else
            log "Auto mode: Tailscale setup will need manual completion after installation"
        fi
    else
        local tailscale_ip=$(tailscale ip)
        info "Tailscale configured with IP: $tailscale_ip"
    fi

    success "Tailscale setup completed"
}

# AgenticDosNode deployment
agenticnode_deployment() {
    ((CURRENT_STEP++))
    show_progress $CURRENT_STEP $TOTAL_STEPS "Deploying AgenticDosNode services..."

    log "Deploying AgenticDosNode services..."

    cd "$INSTALL_DIR"

    # Copy environment template
    if [[ -f ".env.example" && ! -f ".env" ]]; then
        log "Creating environment configuration..."
        cp .env.example .env

        # Generate secure passwords
        local postgres_pass=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-32)
        local redis_pass=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-32)
        local jwt_secret=$(openssl rand -base64 64 | tr -d "=+/")

        sed -i "s/POSTGRES_PASSWORD=.*/POSTGRES_PASSWORD=$postgres_pass/" .env
        sed -i "s/REDIS_PASSWORD=.*/REDIS_PASSWORD=$redis_pass/" .env
        sed -i "s/JWT_SECRET=.*/JWT_SECRET=$jwt_secret/" .env
    fi

    # Run bootstrap script
    if [[ -f "scripts/bootstrap-complete.sh" ]]; then
        log "Running AgenticDosNode bootstrap..."
        bash scripts/bootstrap-complete.sh 2>&1 | tee -a "$LOG_FILE"
    else
        error "Bootstrap script not found"
    fi

    success "AgenticDosNode deployment completed"
}

# Final validation
final_validation() {
    log "Running final validation and health checks..."

    cd "$INSTALL_DIR"

    # Run validation script
    if [[ -f "scripts/validate-deployment.sh" ]]; then
        log "Running comprehensive validation..."
        bash scripts/validate-deployment.sh 2>&1 | tee -a "$LOG_FILE"
    fi

    # Generate installation report
    generate_installation_report

    success "Installation validation completed"
}

# Generate installation report
generate_installation_report() {
    local report_file="$INSTALL_DIR/installation-report.txt"

    cat > "$report_file" << EOF
AgenticDosNode Installation Report
=================================

Installation Date: $(date)
Machine Type: $MACHINE_TYPE
Install Mode: $INSTALL_MODE
Installation Directory: $INSTALL_DIR

System Information:
- OS: $(cat /etc/os-release | grep PRETTY_NAME | cut -d'"' -f2)
- Kernel: $(uname -r)
- CPU: $(nproc) cores
- Memory: $(free -h | awk '/^Mem:/{print $2}')
- Disk Available: $(df -h / | tail -1 | awk '{print $4}')

Services Status:
$(systemctl status docker --no-pager -l 2>/dev/null || echo "Docker: Not available")

Network Configuration:
$(ip addr show | grep -E "inet.*scope global" || echo "Network: Basic configuration")

Tailscale Status:
$(tailscale status 2>/dev/null || echo "Tailscale: Not configured")

Installation Log: $LOG_FILE

Next Steps:
1. Configure API keys in $INSTALL_DIR/.env
2. Access demo at http://localhost:3000
3. Configure n8n at http://localhost:5678
4. Monitor at http://localhost:9090

EOF

    log "Installation report generated: $report_file"
}

# Show completion message
show_completion() {
    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                    🎉 INSTALLATION COMPLETE! 🎉              ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    echo -e "${BLUE}🏠 Installation Directory:${NC} $INSTALL_DIR"
    echo -e "${BLUE}📋 Installation Log:${NC} $LOG_FILE"
    echo -e "${BLUE}📊 Installation Report:${NC} $INSTALL_DIR/installation-report.txt"
    echo ""

    echo -e "${CYAN}🌐 Access Points:${NC}"
    echo -e "  • Demo Application:    http://localhost:3000"
    echo -e "  • n8n Automation:      http://localhost:5678"
    echo -e "  • Monitoring:          http://localhost:9090"
    echo -e "  • Vector Database:     http://localhost:6333"
    echo -e "  • API Orchestrator:    http://localhost:8000"

    if command -v tailscale &> /dev/null && tailscale status &> /dev/null; then
        local hostname=$(hostname)
        local tailscale_domain=$(tailscale status | grep "$(hostname)" | awk '{print $2}' | head -1)
        if [[ -n "$tailscale_domain" ]]; then
            echo ""
            echo -e "${PURPLE}🔗 Tailscale Access:${NC}"
            echo -e "  • Demo Application:    http://$tailscale_domain:3000"
            echo -e "  • n8n Automation:      http://$tailscale_domain:5678"
            echo -e "  • Monitoring:          http://$tailscale_domain:9090"
        fi
    fi

    echo ""
    echo -e "${YELLOW}📝 Next Steps:${NC}"
    echo -e "  1. Configure your API keys in ${INSTALL_DIR}/.env"
    echo -e "  2. Test the demo application"
    echo -e "  3. Set up automation workflows in n8n"
    echo -e "  4. Monitor costs and performance"
    echo ""

    echo -e "${GREEN}🚀 Your AgenticDosNode deployment is ready!${NC}"
    echo ""
}

# Error handler
handle_error() {
    local exit_code=$?
    echo ""
    error "Installation failed at step $CURRENT_STEP with exit code $exit_code"
    echo -e "${YELLOW}Check the log file for details: $LOG_FILE${NC}"
    echo -e "${YELLOW}You can retry the installation or run individual components manually.${NC}"
    exit $exit_code
}

# Set error handler
trap handle_error ERR

# Main installation function
main() {
    show_banner
    validate_inputs

    log "Starting AgenticDosNode automated installation..."
    log "Machine Type: $MACHINE_TYPE, Install Mode: $INSTALL_MODE"

    check_prerequisites
    clone_repository
    system_preparation
    performance_optimization
    tailscale_setup
    agenticnode_deployment
    final_validation

    show_completion

    success "AgenticDosNode installation completed successfully!"
}

# Execute main function
main "$@"