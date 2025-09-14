#!/bin/bash

################################################################################
# AgenticDosNode Post-Cleanup Validation Script
#
# This script validates that the system is properly cleaned and ready for
# AgenticDosNode deployment after running cleanup procedures.
#
# Validation checks:
# - All required ports are free
# - No conflicting services running
# - Docker environment is clean
# - GPU resources are available
# - Network configuration is correct
# - System resources are adequate
################################################################################

set -euo pipefail

# Configuration
VALIDATION_REPORT="./validation-report-$(date +%Y%m%d-%H%M%S).json"
LOG_FILE="/var/log/agenticdos-validation.log"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'
BOLD='\033[1m'

# Validation counters
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0
WARNING_CHECKS=0

# Initialize validation report
VALIDATION_JSON='{"timestamp":"'$(date -u +"%Y-%m-%dT%H:%M:%SZ")'","hostname":"'$(hostname)'","checks":[],"summary":{}}'

################################################################################
# Utility Functions
################################################################################

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

log_pass() {
    echo -e "${GREEN}[PASS]${NC} $1" | tee -a "$LOG_FILE"
    ((PASSED_CHECKS++))
    ((TOTAL_CHECKS++))
    add_check "$1" "pass" ""
}

log_fail() {
    echo -e "${RED}[FAIL]${NC} $1" | tee -a "$LOG_FILE"
    ((FAILED_CHECKS++))
    ((TOTAL_CHECKS++))
    add_check "$1" "fail" "$2"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1" | tee -a "$LOG_FILE"
    ((WARNING_CHECKS++))
    ((TOTAL_CHECKS++))
    add_check "$1" "warning" "$2"
}

log_section() {
    echo -e "\n${PURPLE}========================================${NC}" | tee -a "$LOG_FILE"
    echo -e "${PURPLE}$1${NC}" | tee -a "$LOG_FILE"
    echo -e "${PURPLE}========================================${NC}\n" | tee -a "$LOG_FILE"
}

add_check() {
    local description="$1"
    local status="$2"
    local details="${3:-}"

    local check_json=$(cat <<EOF
{
    "description": "$description",
    "status": "$status",
    "details": "$details",
    "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
}
EOF
)

    VALIDATION_JSON=$(echo "$VALIDATION_JSON" | jq ".checks += [$check_json]")
}

################################################################################
# Port Availability Validation
################################################################################

validate_ports() {
    log_section "VALIDATING PORT AVAILABILITY"

    # Define required ports by node type
    declare -A THANOS_PORTS=(
        ["8000"]="vLLM Server"
        ["6333"]="Qdrant Primary"
        ["8001"]="Embedding Service"
        ["8002"]="Whisper API"
        ["8003"]="ComfyUI"
        ["8004"]="Code Interpreter"
        ["11434"]="Ollama API"
    )

    declare -A ORACLE1_PORTS=(
        ["5678"]="n8n"
        ["8080"]="LangGraph API"
        ["6379"]="Redis"
        ["6334"]="Qdrant Replica"
        ["9090"]="Prometheus"
        ["3000"]="Grafana"
        ["8081"]="Claude Proxy"
    )

    # Determine which node type we're validating
    local node_type=""
    if [[ -f "/etc/hostname" ]]; then
        local hostname=$(cat /etc/hostname)
        if [[ "$hostname" == *"thanos"* ]]; then
            node_type="thanos"
        elif [[ "$hostname" == *"oracle"* ]]; then
            node_type="oracle1"
        fi
    fi

    # Check ports based on node type or all if unknown
    local ports_to_check=()

    if [[ "$node_type" == "thanos" ]]; then
        log_info "Validating ports for thanos node"
        for port in "${!THANOS_PORTS[@]}"; do
            ports_to_check+=("$port:${THANOS_PORTS[$port]}")
        done
    elif [[ "$node_type" == "oracle1" ]]; then
        log_info "Validating ports for oracle1 node"
        for port in "${!ORACLE1_PORTS[@]}"; do
            ports_to_check+=("$port:${ORACLE1_PORTS[$port]}")
        done
    else
        log_info "Node type unknown, checking all ports"
        for port in "${!THANOS_PORTS[@]}"; do
            ports_to_check+=("$port:${THANOS_PORTS[$port]}")
        done
        for port in "${!ORACLE1_PORTS[@]}"; do
            ports_to_check+=("$port:${ORACLE1_PORTS[$port]}")
        done
    fi

    # Validate each port
    local all_ports_free=true
    for port_entry in "${ports_to_check[@]}"; do
        local port="${port_entry%%:*}"
        local service="${port_entry##*:}"

        if ss -tuln | grep -q ":$port "; then
            local process_info=$(ss -tulnp 2>/dev/null | grep ":$port " | head -1 || echo "unknown")
            log_fail "Port $port needed for $service is in use" "$process_info"
            all_ports_free=false
        else
            log_pass "Port $port is free for $service"
        fi
    done

    if $all_ports_free; then
        log_pass "All required ports are available"
    fi
}

################################################################################
# Service Validation
################################################################################

validate_services() {
    log_section "VALIDATING SERVICE STATUS"

    # Check for conflicting web servers
    local web_servers=("nginx" "apache2" "httpd" "caddy" "traefik")
    local web_server_running=false

    for server in "${web_servers[@]}"; do
        if systemctl is-active "$server" &>/dev/null; then
            log_fail "Web server $server is still running" "systemctl stop $server"
            web_server_running=true
        fi
    done

    if ! $web_server_running; then
        log_pass "No conflicting web servers running"
    fi

    # Check for conflicting databases
    local databases=("postgresql" "mysql" "mariadb" "mongodb")
    local db_conflicts=false

    for db in "${databases[@]}"; do
        if systemctl is-active "$db" &>/dev/null; then
            # Check if it's actually using a conflicting port
            local port=""
            case "$db" in
                postgresql) port="5432" ;;
                mysql|mariadb) port="3306" ;;
                mongodb) port="27017" ;;
            esac

            if [[ -n "$port" ]] && ss -tuln | grep -q ":$port "; then
                log_warn "Database $db is running on port $port" "May conflict if AgenticDosNode needs this port"
                db_conflicts=true
            else
                log_info "Database $db is running but not on conflicting port"
            fi
        fi
    done

    if ! $db_conflicts; then
        log_pass "No conflicting database services"
    fi

    # Check for Kubernetes/Swarm
    if systemctl is-active "k3s" &>/dev/null || systemctl is-active "k3s-agent" &>/dev/null; then
        log_warn "K3s Kubernetes is running" "May conflict with Docker orchestration"
    else
        log_pass "No Kubernetes running"
    fi

    if docker info 2>/dev/null | grep -q "Swarm: active"; then
        log_warn "Docker Swarm is active" "May complicate container management"
    else
        log_pass "Docker Swarm not active"
    fi
}

################################################################################
# Docker Environment Validation
################################################################################

validate_docker() {
    log_section "VALIDATING DOCKER ENVIRONMENT"

    if ! command -v docker &> /dev/null; then
        log_fail "Docker is not installed" "Install Docker CE"
        return
    fi

    # Check Docker daemon
    if ! docker info &>/dev/null; then
        log_fail "Docker daemon is not running" "systemctl start docker"
        return
    fi

    log_pass "Docker is installed and running"

    # Check Docker version
    local docker_version=$(docker version --format '{{.Server.Version}}' 2>/dev/null)
    log_info "Docker version: $docker_version"

    # Check for running containers
    local running_containers=$(docker ps -q | wc -l)
    if [[ $running_containers -gt 0 ]]; then
        log_warn "$running_containers container(s) still running" "Review with: docker ps"
    else
        log_pass "No containers running"
    fi

    # Check Docker disk usage
    local docker_disk=$(docker system df --format json 2>/dev/null | jq -r '.Images[0].Size' | sed 's/GB//')
    if [[ -n "$docker_disk" ]]; then
        log_info "Docker disk usage: $(docker system df 2>/dev/null | grep Images | awk '{print $4}')"
    fi

    # Check Docker Compose
    if command -v docker-compose &> /dev/null; then
        local compose_version=$(docker-compose version --short 2>/dev/null)
        log_pass "Docker Compose installed: $compose_version"
    elif docker compose version &>/dev/null; then
        local compose_version=$(docker compose version --short 2>/dev/null)
        log_pass "Docker Compose plugin installed: $compose_version"
    else
        log_fail "Docker Compose not found" "Install docker-compose or docker-compose-plugin"
    fi

    # Check for Docker networks conflicting with Tailscale
    local conflicting_networks=$(docker network ls --format "{{.Name}}" | while read -r network; do
        local subnet=$(docker network inspect "$network" 2>/dev/null | jq -r '.[0].IPAM.Config[0].Subnet' || echo "")
        if [[ "$subnet" == "100.64."* ]]; then
            echo "$network"
        fi
    done)

    if [[ -n "$conflicting_networks" ]]; then
        log_fail "Docker networks conflict with Tailscale subnet" "$conflicting_networks"
    else
        log_pass "No Docker network conflicts"
    fi
}

################################################################################
# GPU Validation
################################################################################

validate_gpu() {
    log_section "VALIDATING GPU RESOURCES"

    if command -v nvidia-smi &> /dev/null; then
        log_pass "NVIDIA drivers installed"

        # Check GPU availability
        if nvidia-smi &>/dev/null; then
            local gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
            log_pass "Found $gpu_count NVIDIA GPU(s)"

            # Check GPU memory
            local gpu_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
            log_info "GPU memory: ${gpu_memory}MB"

            # Check for processes using GPU
            local gpu_processes=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
            if [[ $gpu_processes -gt 0 ]]; then
                log_warn "$gpu_processes process(es) using GPU" "Check with: nvidia-smi"
            else
                log_pass "GPU is free"
            fi

            # Check CUDA
            if [[ -d "/usr/local/cuda" ]]; then
                local cuda_version=$(/usr/local/cuda/bin/nvcc --version 2>/dev/null | grep "release" | awk '{print $6}' | cut -d',' -f1)
                log_pass "CUDA installed: $cuda_version"
            else
                log_warn "CUDA not found at /usr/local/cuda" "May be needed for some AI workloads"
            fi

            # Check Docker GPU runtime
            if docker info 2>/dev/null | grep -q "nvidia"; then
                log_pass "NVIDIA Docker runtime configured"
            else
                log_warn "NVIDIA Docker runtime not configured" "Install nvidia-docker2"
            fi

        else
            log_fail "nvidia-smi failed" "Check GPU drivers"
        fi
    elif command -v rocm-smi &> /dev/null; then
        log_pass "AMD ROCm drivers installed"
        log_warn "AMD GPU support is experimental" "Ensure ROCm-compatible containers"
    else
        log_warn "No GPU detected" "GPU acceleration will not be available"
    fi
}

################################################################################
# Network Validation
################################################################################

validate_network() {
    log_section "VALIDATING NETWORK CONFIGURATION"

    # Check Tailscale
    if command -v tailscale &> /dev/null; then
        if systemctl is-active tailscaled &>/dev/null; then
            log_pass "Tailscale is installed and running"

            # Check Tailscale status
            local tailscale_status=$(tailscale status 2>/dev/null | head -1)
            if [[ -n "$tailscale_status" ]]; then
                log_info "Tailscale status: Connected"
            else
                log_warn "Tailscale not authenticated" "Run: tailscale up"
            fi
        else
            log_warn "Tailscale daemon not running" "systemctl start tailscaled"
        fi
    else
        log_fail "Tailscale not installed" "Required for mesh networking"
    fi

    # Check for conflicting VPNs
    local vpn_services=("zerotier-one" "wireguard" "openvpn")
    local vpn_conflicts=false

    for vpn in "${vpn_services[@]}"; do
        if systemctl is-active "$vpn" &>/dev/null; then
            log_warn "Conflicting VPN service running: $vpn" "May interfere with Tailscale"
            vpn_conflicts=true
        fi
    done

    if ! $vpn_conflicts; then
        log_pass "No conflicting VPN services"
    fi

    # Check firewall
    if command -v ufw &> /dev/null; then
        if ufw status | grep -q "Status: active"; then
            # Check if Tailscale port is allowed
            if ufw status | grep -q "41641/udp"; then
                log_pass "UFW allows Tailscale port"
            else
                log_warn "UFW may block Tailscale" "ufw allow 41641/udp"
            fi
        else
            log_info "UFW is inactive"
        fi
    fi

    # Check IP forwarding
    local ip_forward=$(cat /proc/sys/net/ipv4/ip_forward)
    if [[ "$ip_forward" == "1" ]]; then
        log_pass "IP forwarding enabled"
    else
        log_warn "IP forwarding disabled" "May be needed for container networking"
    fi

    # Check DNS resolution
    if host google.com &>/dev/null; then
        log_pass "DNS resolution working"
    else
        log_fail "DNS resolution failed" "Check /etc/resolv.conf"
    fi
}

################################################################################
# System Resources Validation
################################################################################

validate_system_resources() {
    log_section "VALIDATING SYSTEM RESOURCES"

    # Check CPU
    local cpu_count=$(nproc)
    local cpu_model=$(lscpu | grep "Model name" | cut -d: -f2 | xargs)
    log_info "CPU: $cpu_model ($cpu_count cores)"

    if [[ $cpu_count -lt 4 ]]; then
        log_warn "Only $cpu_count CPU cores available" "Recommended: 4+ cores"
    else
        log_pass "Adequate CPU resources: $cpu_count cores"
    fi

    # Check RAM
    local total_ram_gb=$(free -g | grep "^Mem:" | awk '{print $2}')
    local available_ram_gb=$(free -g | grep "^Mem:" | awk '{print $7}')

    log_info "RAM: ${available_ram_gb}GB available / ${total_ram_gb}GB total"

    if [[ $available_ram_gb -lt 8 ]]; then
        log_warn "Low available RAM: ${available_ram_gb}GB" "Recommended: 8GB+ available"
    elif [[ $available_ram_gb -lt 16 ]]; then
        log_pass "Adequate RAM available: ${available_ram_gb}GB"
    else
        log_pass "Excellent RAM available: ${available_ram_gb}GB"
    fi

    # Check disk space
    local root_available=$(df -BG / | tail -1 | awk '{print $4}' | sed 's/G//')
    log_info "Root partition: ${root_available}GB available"

    if [[ $root_available -lt 50 ]]; then
        log_warn "Low disk space: ${root_available}GB" "Recommended: 50GB+ for models and data"
    elif [[ $root_available -lt 100 ]]; then
        log_pass "Adequate disk space: ${root_available}GB"
    else
        log_pass "Excellent disk space: ${root_available}GB"
    fi

    # Check swap
    local swap_total=$(free -g | grep "^Swap:" | awk '{print $2}')
    if [[ $swap_total -gt 0 ]]; then
        log_pass "Swap configured: ${swap_total}GB"
    else
        log_warn "No swap configured" "Consider adding swap for stability"
    fi

    # Check load average
    local load_avg=$(uptime | awk -F'load average:' '{print $2}')
    log_info "Load average:$load_avg"

    local load_1min=$(echo "$load_avg" | cut -d',' -f1 | xargs)
    if (( $(echo "$load_1min > $cpu_count" | bc -l) )); then
        log_warn "High system load: $load_1min" "System may be under stress"
    else
        log_pass "System load normal: $load_1min"
    fi
}

################################################################################
# Python Environment Validation
################################################################################

validate_python() {
    log_section "VALIDATING PYTHON ENVIRONMENT"

    # Check Python version
    if command -v python3 &> /dev/null; then
        local python_version=$(python3 --version 2>&1 | awk '{print $2}')
        log_pass "Python3 installed: $python_version"

        # Check if version is recent enough
        local major=$(echo "$python_version" | cut -d. -f1)
        local minor=$(echo "$python_version" | cut -d. -f2)
        if [[ $major -eq 3 ]] && [[ $minor -ge 8 ]]; then
            log_pass "Python version is adequate"
        else
            log_warn "Python version is old: $python_version" "Recommended: Python 3.8+"
        fi
    else
        log_fail "Python3 not found" "Install python3"
    fi

    # Check pip
    if command -v pip3 &> /dev/null; then
        log_pass "pip3 is installed"
    else
        log_warn "pip3 not found" "Install python3-pip"
    fi

    # Check virtual environment tools
    if command -v virtualenv &> /dev/null || python3 -m venv --help &>/dev/null 2>&1; then
        log_pass "Virtual environment support available"
    else
        log_warn "Virtual environment tools not found" "Install python3-venv"
    fi

    # Check for conflicting global packages
    local ai_packages=$(pip3 list 2>/dev/null | grep -E "torch|tensorflow|transformers" | wc -l)
    if [[ $ai_packages -gt 0 ]]; then
        log_warn "$ai_packages AI packages installed globally" "Consider using virtual environments"
    else
        log_pass "No conflicting global AI packages"
    fi
}

################################################################################
# Security Validation
################################################################################

validate_security() {
    log_section "VALIDATING SECURITY SETTINGS"

    # Check SSH configuration
    if [[ -f "/etc/ssh/sshd_config" ]]; then
        if grep -q "^PermitRootLogin no" /etc/ssh/sshd_config; then
            log_pass "Root SSH login disabled"
        else
            log_warn "Root SSH login may be enabled" "Set: PermitRootLogin no"
        fi

        if grep -q "^PasswordAuthentication no" /etc/ssh/sshd_config; then
            log_pass "SSH password authentication disabled"
        else
            log_warn "SSH password authentication enabled" "Consider using key-based auth only"
        fi
    fi

    # Check for unattended upgrades
    if systemctl is-enabled unattended-upgrades &>/dev/null; then
        log_pass "Automatic security updates enabled"
    else
        log_warn "Automatic updates not configured" "Enable unattended-upgrades"
    fi

    # Check fail2ban
    if systemctl is-active fail2ban &>/dev/null; then
        log_pass "fail2ban is active"
    else
        log_warn "fail2ban not running" "Consider enabling for brute-force protection"
    fi

    # Check file permissions on sensitive directories
    if [[ -d "/etc/docker" ]]; then
        local docker_perms=$(stat -c %a /etc/docker)
        if [[ "$docker_perms" == "755" ]] || [[ "$docker_perms" == "750" ]]; then
            log_pass "Docker config permissions correct"
        else
            log_warn "Docker config has unusual permissions: $docker_perms" "Expected 755 or 750"
        fi
    fi
}

################################################################################
# Generate Readiness Report
################################################################################

generate_readiness_report() {
    log_section "GENERATING READINESS REPORT"

    # Update summary in JSON
    VALIDATION_JSON=$(echo "$VALIDATION_JSON" | jq ".summary = {
        \"total_checks\": $TOTAL_CHECKS,
        \"passed\": $PASSED_CHECKS,
        \"failed\": $FAILED_CHECKS,
        \"warnings\": $WARNING_CHECKS,
        \"pass_rate\": $(echo "scale=2; $PASSED_CHECKS * 100 / $TOTAL_CHECKS" | bc)
    }")

    # Save JSON report
    echo "$VALIDATION_JSON" | jq '.' > "$VALIDATION_REPORT"
    log_info "Validation report saved to: $VALIDATION_REPORT"

    # Generate readiness score
    local readiness_score=$(echo "scale=2; $PASSED_CHECKS * 100 / $TOTAL_CHECKS" | bc)

    echo ""
    echo -e "${BOLD}===== SYSTEM READINESS SUMMARY =====${NC}"
    echo ""
    echo "Total Checks: $TOTAL_CHECKS"
    echo -e "${GREEN}Passed: $PASSED_CHECKS${NC}"
    echo -e "${RED}Failed: $FAILED_CHECKS${NC}"
    echo -e "${YELLOW}Warnings: $WARNING_CHECKS${NC}"
    echo ""
    echo -e "${BOLD}Readiness Score: ${readiness_score}%${NC}"
    echo ""

    if [[ $FAILED_CHECKS -eq 0 ]]; then
        echo -e "${GREEN}${BOLD}✓ SYSTEM IS READY FOR AGENTICDOSNODE DEPLOYMENT${NC}"
        echo ""
        echo "Next steps:"
        echo "1. Review any warnings above"
        echo "2. Configure environment variables in .env file"
        echo "3. Run the appropriate setup script for your node type:"
        echo "   - For thanos (GPU node): ./setup-thanos.sh"
        echo "   - For oracle1 (CPU node): ./setup-oracle1.sh"
        return 0
    elif [[ $FAILED_CHECKS -le 2 ]]; then
        echo -e "${YELLOW}${BOLD}⚠ SYSTEM IS MOSTLY READY${NC}"
        echo ""
        echo "Please address the failed checks before deployment:"
        echo "$VALIDATION_JSON" | jq -r '.checks[] | select(.status == "fail") | "- " + .description + ": " + .details'
        return 1
    else
        echo -e "${RED}${BOLD}✗ SYSTEM IS NOT READY${NC}"
        echo ""
        echo "Multiple issues must be resolved before deployment."
        echo "Review the full report: $VALIDATION_REPORT"
        return 2
    fi
}

################################################################################
# Main Execution
################################################################################

main() {
    log_info "AgenticDosNode Post-Cleanup Validation"
    log_info "Starting validation at $(date)"
    echo ""

    # Create log directory
    mkdir -p "$(dirname "$LOG_FILE")"

    # Run all validations
    validate_ports
    validate_services
    validate_docker
    validate_gpu
    validate_network
    validate_system_resources
    validate_python
    validate_security

    # Generate final report
    generate_readiness_report
}

# Run main function
main "$@"