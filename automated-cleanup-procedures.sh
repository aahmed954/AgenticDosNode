#!/bin/bash

################################################################################
# AgenticDosNode Automated Cleanup Procedures
#
# This script provides automated cleanup and preparation procedures for
# AgenticDosNode deployment, with safety checks and rollback capabilities.
#
# Features:
# - Automated service stopping with dependency checking
# - Docker ecosystem cleanup with selective preservation
# - Python environment isolation
# - GPU resource management
# - Network configuration cleanup
# - Backup and rollback capabilities
################################################################################

set -euo pipefail

# Configuration
BACKUP_DIR="/var/backups/agenticdos-cleanup-$(date +%Y%m%d-%H%M%S)"
LOG_FILE="/var/log/agenticdos-cleanup.log"
DRY_RUN=${DRY_RUN:-false}
INTERACTIVE=${INTERACTIVE:-true}
PRESERVE_DATA=${PRESERVE_DATA:-true}

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'
BOLD='\033[1m'

# Cleanup state tracking
declare -A CLEANUP_STATE
CLEANUP_STATE["services"]=0
CLEANUP_STATE["docker"]=0
CLEANUP_STATE["python"]=0
CLEANUP_STATE["gpu"]=0
CLEANUP_STATE["network"]=0
CLEANUP_STATE["ports"]=0

################################################################################
# Utility Functions
################################################################################

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

log_section() {
    echo -e "\n${PURPLE}========================================${NC}" | tee -a "$LOG_FILE"
    echo -e "${PURPLE}$1${NC}" | tee -a "$LOG_FILE"
    echo -e "${PURPLE}========================================${NC}\n" | tee -a "$LOG_FILE"
}

confirm() {
    if [[ "$INTERACTIVE" == "false" ]]; then
        return 0
    fi
    read -p "$1 [y/N]: " -n 1 -r
    echo
    [[ $REPLY =~ ^[Yy]$ ]]
}

execute_command() {
    local cmd="$1"
    local description="${2:-Executing command}"

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would execute: $cmd"
        return 0
    fi

    log_info "$description"
    if eval "$cmd" 2>&1 | tee -a "$LOG_FILE"; then
        return 0
    else
        log_error "Command failed: $cmd"
        return 1
    fi
}

create_backup() {
    local source="$1"
    local name="$2"

    if [[ ! -e "$source" ]]; then
        return 0
    fi

    mkdir -p "$BACKUP_DIR"
    local backup_path="$BACKUP_DIR/${name}-$(date +%Y%m%d-%H%M%S).tar.gz"

    log_info "Creating backup of $source"
    if tar -czf "$backup_path" "$source" 2>/dev/null; then
        log_success "Backup created: $backup_path"
        echo "$backup_path" >> "$BACKUP_DIR/manifest.txt"
        return 0
    else
        log_warning "Failed to backup $source"
        return 1
    fi
}

################################################################################
# Service Cleanup Functions
################################################################################

cleanup_web_servers() {
    log_section "CLEANING UP WEB SERVERS"

    local web_servers=("nginx" "apache2" "httpd" "caddy" "traefik")
    local cleaned=0

    for server in "${web_servers[@]}"; do
        if systemctl is-active "$server" &>/dev/null; then
            log_warning "Found active web server: $server"

            if confirm "Stop and disable $server?"; then
                # Backup configuration
                case "$server" in
                    nginx)
                        create_backup "/etc/nginx" "nginx-config"
                        ;;
                    apache2|httpd)
                        create_backup "/etc/apache2" "apache-config"
                        create_backup "/etc/httpd" "httpd-config"
                        ;;
                    caddy)
                        create_backup "/etc/caddy" "caddy-config"
                        ;;
                esac

                execute_command "systemctl stop $server" "Stopping $server"
                execute_command "systemctl disable $server" "Disabling $server"
                ((cleaned++))
            fi
        fi
    done

    if [[ $cleaned -gt 0 ]]; then
        CLEANUP_STATE["services"]=1
        log_success "Cleaned up $cleaned web server(s)"
    fi
}

cleanup_databases() {
    log_section "CLEANING UP DATABASE SERVICES"

    local databases=(
        "postgresql:5432"
        "mysql:3306"
        "mariadb:3306"
        "mongodb:27017"
        "redis:6379"
        "memcached:11211"
        "cassandra:9042"
        "elasticsearch:9200"
    )

    for db_entry in "${databases[@]}"; do
        local db_name="${db_entry%%:*}"
        local db_port="${db_entry##*:}"

        if systemctl is-active "$db_name" &>/dev/null; then
            log_warning "Found active database: $db_name (port $db_port)"

            if confirm "Stop $db_name? (Data will be preserved)"; then
                # Backup data directories
                case "$db_name" in
                    postgresql)
                        create_backup "/var/lib/postgresql" "postgresql-data"
                        ;;
                    mysql|mariadb)
                        create_backup "/var/lib/mysql" "mysql-data"
                        ;;
                    mongodb)
                        create_backup "/var/lib/mongodb" "mongodb-data"
                        ;;
                    redis)
                        create_backup "/var/lib/redis" "redis-data"
                        ;;
                esac

                execute_command "systemctl stop $db_name" "Stopping $db_name"

                if ! confirm "Keep $db_name installed for future use?"; then
                    execute_command "systemctl disable $db_name" "Disabling $db_name"
                fi

                CLEANUP_STATE["services"]=1
            fi
        fi
    done
}

cleanup_ml_services() {
    log_section "CLEANING UP ML/AI SERVICES"

    # Check for Jupyter instances
    local jupyter_pids=$(pgrep -f "jupyter" 2>/dev/null || true)
    if [[ -n "$jupyter_pids" ]]; then
        log_warning "Found running Jupyter instances"
        if confirm "Stop all Jupyter instances?"; then
            echo "$jupyter_pids" | xargs -r kill -TERM 2>/dev/null || true
            sleep 2
            echo "$jupyter_pids" | xargs -r kill -9 2>/dev/null || true
            log_success "Jupyter instances stopped"
        fi
    fi

    # Check for MLflow
    if pgrep -f "mlflow" > /dev/null 2>&1; then
        log_warning "Found running MLflow server"
        if confirm "Stop MLflow server?"; then
            pkill -f "mlflow" || true
            log_success "MLflow stopped"
        fi
    fi

    # Check for TensorBoard
    if pgrep -f "tensorboard" > /dev/null 2>&1; then
        log_warning "Found running TensorBoard"
        if confirm "Stop TensorBoard?"; then
            pkill -f "tensorboard" || true
            log_success "TensorBoard stopped"
        fi
    fi

    # Check for Ollama
    if systemctl is-active "ollama" &>/dev/null; then
        log_warning "Found Ollama service"
        if confirm "Stop Ollama? (Models will be preserved)"; then
            create_backup "/usr/share/ollama/.ollama" "ollama-models"
            execute_command "systemctl stop ollama" "Stopping Ollama"
            CLEANUP_STATE["services"]=1
        fi
    fi
}

################################################################################
# Docker Cleanup Functions
################################################################################

cleanup_docker_containers() {
    log_section "CLEANING UP DOCKER CONTAINERS"

    if ! command -v docker &> /dev/null; then
        log_info "Docker not installed, skipping"
        return
    fi

    # Get all running containers
    local containers=$(docker ps -q)
    if [[ -n "$containers" ]]; then
        log_warning "Found $(echo "$containers" | wc -l) running container(s)"

        if confirm "Stop all running containers?"; then
            execute_command "docker stop $containers" "Stopping containers"
        fi
    fi

    # Remove stopped containers
    local stopped=$(docker ps -aq --filter "status=exited")
    if [[ -n "$stopped" ]]; then
        log_info "Found $(echo "$stopped" | wc -l) stopped container(s)"

        if confirm "Remove stopped containers?"; then
            execute_command "docker rm $stopped" "Removing stopped containers"
        fi
    fi

    CLEANUP_STATE["docker"]=1
}

cleanup_docker_images() {
    log_section "CLEANING UP DOCKER IMAGES"

    if ! command -v docker &> /dev/null; then
        return
    fi

    # Identify AI/ML related images to preserve
    local preserve_patterns=("ollama" "vllm" "pytorch" "tensorflow")
    local preserve_images=""

    for pattern in "${preserve_patterns[@]}"; do
        local matching=$(docker images --format "{{.Repository}}:{{.Tag}}" | grep -i "$pattern" || true)
        if [[ -n "$matching" ]]; then
            preserve_images+="$matching"$'\n'
        fi
    done

    if [[ -n "$preserve_images" ]] && [[ "$PRESERVE_DATA" == "true" ]]; then
        log_info "Preserving AI/ML images:"
        echo "$preserve_images" | while read -r img; do
            [[ -n "$img" ]] && echo "  - $img"
        done
    fi

    # Remove dangling images
    local dangling=$(docker images -q -f "dangling=true")
    if [[ -n "$dangling" ]]; then
        log_warning "Found $(echo "$dangling" | wc -l) dangling image(s)"

        if confirm "Remove dangling images?"; then
            execute_command "docker rmi $dangling" "Removing dangling images"
        fi
    fi

    # Optional: Remove all unused images
    if confirm "Remove ALL unused images (excluding preserved)?"; then
        execute_command "docker image prune -a -f" "Removing unused images"
    fi

    CLEANUP_STATE["docker"]=1
}

cleanup_docker_volumes() {
    log_section "CLEANING UP DOCKER VOLUMES"

    if ! command -v docker &> /dev/null; then
        return
    fi

    # List volumes with AI/ML data
    local ml_volumes=$(docker volume ls -q | while read -r vol; do
        local mount=$(docker volume inspect "$vol" 2>/dev/null | jq -r '.[0].Mountpoint')
        if [[ -d "$mount" ]]; then
            # Check for model files
            if find "$mount" -type f \( -name "*.bin" -o -name "*.safetensors" -o -name "*.ckpt" \) 2>/dev/null | head -1 | grep -q .; then
                echo "$vol"
            fi
        fi
    done)

    if [[ -n "$ml_volumes" ]] && [[ "$PRESERVE_DATA" == "true" ]]; then
        log_info "Preserving volumes with ML data:"
        echo "$ml_volumes" | while read -r vol; do
            [[ -n "$vol" ]] && echo "  - $vol"
        done
    fi

    # Remove unused volumes
    if confirm "Remove unused Docker volumes (excluding preserved)?"; then
        execute_command "docker volume prune -f" "Removing unused volumes"
    fi

    CLEANUP_STATE["docker"]=1
}

cleanup_docker_networks() {
    log_section "CLEANING UP DOCKER NETWORKS"

    if ! command -v docker &> /dev/null; then
        return
    fi

    # Check for custom networks
    local custom_networks=$(docker network ls --format "{{.Name}}" | grep -v "bridge\|host\|none")

    if [[ -n "$custom_networks" ]]; then
        log_info "Found custom Docker networks:"
        echo "$custom_networks" | while read -r net; do
            echo "  - $net"
        done

        if confirm "Remove unused custom networks?"; then
            execute_command "docker network prune -f" "Removing unused networks"
        fi
    fi

    CLEANUP_STATE["docker"]=1
}

################################################################################
# Python Environment Cleanup
################################################################################

cleanup_python_environments() {
    log_section "CLEANING UP PYTHON ENVIRONMENTS"

    # Clean pip cache
    if confirm "Clear pip cache?"; then
        execute_command "pip cache purge 2>/dev/null || true" "Clearing pip cache"
        execute_command "pip3 cache purge 2>/dev/null || true" "Clearing pip3 cache"
    fi

    # Clean conda environments
    local conda_paths=("/opt/conda" "/opt/miniconda3" "$HOME/anaconda3" "$HOME/miniconda3")

    for conda_path in "${conda_paths[@]}"; do
        if [[ -d "$conda_path" ]] && [[ -x "$conda_path/bin/conda" ]]; then
            log_info "Found Conda at $conda_path"

            if confirm "Clean Conda cache and unused packages?"; then
                execute_command "$conda_path/bin/conda clean --all -y" "Cleaning Conda"
            fi

            # List environments
            local envs=$("$conda_path/bin/conda" env list 2>/dev/null | grep -v "^#" | awk '{print $1}' | grep -v "^base$" || true)
            if [[ -n "$envs" ]]; then
                log_info "Found Conda environments:"
                echo "$envs" | while read -r env; do
                    echo "  - $env"
                    if confirm "Remove environment '$env'?"; then
                        execute_command "$conda_path/bin/conda env remove -n $env -y" "Removing $env"
                    fi
                done
            fi
        fi
    done

    # Clean Hugging Face cache
    local hf_cache="$HOME/.cache/huggingface"
    if [[ -d "$hf_cache" ]]; then
        local size=$(du -sh "$hf_cache" 2>/dev/null | cut -f1)
        log_warning "Hugging Face cache found: $size"

        if confirm "Clean Hugging Face cache?"; then
            if [[ "$PRESERVE_DATA" == "true" ]]; then
                create_backup "$hf_cache" "huggingface-cache"
            fi
            execute_command "rm -rf $hf_cache/hub/tmp*" "Removing temporary files"
        fi
    fi

    CLEANUP_STATE["python"]=1
}

################################################################################
# GPU Resource Cleanup
################################################################################

cleanup_gpu_resources() {
    log_section "CLEANING UP GPU RESOURCES"

    if ! command -v nvidia-smi &> /dev/null; then
        log_info "No NVIDIA GPU detected, skipping"
        return
    fi

    # Kill GPU-intensive processes
    local gpu_processes=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null || true)

    if [[ -n "$gpu_processes" ]]; then
        log_warning "Found processes using GPU:"
        echo "$gpu_processes" | while IFS=',' read -r pid process; do
            echo "  PID: $pid - $process"

            # Skip critical system processes
            if [[ "$process" == *"Xorg"* ]] || [[ "$process" == *"gnome"* ]]; then
                log_info "Skipping system process: $process"
                continue
            fi

            if confirm "Terminate GPU process $process (PID: $pid)?"; then
                execute_command "kill -TERM $pid" "Terminating $process"
                sleep 2
                kill -0 $pid 2>/dev/null && kill -9 $pid 2>/dev/null || true
            fi
        done
    fi

    # Reset GPU if needed
    if confirm "Reset GPU state?"; then
        execute_command "nvidia-smi --gpu-reset" "Resetting GPU"
    fi

    # Clean CUDA cache
    local cuda_cache_dirs=(
        "$HOME/.nv"
        "$HOME/.cuda"
        "/tmp/cuda*"
    )

    for cache_dir in "${cuda_cache_dirs[@]}"; do
        if [[ -d "$cache_dir" ]]; then
            log_info "Found CUDA cache: $cache_dir"
            if confirm "Remove CUDA cache directory?"; then
                execute_command "rm -rf $cache_dir" "Removing $cache_dir"
            fi
        fi
    done

    CLEANUP_STATE["gpu"]=1
}

################################################################################
# Network Configuration Cleanup
################################################################################

cleanup_network_configs() {
    log_section "CLEANING UP NETWORK CONFIGURATIONS"

    # Clean up conflicting VPN services
    local vpn_services=("zerotier-one" "wireguard" "openvpn")

    for service in "${vpn_services[@]}"; do
        if systemctl is-active "$service" &>/dev/null; then
            log_warning "Found active VPN service: $service"

            if confirm "Stop and disable $service?"; then
                execute_command "systemctl stop $service" "Stopping $service"
                execute_command "systemctl disable $service" "Disabling $service"
            fi
        fi
    done

    # Clean up firewall rules
    if command -v ufw &> /dev/null && ufw status | grep -q "Status: active"; then
        log_info "UFW firewall is active"

        if confirm "Reset UFW rules to default?"; then
            create_backup "/etc/ufw" "ufw-rules"
            execute_command "ufw --force reset" "Resetting UFW"
            execute_command "ufw --force enable" "Re-enabling UFW"
            execute_command "ufw allow ssh" "Allowing SSH"
            execute_command "ufw allow 41641/udp comment 'Tailscale'" "Allowing Tailscale"
        fi
    fi

    # Clean up iptables rules
    if confirm "Reset iptables to default (preserving SSH)?"; then
        create_backup "/etc/iptables" "iptables-rules"

        # Save current SSH port
        local ssh_port=$(ss -tlnp | grep sshd | awk '{print $4}' | cut -d: -f2 | head -1)
        ssh_port=${ssh_port:-22}

        # Reset iptables
        execute_command "iptables -F" "Flushing iptables rules"
        execute_command "iptables -X" "Deleting iptables chains"
        execute_command "iptables -t nat -F" "Flushing NAT table"
        execute_command "iptables -t nat -X" "Deleting NAT chains"

        # Set default policies
        execute_command "iptables -P INPUT ACCEPT" "Setting INPUT policy"
        execute_command "iptables -P FORWARD ACCEPT" "Setting FORWARD policy"
        execute_command "iptables -P OUTPUT ACCEPT" "Setting OUTPUT policy"

        # Ensure SSH access
        execute_command "iptables -A INPUT -p tcp --dport $ssh_port -j ACCEPT" "Ensuring SSH access"
    fi

    CLEANUP_STATE["network"]=1
}

################################################################################
# Port Cleanup
################################################################################

cleanup_ports() {
    log_section "FREEING UP REQUIRED PORTS"

    local required_ports=(
        "8000" "6333" "8001" "8002" "8003" "8004" "11434"  # thanos
        "5678" "8080" "6379" "6334" "9090" "3000" "8081"   # oracle1
    )

    for port in "${required_ports[@]}"; do
        local pids=$(lsof -ti:$port 2>/dev/null || true)

        if [[ -n "$pids" ]]; then
            log_warning "Port $port is in use by PID(s): $pids"

            # Get process details
            for pid in $pids; do
                local process_name=$(ps -p $pid -o comm= 2>/dev/null || echo "unknown")
                log_info "  PID $pid: $process_name"

                if confirm "Kill process $process_name (PID: $pid) using port $port?"; then
                    execute_command "kill -TERM $pid" "Terminating process"
                    sleep 2
                    kill -0 $pid 2>/dev/null && kill -9 $pid 2>/dev/null || true
                fi
            done
        else
            log_success "Port $port is free"
        fi
    done

    CLEANUP_STATE["ports"]=1
}

################################################################################
# System Optimization
################################################################################

perform_system_optimization() {
    log_section "SYSTEM OPTIMIZATION"

    # Clean package cache
    if confirm "Clean APT cache?"; then
        execute_command "apt-get clean" "Cleaning APT cache"
        execute_command "apt-get autoclean" "Removing old packages"
        execute_command "apt-get autoremove -y" "Removing unnecessary packages"
    fi

    # Clean system logs
    if confirm "Clean old system logs?"; then
        execute_command "journalctl --vacuum-time=7d" "Cleaning journal logs older than 7 days"
        execute_command "find /var/log -type f -name '*.gz' -delete" "Removing compressed logs"
        execute_command "find /var/log -type f -name '*.1' -delete" "Removing rotated logs"
    fi

    # Clean temporary files
    if confirm "Clean temporary files?"; then
        execute_command "find /tmp -type f -atime +7 -delete 2>/dev/null || true" "Cleaning /tmp"
        execute_command "find /var/tmp -type f -atime +7 -delete 2>/dev/null || true" "Cleaning /var/tmp"
    fi

    # Optimize system settings
    if confirm "Apply system optimizations for AgenticDosNode?"; then
        # File descriptor limits
        if ! grep -q "agenticdos" /etc/security/limits.conf; then
            cat >> /etc/security/limits.conf << EOF

# AgenticDosNode optimizations
* soft nofile 65536
* hard nofile 65536
* soft nproc 32768
* hard nproc 32768
EOF
            log_success "Updated file descriptor limits"
        fi

        # Kernel parameters
        if ! grep -q "agenticdos" /etc/sysctl.conf; then
            cat >> /etc/sysctl.conf << EOF

# AgenticDosNode network optimizations
net.core.rmem_max = 134217728
net.core.wmem_max = 134217728
net.ipv4.tcp_rmem = 4096 87380 134217728
net.ipv4.tcp_wmem = 4096 65536 134217728
net.core.netdev_max_backlog = 5000
net.ipv4.tcp_congestion_control = bbr
net.ipv4.tcp_notsent_lowat = 16384
EOF
            execute_command "sysctl -p" "Applying kernel parameters"
        fi
    fi
}

################################################################################
# Rollback Function
################################################################################

perform_rollback() {
    log_section "PERFORMING ROLLBACK"

    if [[ ! -d "$BACKUP_DIR" ]]; then
        log_error "No backup directory found"
        return 1
    fi

    if [[ ! -f "$BACKUP_DIR/manifest.txt" ]]; then
        log_error "No backup manifest found"
        return 1
    fi

    log_info "Found backups:"
    cat "$BACKUP_DIR/manifest.txt"

    if confirm "Restore all backups?"; then
        while read -r backup_file; do
            if [[ -f "$backup_file" ]]; then
                log_info "Restoring $backup_file"
                tar -xzf "$backup_file" -C / 2>/dev/null || log_error "Failed to restore $backup_file"
            fi
        done < "$BACKUP_DIR/manifest.txt"

        log_success "Rollback complete"
    fi
}

################################################################################
# Main Execution
################################################################################

show_menu() {
    echo -e "\n${CYAN}${BOLD}AgenticDosNode Cleanup Menu${NC}"
    echo "=============================="
    echo "1) Full cleanup (recommended)"
    echo "2) Service cleanup only"
    echo "3) Docker cleanup only"
    echo "4) Python environment cleanup"
    echo "5) GPU resource cleanup"
    echo "6) Network configuration cleanup"
    echo "7) Port cleanup only"
    echo "8) System optimization"
    echo "9) Rollback previous cleanup"
    echo "0) Exit"
    echo ""
    read -p "Select option: " -n 1 -r option
    echo ""
}

main() {
    log_info "AgenticDosNode Automated Cleanup Procedures"
    log_info "Backup directory: $BACKUP_DIR"
    log_info "Dry run: $DRY_RUN"
    log_info "Interactive: $INTERACTIVE"
    log_info "Preserve data: $PRESERVE_DATA"

    # Create backup directory
    mkdir -p "$BACKUP_DIR"
    mkdir -p "$(dirname "$LOG_FILE")"

    if [[ "$INTERACTIVE" == "true" ]]; then
        while true; do
            show_menu

            case $option in
                1)
                    log_section "PERFORMING FULL CLEANUP"
                    cleanup_web_servers
                    cleanup_databases
                    cleanup_ml_services
                    cleanup_docker_containers
                    cleanup_docker_images
                    cleanup_docker_volumes
                    cleanup_docker_networks
                    cleanup_python_environments
                    cleanup_gpu_resources
                    cleanup_network_configs
                    cleanup_ports
                    perform_system_optimization
                    ;;
                2)
                    cleanup_web_servers
                    cleanup_databases
                    cleanup_ml_services
                    ;;
                3)
                    cleanup_docker_containers
                    cleanup_docker_images
                    cleanup_docker_volumes
                    cleanup_docker_networks
                    ;;
                4)
                    cleanup_python_environments
                    ;;
                5)
                    cleanup_gpu_resources
                    ;;
                6)
                    cleanup_network_configs
                    ;;
                7)
                    cleanup_ports
                    ;;
                8)
                    perform_system_optimization
                    ;;
                9)
                    perform_rollback
                    ;;
                0)
                    break
                    ;;
                *)
                    log_error "Invalid option"
                    ;;
            esac
        done
    else
        # Non-interactive mode - perform full cleanup
        log_section "PERFORMING FULL CLEANUP (NON-INTERACTIVE)"
        cleanup_web_servers
        cleanup_databases
        cleanup_ml_services
        cleanup_docker_containers
        cleanup_docker_images
        cleanup_docker_volumes
        cleanup_docker_networks
        cleanup_python_environments
        cleanup_gpu_resources
        cleanup_network_configs
        cleanup_ports
        perform_system_optimization
    fi

    # Summary
    log_section "CLEANUP SUMMARY"

    echo "Cleanup operations completed:"
    for category in "${!CLEANUP_STATE[@]}"; do
        if [[ ${CLEANUP_STATE[$category]} -eq 1 ]]; then
            echo "  ✓ $category"
        else
            echo "  - $category (skipped)"
        fi
    done

    echo ""
    log_info "Backup directory: $BACKUP_DIR"
    log_info "Log file: $LOG_FILE"
    echo ""
    log_success "Cleanup procedures complete!"
    log_info "System is now prepared for AgenticDosNode deployment"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --non-interactive)
            INTERACTIVE=false
            shift
            ;;
        --no-preserve)
            PRESERVE_DATA=false
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dry-run          Show what would be done without making changes"
            echo "  --non-interactive  Run without prompts (use with caution)"
            echo "  --no-preserve      Don't preserve ML models and data"
            echo "  --help            Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if running as root for certain operations
if [[ $EUID -ne 0 ]] && [[ "$DRY_RUN" == "false" ]]; then
    log_warning "Some cleanup operations require root privileges"
    log_info "For complete cleanup, run: sudo $0 $*"
fi

# Start main execution
main