#!/bin/bash

################################################################################
# AgenticDosNode Resource Conflict Detection and Remediation System
#
# This script performs comprehensive analysis of potential resource conflicts
# before deploying AgenticDosNode on dedicated Ubuntu machines.
#
# Categories analyzed:
# 1. Service port conflicts
# 2. Docker ecosystem conflicts
# 3. Python/AI environment conflicts
# 4. GPU resource conflicts
# 5. Network and firewall conflicts
################################################################################

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Log levels
LOG_FILE="/var/log/agenticdos-conflict-detection.log"
REPORT_FILE="./conflict-analysis-report-$(date +%Y%m%d-%H%M%S).json"

# Initialize report structure
REPORT_JSON='{"timestamp":"'$(date -u +"%Y-%m-%dT%H:%M:%SZ")'","hostname":"'$(hostname)'","conflicts":[],"remediations":[]}'

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

# Add conflict to JSON report
add_conflict() {
    local category="$1"
    local service="$2"
    local details="$3"
    local severity="$4"
    local remediation="$5"

    local conflict_json=$(cat <<EOF
{
    "category": "$category",
    "service": "$service",
    "details": "$details",
    "severity": "$severity",
    "remediation": "$remediation"
}
EOF
)

    REPORT_JSON=$(echo "$REPORT_JSON" | jq ".conflicts += [$conflict_json]")
}

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_error "This script must be run as root for complete system analysis"
        exit 1
    fi
}

################################################################################
# Port Conflict Detection
################################################################################

check_port_conflicts() {
    log_section "CHECKING PORT CONFLICTS"

    # Define AgenticDosNode required ports
    declare -A REQUIRED_PORTS=(
        # thanos node services
        ["8000"]="vLLM Server"
        ["6333"]="Qdrant Primary"
        ["8001"]="Embedding Service"
        ["8002"]="Whisper API"
        ["8003"]="ComfyUI"
        ["8004"]="Code Interpreter"
        ["11434"]="Ollama API"

        # oracle1 node services
        ["5678"]="n8n"
        ["8080"]="LangGraph API"
        ["8000"]="Kong Gateway"
        ["6379"]="Redis"
        ["6334"]="Qdrant Replica"
        ["9090"]="Prometheus"
        ["3000"]="Grafana"
        ["3001"]="Alternative Grafana"
        ["8081"]="Claude Proxy"

        # Common services
        ["80"]="HTTP"
        ["443"]="HTTPS"
        ["5432"]="PostgreSQL"
        ["27017"]="MongoDB"
        ["5000"]="MLflow/Registry"
        ["8888"]="Jupyter"
    )

    local conflicts_found=false

    for port in "${!REQUIRED_PORTS[@]}"; do
        local service="${REQUIRED_PORTS[$port]}"

        # Check if port is in use
        if ss -tuln | grep -q ":$port "; then
            conflicts_found=true
            local process_info=$(ss -tulnp | grep ":$port " | head -1)
            local current_service=$(echo "$process_info" | awk '{print $NF}')

            log_warning "Port $port (needed for $service) is already in use"
            log_info "Current service: $current_service"

            # Determine remediation
            local remediation=""
            if [[ "$current_service" == *"docker"* ]]; then
                remediation="Stop Docker container using port $port or reconfigure AgenticDosNode service to use alternative port"
            elif [[ "$current_service" == *"systemd"* ]]; then
                remediation="Stop systemd service: systemctl stop <service> or reconfigure port binding"
            else
                remediation="Kill process using port $port: lsof -ti:$port | xargs kill -9"
            fi

            add_conflict "port" "$service" "Port $port in use by $current_service" "high" "$remediation"
        else
            log_success "Port $port is available for $service"
        fi
    done

    if ! $conflicts_found; then
        log_success "No port conflicts detected"
    fi
}

################################################################################
# Docker Ecosystem Analysis
################################################################################

check_docker_conflicts() {
    log_section "CHECKING DOCKER ECOSYSTEM"

    if ! command -v docker &> /dev/null; then
        log_warning "Docker is not installed"
        return
    fi

    # Check for AI-related Docker images
    log_info "Scanning for AI-related Docker images..."

    local ai_images=(
        "pytorch"
        "tensorflow"
        "nvidia"
        "ollama"
        "langchain"
        "huggingface"
        "vllm"
        "qdrant"
        "mlflow"
        "jupyter"
        "comfyui"
    )

    for image_pattern in "${ai_images[@]}"; do
        local images=$(docker images --format "{{.Repository}}:{{.Tag}}" | grep -i "$image_pattern" || true)
        if [[ -n "$images" ]]; then
            log_warning "Found existing $image_pattern images:"
            echo "$images" | while read -r img; do
                echo "  - $img"
                local size=$(docker images --format "{{.Repository}}:{{.Tag}} {{.Size}}" | grep "^$img " | awk '{print $2}')
                echo "    Size: $size"
            done

            add_conflict "docker" "$image_pattern" "Existing AI images may conflict with new deployments" "medium" \
                "Review and potentially remove old images: docker rmi <image>"
        fi
    done

    # Check for GPU-enabled containers
    log_info "Checking for GPU-enabled containers..."

    local gpu_containers=$(docker ps -a --format "{{.Names}}" | while read -r container; do
        if docker inspect "$container" 2>/dev/null | jq -r '.[0].HostConfig.DeviceRequests' | grep -q "gpu"; then
            echo "$container"
        fi
    done)

    if [[ -n "$gpu_containers" ]]; then
        log_warning "Found GPU-enabled containers:"
        echo "$gpu_containers" | while read -r container; do
            echo "  - $container"
            local status=$(docker inspect "$container" --format '{{.State.Status}}')
            echo "    Status: $status"

            if [[ "$status" == "running" ]]; then
                add_conflict "docker" "$container" "Running GPU container may compete for resources" "high" \
                    "Stop container: docker stop $container"
            fi
        done
    fi

    # Check Docker networks
    log_info "Analyzing Docker networks for conflicts..."

    local networks=$(docker network ls --format "{{.Name}}" | grep -v "bridge\|host\|none")
    if [[ -n "$networks" ]]; then
        echo "$networks" | while read -r network; do
            local subnet=$(docker network inspect "$network" 2>/dev/null | jq -r '.[0].IPAM.Config[0].Subnet' || echo "unknown")
            if [[ "$subnet" == "100.64."* ]]; then
                log_warning "Network $network uses Tailscale-conflicting subnet: $subnet"
                add_conflict "docker" "$network" "Docker network conflicts with Tailscale subnet" "high" \
                    "Remove or reconfigure network: docker network rm $network"
            fi
        done
    fi

    # Check Docker volumes with AI data
    log_info "Checking Docker volumes for AI data..."

    local volumes=$(docker volume ls --format "{{.Name}}")
    if [[ -n "$volumes" ]]; then
        echo "$volumes" | while read -r volume; do
            local mount_point=$(docker volume inspect "$volume" 2>/dev/null | jq -r '.[0].Mountpoint')
            if [[ -d "$mount_point" ]]; then
                # Check for model files
                local model_files=$(find "$mount_point" -type f \( -name "*.bin" -o -name "*.safetensors" -o -name "*.ckpt" -o -name "*.pth" \) 2>/dev/null | head -5)
                if [[ -n "$model_files" ]]; then
                    log_warning "Volume $volume contains AI model files"
                    local size=$(du -sh "$mount_point" 2>/dev/null | cut -f1)
                    echo "  Size: $size"
                    add_conflict "docker" "$volume" "Docker volume contains AI models ($size)" "low" \
                        "Backup if needed: docker run --rm -v $volume:/data alpine tar czf /backup.tar.gz /data"
                fi
            fi
        done
    fi
}

################################################################################
# Python/AI Environment Detection
################################################################################

check_python_environments() {
    log_section "CHECKING PYTHON/AI ENVIRONMENTS"

    # Check for Conda/Miniconda
    log_info "Checking for Conda installations..."

    local conda_paths=(
        "/opt/conda"
        "/opt/miniconda3"
        "$HOME/anaconda3"
        "$HOME/miniconda3"
        "/usr/local/anaconda3"
    )

    for conda_path in "${conda_paths[@]}"; do
        if [[ -d "$conda_path" ]]; then
            log_warning "Found Conda installation at: $conda_path"

            # List environments
            if [[ -x "$conda_path/bin/conda" ]]; then
                local envs=$("$conda_path/bin/conda" env list 2>/dev/null | grep -v "^#" | awk '{print $1}' | grep -v "^base$" || true)
                if [[ -n "$envs" ]]; then
                    echo "  Environments:"
                    echo "$envs" | while read -r env; do
                        echo "    - $env"
                    done
                fi
            fi

            add_conflict "python" "conda" "Existing Conda installation at $conda_path" "low" \
                "Can coexist but may cause PATH conflicts. Consider using separate environments"
        fi
    done

    # Check for virtual environments
    log_info "Checking for Python virtual environments..."

    local venv_indicators=(".venv" "venv" "env" ".env")
    local search_paths=("/home" "/opt" "/var/www" "/srv")

    for search_path in "${search_paths[@]}"; do
        if [[ -d "$search_path" ]]; then
            for indicator in "${venv_indicators[@]}"; do
                local venvs=$(find "$search_path" -maxdepth 3 -type d -name "$indicator" 2>/dev/null | head -10)
                if [[ -n "$venvs" ]]; then
                    echo "$venvs" | while read -r venv_path; do
                        if [[ -f "$venv_path/bin/python" ]]; then
                            log_info "Found virtual environment: $venv_path"

                            # Check for AI packages
                            local ai_packages=$("$venv_path/bin/pip" list 2>/dev/null | grep -E "torch|tensorflow|transformers|langchain|openai" || true)
                            if [[ -n "$ai_packages" ]]; then
                                log_warning "  Contains AI packages:"
                                echo "$ai_packages" | head -5 | while read -r pkg; do
                                    echo "    - $pkg"
                                done
                            fi
                        fi
                    done
                fi
            done
        fi
    done

    # Check system-wide AI packages
    log_info "Checking system-wide Python AI packages..."

    if command -v pip3 &> /dev/null; then
        local ai_packages=$(pip3 list 2>/dev/null | grep -E "torch|tensorflow|transformers|langchain|openai|huggingface" || true)
        if [[ -n "$ai_packages" ]]; then
            log_warning "System-wide AI packages installed:"
            echo "$ai_packages" | while read -r pkg; do
                echo "  - $pkg"
            done

            add_conflict "python" "system-packages" "System-wide AI packages may conflict" "medium" \
                "Consider using virtual environments to isolate dependencies"
        fi
    fi
}

################################################################################
# GPU Resource Conflicts
################################################################################

check_gpu_conflicts() {
    log_section "CHECKING GPU RESOURCES"

    # Check for NVIDIA GPUs
    if command -v nvidia-smi &> /dev/null; then
        log_info "NVIDIA GPU detected"

        # Get GPU info
        local gpu_info=$(nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu --format=csv,noheader)
        echo "GPU Status:"
        echo "$gpu_info" | while IFS=',' read -r name mem_total mem_used mem_free temp util; do
            echo "  Name: $name"
            echo "  Memory: $mem_used / $mem_total (Free: $mem_free)"
            echo "  Temperature: $temp"
            echo "  Utilization: $util"
        done

        # Check for processes using GPU
        local gpu_processes=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true)
        if [[ -n "$gpu_processes" ]]; then
            log_warning "Processes currently using GPU:"
            echo "$gpu_processes" | while IFS=',' read -r pid process memory; do
                echo "  PID: $pid, Process: $process, Memory: $memory"

                # Check if it's a mining process
                if echo "$process" | grep -iE "miner|mining|ethminer|nicehash" > /dev/null; then
                    add_conflict "gpu" "$process" "Mining software detected using GPU" "critical" \
                        "Stop mining: kill -9 $pid"
                fi

                # Check if it's an ML training process
                if echo "$process" | grep -iE "python|jupyter|train" > /dev/null; then
                    add_conflict "gpu" "$process" "ML training process using GPU" "high" \
                        "Wait for completion or stop: kill -TERM $pid"
                fi
            done
        fi

        # Check CUDA versions
        log_info "Checking CUDA installations..."

        local cuda_paths=(
            "/usr/local/cuda"
            "/usr/local/cuda-11"
            "/usr/local/cuda-12"
            "/opt/cuda"
        )

        for cuda_path in "${cuda_paths[@]}"; do
            if [[ -d "$cuda_path" ]]; then
                local version=$("$cuda_path/bin/nvcc" --version 2>/dev/null | grep "release" | awk '{print $6}' | cut -d',' -f1)
                log_info "Found CUDA $version at $cuda_path"

                # Check for version conflicts
                if [[ -L "/usr/local/cuda" ]]; then
                    local current_link=$(readlink "/usr/local/cuda")
                    if [[ "$current_link" != "$cuda_path" ]]; then
                        log_warning "Multiple CUDA versions detected"
                        add_conflict "gpu" "cuda" "Multiple CUDA versions installed" "medium" \
                            "Set correct version: ln -sfn $cuda_path /usr/local/cuda"
                    fi
                fi
            fi
        done

        # Check for Docker GPU runtime
        if command -v docker &> /dev/null; then
            if docker info 2>/dev/null | grep -q "nvidia"; then
                log_success "NVIDIA Docker runtime is configured"
            else
                log_warning "NVIDIA Docker runtime not configured"
                add_conflict "gpu" "docker-runtime" "NVIDIA Docker runtime not configured" "high" \
                    "Install nvidia-docker2 and restart Docker"
            fi
        fi

    elif command -v rocm-smi &> /dev/null; then
        log_info "AMD GPU detected (ROCm)"

        # Get AMD GPU info
        local amd_info=$(rocm-smi --showtemp --showuse --showmeminfo vram 2>/dev/null || true)
        if [[ -n "$amd_info" ]]; then
            echo "AMD GPU Status:"
            echo "$amd_info"
        fi

        add_conflict "gpu" "rocm" "AMD GPU detected - may need ROCm-specific configurations" "medium" \
            "Ensure ROCm-compatible containers are used"
    else
        log_warning "No GPU detected or GPU drivers not installed"
    fi
}

################################################################################
# Network and Firewall Analysis
################################################################################

check_network_conflicts() {
    log_section "CHECKING NETWORK AND FIREWALL"

    # Check for existing VPN/mesh networks
    log_info "Checking for VPN/mesh networking software..."

    local vpn_services=("tailscale" "zerotier-one" "wireguard" "openvpn" "nebula")

    for service in "${vpn_services[@]}"; do
        if systemctl is-active "$service" &>/dev/null; then
            log_warning "Found active VPN service: $service"

            # Get interface info
            local interface=""
            case "$service" in
                tailscale)
                    interface="tailscale0"
                    ;;
                zerotier-one)
                    interface=$(ip link show | grep "zt" | cut -d: -f2 | tr -d ' ' | head -1)
                    ;;
                wireguard)
                    interface=$(ip link show | grep "wg" | cut -d: -f2 | tr -d ' ' | head -1)
                    ;;
            esac

            if [[ -n "$interface" ]] && ip link show "$interface" &>/dev/null; then
                local ip_addr=$(ip -4 addr show "$interface" 2>/dev/null | grep inet | awk '{print $2}')
                echo "  Interface: $interface"
                echo "  IP: $ip_addr"

                if [[ "$service" != "tailscale" ]]; then
                    add_conflict "network" "$service" "Existing VPN may conflict with Tailscale mesh" "medium" \
                        "Consider consolidating to Tailscale or ensure no subnet conflicts"
                fi
            fi
        fi
    done

    # Check firewall rules
    log_info "Analyzing firewall configuration..."

    if command -v ufw &> /dev/null; then
        if ufw status | grep -q "Status: active"; then
            log_info "UFW firewall is active"

            # Check if required ports are allowed
            local ufw_rules=$(ufw status numbered | grep -E "^\[")
            echo "Current UFW rules:"
            echo "$ufw_rules" | head -10

            # Check for Tailscale port
            if ! echo "$ufw_rules" | grep -q "41641"; then
                log_warning "Tailscale port 41641/udp not explicitly allowed"
                add_conflict "network" "ufw" "Tailscale port may be blocked" "medium" \
                    "Allow Tailscale: ufw allow 41641/udp"
            fi
        fi
    fi

    if command -v iptables &> /dev/null; then
        local iptables_rules=$(iptables -L -n 2>/dev/null | wc -l)
        if [[ $iptables_rules -gt 20 ]]; then
            log_warning "Complex iptables rules detected ($iptables_rules lines)"
            add_conflict "network" "iptables" "Complex iptables rules may interfere" "low" \
                "Review rules: iptables -L -n -v"
        fi
    fi

    # Check for port forwarding conflicts
    log_info "Checking for existing port forwarding..."

    if [[ -f /proc/sys/net/ipv4/ip_forward ]]; then
        local forwarding=$(cat /proc/sys/net/ipv4/ip_forward)
        if [[ "$forwarding" == "1" ]]; then
            log_info "IP forwarding is enabled"

            # Check NAT rules
            local nat_rules=$(iptables -t nat -L -n 2>/dev/null | grep -E "DNAT|SNAT|MASQUERADE" | wc -l)
            if [[ $nat_rules -gt 0 ]]; then
                log_warning "Found $nat_rules NAT rules"
                add_conflict "network" "nat" "Existing NAT rules may conflict" "medium" \
                    "Review NAT table: iptables -t nat -L -n -v"
            fi
        fi
    fi

    # Check network interfaces
    log_info "Analyzing network interfaces..."

    local interfaces=$(ip -o link show | awk '{print $2}' | sed 's/://' | grep -v "lo")
    echo "Network interfaces:"
    echo "$interfaces" | while read -r iface; do
        local state=$(ip link show "$iface" | grep -oP 'state \K\w+')
        local ip_addrs=$(ip -4 addr show "$iface" 2>/dev/null | grep inet | awk '{print $2}')
        echo "  $iface: $state"
        if [[ -n "$ip_addrs" ]]; then
            echo "$ip_addrs" | while read -r ip; do
                echo "    IP: $ip"
            done
        fi
    done
}

################################################################################
# Service-Specific Checks
################################################################################

check_specific_services() {
    log_section "CHECKING SPECIFIC SERVICE CONFLICTS"

    # Web servers
    local web_servers=("nginx" "apache2" "httpd" "caddy" "traefik")
    for server in "${web_servers[@]}"; do
        if systemctl is-active "$server" &>/dev/null; then
            log_warning "Web server $server is active"
            add_conflict "service" "$server" "Web server using ports 80/443" "high" \
                "Stop service: systemctl stop $server"
        fi
    done

    # Databases
    local databases=("postgresql" "mysql" "mariadb" "mongodb" "redis" "memcached")
    for db in "${databases[@]}"; do
        if systemctl is-active "$db" &>/dev/null; then
            log_warning "Database $db is active"
            local port=""
            case "$db" in
                postgresql) port="5432" ;;
                mysql|mariadb) port="3306" ;;
                mongodb) port="27017" ;;
                redis) port="6379" ;;
                memcached) port="11211" ;;
            esac
            add_conflict "service" "$db" "Database service on port $port" "medium" \
                "Stop if not needed: systemctl stop $db"
        fi
    done

    # Container orchestration
    if systemctl is-active "k3s" &>/dev/null || systemctl is-active "k3s-agent" &>/dev/null; then
        log_warning "K3s Kubernetes is running"
        add_conflict "service" "k3s" "Kubernetes may conflict with Docker Swarm" "high" \
            "Choose one orchestration platform"
    fi

    if docker info 2>/dev/null | grep -q "Swarm: active"; then
        log_warning "Docker Swarm is active"
        add_conflict "service" "docker-swarm" "Docker Swarm mode is active" "medium" \
            "Leave swarm if not needed: docker swarm leave --force"
    fi

    # AI/ML specific services
    local ml_services=("jupyter" "jupyterhub" "mlflow" "tensorboard" "ray")
    for service in "${ml_services[@]}"; do
        if pgrep -f "$service" > /dev/null 2>&1; then
            log_warning "ML service $service is running"
            local pids=$(pgrep -f "$service" | head -5)
            add_conflict "service" "$service" "ML service is running (PIDs: $(echo $pids | tr '\n' ' '))" "medium" \
                "Stop service or integrate with AgenticDosNode"
        fi
    done
}

################################################################################
# Disk Space Analysis
################################################################################

check_disk_space() {
    log_section "CHECKING DISK SPACE"

    # Check available space
    local disk_usage=$(df -h / | tail -1)
    local available=$(echo "$disk_usage" | awk '{print $4}')
    local percent=$(echo "$disk_usage" | awk '{print $5}' | sed 's/%//')

    log_info "Root filesystem: $available available (${percent}% used)"

    if [[ $percent -gt 80 ]]; then
        log_warning "Disk usage is high (${percent}%)"
        add_conflict "disk" "root" "Low disk space available" "high" \
            "Free up space or expand volume"
    fi

    # Check for large ML model directories
    local model_dirs=(
        "/models"
        "/opt/models"
        "/var/lib/ollama"
        "$HOME/.cache/huggingface"
        "$HOME/.ollama"
        "/usr/local/share/models"
    )

    for dir in "${model_dirs[@]}"; do
        if [[ -d "$dir" ]]; then
            local size=$(du -sh "$dir" 2>/dev/null | cut -f1)
            log_info "Found model directory: $dir (Size: $size)"

            # Check if it's very large
            local size_gb=$(du -s "$dir" 2>/dev/null | awk '{print int($1/1048576)}')
            if [[ $size_gb -gt 50 ]]; then
                add_conflict "disk" "$dir" "Large model directory using ${size_gb}GB" "medium" \
                    "Consider moving to dedicated storage or cleanup old models"
            fi
        fi
    done

    # Check Docker space usage
    if command -v docker &> /dev/null; then
        local docker_usage=$(docker system df 2>/dev/null | tail -n +2 | head -3)
        log_info "Docker disk usage:"
        echo "$docker_usage"

        # Check if cleanup is needed
        local reclaimable=$(docker system df 2>/dev/null | grep "RECLAIMABLE" -A3 | tail -3 | awk '{print $4}' | sed 's/[^0-9.]//g')
        local total_reclaimable=0
        for size in $reclaimable; do
            total_reclaimable=$(echo "$total_reclaimable + $size" | bc)
        done

        if (( $(echo "$total_reclaimable > 10" | bc -l) )); then
            log_warning "Docker has ${total_reclaimable}GB reclaimable space"
            add_conflict "disk" "docker" "Docker using excessive disk space" "medium" \
                "Run cleanup: docker system prune -a --volumes"
        fi
    fi
}

################################################################################
# Memory Analysis
################################################################################

check_memory_resources() {
    log_section "CHECKING MEMORY RESOURCES"

    # Get memory info
    local total_mem=$(free -h | grep "^Mem:" | awk '{print $2}')
    local used_mem=$(free -h | grep "^Mem:" | awk '{print $3}')
    local available_mem=$(free -h | grep "^Mem:" | awk '{print $7}')

    log_info "Memory: $used_mem used / $total_mem total (Available: $available_mem)"

    # Check if memory is low
    local available_gb=$(free -g | grep "^Mem:" | awk '{print $7}')
    if [[ $available_gb -lt 8 ]]; then
        log_warning "Less than 8GB RAM available"
        add_conflict "memory" "system" "Insufficient available memory" "high" \
            "Free up memory or add more RAM"
    fi

    # Check for memory-intensive processes
    log_info "Top memory-consuming processes:"
    ps aux --sort=-%mem | head -6 | tail -5 | while read -r line; do
        local proc=$(echo "$line" | awk '{print $11}')
        local mem=$(echo "$line" | awk '{print $4}')
        local pid=$(echo "$line" | awk '{print $2}')
        echo "  $proc (PID: $pid): ${mem}% memory"

        # Flag if using too much memory
        if (( $(echo "$mem > 20" | bc -l) )); then
            add_conflict "memory" "$proc" "Process using ${mem}% of memory" "medium" \
                "Consider stopping or optimizing process"
        fi
    done

    # Check swap usage
    local swap_total=$(free -h | grep "^Swap:" | awk '{print $2}')
    local swap_used=$(free -h | grep "^Swap:" | awk '{print $3}')

    if [[ "$swap_total" != "0B" ]]; then
        log_info "Swap: $swap_used used / $swap_total total"

        local swap_percent=$(free | grep "^Swap:" | awk '{if ($2 > 0) print int($3/$2*100)}')
        if [[ -n "$swap_percent" ]] && [[ $swap_percent -gt 50 ]]; then
            log_warning "High swap usage (${swap_percent}%)"
            add_conflict "memory" "swap" "High swap usage indicates memory pressure" "medium" \
                "Add more RAM or optimize memory usage"
        fi
    fi
}

################################################################################
# Generate Remediation Script
################################################################################

generate_remediation_script() {
    log_section "GENERATING REMEDIATION SCRIPT"

    local script_file="./remediate-conflicts-$(date +%Y%m%d-%H%M%S).sh"

    cat > "$script_file" << 'EOF'
#!/bin/bash
#
# AgenticDosNode Conflict Remediation Script
# Generated: $(date)
#
# This script contains suggested remediation actions for detected conflicts.
# Review each action before executing.
#

set -euo pipefail

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}AgenticDosNode Conflict Remediation${NC}"
echo "======================================="
echo ""
echo "This script will help resolve conflicts detected on your system."
echo "Each action will be confirmed before execution."
echo ""

confirm() {
    read -p "$1 [y/N]: " -n 1 -r
    echo
    [[ $REPLY =~ ^[Yy]$ ]]
}

# Function to safely stop services
stop_service() {
    local service=$1
    if systemctl is-active "$service" &>/dev/null; then
        echo -e "${YELLOW}Stopping $service...${NC}"
        sudo systemctl stop "$service"
        sudo systemctl disable "$service"
        echo -e "${GREEN}$service stopped and disabled${NC}"
    else
        echo "$service is not active"
    fi
}

# Function to free ports
free_port() {
    local port=$1
    local pids=$(lsof -ti:$port 2>/dev/null || true)
    if [[ -n "$pids" ]]; then
        echo -e "${YELLOW}Processes using port $port: $pids${NC}"
        if confirm "Kill these processes?"; then
            echo "$pids" | xargs -r kill -TERM
            sleep 2
            # Force kill if still running
            echo "$pids" | xargs -r kill -9 2>/dev/null || true
            echo -e "${GREEN}Port $port freed${NC}"
        fi
    else
        echo "Port $port is already free"
    fi
}

# Function to clean Docker resources
clean_docker() {
    if command -v docker &> /dev/null; then
        echo -e "${YELLOW}Cleaning Docker resources...${NC}"

        if confirm "Stop all Docker containers?"; then
            docker stop $(docker ps -aq) 2>/dev/null || true
        fi

        if confirm "Remove stopped containers?"; then
            docker container prune -f
        fi

        if confirm "Remove unused images?"; then
            docker image prune -a -f
        fi

        if confirm "Remove unused volumes?"; then
            docker volume prune -f
        fi

        if confirm "Remove unused networks?"; then
            docker network prune -f
        fi

        echo -e "${GREEN}Docker cleanup complete${NC}"
    fi
}

# Main remediation actions
echo ""
echo "=== SERVICE CONFLICTS ==="
EOF

    # Add specific remediation commands based on conflicts
    local conflicts=$(echo "$REPORT_JSON" | jq -r '.conflicts[] | @base64')

    for conflict in $conflicts; do
        local category=$(echo "$conflict" | base64 -d | jq -r '.category')
        local service=$(echo "$conflict" | base64 -d | jq -r '.service')
        local remediation=$(echo "$conflict" | base64 -d | jq -r '.remediation')

        case "$category" in
            "port")
                local port=$(echo "$remediation" | grep -oP '\d+' | head -1)
                if [[ -n "$port" ]]; then
                    echo "if confirm \"Free port $port used by $service?\"; then" >> "$script_file"
                    echo "    free_port $port" >> "$script_file"
                    echo "fi" >> "$script_file"
                    echo "" >> "$script_file"
                fi
                ;;
            "service")
                echo "if confirm \"Stop $service service?\"; then" >> "$script_file"
                echo "    stop_service \"$service\"" >> "$script_file"
                echo "fi" >> "$script_file"
                echo "" >> "$script_file"
                ;;
        esac
    done

    cat >> "$script_file" << 'EOF'

echo ""
echo "=== DOCKER CLEANUP ==="
if confirm "Perform Docker cleanup?"; then
    clean_docker
fi

echo ""
echo "=== FIREWALL CONFIGURATION ==="
if confirm "Configure firewall for AgenticDosNode?"; then
    # Allow Tailscale
    sudo ufw allow 41641/udp comment 'Tailscale'

    # Allow required ports (adjust based on your node type)
    # thanos node
    #sudo ufw allow 8000/tcp comment 'vLLM Server'
    #sudo ufw allow 6333/tcp comment 'Qdrant'

    # oracle1 node
    #sudo ufw allow 5678/tcp comment 'n8n'
    #sudo ufw allow 8080/tcp comment 'LangGraph'

    echo -e "${GREEN}Firewall rules updated${NC}"
fi

echo ""
echo "=== SYSTEM OPTIMIZATION ==="
if confirm "Optimize system settings for AgenticDosNode?"; then
    # Increase file descriptor limits
    echo "* soft nofile 65536" | sudo tee -a /etc/security/limits.conf
    echo "* hard nofile 65536" | sudo tee -a /etc/security/limits.conf

    # Optimize kernel parameters
    echo "net.core.rmem_max = 134217728" | sudo tee -a /etc/sysctl.conf
    echo "net.core.wmem_max = 134217728" | sudo tee -a /etc/sysctl.conf
    echo "net.ipv4.tcp_rmem = 4096 87380 134217728" | sudo tee -a /etc/sysctl.conf
    echo "net.ipv4.tcp_wmem = 4096 65536 134217728" | sudo tee -a /etc/sysctl.conf

    sudo sysctl -p

    echo -e "${GREEN}System optimizations applied${NC}"
fi

echo ""
echo -e "${GREEN}Remediation complete!${NC}"
echo "Please review the changes and restart services as needed."
EOF

    chmod +x "$script_file"
    log_success "Remediation script generated: $script_file"
}

################################################################################
# Main Execution
################################################################################

main() {
    log_info "Starting AgenticDosNode Resource Conflict Detection"
    log_info "Hostname: $(hostname)"
    log_info "OS: $(lsb_release -d 2>/dev/null | cut -f2 || cat /etc/os-release | grep PRETTY_NAME | cut -d'"' -f2)"
    log_info "Kernel: $(uname -r)"
    log_info "Architecture: $(uname -m)"
    echo ""

    # Run all checks
    check_port_conflicts
    check_docker_conflicts
    check_python_environments
    check_gpu_conflicts
    check_network_conflicts
    check_specific_services
    check_disk_space
    check_memory_resources

    # Generate report
    log_section "GENERATING REPORT"

    # Save JSON report
    echo "$REPORT_JSON" | jq '.' > "$REPORT_FILE"
    log_success "Detailed report saved to: $REPORT_FILE"

    # Generate remediation script
    generate_remediation_script

    # Summary
    log_section "SUMMARY"

    local total_conflicts=$(echo "$REPORT_JSON" | jq '.conflicts | length')
    local critical=$(echo "$REPORT_JSON" | jq '.conflicts | map(select(.severity == "critical")) | length')
    local high=$(echo "$REPORT_JSON" | jq '.conflicts | map(select(.severity == "high")) | length')
    local medium=$(echo "$REPORT_JSON" | jq '.conflicts | map(select(.severity == "medium")) | length')
    local low=$(echo "$REPORT_JSON" | jq '.conflicts | map(select(.severity == "low")) | length')

    echo "Total conflicts detected: $total_conflicts"
    echo "  Critical: $critical"
    echo "  High: $high"
    echo "  Medium: $medium"
    echo "  Low: $low"
    echo ""

    if [[ $total_conflicts -eq 0 ]]; then
        log_success "No conflicts detected! System is ready for AgenticDosNode deployment."
    elif [[ $critical -gt 0 ]]; then
        log_error "Critical conflicts detected. Please resolve before deployment."
        exit 1
    elif [[ $high -gt 0 ]]; then
        log_warning "High priority conflicts detected. Review and resolve recommended."
        exit 2
    else
        log_warning "Some conflicts detected. Review report for details."
    fi

    echo ""
    log_info "Next steps:"
    echo "  1. Review the detailed report: $REPORT_FILE"
    echo "  2. Run remediation script: bash remediate-conflicts-*.sh"
    echo "  3. Re-run this detector after remediation"
    echo "  4. Proceed with AgenticDosNode deployment"
}

# Check if running as root (optional, remove if not needed)
if [[ $EUID -ne 0 ]]; then
    log_warning "Running as non-root. Some checks may be limited."
    log_info "For complete analysis, run: sudo $0"
fi

# Create log directory if it doesn't exist
mkdir -p "$(dirname "$LOG_FILE")"

# Run main function
main "$@"