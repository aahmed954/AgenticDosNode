#!/bin/bash

# AgenticDosNode Environment Preparation Script
# Common preparation tasks for both GPU and CPU nodes
# Version: 1.0.0
# Author: Claude Code DevOps Troubleshooter

set -euo pipefail

# Get script directory and source utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/cleanup-utils.sh"

# Configuration
readonly AGENTICNODE_USER="${AGENTICNODE_USER:-agenticnode}"
readonly AGENTICNODE_GROUP="${AGENTICNODE_GROUP:-agenticnode}"
readonly INSTALL_DIR="${INSTALL_DIR:-/opt/agenticnode}"
readonly DATA_DIR="${DATA_DIR:-/var/lib/agenticnode}"
readonly LOG_DIR_APP="${LOG_DIR_APP:-/var/log/agenticnode}"

# Docker configuration
readonly DOCKER_COMPOSE_VERSION="2.24.0"
readonly DOCKER_VERSION="24.0"

# Required packages
readonly REQUIRED_PACKAGES=(
    "curl" "wget" "git" "jq" "htop" "iotop" "netstat-nat" "ss"
    "build-essential" "ca-certificates" "gnupg" "lsb-release"
    "software-properties-common" "apt-transport-https"
    "python3" "python3-pip" "python3-venv"
    "nodejs" "npm"
)

# System limits and optimizations
readonly SYSTEM_OPTIMIZATIONS=(
    "net.core.somaxconn=65535"
    "net.core.netdev_max_backlog=5000"
    "net.ipv4.tcp_max_syn_backlog=4096"
    "vm.swappiness=10"
    "vm.dirty_ratio=60"
    "vm.dirty_background_ratio=2"
    "fs.file-max=2097152"
)

main() {
    log "INFO" "Starting AgenticDosNode environment preparation"
    log "INFO" "Timestamp: $(date)"
    log "INFO" "User: $(whoami)"
    log "INFO" "Working directory: $PWD"

    # Detect system information
    detect_system_info

    # Check if cleanup was run
    check_cleanup_status

    cat << EOF

${GREEN}AgenticDosNode Environment Preparation${NC}
This script will prepare your system for AgenticDosNode deployment.

WHAT WILL BE INSTALLED/CONFIGURED:
- Docker and Docker Compose
- System packages and dependencies
- User accounts and permissions
- Directory structure
- System optimizations
- Security configurations
- Monitoring tools

SYSTEM MODIFICATIONS:
- Install packages via apt
- Create system user and group
- Modify system limits and kernel parameters
- Configure Docker daemon
- Set up systemd services
- Create directories with proper permissions

EOF

    if ! confirm "Proceed with environment preparation?"; then
        log "INFO" "Preparation aborted by user"
        exit 0
    fi

    # Preparation phases
    update_system_packages
    install_docker
    install_docker_compose
    create_user_accounts
    setup_directory_structure
    configure_system_limits
    install_monitoring_tools
    configure_security
    setup_logrotation
    optimize_system_performance
    install_agenticnode_dependencies
    setup_systemd_services
    configure_firewall

    # Final validation
    validate_installation

    show_preparation_summary
    log "INFO" "Environment preparation completed successfully"
}

detect_system_info() {
    log "STEP" "Detecting system information..."

    # Basic system info
    local os_info=$(lsb_release -d 2>/dev/null | cut -f2 || echo "Unknown")
    local kernel_version=$(uname -r)
    local architecture=$(uname -m)
    local cpu_count=$(nproc)
    local total_memory=$(free -h | grep "Mem:" | awk '{print $2}')
    local disk_space=$(df -h / | tail -1 | awk '{print $4}')

    log "INFO" "Operating System: $os_info"
    log "INFO" "Kernel Version: $kernel_version"
    log "INFO" "Architecture: $architecture"
    log "INFO" "CPU Cores: $cpu_count"
    log "INFO" "Total Memory: $total_memory"
    log "INFO" "Available Disk Space: $disk_space"

    # Check for GPU
    if command -v nvidia-smi >/dev/null 2>&1; then
        local gpu_info=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)
        log "INFO" "GPU: $gpu_info"
        export HAS_GPU=true
    else
        log "INFO" "GPU: Not detected (CPU-only node)"
        export HAS_GPU=false
    fi

    # Network information
    local primary_interface=$(ip route | grep default | awk '{print $5}' | head -1)
    local ip_address=$(ip addr show "$primary_interface" | grep "inet " | awk '{print $2}' | cut -d'/' -f1 | head -1)
    log "INFO" "Primary Interface: $primary_interface"
    log "INFO" "IP Address: $ip_address"

    # Export for other functions
    export PRIMARY_IP="$ip_address"
    export CPU_COUNT="$cpu_count"
}

check_cleanup_status() {
    log "STEP" "Checking cleanup status..."

    # Check if cleanup logs exist
    local cleanup_logs=$(find "${SCRIPT_DIR}/cleanup-logs" -name "cleanup_*.log" 2>/dev/null | head -1 || true)
    if [[ -n "$cleanup_logs" ]]; then
        local last_cleanup=$(basename "$cleanup_logs" | sed 's/cleanup_\(.*\)\.log/\1/')
        log "INFO" "Found cleanup log from: $last_cleanup"
    else
        log "WARN" "No cleanup logs found. It's recommended to run cleanup first."
        if confirm "Continue without prior cleanup?"; then
            log "INFO" "Proceeding without cleanup"
        else
            log "INFO" "Please run the appropriate cleanup script first"
            exit 1
        fi
    fi

    # Check for common conflicting services
    local conflicting_services=()

    # Check Docker
    if systemctl is-active --quiet docker 2>/dev/null; then
        local docker_containers=$(docker ps -q | wc -l)
        if [[ "$docker_containers" -gt 0 ]]; then
            log "WARN" "Docker is running with $docker_containers containers"
            conflicting_services+=("docker-containers")
        fi
    fi

    # Check required ports
    local port_conflicts=()
    for port in 3000 8000 8001 5432; do
        if check_port "$port"; then
            port_conflicts+=("$port")
        fi
    done

    if [[ ${#port_conflicts[@]} -gt 0 ]]; then
        log "WARN" "Port conflicts detected: ${port_conflicts[*]}"
        if ! confirm "Continue with port conflicts? (This may cause issues)"; then
            log "ERROR" "Please resolve port conflicts and run again"
            exit 1
        fi
    fi
}

update_system_packages() {
    log "STEP" "Updating system packages..."

    # Update package lists
    log "INFO" "Updating package lists..."
    apt-get update -qq || {
        log "ERROR" "Failed to update package lists"
        return 1
    }

    # Upgrade system packages
    if confirm "Upgrade system packages? (Recommended)"; then
        log "INFO" "Upgrading system packages..."
        DEBIAN_FRONTEND=noninteractive apt-get upgrade -y -qq || {
            log "WARN" "Some packages failed to upgrade"
        }
    fi

    # Install required packages
    log "INFO" "Installing required packages..."
    local missing_packages=()

    for package in "${REQUIRED_PACKAGES[@]}"; do
        if ! dpkg -l "$package" >/dev/null 2>&1; then
            missing_packages+=("$package")
        fi
    done

    if [[ ${#missing_packages[@]} -gt 0 ]]; then
        log "INFO" "Installing missing packages: ${missing_packages[*]}"
        DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "${missing_packages[@]}" || {
            log "ERROR" "Failed to install required packages"
            return 1
        }
    else
        log "INFO" "All required packages are already installed"
    fi

    log "INFO" "System package update completed"
}

install_docker() {
    log "STEP" "Installing Docker..."

    if command -v docker >/dev/null 2>&1; then
        local docker_version=$(docker --version | awk '{print $3}' | sed 's/,//')
        log "INFO" "Docker is already installed: $docker_version"

        # Check if version is recent enough
        local version_major=$(echo "$docker_version" | cut -d'.' -f1)
        if [[ "$version_major" -ge 24 ]]; then
            log "INFO" "Docker version is sufficient"
            return 0
        else
            log "WARN" "Docker version is old, will upgrade"
        fi
    fi

    # Remove old Docker versions
    log "INFO" "Removing old Docker installations..."
    apt-get remove -y -qq docker docker-engine docker.io containerd runc 2>/dev/null || true

    # Add Docker's official GPG key
    log "INFO" "Adding Docker repository..."
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    chmod a+r /etc/apt/keyrings/docker.gpg

    # Add Docker repository
    echo \
        "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
        $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
        tee /etc/apt/sources.list.d/docker.list > /dev/null

    # Update package lists
    apt-get update -qq

    # Install Docker
    log "INFO" "Installing Docker CE..."
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
        docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

    # Configure Docker daemon
    log "INFO" "Configuring Docker daemon..."
    local docker_config="/etc/docker/daemon.json"
    backup_item "$docker_config" "docker_daemon_config" 2>/dev/null || true

    local docker_daemon_config='{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  },
  "storage-driver": "overlay2",
  "exec-opts": ["native.cgroupdriver=systemd"],
  "live-restore": true,
  "userland-proxy": false,
  "experimental": false,
  "features": {
    "buildkit": true
  }
}'

    # Add GPU support if available
    if [[ "$HAS_GPU" == "true" ]]; then
        log "INFO" "Adding GPU support to Docker daemon config..."
        docker_daemon_config=$(echo "$docker_daemon_config" | jq '. + {
          "default-runtime": "nvidia",
          "runtimes": {
            "nvidia": {
              "path": "nvidia-container-runtime",
              "runtimeArgs": []
            }
          }
        }')
    fi

    echo "$docker_daemon_config" > "$docker_config"

    # Create docker group and add user
    if ! getent group docker >/dev/null 2>&1; then
        groupadd docker
    fi

    # Add current user to docker group
    if [[ -n "${SUDO_USER:-}" ]]; then
        usermod -aG docker "$SUDO_USER"
        log "INFO" "Added user $SUDO_USER to docker group"
    fi

    # Start and enable Docker
    systemctl daemon-reload
    systemctl enable docker
    systemctl start docker

    # Wait for Docker to be ready
    log "INFO" "Waiting for Docker to be ready..."
    local timeout=30
    local count=0
    while ! docker info >/dev/null 2>&1; do
        sleep 1
        ((count++))
        if [[ $count -ge $timeout ]]; then
            log "ERROR" "Docker failed to start within $timeout seconds"
            return 1
        fi
    done

    log "INFO" "Docker installation completed successfully"
}

install_docker_compose() {
    log "STEP" "Installing Docker Compose..."

    # Remove old docker-compose
    if command -v docker-compose >/dev/null 2>&1; then
        log "INFO" "Removing old docker-compose installation..."
        rm -f /usr/local/bin/docker-compose 2>/dev/null || true
        rm -f /usr/bin/docker-compose 2>/dev/null || true
    fi

    # Docker Compose is now included as a plugin, verify it works
    if docker compose version >/dev/null 2>&1; then
        local compose_version=$(docker compose version --short)
        log "INFO" "Docker Compose plugin is available: $compose_version"
    else
        log "ERROR" "Docker Compose plugin is not available"
        return 1
    fi

    # Create compatibility symlink
    if [[ ! -f /usr/local/bin/docker-compose ]]; then
        log "INFO" "Creating docker-compose compatibility symlink..."
        cat > /usr/local/bin/docker-compose << 'EOF'
#!/bin/bash
docker compose "$@"
EOF
        chmod +x /usr/local/bin/docker-compose
    fi

    log "INFO" "Docker Compose installation completed"
}

create_user_accounts() {
    log "STEP" "Creating user accounts..."

    # Create agenticnode group
    if ! getent group "$AGENTICNODE_GROUP" >/dev/null 2>&1; then
        log "INFO" "Creating group: $AGENTICNODE_GROUP"
        groupadd --system "$AGENTICNODE_GROUP"
    else
        log "INFO" "Group already exists: $AGENTICNODE_GROUP"
    fi

    # Create agenticnode user
    if ! getent passwd "$AGENTICNODE_USER" >/dev/null 2>&1; then
        log "INFO" "Creating user: $AGENTICNODE_USER"
        useradd --system \
            --gid "$AGENTICNODE_GROUP" \
            --home-dir /home/"$AGENTICNODE_USER" \
            --create-home \
            --shell /bin/bash \
            "$AGENTICNODE_USER"
    else
        log "INFO" "User already exists: $AGENTICNODE_USER"
    fi

    # Add user to necessary groups
    local groups_to_add=("docker")
    if [[ "$HAS_GPU" == "true" ]]; then
        groups_to_add+=("video")
    fi

    for group in "${groups_to_add[@]}"; do
        if getent group "$group" >/dev/null 2>&1; then
            usermod -aG "$group" "$AGENTICNODE_USER"
            log "INFO" "Added $AGENTICNODE_USER to group: $group"
        else
            log "WARN" "Group does not exist: $group"
        fi
    done

    # Create SSH key for the user if it doesn't exist
    local ssh_dir="/home/$AGENTICNODE_USER/.ssh"
    if [[ ! -d "$ssh_dir" ]]; then
        log "INFO" "Creating SSH directory for $AGENTICNODE_USER"
        sudo -u "$AGENTICNODE_USER" mkdir -p "$ssh_dir"
        sudo -u "$AGENTICNODE_USER" chmod 700 "$ssh_dir"

        # Generate SSH key
        if [[ ! -f "$ssh_dir/id_rsa" ]]; then
            log "INFO" "Generating SSH key for $AGENTICNODE_USER"
            sudo -u "$AGENTICNODE_USER" ssh-keygen -t rsa -b 4096 -f "$ssh_dir/id_rsa" -N "" -q
        fi
    fi

    log "INFO" "User account creation completed"
}

setup_directory_structure() {
    log "STEP" "Setting up directory structure..."

    # Main directories
    local directories=(
        "$INSTALL_DIR"
        "$DATA_DIR"
        "$LOG_DIR_APP"
        "$DATA_DIR/models"
        "$DATA_DIR/data"
        "$DATA_DIR/config"
        "$DATA_DIR/backups"
        "$DATA_DIR/temp"
        "/etc/agenticnode"
    )

    for dir in "${directories[@]}"; do
        if [[ ! -d "$dir" ]]; then
            log "INFO" "Creating directory: $dir"
            mkdir -p "$dir"
        else
            log "DEBUG" "Directory already exists: $dir"
        fi

        # Set ownership
        chown "$AGENTICNODE_USER:$AGENTICNODE_GROUP" "$dir"

        # Set permissions
        case "$dir" in
            */config|*/backups)
                chmod 750 "$dir"
                ;;
            */temp)
                chmod 755 "$dir"
                ;;
            */logs|"$LOG_DIR_APP")
                chmod 755 "$dir"
                ;;
            *)
                chmod 755 "$dir"
                ;;
        esac
    done

    # Create systemd directory for user services
    local systemd_dir="/home/$AGENTICNODE_USER/.config/systemd/user"
    if [[ ! -d "$systemd_dir" ]]; then
        log "INFO" "Creating systemd user directory"
        sudo -u "$AGENTICNODE_USER" mkdir -p "$systemd_dir"
    fi

    # Create a basic config file
    local config_file="$DATA_DIR/config/agenticnode.conf"
    if [[ ! -f "$config_file" ]]; then
        log "INFO" "Creating initial configuration file"
        cat > "$config_file" << EOF
# AgenticDosNode Configuration
# Generated during environment preparation

[general]
install_dir = $INSTALL_DIR
data_dir = $DATA_DIR
log_dir = $LOG_DIR_APP
user = $AGENTICNODE_USER
group = $AGENTICNODE_GROUP

[system]
has_gpu = $HAS_GPU
cpu_count = $CPU_COUNT
primary_ip = $PRIMARY_IP

[docker]
compose_file = $INSTALL_DIR/docker-compose.yml

[network]
frontend_port = 3000
api_port = 8000
admin_port = 8001
EOF

        chown "$AGENTICNODE_USER:$AGENTICNODE_GROUP" "$config_file"
        chmod 640 "$config_file"
    fi

    log "INFO" "Directory structure setup completed"
}

configure_system_limits() {
    log "STEP" "Configuring system limits..."

    # Configure /etc/security/limits.conf
    local limits_file="/etc/security/limits.conf"
    backup_item "$limits_file" "security_limits_conf"

    log "INFO" "Updating system limits..."
    cat >> "$limits_file" << EOF

# AgenticDosNode limits
$AGENTICNODE_USER soft nofile 65536
$AGENTICNODE_USER hard nofile 65536
$AGENTICNODE_USER soft nproc 32768
$AGENTICNODE_USER hard nproc 32768

# Docker limits
root soft nofile 65536
root hard nofile 65536
EOF

    # Configure systemd limits
    local systemd_limits="/etc/systemd/system.conf"
    backup_item "$systemd_limits" "systemd_system_conf"

    if ! grep -q "DefaultLimitNOFILE=65536" "$systemd_limits"; then
        log "INFO" "Updating systemd limits..."
        sed -i 's/#DefaultLimitNOFILE=/DefaultLimitNOFILE=65536/' "$systemd_limits"
        sed -i 's/#DefaultLimitNPROC=/DefaultLimitNPROC=32768/' "$systemd_limits"
    fi

    # Configure sysctl parameters
    local sysctl_file="/etc/sysctl.d/99-agenticnode.conf"
    log "INFO" "Configuring kernel parameters..."

    cat > "$sysctl_file" << EOF
# AgenticDosNode kernel parameters
# Network optimizations
net.core.somaxconn = 65535
net.core.netdev_max_backlog = 5000
net.ipv4.tcp_max_syn_backlog = 4096
net.ipv4.tcp_fin_timeout = 30
net.ipv4.tcp_keepalive_time = 1800
net.ipv4.tcp_keepalive_probes = 7
net.ipv4.tcp_keepalive_intvl = 30

# Memory management
vm.swappiness = 10
vm.dirty_ratio = 60
vm.dirty_background_ratio = 2
vm.overcommit_memory = 1

# File system
fs.file-max = 2097152
fs.inotify.max_user_watches = 524288
fs.inotify.max_user_instances = 256
EOF

    # GPU-specific optimizations
    if [[ "$HAS_GPU" == "true" ]]; then
        cat >> "$sysctl_file" << EOF

# GPU optimizations
vm.mmap_min_addr = 65536
kernel.yama.ptrace_scope = 0
EOF
    fi

    # Apply sysctl settings
    sysctl -p "$sysctl_file" >/dev/null 2>&1 || log "WARN" "Some sysctl parameters could not be applied"

    log "INFO" "System limits configuration completed"
}

install_monitoring_tools() {
    log "STEP" "Installing monitoring tools..."

    # Install additional monitoring packages
    local monitoring_packages=("htop" "iotop" "nethogs" "dstat" "ncdu" "tree")
    local missing_monitoring=()

    for package in "${monitoring_packages[@]}"; do
        if ! command -v "$package" >/dev/null 2>&1; then
            missing_monitoring+=("$package")
        fi
    done

    if [[ ${#missing_monitoring[@]} -gt 0 ]]; then
        log "INFO" "Installing monitoring packages: ${missing_monitoring[*]}"
        DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "${missing_monitoring[@]}" || {
            log "WARN" "Some monitoring tools failed to install"
        }
    fi

    # Install Docker monitoring tools
    log "INFO" "Installing ctop (Docker container monitoring)..."
    if ! command -v ctop >/dev/null 2>&1; then
        wget https://github.com/bcicen/ctop/releases/download/v0.7.7/ctop-0.7.7-linux-amd64 -O /usr/local/bin/ctop
        chmod +x /usr/local/bin/ctop
    fi

    # GPU monitoring (if available)
    if [[ "$HAS_GPU" == "true" ]]; then
        log "INFO" "Installing GPU monitoring tools..."
        if ! command -v nvtop >/dev/null 2>&1; then
            DEBIAN_FRONTEND=noninteractive apt-get install -y -qq nvtop 2>/dev/null || {
                log "WARN" "nvtop installation failed, will install from source"
                # Could add nvtop compilation here if needed
            }
        fi

        if ! command -v gpustat >/dev/null 2>&1; then
            pip3 install gpustat 2>/dev/null || log "WARN" "gpustat installation failed"
        fi
    fi

    log "INFO" "Monitoring tools installation completed"
}

configure_security() {
    log "STEP" "Configuring security settings..."

    # Configure UFW firewall (if installed)
    if command -v ufw >/dev/null 2>&1; then
        log "INFO" "Configuring UFW firewall..."

        # Reset UFW to defaults
        ufw --force reset >/dev/null 2>&1

        # Default policies
        ufw default deny incoming
        ufw default allow outgoing

        # SSH access
        ufw allow ssh

        # AgenticDosNode ports (from specific IPs if configured)
        local allowed_ips="${ALLOWED_IPS:-any}"
        if [[ "$allowed_ips" != "any" ]]; then
            IFS=',' read -ra IP_ARRAY <<< "$allowed_ips"
            for ip in "${IP_ARRAY[@]}"; do
                ufw allow from "$ip" to any port 3000
                ufw allow from "$ip" to any port 8000
                ufw allow from "$ip" to any port 8001
                log "INFO" "Allowed access from $ip to AgenticDosNode ports"
            done
        else
            # Allow from anywhere (less secure)
            ufw allow 3000
            ufw allow 8000
            ufw allow 8001
        fi

        # Don't enable automatically - let user decide
        log "WARN" "UFW configured but not enabled. Run 'ufw enable' to activate"
    fi

    # Configure fail2ban (if available)
    if command -v fail2ban-client >/dev/null 2>&1; then
        log "INFO" "Configuring fail2ban..."

        systemctl enable fail2ban
        systemctl start fail2ban || log "WARN" "Failed to start fail2ban"
    else
        log "INFO" "Installing fail2ban..."
        DEBIAN_FRONTEND=noninteractive apt-get install -y -qq fail2ban || {
            log "WARN" "fail2ban installation failed"
        }
    fi

    # Set up automatic security updates
    if confirm "Enable automatic security updates?"; then
        log "INFO" "Configuring automatic security updates..."
        DEBIAN_FRONTEND=noninteractive apt-get install -y -qq unattended-upgrades

        # Configure unattended-upgrades
        cat > /etc/apt/apt.conf.d/20auto-upgrades << EOF
APT::Periodic::Update-Package-Lists "1";
APT::Periodic::Unattended-Upgrade "1";
APT::Periodic::AutocleanInterval "7";
EOF
    fi

    # Secure shared memory
    if ! grep -q "tmpfs /run/shm tmpfs defaults,noexec,nosuid 0 0" /etc/fstab; then
        log "INFO" "Securing shared memory..."
        echo "tmpfs /run/shm tmpfs defaults,noexec,nosuid 0 0" >> /etc/fstab
    fi

    log "INFO" "Security configuration completed"
}

setup_logrotation() {
    log "STEP" "Setting up log rotation..."

    # Create logrotate configuration for AgenticDosNode
    local logrotate_config="/etc/logrotate.d/agenticnode"
    cat > "$logrotate_config" << EOF
$LOG_DIR_APP/*.log {
    daily
    missingok
    rotate 7
    compress
    delaycompress
    notifempty
    sharedscripts
    copytruncate
    su $AGENTICNODE_USER $AGENTICNODE_GROUP
}

$DATA_DIR/logs/*.log {
    daily
    missingok
    rotate 14
    compress
    delaycompress
    notifempty
    sharedscripts
    copytruncate
    su $AGENTICNODE_USER $AGENTICNODE_GROUP
}

# Docker logs are handled by Docker daemon configuration
EOF

    # Test logrotate configuration
    if logrotate -d "$logrotate_config" >/dev/null 2>&1; then
        log "INFO" "Logrotate configuration is valid"
    else
        log "WARN" "Logrotate configuration may have issues"
    fi

    log "INFO" "Log rotation setup completed"
}

optimize_system_performance() {
    log "STEP" "Optimizing system performance..."

    # I/O scheduler optimization
    log "INFO" "Configuring I/O scheduler..."
    local schedulers_config="/etc/udev/rules.d/60-schedulers.rules"
    cat > "$schedulers_config" << 'EOF'
# Set I/O scheduler
ACTION=="add|change", KERNEL=="sd[a-z]*|nvme*", ATTR{queue/scheduler}="mq-deadline"
ACTION=="add|change", KERNEL=="sd[a-z]*|nvme*", ATTR{queue/read_ahead_kb}="128"
ACTION=="add|change", KERNEL=="sd[a-z]*|nvme*", ATTR{queue/max_sectors_kb}="1024"
EOF

    # CPU governor (if available)
    if [[ -d /sys/devices/system/cpu/cpu0/cpufreq ]]; then
        log "INFO" "Configuring CPU governor..."
        local cpufreq_config="/etc/systemd/system/cpufreq.service"
        cat > "$cpufreq_config" << EOF
[Unit]
Description=Set CPU frequency scaling governor
DefaultDependencies=false
After=sysinit.target local-fs.target
Before=basic.target

[Service]
Type=oneshot
ExecStart=/bin/bash -c 'echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor'
RemainAfterExit=yes

[Install]
WantedBy=basic.target
EOF

        systemctl daemon-reload
        systemctl enable cpufreq
    fi

    # Transparent Huge Pages (THP) configuration
    log "INFO" "Configuring Transparent Huge Pages..."
    local thp_config="/etc/systemd/system/disable-thp.service"
    cat > "$thp_config" << EOF
[Unit]
Description=Disable Transparent Huge Pages (THP)
DefaultDependencies=false
After=sysinit.target local-fs.target
Before=basic.target

[Service]
Type=oneshot
ExecStart=/bin/bash -c 'echo never | tee /sys/kernel/mm/transparent_hugepage/enabled'
ExecStart=/bin/bash -c 'echo never | tee /sys/kernel/mm/transparent_hugepage/defrag'
RemainAfterExit=yes

[Install]
WantedBy=basic.target
EOF

    systemctl daemon-reload
    systemctl enable disable-thp

    # Configure kernel modules for better performance
    local modules_config="/etc/modules-load.d/agenticnode.conf"
    cat > "$modules_config" << 'EOF'
# Modules for AgenticDosNode
tcp_bbr
EOF

    # Enable BBR congestion control
    if ! grep -q "tcp_bbr" /etc/sysctl.d/99-agenticnode.conf; then
        cat >> /etc/sysctl.d/99-agenticnode.conf << EOF

# TCP congestion control
net.core.default_qdisc = fq
net.ipv4.tcp_congestion_control = bbr
EOF
    fi

    log "INFO" "System performance optimization completed"
}

install_agenticnode_dependencies() {
    log "STEP" "Installing AgenticDosNode dependencies..."

    # Python dependencies
    log "INFO" "Installing Python dependencies..."
    pip3 install --upgrade pip setuptools wheel

    # Node.js version management
    if command -v npm >/dev/null 2>&1; then
        local node_version=$(node --version 2>/dev/null || echo "unknown")
        local npm_version=$(npm --version 2>/dev/null || echo "unknown")
        log "INFO" "Node.js: $node_version, npm: $npm_version"

        # Update npm to latest version
        npm install -g npm@latest 2>/dev/null || log "WARN" "Failed to update npm"
    else
        log "ERROR" "Node.js/npm not found"
        return 1
    fi

    # Install useful global npm packages
    local npm_packages=("pm2" "nodemon")
    for package in "${npm_packages[@]}"; do
        if ! npm list -g "$package" >/dev/null 2>&1; then
            log "INFO" "Installing npm package: $package"
            npm install -g "$package" 2>/dev/null || log "WARN" "Failed to install $package"
        fi
    done

    # Create Python virtual environment for AgenticDosNode
    local venv_dir="$INSTALL_DIR/venv"
    if [[ ! -d "$venv_dir" ]]; then
        log "INFO" "Creating Python virtual environment..."
        sudo -u "$AGENTICNODE_USER" python3 -m venv "$venv_dir"
        sudo -u "$AGENTICNODE_USER" "$venv_dir/bin/pip" install --upgrade pip setuptools wheel
    fi

    log "INFO" "AgenticDosNode dependencies installation completed"
}

setup_systemd_services() {
    log "STEP" "Setting up systemd services..."

    # Create AgenticDosNode main service
    local service_file="/etc/systemd/system/agenticnode.service"
    cat > "$service_file" << EOF
[Unit]
Description=AgenticDosNode Service
After=network.target docker.service
Wants=network.target docker.service
Requires=docker.service

[Service]
Type=forking
User=$AGENTICNODE_USER
Group=$AGENTICNODE_GROUP
WorkingDirectory=$INSTALL_DIR
Environment=NODE_ENV=production
Environment=PYTHONPATH=$INSTALL_DIR
ExecStart=/usr/local/bin/docker-compose up -d
ExecReload=/usr/local/bin/docker-compose restart
ExecStop=/usr/local/bin/docker-compose down
TimeoutStartSec=300
TimeoutStopSec=120
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    # Create health check service
    local healthcheck_service="/etc/systemd/system/agenticnode-healthcheck.service"
    cat > "$healthcheck_service" << EOF
[Unit]
Description=AgenticDosNode Health Check
After=agenticnode.service

[Service]
Type=oneshot
User=$AGENTICNODE_USER
Group=$AGENTICNODE_GROUP
ExecStart=/bin/bash -c 'curl -f http://localhost:8000/health || exit 1'
EOF

    local healthcheck_timer="/etc/systemd/system/agenticnode-healthcheck.timer"
    cat > "$healthcheck_timer" << EOF
[Unit]
Description=Run AgenticDosNode Health Check every 5 minutes
Requires=agenticnode-healthcheck.service

[Timer]
OnCalendar=*:0/5
Persistent=true

[Install]
WantedBy=timers.target
EOF

    # Create backup service
    local backup_service="/etc/systemd/system/agenticnode-backup.service"
    cat > "$backup_service" << EOF
[Unit]
Description=AgenticDosNode Backup
After=agenticnode.service

[Service]
Type=oneshot
User=$AGENTICNODE_USER
Group=$AGENTICNODE_GROUP
WorkingDirectory=$DATA_DIR
ExecStart=/bin/bash -c 'tar -czf backups/agenticnode-backup-\$(date +%%Y%%m%%d-%%H%%M%%S).tar.gz data config'
EOF

    local backup_timer="/etc/systemd/system/agenticnode-backup.timer"
    cat > "$backup_timer" << EOF
[Unit]
Description=Run AgenticDosNode Backup daily
Requires=agenticnode-backup.service

[Timer]
OnCalendar=daily
Persistent=true

[Install]
WantedBy=timers.target
EOF

    # Reload systemd
    systemctl daemon-reload

    # Enable services (but don't start them yet)
    systemctl enable agenticnode-healthcheck.timer
    systemctl enable agenticnode-backup.timer

    log "INFO" "Systemd services setup completed"
}

configure_firewall() {
    log "STEP" "Configuring firewall rules..."

    if ! command -v ufw >/dev/null 2>&1; then
        log "INFO" "Installing UFW firewall..."
        DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ufw
    fi

    # Configure but don't enable UFW yet
    log "INFO" "Configuring UFW rules (not enabling yet)..."

    # Reset to defaults
    ufw --force reset >/dev/null 2>&1

    # Set default policies
    ufw default deny incoming
    ufw default allow outgoing

    # Allow SSH
    ufw allow ssh

    # AgenticDosNode ports
    ufw allow 3000 comment "AgenticDosNode Frontend"
    ufw allow 8000 comment "AgenticDosNode API"
    ufw allow 8001 comment "AgenticDosNode Admin"

    # Docker network (if needed)
    ufw allow from 172.16.0.0/12

    log "WARN" "UFW is configured but not enabled"
    log "INFO" "To enable firewall, run: ufw enable"

    log "INFO" "Firewall configuration completed"
}

validate_installation() {
    log "STEP" "Validating installation..."

    local errors=0
    local warnings=0

    # Check Docker
    if ! docker info >/dev/null 2>&1; then
        log "ERROR" "Docker is not running"
        ((errors++))
    else
        log "INFO" "Docker is running"
    fi

    # Check Docker Compose
    if ! docker compose version >/dev/null 2>&1; then
        log "ERROR" "Docker Compose is not available"
        ((errors++))
    else
        log "INFO" "Docker Compose is available"
    fi

    # Check user account
    if ! getent passwd "$AGENTICNODE_USER" >/dev/null 2>&1; then
        log "ERROR" "AgenticDosNode user does not exist"
        ((errors++))
    else
        log "INFO" "AgenticDosNode user exists"
    fi

    # Check directories
    local critical_dirs=("$INSTALL_DIR" "$DATA_DIR" "$LOG_DIR_APP")
    for dir in "${critical_dirs[@]}"; do
        if [[ ! -d "$dir" ]]; then
            log "ERROR" "Critical directory missing: $dir"
            ((errors++))
        elif [[ ! -w "$dir" ]]; then
            log "WARN" "Directory not writable: $dir"
            ((warnings++))
        else
            log "DEBUG" "Directory OK: $dir"
        fi
    done

    # Check system limits
    local current_nofile=$(ulimit -n)
    if [[ "$current_nofile" -lt 65536 ]]; then
        log "WARN" "File descriptor limit is low: $current_nofile"
        ((warnings++))
    else
        log "INFO" "File descriptor limit OK: $current_nofile"
    fi

    # Check ports
    local required_ports=(3000 8000 8001)
    for port in "${required_ports[@]}"; do
        if check_port "$port"; then
            log "WARN" "Required port is in use: $port"
            ((warnings++))
        else
            log "DEBUG" "Port available: $port"
        fi
    done

    # Check system resources
    local available_memory=$(free -m | grep "Mem:" | awk '{print $7}')
    if [[ "$available_memory" -lt 2048 ]]; then
        log "WARN" "Low available memory: ${available_memory}MB"
        ((warnings++))
    else
        log "INFO" "Available memory OK: ${available_memory}MB"
    fi

    local available_disk=$(df / | tail -1 | awk '{print $4}')
    available_disk=$((available_disk / 1024 / 1024))  # Convert to GB
    if [[ "$available_disk" -lt 10 ]]; then
        log "WARN" "Low available disk space: ${available_disk}GB"
        ((warnings++))
    else
        log "INFO" "Available disk space OK: ${available_disk}GB"
    fi

    # Summary
    if [[ "$errors" -eq 0 && "$warnings" -eq 0 ]]; then
        log "INFO" "Validation passed - system is ready for AgenticDosNode"
        return 0
    elif [[ "$errors" -eq 0 ]]; then
        log "WARN" "Validation completed with $warnings warnings - system should be usable"
        return 0
    else
        log "ERROR" "Validation failed with $errors errors and $warnings warnings"
        log "ERROR" "Please resolve issues before deploying AgenticDosNode"
        return 1
    fi
}

show_preparation_summary() {
    log "STEP" "Preparation Summary"

    cat << EOF

${GREEN}AgenticDosNode Environment Preparation Complete${NC}

SYSTEM INFORMATION:
- OS: $(lsb_release -d 2>/dev/null | cut -f2 || echo "Unknown")
- Architecture: $(uname -m)
- CPU Cores: $CPU_COUNT
- GPU Support: $HAS_GPU
- Primary IP: $PRIMARY_IP

INSTALLED COMPONENTS:
- Docker Engine: $(docker --version 2>/dev/null || echo "Not available")
- Docker Compose: $(docker compose version --short 2>/dev/null || echo "Not available")
- Node.js: $(node --version 2>/dev/null || echo "Not available")
- Python: $(python3 --version 2>/dev/null || echo "Not available")

CREATED ACCOUNTS:
- User: $AGENTICNODE_USER
- Group: $AGENTICNODE_GROUP

DIRECTORY STRUCTURE:
- Install Directory: $INSTALL_DIR
- Data Directory: $DATA_DIR
- Log Directory: $LOG_DIR_APP
- Config File: $DATA_DIR/config/agenticnode.conf

SYSTEMD SERVICES:
- agenticnode.service (main service)
- agenticnode-healthcheck.timer (health monitoring)
- agenticnode-backup.timer (daily backups)

NEXT STEPS:
1. Deploy AgenticDosNode using docker-compose
2. Configure application-specific settings
3. Start services: systemctl start agenticnode
4. Enable firewall: ufw enable (optional)
5. Monitor logs: journalctl -u agenticnode -f

LOG FILES:
- Preparation log: $LOG_FILE
- Backup directory: $BACKUP_DIR

EOF

    if [[ -f "${BACKUP_DIR}/restore_map_${TIMESTAMP}.txt" ]]; then
        log "INFO" "Configuration backups created:"
        cat "${BACKUP_DIR}/restore_map_${TIMESTAMP}.txt" | while IFS=: read original backup; do
            log "INFO" "  $original -> $backup"
        done
    fi
}

# Cleanup function for script interruption
cleanup_on_exit() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        log "ERROR" "Script interrupted with exit code $exit_code"
        log "INFO" "Partial preparation may have occurred. Check logs: $LOG_FILE"
    fi
    exit $exit_code
}

# Set up signal handlers
trap cleanup_on_exit INT TERM EXIT

# Usage information
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

AgenticDosNode Environment Preparation Script

OPTIONS:
    -h, --help              Show this help message
    -y, --yes               Automatic yes to all prompts
    --skip-updates          Skip system package updates
    --skip-security         Skip security configuration
    --allowed-ips IPS       Comma-separated list of IPs allowed to access AgenticDosNode
    --install-dir DIR       Installation directory (default: /opt/agenticnode)
    --data-dir DIR          Data directory (default: /var/lib/agenticnode)
    --user USER             System user name (default: agenticnode)

EXAMPLES:
    $0                              Interactive preparation
    $0 -y                           Automatic preparation
    $0 --allowed-ips 192.168.1.0/24 Restrict access to local network
    $0 --skip-updates               Skip system updates

EOF
}

# Command line argument parsing
AUTO_YES=false
SKIP_UPDATES=false
SKIP_SECURITY=false
ALLOWED_IPS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            exit 0
            ;;
        -y|--yes)
            AUTO_YES=true
            shift
            ;;
        --skip-updates)
            SKIP_UPDATES=true
            shift
            ;;
        --skip-security)
            SKIP_SECURITY=true
            shift
            ;;
        --allowed-ips)
            ALLOWED_IPS="$2"
            shift 2
            ;;
        --install-dir)
            INSTALL_DIR="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --user)
            AGENTICNODE_USER="$2"
            AGENTICNODE_GROUP="$2"
            shift 2
            ;;
        *)
            log "ERROR" "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Export environment variables
export ALLOWED_IPS INSTALL_DIR DATA_DIR AGENTICNODE_USER AGENTICNODE_GROUP

# Override confirm function if auto-yes is enabled
if [[ "$AUTO_YES" == "true" ]]; then
    confirm() {
        log "INFO" "Auto-confirming: $1"
        return 0
    }
fi

# Check if running as root (required for system modifications)
if [[ $EUID -ne 0 ]]; then
    log "ERROR" "This script must be run as root (use sudo)"
    exit 1
fi

# Run main function
main "$@"