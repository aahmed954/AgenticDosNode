#!/bin/bash

# AgenticDosNode Clean State Validation Script
# Comprehensive verification of system readiness for deployment
# Version: 1.0.0
# Author: Claude Code DevOps Troubleshooter

set -euo pipefail

# Get script directory and source utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/cleanup-utils.sh"

# Configuration
readonly MACHINE_TYPE="${MACHINE_TYPE:-auto}"
readonly REQUIRED_PORTS=(3000 5678 6333 8000 8001 9090 8080 5432 6379 27017 11434)
readonly AGENTICNODE_PORTS=(3000 8000 8001)
readonly OPTIONAL_PORTS=(5678 6333 9090 8080 5432 6379 27017 11434)

# System requirements
readonly MIN_RAM_GB=4
readonly RECOMMENDED_RAM_GB=8
readonly MIN_DISK_GB=20
readonly RECOMMENDED_DISK_GB=50
readonly MIN_CPU_CORES=2

# Validation categories
declare -A VALIDATION_RESULTS
declare -A VALIDATION_DETAILS
declare -i TOTAL_CHECKS=0
declare -i PASSED_CHECKS=0
declare -i FAILED_CHECKS=0
declare -i WARNING_CHECKS=0

main() {
    log "INFO" "Starting AgenticDosNode clean state validation"
    log "INFO" "Timestamp: $(date)"
    log "INFO" "User: $(whoami)"
    log "INFO" "Working directory: $PWD"

    # Detect machine type if not specified
    if [[ "$MACHINE_TYPE" == "auto" ]]; then
        detect_machine_type
    fi

    log "INFO" "Machine type: $MACHINE_TYPE"

    cat << EOF

${CYAN}AgenticDosNode Clean State Validation${NC}
This script validates that your system is properly cleaned and ready for AgenticDosNode deployment.

VALIDATION CATEGORIES:
- System Resources (CPU, Memory, Disk)
- Network Ports and Connectivity
- Docker Environment
- Process and Service State
- System Configuration
- Security Settings
- Performance Optimization
- Directory Structure
- Machine-Specific Checks ($MACHINE_TYPE)

EOF

    if ! confirm "Proceed with validation?"; then
        log "INFO" "Validation aborted by user"
        exit 0
    fi

    # Run validation categories
    validate_system_resources
    validate_network_ports
    validate_docker_environment
    validate_process_state
    validate_system_configuration
    validate_security_settings
    validate_performance_optimization
    validate_directory_structure
    validate_machine_specific

    # Generate comprehensive report
    generate_validation_report

    # Show results summary
    show_validation_summary

    # Exit with appropriate code
    if [[ $FAILED_CHECKS -gt 0 ]]; then
        log "ERROR" "Validation failed - system is not ready for AgenticDosNode deployment"
        exit 1
    elif [[ $WARNING_CHECKS -gt 0 ]]; then
        log "WARN" "Validation completed with warnings - proceed with caution"
        exit 2
    else
        log "INFO" "Validation passed - system is ready for AgenticDosNode deployment"
        exit 0
    fi
}

detect_machine_type() {
    log "STEP" "Detecting machine type..."

    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
        export MACHINE_TYPE="GPU"
        log "INFO" "Detected GPU-capable machine"
    else
        export MACHINE_TYPE="CPU"
        log "INFO" "Detected CPU-only machine"
    fi
}

add_check() {
    local category="$1"
    local name="$2"
    local status="$3"
    local details="$4"

    ((TOTAL_CHECKS++))

    case "$status" in
        "PASS")
            ((PASSED_CHECKS++))
            ;;
        "FAIL")
            ((FAILED_CHECKS++))
            ;;
        "WARN")
            ((WARNING_CHECKS++))
            ;;
    esac

    VALIDATION_RESULTS["${category}:${name}"]="$status"
    VALIDATION_DETAILS["${category}:${name}"]="$details"

    log "DEBUG" "[$status] $category: $name - $details"
}

validate_system_resources() {
    log "STEP" "Validating system resources..."

    # CPU validation
    local cpu_count=$(nproc)
    if [[ $cpu_count -ge $MIN_CPU_CORES ]]; then
        add_check "System Resources" "CPU Cores" "PASS" "$cpu_count cores (minimum: $MIN_CPU_CORES)"
    else
        add_check "System Resources" "CPU Cores" "FAIL" "$cpu_count cores (minimum: $MIN_CPU_CORES required)"
    fi

    # Memory validation
    local total_mem_kb=$(grep MemTotal /proc/meminfo | awk '{print $2}')
    local total_mem_gb=$((total_mem_kb / 1024 / 1024))
    local available_mem_kb=$(grep MemAvailable /proc/meminfo | awk '{print $2}')
    local available_mem_gb=$((available_mem_kb / 1024 / 1024))

    if [[ $total_mem_gb -ge $RECOMMENDED_RAM_GB ]]; then
        add_check "System Resources" "Total Memory" "PASS" "${total_mem_gb}GB (recommended: ${RECOMMENDED_RAM_GB}GB)"
    elif [[ $total_mem_gb -ge $MIN_RAM_GB ]]; then
        add_check "System Resources" "Total Memory" "WARN" "${total_mem_gb}GB (minimum met, recommended: ${RECOMMENDED_RAM_GB}GB)"
    else
        add_check "System Resources" "Total Memory" "FAIL" "${total_mem_gb}GB (minimum: ${MIN_RAM_GB}GB required)"
    fi

    if [[ $available_mem_gb -ge 2 ]]; then
        add_check "System Resources" "Available Memory" "PASS" "${available_mem_gb}GB available"
    elif [[ $available_mem_gb -ge 1 ]]; then
        add_check "System Resources" "Available Memory" "WARN" "${available_mem_gb}GB available (low)"
    else
        add_check "System Resources" "Available Memory" "FAIL" "${available_mem_gb}GB available (insufficient)"
    fi

    # Disk space validation
    local disk_info=$(df / | tail -1)
    local total_disk_kb=$(echo "$disk_info" | awk '{print $2}')
    local available_disk_kb=$(echo "$disk_info" | awk '{print $4}')
    local total_disk_gb=$((total_disk_kb / 1024 / 1024))
    local available_disk_gb=$((available_disk_kb / 1024 / 1024))
    local disk_usage=$(echo "$disk_info" | awk '{print $5}' | sed 's/%//')

    if [[ $available_disk_gb -ge $RECOMMENDED_DISK_GB ]]; then
        add_check "System Resources" "Disk Space" "PASS" "${available_disk_gb}GB available (recommended: ${RECOMMENDED_DISK_GB}GB)"
    elif [[ $available_disk_gb -ge $MIN_DISK_GB ]]; then
        add_check "System Resources" "Disk Space" "WARN" "${available_disk_gb}GB available (minimum met, recommended: ${RECOMMENDED_DISK_GB}GB)"
    else
        add_check "System Resources" "Disk Space" "FAIL" "${available_disk_gb}GB available (minimum: ${MIN_DISK_GB}GB required)"
    fi

    if [[ $disk_usage -le 80 ]]; then
        add_check "System Resources" "Disk Usage" "PASS" "${disk_usage}% used"
    elif [[ $disk_usage -le 90 ]]; then
        add_check "System Resources" "Disk Usage" "WARN" "${disk_usage}% used (high usage)"
    else
        add_check "System Resources" "Disk Usage" "FAIL" "${disk_usage}% used (critical usage)"
    fi

    # Load average validation
    local load_avg=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | sed 's/,//')
    local load_threshold=$(echo "$cpu_count * 0.8" | bc -l 2>/dev/null || echo "$cpu_count")

    if (( $(echo "$load_avg < $load_threshold" | bc -l 2>/dev/null || echo 1) )); then
        add_check "System Resources" "Load Average" "PASS" "${load_avg} (threshold: ${load_threshold})"
    else
        add_check "System Resources" "Load Average" "WARN" "${load_avg} (high load, threshold: ${load_threshold})"
    fi

    log "INFO" "System resources validation completed"
}

validate_network_ports() {
    log "STEP" "Validating network ports..."

    # Check critical AgenticDosNode ports
    for port in "${AGENTICNODE_PORTS[@]}"; do
        if check_port "$port"; then
            local processes=$(find_port_processes "$port")
            local process_info=""
            if [[ -n "$processes" ]]; then
                for pid in $processes; do
                    if kill -0 "$pid" 2>/dev/null; then
                        local cmd=$(ps -p "$pid" -o comm --no-headers 2>/dev/null || echo "unknown")
                        process_info="$process_info $cmd($pid)"
                    fi
                done
            fi
            add_check "Network Ports" "Critical Port $port" "FAIL" "In use by:$process_info"
        else
            add_check "Network Ports" "Critical Port $port" "PASS" "Available"
        fi
    done

    # Check optional ports
    local optional_conflicts=0
    for port in "${OPTIONAL_PORTS[@]}"; do
        if check_port "$port"; then
            ((optional_conflicts++))
            local processes=$(find_port_processes "$port")
            local process_info=""
            if [[ -n "$processes" ]]; then
                for pid in $processes; do
                    if kill -0 "$pid" 2>/dev/null; then
                        local cmd=$(ps -p "$pid" -o comm --no-headers 2>/dev/null || echo "unknown")
                        process_info="$process_info $cmd($pid)"
                    fi
                done
            fi
            log "DEBUG" "Optional port $port in use by:$process_info"
        fi
    done

    if [[ $optional_conflicts -eq 0 ]]; then
        add_check "Network Ports" "Optional Ports" "PASS" "All optional ports available"
    elif [[ $optional_conflicts -le 3 ]]; then
        add_check "Network Ports" "Optional Ports" "WARN" "$optional_conflicts optional ports in use"
    else
        add_check "Network Ports" "Optional Ports" "FAIL" "$optional_conflicts optional ports in use (too many conflicts)"
    fi

    # Network connectivity validation
    if ping -c 1 8.8.8.8 >/dev/null 2>&1; then
        add_check "Network Connectivity" "Internet Access" "PASS" "Can reach 8.8.8.8"
    else
        add_check "Network Connectivity" "Internet Access" "FAIL" "Cannot reach internet"
    fi

    # DNS resolution
    if nslookup google.com >/dev/null 2>&1; then
        add_check "Network Connectivity" "DNS Resolution" "PASS" "DNS working"
    else
        add_check "Network Connectivity" "DNS Resolution" "FAIL" "DNS resolution failed"
    fi

    # Docker Hub connectivity (if Docker is available)
    if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
        if docker pull hello-world >/dev/null 2>&1; then
            add_check "Network Connectivity" "Docker Hub" "PASS" "Can pull from Docker Hub"
            docker rmi hello-world >/dev/null 2>&1 || true
        else
            add_check "Network Connectivity" "Docker Hub" "WARN" "Cannot pull from Docker Hub"
        fi
    fi

    log "INFO" "Network ports validation completed"
}

validate_docker_environment() {
    log "STEP" "Validating Docker environment..."

    # Docker installation
    if command -v docker >/dev/null 2>&1; then
        local docker_version=$(docker --version | awk '{print $3}' | sed 's/,//')
        add_check "Docker" "Installation" "PASS" "Docker $docker_version installed"
    else
        add_check "Docker" "Installation" "FAIL" "Docker not installed"
        return
    fi

    # Docker service status
    if systemctl is-active --quiet docker; then
        add_check "Docker" "Service Status" "PASS" "Docker service running"
    else
        add_check "Docker" "Service Status" "FAIL" "Docker service not running"
    fi

    # Docker daemon accessibility
    if docker info >/dev/null 2>&1; then
        add_check "Docker" "Daemon Access" "PASS" "Docker daemon accessible"
    else
        add_check "Docker" "Daemon Access" "FAIL" "Cannot access Docker daemon"
        return
    fi

    # Docker Compose
    if docker compose version >/dev/null 2>&1; then
        local compose_version=$(docker compose version --short)
        add_check "Docker" "Compose Plugin" "PASS" "Docker Compose $compose_version available"
    else
        add_check "Docker" "Compose Plugin" "FAIL" "Docker Compose not available"
    fi

    # Docker containers check
    local running_containers=$(docker ps -q | wc -l)
    local all_containers=$(docker ps -aq | wc -l)

    if [[ $running_containers -eq 0 ]]; then
        add_check "Docker" "Running Containers" "PASS" "No containers running"
    else
        add_check "Docker" "Running Containers" "WARN" "$running_containers containers still running"
    fi

    if [[ $all_containers -eq 0 ]]; then
        add_check "Docker" "Container Cleanup" "PASS" "No containers present"
    elif [[ $all_containers -le 5 ]]; then
        add_check "Docker" "Container Cleanup" "WARN" "$all_containers containers present"
    else
        add_check "Docker" "Container Cleanup" "FAIL" "$all_containers containers present (cleanup incomplete)"
    fi

    # Docker images check
    local images=$(docker images -q | wc -l)
    if [[ $images -eq 0 ]]; then
        add_check "Docker" "Image Cleanup" "PASS" "No images present"
    elif [[ $images -le 10 ]]; then
        add_check "Docker" "Image Cleanup" "WARN" "$images images present"
    else
        add_check "Docker" "Image Cleanup" "FAIL" "$images images present (cleanup incomplete)"
    fi

    # Docker volumes check
    local volumes=$(docker volume ls -q | wc -l)
    if [[ $volumes -eq 0 ]]; then
        add_check "Docker" "Volume Cleanup" "PASS" "No volumes present"
    elif [[ $volumes -le 3 ]]; then
        add_check "Docker" "Volume Cleanup" "WARN" "$volumes volumes present"
    else
        add_check "Docker" "Volume Cleanup" "FAIL" "$volumes volumes present (cleanup incomplete)"
    fi

    # Docker network check
    local networks=$(docker network ls --format "{{.Name}}" | grep -v -E "^(bridge|host|none)$" | wc -l)
    if [[ $networks -eq 0 ]]; then
        add_check "Docker" "Network Cleanup" "PASS" "Only default networks present"
    else
        add_check "Docker" "Network Cleanup" "WARN" "$networks custom networks present"
    fi

    # Docker storage driver
    local storage_driver=$(docker info --format '{{.Driver}}' 2>/dev/null || echo "unknown")
    if [[ "$storage_driver" == "overlay2" ]]; then
        add_check "Docker" "Storage Driver" "PASS" "Using overlay2 driver"
    else
        add_check "Docker" "Storage Driver" "WARN" "Using $storage_driver driver (overlay2 recommended)"
    fi

    # Docker daemon configuration
    if [[ -f /etc/docker/daemon.json ]]; then
        if jq empty /etc/docker/daemon.json >/dev/null 2>&1; then
            add_check "Docker" "Daemon Config" "PASS" "Valid daemon.json present"
        else
            add_check "Docker" "Daemon Config" "WARN" "Invalid daemon.json format"
        fi
    else
        add_check "Docker" "Daemon Config" "WARN" "No daemon.json configuration"
    fi

    log "INFO" "Docker environment validation completed"
}

validate_process_state() {
    log "STEP" "Validating process state..."

    # Common AI/ML process patterns
    local ai_patterns=(
        "ollama" "localai" "pytorch" "tensorflow" "jupyter"
        "gradio" "streamlit" "redis-server" "postgres" "mongod"
        "qdrant" "weaviate" "elasticsearch"
    )

    local total_ai_processes=0
    for pattern in "${ai_patterns[@]}"; do
        local count=$(pgrep -f "$pattern" 2>/dev/null | wc -l || echo "0")
        total_ai_processes=$((total_ai_processes + count))
        if [[ $count -gt 0 ]]; then
            log "DEBUG" "Found $count processes matching pattern: $pattern"
        fi
    done

    if [[ $total_ai_processes -eq 0 ]]; then
        add_check "Process State" "AI/ML Processes" "PASS" "No AI/ML processes running"
    elif [[ $total_ai_processes -le 3 ]]; then
        add_check "Process State" "AI/ML Processes" "WARN" "$total_ai_processes AI/ML processes still running"
    else
        add_check "Process State" "AI/ML Processes" "FAIL" "$total_ai_processes AI/ML processes still running (cleanup incomplete)"
    fi

    # High memory usage processes
    local high_mem_processes=$(ps aux --no-headers | awk '$4 > 10 {print $11}' | sort | uniq -c | sort -nr | head -5)
    local high_mem_count=$(echo "$high_mem_processes" | wc -l)

    if [[ $high_mem_count -eq 0 ]]; then
        add_check "Process State" "Memory Usage" "PASS" "No high memory processes"
    else
        add_check "Process State" "Memory Usage" "WARN" "$high_mem_count processes using >10% memory"
        log "DEBUG" "High memory processes:"
        echo "$high_mem_processes" | while read line; do
            log "DEBUG" "  $line"
        done
    fi

    # High CPU usage processes
    local high_cpu_processes=$(ps aux --no-headers | awk '$3 > 50 {print $11}' | sort | uniq -c | sort -nr | head -3)
    local high_cpu_count=$(echo "$high_cpu_processes" | wc -l)

    if [[ $high_cpu_count -eq 0 ]]; then
        add_check "Process State" "CPU Usage" "PASS" "No high CPU processes"
    else
        add_check "Process State" "CPU Usage" "WARN" "$high_cpu_count processes using >50% CPU"
    fi

    # Zombie processes
    local zombie_count=$(ps aux | awk '$8 ~ /^Z/ {print $2}' | wc -l)
    if [[ $zombie_count -eq 0 ]]; then
        add_check "Process State" "Zombie Processes" "PASS" "No zombie processes"
    else
        add_check "Process State" "Zombie Processes" "WARN" "$zombie_count zombie processes present"
    fi

    log "INFO" "Process state validation completed"
}

validate_system_configuration() {
    log "STEP" "Validating system configuration..."

    # File descriptor limits
    local nofile_limit=$(ulimit -n)
    if [[ $nofile_limit -ge 65536 ]]; then
        add_check "System Config" "File Descriptors" "PASS" "$nofile_limit limit"
    elif [[ $nofile_limit -ge 32768 ]]; then
        add_check "System Config" "File Descriptors" "WARN" "$nofile_limit limit (recommended: 65536)"
    else
        add_check "System Config" "File Descriptors" "FAIL" "$nofile_limit limit (too low for production)"
    fi

    # Process limits
    local nproc_limit=$(ulimit -u)
    if [[ $nproc_limit -ge 32768 ]]; then
        add_check "System Config" "Process Limit" "PASS" "$nproc_limit limit"
    elif [[ $nproc_limit -ge 16384 ]]; then
        add_check "System Config" "Process Limit" "WARN" "$nproc_limit limit (recommended: 32768)"
    else
        add_check "System Config" "Process Limit" "FAIL" "$nproc_limit limit (too low)"
    fi

    # Swap configuration
    local swap_usage=$(free | grep Swap | awk '{if ($2 > 0) print int($3/$2*100); else print 0}')
    local swappiness=$(cat /proc/sys/vm/swappiness)

    if [[ $swap_usage -le 10 ]]; then
        add_check "System Config" "Swap Usage" "PASS" "${swap_usage}% swap used"
    elif [[ $swap_usage -le 30 ]]; then
        add_check "System Config" "Swap Usage" "WARN" "${swap_usage}% swap used (moderate)"
    else
        add_check "System Config" "Swap Usage" "FAIL" "${swap_usage}% swap used (high)"
    fi

    if [[ $swappiness -le 10 ]]; then
        add_check "System Config" "Swappiness" "PASS" "Swappiness: $swappiness"
    elif [[ $swappiness -le 30 ]]; then
        add_check "System Config" "Swappiness" "WARN" "Swappiness: $swappiness (recommended: ≤10)"
    else
        add_check "System Config" "Swappiness" "FAIL" "Swappiness: $swappiness (too high for AI workloads)"
    fi

    # Network configuration
    local somaxconn=$(cat /proc/sys/net/core/somaxconn 2>/dev/null || echo "128")
    if [[ $somaxconn -ge 32768 ]]; then
        add_check "System Config" "Network Backlog" "PASS" "somaxconn: $somaxconn"
    elif [[ $somaxconn -ge 1024 ]]; then
        add_check "System Config" "Network Backlog" "WARN" "somaxconn: $somaxconn (recommended: ≥32768)"
    else
        add_check "System Config" "Network Backlog" "FAIL" "somaxconn: $somaxconn (too low)"
    fi

    # Transparent Huge Pages
    local thp_enabled="unknown"
    if [[ -f /sys/kernel/mm/transparent_hugepage/enabled ]]; then
        thp_enabled=$(cat /sys/kernel/mm/transparent_hugepage/enabled | grep -o '\[.*\]' | tr -d '[]')
    fi

    case "$thp_enabled" in
        "never")
            add_check "System Config" "Transparent Huge Pages" "PASS" "THP disabled"
            ;;
        "always")
            add_check "System Config" "Transparent Huge Pages" "WARN" "THP always enabled (may affect performance)"
            ;;
        "madvise")
            add_check "System Config" "Transparent Huge Pages" "PASS" "THP on madvise (good)"
            ;;
        *)
            add_check "System Config" "Transparent Huge Pages" "WARN" "THP status unknown: $thp_enabled"
            ;;
    esac

    # Time synchronization
    if systemctl is-active --quiet ntp || systemctl is-active --quiet systemd-timesyncd || systemctl is-active --quiet chrony; then
        add_check "System Config" "Time Sync" "PASS" "Time synchronization active"
    else
        add_check "System Config" "Time Sync" "WARN" "No time synchronization service active"
    fi

    log "INFO" "System configuration validation completed"
}

validate_security_settings() {
    log "STEP" "Validating security settings..."

    # Firewall status
    if command -v ufw >/dev/null 2>&1; then
        local ufw_status=$(ufw status | head -1 | awk '{print $2}')
        case "$ufw_status" in
            "active")
                add_check "Security" "Firewall" "PASS" "UFW is active"
                ;;
            "inactive")
                add_check "Security" "Firewall" "WARN" "UFW is configured but inactive"
                ;;
            *)
                add_check "Security" "Firewall" "WARN" "UFW status unclear: $ufw_status"
                ;;
        esac
    elif command -v iptables >/dev/null 2>&1; then
        local iptables_rules=$(iptables -L | wc -l)
        if [[ $iptables_rules -gt 10 ]]; then
            add_check "Security" "Firewall" "PASS" "iptables rules configured"
        else
            add_check "Security" "Firewall" "WARN" "Minimal iptables configuration"
        fi
    else
        add_check "Security" "Firewall" "FAIL" "No firewall detected"
    fi

    # SSH configuration
    if [[ -f /etc/ssh/sshd_config ]]; then
        local ssh_config_issues=0

        if grep -q "PermitRootLogin yes" /etc/ssh/sshd_config; then
            ((ssh_config_issues++))
        fi

        if grep -q "PasswordAuthentication yes" /etc/ssh/sshd_config; then
            ((ssh_config_issues++))
        fi

        if [[ $ssh_config_issues -eq 0 ]]; then
            add_check "Security" "SSH Config" "PASS" "SSH configuration secure"
        elif [[ $ssh_config_issues -eq 1 ]]; then
            add_check "Security" "SSH Config" "WARN" "Minor SSH security issues"
        else
            add_check "Security" "SSH Config" "FAIL" "Multiple SSH security issues"
        fi
    else
        add_check "Security" "SSH Config" "WARN" "SSH config file not found"
    fi

    # fail2ban status
    if systemctl is-active --quiet fail2ban; then
        add_check "Security" "Intrusion Prevention" "PASS" "fail2ban is active"
    elif command -v fail2ban-client >/dev/null 2>&1; then
        add_check "Security" "Intrusion Prevention" "WARN" "fail2ban installed but not active"
    else
        add_check "Security" "Intrusion Prevention" "WARN" "fail2ban not installed"
    fi

    # Automatic updates
    if [[ -f /etc/apt/apt.conf.d/20auto-upgrades ]]; then
        local auto_updates=$(grep "Unattended-Upgrade" /etc/apt/apt.conf.d/20auto-upgrades | grep -c '"1"' || echo "0")
        if [[ $auto_updates -gt 0 ]]; then
            add_check "Security" "Auto Updates" "PASS" "Automatic security updates enabled"
        else
            add_check "Security" "Auto Updates" "WARN" "Automatic security updates not configured"
        fi
    else
        add_check "Security" "Auto Updates" "WARN" "Unattended-upgrades not configured"
    fi

    # System users check
    local system_users=$(awk -F: '$3 >= 1000 && $3 < 65534 {print $1}' /etc/passwd | wc -l)
    if [[ $system_users -le 3 ]]; then
        add_check "Security" "User Accounts" "PASS" "$system_users regular user accounts"
    elif [[ $system_users -le 10 ]]; then
        add_check "Security" "User Accounts" "WARN" "$system_users regular user accounts (review if necessary)"
    else
        add_check "Security" "User Accounts" "WARN" "$system_users regular user accounts (many accounts present)"
    fi

    # SUDO configuration
    if [[ -f /etc/sudoers ]]; then
        local sudo_nopasswd=$(grep -c "NOPASSWD" /etc/sudoers /etc/sudoers.d/* 2>/dev/null || echo "0")
        if [[ $sudo_nopasswd -eq 0 ]]; then
            add_check "Security" "SUDO Config" "PASS" "No passwordless sudo configured"
        else
            add_check "Security" "SUDO Config" "WARN" "$sudo_nopasswd passwordless sudo entries (review security)"
        fi
    fi

    log "INFO" "Security settings validation completed"
}

validate_performance_optimization() {
    log "STEP" "Validating performance optimization..."

    # I/O scheduler
    local io_scheduler="unknown"
    local primary_disk=$(lsblk -no NAME,TYPE | grep disk | head -1 | awk '{print $1}')
    if [[ -n "$primary_disk" && -f "/sys/block/$primary_disk/queue/scheduler" ]]; then
        io_scheduler=$(cat "/sys/block/$primary_disk/queue/scheduler" | grep -o '\[.*\]' | tr -d '[]')
    fi

    case "$io_scheduler" in
        "mq-deadline"|"deadline")
            add_check "Performance" "I/O Scheduler" "PASS" "Using $io_scheduler scheduler"
            ;;
        "cfq"|"noop"|"none")
            add_check "Performance" "I/O Scheduler" "WARN" "Using $io_scheduler scheduler (mq-deadline recommended)"
            ;;
        *)
            add_check "Performance" "I/O Scheduler" "WARN" "Unknown scheduler: $io_scheduler"
            ;;
    esac

    # CPU governor
    local cpu_governor="unknown"
    if [[ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]]; then
        cpu_governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
    fi

    case "$cpu_governor" in
        "performance")
            add_check "Performance" "CPU Governor" "PASS" "Using performance governor"
            ;;
        "ondemand"|"schedutil")
            add_check "Performance" "CPU Governor" "WARN" "Using $cpu_governor governor (performance recommended for AI workloads)"
            ;;
        "powersave")
            add_check "Performance" "CPU Governor" "FAIL" "Using powersave governor (not optimal for AI workloads)"
            ;;
        *)
            add_check "Performance" "CPU Governor" "WARN" "CPU governor: $cpu_governor"
            ;;
    esac

    # TCP congestion control
    local tcp_congestion=$(cat /proc/sys/net/ipv4/tcp_congestion_control 2>/dev/null || echo "unknown")
    case "$tcp_congestion" in
        "bbr")
            add_check "Performance" "TCP Congestion" "PASS" "Using BBR congestion control"
            ;;
        "cubic")
            add_check "Performance" "TCP Congestion" "WARN" "Using CUBIC (BBR recommended)"
            ;;
        *)
            add_check "Performance" "TCP Congestion" "WARN" "TCP congestion control: $tcp_congestion"
            ;;
    esac

    # Huge pages configuration (for memory-intensive AI workloads)
    local hugepages_total=$(cat /proc/sys/vm/nr_hugepages 2>/dev/null || echo "0")
    local hugepages_free=$(cat /proc/meminfo | grep HugePages_Free | awk '{print $2}' || echo "0")

    if [[ $hugepages_total -gt 0 ]]; then
        add_check "Performance" "Huge Pages" "PASS" "$hugepages_total huge pages configured ($hugepages_free free)"
    else
        add_check "Performance" "Huge Pages" "WARN" "No huge pages configured (may benefit AI workloads)"
    fi

    # NUMA configuration
    if command -v numactl >/dev/null 2>&1; then
        local numa_nodes=$(numactl --hardware | grep "available:" | awk '{print $2}')
        if [[ $numa_nodes -gt 1 ]]; then
            add_check "Performance" "NUMA" "WARN" "Multi-NUMA system ($numa_nodes nodes) - consider NUMA-aware deployment"
        else
            add_check "Performance" "NUMA" "PASS" "Single NUMA node system"
        fi
    else
        add_check "Performance" "NUMA" "WARN" "numactl not available (install for NUMA systems)"
    fi

    log "INFO" "Performance optimization validation completed"
}

validate_directory_structure() {
    log "STEP" "Validating directory structure..."

    # Common AgenticDosNode directories
    local expected_dirs=(
        "/opt/agenticnode"
        "/var/lib/agenticnode"
        "/var/log/agenticnode"
        "/etc/agenticnode"
    )

    local missing_dirs=0
    for dir in "${expected_dirs[@]}"; do
        if [[ -d "$dir" ]]; then
            local permissions=$(stat -c "%a" "$dir")
            local owner=$(stat -c "%U:%G" "$dir")
            add_check "Directory Structure" "$(basename "$dir")" "PASS" "$dir exists ($permissions, $owner)"
        else
            ((missing_dirs++))
            log "DEBUG" "Directory does not exist: $dir"
        fi
    done

    if [[ $missing_dirs -eq 0 ]]; then
        add_check "Directory Structure" "Preparation Status" "PASS" "All directories exist (environment prepared)"
    elif [[ $missing_dirs -le 2 ]]; then
        add_check "Directory Structure" "Preparation Status" "WARN" "$missing_dirs directories missing (partial preparation)"
    else
        add_check "Directory Structure" "Preparation Status" "FAIL" "$missing_dirs directories missing (not prepared)"
    fi

    # Check for remnant AI directories that should be cleaned
    local ai_dirs_to_check=(
        "$HOME/.ollama"
        "$HOME/.cache/huggingface"
        "$HOME/.local/share/ollama"
        "/var/lib/ollama"
        "/opt/ollama"
    )

    local remaining_ai_dirs=0
    for dir in "${ai_dirs_to_check[@]}"; do
        if [[ -d "$dir" ]]; then
            ((remaining_ai_dirs++))
            local size=$(du -sh "$dir" 2>/dev/null | cut -f1 || echo "unknown")
            log "DEBUG" "Remaining AI directory: $dir [$size]"
        fi
    done

    if [[ $remaining_ai_dirs -eq 0 ]]; then
        add_check "Directory Structure" "AI Cleanup" "PASS" "No AI directories remaining"
    elif [[ $remaining_ai_dirs -le 2 ]]; then
        add_check "Directory Structure" "AI Cleanup" "WARN" "$remaining_ai_dirs AI directories remain"
    else
        add_check "Directory Structure" "AI Cleanup" "FAIL" "$remaining_ai_dirs AI directories remain (cleanup incomplete)"
    fi

    # Check temp directories
    local temp_ai_files=$(find /tmp -name "*ollama*" -o -name "*ai*" -o -name "*ml*" 2>/dev/null | wc -l || echo "0")
    if [[ $temp_ai_files -eq 0 ]]; then
        add_check "Directory Structure" "Temp Cleanup" "PASS" "No AI temp files"
    elif [[ $temp_ai_files -le 10 ]]; then
        add_check "Directory Structure" "Temp Cleanup" "WARN" "$temp_ai_files AI temp files remain"
    else
        add_check "Directory Structure" "Temp Cleanup" "FAIL" "$temp_ai_files AI temp files remain"
    fi

    log "INFO" "Directory structure validation completed"
}

validate_machine_specific() {
    log "STEP" "Validating $MACHINE_TYPE-specific requirements..."

    if [[ "$MACHINE_TYPE" == "GPU" ]]; then
        validate_gpu_specific
    else
        validate_cpu_specific
    fi
}

validate_gpu_specific() {
    log "INFO" "Validating GPU-specific requirements..."

    # NVIDIA driver
    if command -v nvidia-smi >/dev/null 2>&1; then
        local driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1)
        add_check "GPU" "NVIDIA Driver" "PASS" "Driver version $driver_version"
    else
        add_check "GPU" "NVIDIA Driver" "FAIL" "NVIDIA driver not installed or not working"
        return
    fi

    # GPU status and availability
    local gpu_count=$(nvidia-smi --list-gpus | wc -l)
    if [[ $gpu_count -gt 0 ]]; then
        add_check "GPU" "GPU Count" "PASS" "$gpu_count GPU(s) detected"
    else
        add_check "GPU" "GPU Count" "FAIL" "No GPUs detected"
        return
    fi

    # GPU memory and utilization
    local gpu_processes=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | wc -l)
    if [[ $gpu_processes -eq 0 ]]; then
        add_check "GPU" "GPU Processes" "PASS" "No processes using GPU"
    else
        add_check "GPU" "GPU Processes" "WARN" "$gpu_processes processes using GPU"
    fi

    # GPU memory usage
    local gpu_memory_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    local gpu_memory_total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    local gpu_memory_percent=$((gpu_memory_used * 100 / gpu_memory_total))

    if [[ $gpu_memory_percent -le 10 ]]; then
        add_check "GPU" "GPU Memory" "PASS" "${gpu_memory_percent}% GPU memory used"
    elif [[ $gpu_memory_percent -le 30 ]]; then
        add_check "GPU" "GPU Memory" "WARN" "${gpu_memory_percent}% GPU memory used"
    else
        add_check "GPU" "GPU Memory" "FAIL" "${gpu_memory_percent}% GPU memory used (high usage)"
    fi

    # CUDA installation
    if command -v nvcc >/dev/null 2>&1; then
        local cuda_version=$(nvcc --version | grep "release" | awk '{print $6}' | sed 's/V//' | sed 's/,//')
        add_check "GPU" "CUDA Toolkit" "PASS" "CUDA $cuda_version installed"
    else
        add_check "GPU" "CUDA Toolkit" "WARN" "CUDA toolkit not installed (may be needed for some AI frameworks)"
    fi

    # NVIDIA Container Runtime
    if docker info 2>/dev/null | grep -q nvidia; then
        add_check "GPU" "Docker GPU Support" "PASS" "NVIDIA container runtime configured"
    else
        add_check "GPU" "Docker GPU Support" "WARN" "NVIDIA container runtime not configured in Docker"
    fi

    # GPU temperature and power
    local gpu_temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1)
    if [[ $gpu_temp -le 70 ]]; then
        add_check "GPU" "GPU Temperature" "PASS" "${gpu_temp}°C"
    elif [[ $gpu_temp -le 85 ]]; then
        add_check "GPU" "GPU Temperature" "WARN" "${gpu_temp}°C (warm)"
    else
        add_check "GPU" "GPU Temperature" "FAIL" "${gpu_temp}°C (overheating)"
    fi
}

validate_cpu_specific() {
    log "INFO" "Validating CPU-specific requirements..."

    # CPU features for AI workloads
    local cpu_features=$(cat /proc/cpuinfo | grep flags | head -1 | awk -F: '{print $2}')

    local important_features=("avx" "avx2" "sse4_1" "sse4_2")
    local supported_features=0

    for feature in "${important_features[@]}"; do
        if echo "$cpu_features" | grep -q "$feature"; then
            ((supported_features++))
        fi
    done

    if [[ $supported_features -eq ${#important_features[@]} ]]; then
        add_check "CPU" "CPU Features" "PASS" "All important CPU features supported"
    elif [[ $supported_features -ge 2 ]]; then
        add_check "CPU" "CPU Features" "WARN" "$supported_features/${#important_features[@]} important CPU features supported"
    else
        add_check "CPU" "CPU Features" "FAIL" "Missing important CPU features for AI workloads"
    fi

    # CPU cache size
    local l3_cache=$(cat /proc/cpuinfo | grep "cache size" | head -1 | awk '{print $4}')
    if [[ -n "$l3_cache" ]]; then
        local cache_mb=${l3_cache%?}  # Remove last character (usually 'K')
        cache_mb=$((cache_mb / 1024)) # Convert to MB if it was KB

        if [[ $cache_mb -ge 16 ]]; then
            add_check "CPU" "L3 Cache" "PASS" "${l3_cache} L3 cache"
        elif [[ $cache_mb -ge 8 ]]; then
            add_check "CPU" "L3 Cache" "WARN" "${l3_cache} L3 cache (larger cache benefits AI workloads)"
        else
            add_check "CPU" "L3 Cache" "WARN" "${l3_cache} L3 cache (small for AI workloads)"
        fi
    else
        add_check "CPU" "L3 Cache" "WARN" "Cannot determine L3 cache size"
    fi

    # Memory bandwidth (approximated by memory speed)
    local memory_speed=$(dmidecode -t memory 2>/dev/null | grep "Speed:" | grep -v "Unknown" | head -1 | awk '{print $2}' || echo "unknown")
    if [[ "$memory_speed" != "unknown" ]]; then
        local speed_mhz=${memory_speed%?*}
        if [[ $speed_mhz -ge 3200 ]]; then
            add_check "CPU" "Memory Speed" "PASS" "${memory_speed} RAM"
        elif [[ $speed_mhz -ge 2400 ]]; then
            add_check "CPU" "Memory Speed" "WARN" "${memory_speed} RAM (faster RAM benefits AI workloads)"
        else
            add_check "CPU" "Memory Speed" "WARN" "${memory_speed} RAM (slow for AI workloads)"
        fi
    else
        add_check "CPU" "Memory Speed" "WARN" "Cannot determine RAM speed"
    fi

    # Check for AI-optimized libraries
    local intel_mkl_found=false
    local openblas_found=false

    if ldconfig -p | grep -q libmkl; then
        intel_mkl_found=true
    fi

    if ldconfig -p | grep -q openblas; then
        openblas_found=true
    fi

    if [[ "$intel_mkl_found" == true ]]; then
        add_check "CPU" "Optimized Libraries" "PASS" "Intel MKL detected"
    elif [[ "$openblas_found" == true ]]; then
        add_check "CPU" "Optimized Libraries" "PASS" "OpenBLAS detected"
    else
        add_check "CPU" "Optimized Libraries" "WARN" "No optimized math libraries detected (install MKL or OpenBLAS)"
    fi
}

generate_validation_report() {
    log "STEP" "Generating validation report..."

    local report_file="${LOG_DIR}/validation_report_${TIMESTAMP}.txt"
    local json_report="${LOG_DIR}/validation_report_${TIMESTAMP}.json"

    # Text report
    cat > "$report_file" << EOF
AgenticDosNode Clean State Validation Report
Generated: $(date)
Machine Type: $MACHINE_TYPE
Total Checks: $TOTAL_CHECKS
Passed: $PASSED_CHECKS
Failed: $FAILED_CHECKS
Warnings: $WARNING_CHECKS

========================================
DETAILED RESULTS
========================================

EOF

    # Group results by category
    local categories=(
        "System Resources"
        "Network Ports"
        "Network Connectivity"
        "Docker"
        "Process State"
        "System Config"
        "Security"
        "Performance"
        "Directory Structure"
        "GPU"
        "CPU"
    )

    for category in "${categories[@]}"; do
        local category_found=false

        for key in "${!VALIDATION_RESULTS[@]}"; do
            if [[ "$key" == "$category:"* ]]; then
                if [[ "$category_found" == false ]]; then
                    echo "[$category]" >> "$report_file"
                    category_found=true
                fi

                local test_name="${key#*:}"
                local status="${VALIDATION_RESULTS[$key]}"
                local details="${VALIDATION_DETAILS[$key]}"

                printf "  %-20s [%s] %s\n" "$test_name" "$status" "$details" >> "$report_file"
            fi
        done

        if [[ "$category_found" == true ]]; then
            echo "" >> "$report_file"
        fi
    done

    # JSON report for automation
    cat > "$json_report" << EOF
{
  "timestamp": "$(date -Iseconds)",
  "machine_type": "$MACHINE_TYPE",
  "summary": {
    "total_checks": $TOTAL_CHECKS,
    "passed": $PASSED_CHECKS,
    "failed": $FAILED_CHECKS,
    "warnings": $WARNING_CHECKS
  },
  "results": {
EOF

    local first_entry=true
    for key in "${!VALIDATION_RESULTS[@]}"; do
        local category="${key%:*}"
        local test_name="${key#*:}"
        local status="${VALIDATION_RESULTS[$key]}"
        local details="${VALIDATION_DETAILS[$key]}"

        if [[ "$first_entry" == false ]]; then
            echo "," >> "$json_report"
        fi
        first_entry=false

        cat >> "$json_report" << EOF
    "$key": {
      "category": "$category",
      "test": "$test_name",
      "status": "$status",
      "details": "$details"
    }EOF
    done

    cat >> "$json_report" << EOF

  }
}
EOF

    log "INFO" "Validation reports generated:"
    log "INFO" "  Text report: $report_file"
    log "INFO" "  JSON report: $json_report"
}

show_validation_summary() {
    log "STEP" "Validation Summary"

    local status_color="$GREEN"
    local status_text="READY"

    if [[ $FAILED_CHECKS -gt 0 ]]; then
        status_color="$RED"
        status_text="NOT READY"
    elif [[ $WARNING_CHECKS -gt 0 ]]; then
        status_color="$YELLOW"
        status_text="READY WITH WARNINGS"
    fi

    cat << EOF

${status_color}========================================
VALIDATION SUMMARY
========================================${NC}

Machine Type: $MACHINE_TYPE
Total Checks: $TOTAL_CHECKS
${GREEN}Passed: $PASSED_CHECKS${NC}
${RED}Failed: $FAILED_CHECKS${NC}
${YELLOW}Warnings: $WARNING_CHECKS${NC}

Overall Status: ${status_color}$status_text${NC}

EOF

    if [[ $FAILED_CHECKS -gt 0 ]]; then
        echo -e "${RED}CRITICAL ISSUES:${NC}"
        for key in "${!VALIDATION_RESULTS[@]}"; do
            if [[ "${VALIDATION_RESULTS[$key]}" == "FAIL" ]]; then
                local category="${key%:*}"
                local test_name="${key#*:}"
                local details="${VALIDATION_DETAILS[$key]}"
                echo -e "${RED}  ✗${NC} $category: $test_name - $details"
            fi
        done
        echo ""
    fi

    if [[ $WARNING_CHECKS -gt 0 ]]; then
        echo -e "${YELLOW}WARNINGS:${NC}"
        for key in "${!VALIDATION_RESULTS[@]}"; do
            if [[ "${VALIDATION_RESULTS[$key]}" == "WARN" ]]; then
                local category="${key%:*}"
                local test_name="${key#*:}"
                local details="${VALIDATION_DETAILS[$key]}"
                echo -e "${YELLOW}  ⚠${NC} $category: $test_name - $details"
            fi
        done
        echo ""
    fi

    # Recommendations based on results
    cat << EOF
${CYAN}RECOMMENDATIONS:${NC}

EOF

    if [[ $FAILED_CHECKS -gt 0 ]]; then
        echo "• Address all critical issues before deploying AgenticDosNode"
        echo "• Run the appropriate cleanup script if processes/services are still running"
        echo "• Ensure all required ports are free"
        echo "• Verify system resources meet minimum requirements"
    fi

    if [[ $WARNING_CHECKS -gt 0 ]]; then
        echo "• Review warnings and address if possible for optimal performance"
        echo "• Consider system optimizations for better AI workload performance"
        echo "• Monitor resource usage during initial deployment"
    fi

    if [[ $FAILED_CHECKS -eq 0 && $WARNING_CHECKS -eq 0 ]]; then
        echo "• System is ready for AgenticDosNode deployment"
        echo "• Proceed with docker-compose deployment"
        echo "• Monitor system resources during initial startup"
        echo "• Enable firewall after confirming connectivity"
    fi

    echo ""
    echo "Log files available at: $LOG_DIR"
    echo "For detailed results, see: ${LOG_DIR}/validation_report_${TIMESTAMP}.txt"
}

# Usage information
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

AgenticDosNode Clean State Validation Script

OPTIONS:
    -h, --help              Show this help message
    -t, --type TYPE         Machine type (GPU|CPU|auto) [default: auto]
    -v, --verbose           Verbose output
    --json-output          Output results in JSON format
    --report-only          Generate report without console output

EXAMPLES:
    $0                      Auto-detect machine type and validate
    $0 -t GPU              Validate as GPU node
    $0 -t CPU              Validate as CPU node
    $0 --json-output       Output results in JSON format

EXIT CODES:
    0                       All checks passed
    1                       Critical issues found (not ready)
    2                       Warnings found (ready with cautions)

EOF
}

# Command line argument parsing
JSON_OUTPUT=false
REPORT_ONLY=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            exit 0
            ;;
        -t|--type)
            MACHINE_TYPE="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        --json-output)
            JSON_OUTPUT=true
            shift
            ;;
        --report-only)
            REPORT_ONLY=true
            shift
            ;;
        *)
            log "ERROR" "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate machine type
case "$MACHINE_TYPE" in
    "GPU"|"CPU"|"auto")
        ;;
    *)
        log "ERROR" "Invalid machine type: $MACHINE_TYPE (must be GPU, CPU, or auto)"
        exit 1
        ;;
esac

# Export environment variables
export MACHINE_TYPE VERBOSE

# Redirect output for report-only mode
if [[ "$REPORT_ONLY" == "true" ]]; then
    exec 1>/dev/null
fi

# Run main function
main "$@"