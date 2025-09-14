#!/bin/bash

# Hardware Optimization Script for AgenticDosNode
# Optimizes CPU, GPU, Memory, and I/O for AI workloads

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BACKUP_DIR="/etc/agentic-backup"
LOG_FILE="/var/log/agentic-hardware-optimization.log"
NODE_TYPE="${1:-auto}" # thanos, oracle1, or auto

# Logging function
log() {
    echo -e "${2:-INFO}: $1" | tee -a "$LOG_FILE"
    logger -t "agentic-hw-opt" "$1"
}

# Detect node type
detect_node() {
    if nvidia-smi &>/dev/null; then
        echo "thanos"
    else
        echo "oracle1"
    fi
}

# Backup current configuration
backup_config() {
    log "Creating configuration backup" "${BLUE}"
    mkdir -p "$BACKUP_DIR"

    # Backup CPU governor settings
    if [ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]; then
        cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > "$BACKUP_DIR/cpu_governors.bak" 2>/dev/null || true
    fi

    # Backup current sysctl settings
    sysctl -a > "$BACKUP_DIR/sysctl.bak" 2>/dev/null || true

    # Backup scheduler settings
    for disk in /sys/block/*/queue/scheduler; do
        if [ -f "$disk" ]; then
            echo "$(dirname $disk | xargs basename): $(cat $disk)" >> "$BACKUP_DIR/io_schedulers.bak"
        fi
    done

    log "Backup completed at $BACKUP_DIR" "${GREEN}"
}

# CPU Optimization
optimize_cpu() {
    log "Optimizing CPU for AI workloads" "${BLUE}"

    # Set CPU governor to performance
    if [ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]; then
        for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
            echo "performance" > "$cpu" 2>/dev/null || true
        done
        log "CPU governor set to performance mode" "${GREEN}"
    else
        log "CPU frequency scaling not available" "${YELLOW}"
    fi

    # Disable CPU throttling
    if [ -f /sys/devices/system/cpu/intel_pstate/no_turbo ]; then
        echo "0" > /sys/devices/system/cpu/intel_pstate/no_turbo
        log "Intel Turbo Boost enabled" "${GREEN}"
    fi

    # Set CPU affinity for critical processes
    if [ "$NODE_TYPE" == "thanos" ]; then
        # Reserve CPUs 0-3 for system, 4+ for AI workloads
        echo "0-3" > /sys/fs/cgroup/cpuset/system.slice/cpuset.cpus 2>/dev/null || true
        log "CPU affinity configured for GPU workloads" "${GREEN}"
    else
        # Oracle1: Balance across all cores
        log "CPU affinity: using all cores for balanced workload" "${GREEN}"
    fi

    # Optimize interrupt handling
    if [ -f /proc/irq/default_smp_affinity ]; then
        # Spread interrupts across all CPUs
        echo "ff" > /proc/irq/default_smp_affinity
        log "IRQ affinity optimized" "${GREEN}"
    fi
}

# Memory Optimization
optimize_memory() {
    log "Optimizing memory for large AI models" "${BLUE}"

    # Configure huge pages for better TLB performance
    if [ "$NODE_TYPE" == "thanos" ]; then
        # Allocate 8GB of huge pages for AI models
        echo 4096 > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages
        log "Allocated 8GB huge pages for AI models" "${GREEN}"
    else
        # Oracle1: 2GB for databases
        echo 1024 > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages
        log "Allocated 2GB huge pages for databases" "${GREEN}"
    fi

    # Memory management settings
    cat >> /etc/sysctl.d/99-agentic-memory.conf << EOF
# Memory optimization for AI workloads
vm.swappiness = 10
vm.dirty_ratio = 15
vm.dirty_background_ratio = 5
vm.vfs_cache_pressure = 50
vm.overcommit_memory = 1
vm.overcommit_ratio = 90
vm.max_map_count = 262144
vm.min_free_kbytes = 524288

# Shared memory for containers
kernel.shmmax = 68719476736
kernel.shmall = 16777216
kernel.shmmni = 4096

# NUMA optimization
vm.zone_reclaim_mode = 0
EOF

    sysctl -p /etc/sysctl.d/99-agentic-memory.conf
    log "Memory parameters optimized" "${GREEN}"

    # Disable transparent huge pages (can cause latency spikes)
    echo "never" > /sys/kernel/mm/transparent_hugepage/enabled
    echo "never" > /sys/kernel/mm/transparent_hugepage/defrag
    log "Transparent huge pages disabled" "${GREEN}"
}

# I/O Scheduler Optimization
optimize_io() {
    log "Optimizing I/O scheduler for database and model loading" "${BLUE}"

    for disk in /sys/block/*/queue; do
        DISK_NAME=$(basename $(dirname $disk))

        # Skip loop devices and virtual disks
        if [[ "$DISK_NAME" == loop* ]] || [[ "$DISK_NAME" == ram* ]]; then
            continue
        fi

        # Detect if SSD or HDD
        ROTATIONAL=$(cat /sys/block/$DISK_NAME/queue/rotational)

        if [ "$ROTATIONAL" == "0" ]; then
            # SSD: Use deadline or none scheduler
            if grep -q none "$disk/scheduler" 2>/dev/null; then
                echo "none" > "$disk/scheduler" 2>/dev/null || true
                log "I/O scheduler for $DISK_NAME set to none (NVMe)" "${GREEN}"
            elif grep -q mq-deadline "$disk/scheduler" 2>/dev/null; then
                echo "mq-deadline" > "$disk/scheduler" 2>/dev/null || true
                log "I/O scheduler for $DISK_NAME set to mq-deadline (SSD)" "${GREEN}"
            fi

            # Optimize queue parameters for SSD
            echo 512 > "$disk/nr_requests" 2>/dev/null || true
            echo 0 > "$disk/add_random" 2>/dev/null || true
            echo 2 > "$disk/rq_affinity" 2>/dev/null || true
        else
            # HDD: Use BFQ for better fairness
            if grep -q bfq "$disk/scheduler" 2>/dev/null; then
                echo "bfq" > "$disk/scheduler" 2>/dev/null || true
                log "I/O scheduler for $DISK_NAME set to bfq (HDD)" "${GREEN}"
            fi

            # Optimize queue parameters for HDD
            echo 256 > "$disk/nr_requests" 2>/dev/null || true
            echo 16384 > "$disk/read_ahead_kb" 2>/dev/null || true
        fi
    done
}

# GPU Optimization (Thanos only)
optimize_gpu() {
    if [ "$NODE_TYPE" != "thanos" ]; then
        log "Skipping GPU optimization (not a GPU node)" "${YELLOW}"
        return
    fi

    log "Optimizing GPU for AI inference" "${BLUE}"

    # Check for NVIDIA GPU
    if ! nvidia-smi &>/dev/null; then
        log "NVIDIA GPU not detected" "${RED}"
        return
    fi

    # Set GPU to persistence mode
    nvidia-smi -pm 1
    log "GPU persistence mode enabled" "${GREEN}"

    # Set power limit for optimal performance/watt
    # RTX 4090 default is 450W, we set to 400W for efficiency
    nvidia-smi -pl 400 2>/dev/null || true
    log "GPU power limit set to 400W" "${GREEN}"

    # Configure GPU memory overcommit
    nvidia-smi -c EXCLUSIVE_PROCESS 2>/dev/null || true
    log "GPU compute mode set to EXCLUSIVE_PROCESS" "${GREEN}"

    # Set maximum graphics and memory clocks
    nvidia-smi -ac 10501,2520 2>/dev/null || {
        log "Could not set GPU clocks (may require sudo or different values)" "${YELLOW}"
    }

    # Enable GPU boost
    nvidia-settings -a "[gpu:0]/GPUPowerMizerMode=1" 2>/dev/null || true

    # Configure CUDA settings
    cat >> /etc/environment << EOF
CUDA_CACHE_MAXSIZE=4294967296
CUDA_CACHE_PATH=/var/cache/cuda
CUDA_FORCE_PTX_JIT=0
CUDA_DEVICE_ORDER=PCI_BUS_ID
EOF

    # Create CUDA cache directory
    mkdir -p /var/cache/cuda
    chmod 777 /var/cache/cuda

    log "GPU optimization completed" "${GREEN}"
}

# Thermal Management
optimize_thermal() {
    log "Configuring thermal management" "${BLUE}"

    # Install and configure thermald if available
    if command -v thermald &>/dev/null; then
        systemctl enable thermald
        systemctl start thermald
        log "Thermald enabled for thermal management" "${GREEN}"
    fi

    # Set fan curves for GPU (Thanos)
    if [ "$NODE_TYPE" == "thanos" ] && nvidia-smi &>/dev/null; then
        # Set aggressive fan curve for sustained workloads
        nvidia-settings -a "[gpu:0]/GPUFanControlState=1" 2>/dev/null || true
        nvidia-settings -a "[fan:0]/GPUTargetFanSpeed=70" 2>/dev/null || true
        log "GPU fan curve configured for sustained workloads" "${GREEN}"
    fi

    # CPU thermal configuration
    if [ -f /sys/devices/system/cpu/intel_pstate/max_perf_pct ]; then
        # Allow max performance but with thermal throttling
        echo "100" > /sys/devices/system/cpu/intel_pstate/max_perf_pct
        log "CPU thermal limits configured" "${GREEN}"
    fi
}

# Network Optimization
optimize_network() {
    log "Optimizing network stack for AI APIs" "${BLUE}"

    cat >> /etc/sysctl.d/99-agentic-network.conf << EOF
# Network optimization for high-throughput AI APIs
net.core.rmem_max = 134217728
net.core.wmem_max = 134217728
net.ipv4.tcp_rmem = 4096 87380 134217728
net.ipv4.tcp_wmem = 4096 65536 134217728
net.core.netdev_max_backlog = 5000
net.ipv4.tcp_congestion_control = bbr
net.ipv4.tcp_notsent_lowat = 16384
net.ipv4.tcp_fastopen = 3
net.ipv4.tcp_mtu_probing = 1
net.core.default_qdisc = fq

# Connection handling
net.ipv4.ip_local_port_range = 10000 65000
net.ipv4.tcp_fin_timeout = 30
net.ipv4.tcp_keepalive_time = 300
net.ipv4.tcp_keepalive_probes = 5
net.ipv4.tcp_keepalive_intvl = 30
net.core.somaxconn = 4096
net.ipv4.tcp_max_syn_backlog = 8192
net.ipv4.tcp_syncookies = 1
net.ipv4.tcp_tw_reuse = 1
net.ipv4.tcp_timestamps = 1

# Docker/container optimization
net.bridge.bridge-nf-call-iptables = 1
net.bridge.bridge-nf-call-ip6tables = 1
net.ipv4.conf.all.forwarding = 1
net.ipv6.conf.all.forwarding = 1
EOF

    sysctl -p /etc/sysctl.d/99-agentic-network.conf
    log "Network stack optimized" "${GREEN}"

    # Enable BBR congestion control if available
    if modprobe tcp_bbr 2>/dev/null; then
        log "BBR congestion control enabled" "${GREEN}"
    fi

    # Optimize network interrupts
    if [ -f /proc/irq/default_smp_affinity ]; then
        # Spread network interrupts across CPUs
        for irq in $(grep -E 'eth|eno|ens|enp' /proc/interrupts | cut -d: -f1); do
            echo "ff" > "/proc/irq/$irq/smp_affinity" 2>/dev/null || true
        done
        log "Network interrupt affinity optimized" "${GREEN}"
    fi
}

# File System Optimization
optimize_filesystem() {
    log "Optimizing file system for AI workloads" "${BLUE}"

    # Update mount options for better performance
    # Add noatime,nodiratime to reduce unnecessary writes
    if grep -q "ext4" /proc/mounts; then
        log "Optimizing ext4 filesystems" "${GREEN}"
        # This would need to be done in /etc/fstab and requires reboot
        cat << EOF

To optimize filesystem mounts, add these options to /etc/fstab:
- noatime,nodiratime for all data partitions
- commit=60 for ext4 partitions (longer commit interval)
- nobarrier for SSDs with battery backup

Example:
UUID=xxx / ext4 defaults,noatime,nodiratime,commit=60 0 1
EOF
    fi

    # Increase inotify limits for containers
    cat >> /etc/sysctl.d/99-agentic-fs.conf << EOF
# File system optimization
fs.inotify.max_user_watches = 524288
fs.inotify.max_user_instances = 8192
fs.file-max = 2097152
fs.nr_open = 1048576
EOF

    sysctl -p /etc/sysctl.d/99-agentic-fs.conf
    log "File system parameters optimized" "${GREEN}"
}

# Apply all optimizations
apply_all() {
    log "Starting hardware optimization for $NODE_TYPE node" "${BLUE}"

    backup_config
    optimize_cpu
    optimize_memory
    optimize_io
    optimize_gpu
    optimize_thermal
    optimize_network
    optimize_filesystem

    log "Hardware optimization completed successfully!" "${GREEN}"

    cat << EOF

${GREEN}Optimization Summary:${NC}
- CPU: Performance mode, optimized affinity
- Memory: Huge pages enabled, swappiness reduced
- I/O: Schedulers optimized for SSD/HDD
- GPU: Power/clock optimized (if applicable)
- Network: BBR enabled, buffers increased
- Thermal: Active management configured

${YELLOW}Recommended actions:${NC}
1. Reboot to apply all changes
2. Run benchmarks to verify improvements
3. Monitor system metrics for 24 hours

${BLUE}To revert changes:${NC}
cp -r $BACKUP_DIR/* /etc/
reboot
EOF
}

# Rollback function
rollback() {
    log "Rolling back optimization changes" "${YELLOW}"

    if [ ! -d "$BACKUP_DIR" ]; then
        log "No backup found at $BACKUP_DIR" "${RED}"
        exit 1
    fi

    # Restore CPU governors
    if [ -f "$BACKUP_DIR/cpu_governors.bak" ]; then
        i=0
        while read governor; do
            echo "$governor" > "/sys/devices/system/cpu/cpu$i/cpufreq/scaling_governor" 2>/dev/null || true
            ((i++))
        done < "$BACKUP_DIR/cpu_governors.bak"
    fi

    # Remove custom sysctl configs
    rm -f /etc/sysctl.d/99-agentic-*.conf
    sysctl --system

    log "Rollback completed. Please reboot to apply all changes." "${GREEN}"
}

# Main execution
main() {
    # Check if running as root
    if [ "$EUID" -ne 0 ]; then
        echo "This script must be run as root (use sudo)"
        exit 1
    fi

    # Detect or use specified node type
    if [ "$NODE_TYPE" == "auto" ]; then
        NODE_TYPE=$(detect_node)
        log "Detected node type: $NODE_TYPE" "${BLUE}"
    fi

    # Parse command
    case "${2:-apply}" in
        apply)
            apply_all
            ;;
        rollback)
            rollback
            ;;
        *)
            echo "Usage: $0 [thanos|oracle1|auto] [apply|rollback]"
            exit 1
            ;;
    esac
}

main "$@"