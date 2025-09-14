#!/bin/bash

# Kernel Parameter Tuning Script for AgenticDosNode
# Fine-tunes kernel parameters for AI and database workloads

set -euo pipefail

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
BACKUP_DIR="/etc/agentic-backup/kernel"
LOG_FILE="/var/log/agentic-kernel-tuning.log"
NODE_TYPE="${1:-auto}"

# Logging
log() {
    echo -e "${2:-INFO}: $1" | tee -a "$LOG_FILE"
    logger -t "agentic-kernel" "$1"
}

# Detect node type
detect_node() {
    if nvidia-smi &>/dev/null; then
        echo "thanos"
    else
        echo "oracle1"
    fi
}

# Backup current kernel parameters
backup_kernel_params() {
    log "Backing up current kernel parameters" "${BLUE}"
    mkdir -p "$BACKUP_DIR"

    # Backup sysctl settings
    sysctl -a > "$BACKUP_DIR/sysctl-all.bak" 2>/dev/null

    # Backup limits
    cp /etc/security/limits.conf "$BACKUP_DIR/limits.conf.bak" 2>/dev/null || true
    cp -r /etc/security/limits.d/ "$BACKUP_DIR/limits.d.bak/" 2>/dev/null || true

    # Backup kernel modules config
    cp /etc/modules-load.d/* "$BACKUP_DIR/" 2>/dev/null || true

    log "Kernel parameters backed up to $BACKUP_DIR" "${GREEN}"
}

# Core kernel tuning
tune_kernel_core() {
    log "Applying core kernel optimizations" "${BLUE}"

    cat > /etc/sysctl.d/10-agentic-core.conf << 'EOF'
# Core Kernel Parameters for AgenticDosNode

# Process and Thread Management
kernel.pid_max = 4194304
kernel.threads-max = 600000
kernel.sched_migration_cost_ns = 5000000
kernel.sched_autogroup_enabled = 0

# CPU Scheduling Optimization
kernel.sched_min_granularity_ns = 10000000
kernel.sched_wakeup_granularity_ns = 15000000
kernel.sched_latency_ns = 20000000
kernel.sched_nr_migrate = 128
kernel.sched_rr_timeslice_ms = 100
kernel.sched_rt_runtime_us = 950000

# Memory Management
vm.max_map_count = 2621440
vm.swappiness = 1
vm.dirty_ratio = 40
vm.dirty_background_ratio = 10
vm.dirty_expire_centisecs = 3000
vm.dirty_writeback_centisecs = 500
vm.vfs_cache_pressure = 50
vm.overcommit_memory = 1
vm.overcommit_ratio = 95
vm.min_free_kbytes = 262144
vm.admin_reserve_kbytes = 131072

# Huge Pages Configuration
vm.nr_hugepages = 0
vm.hugetlb_shm_group = 0

# NUMA Optimization
vm.zone_reclaim_mode = 0
vm.numa_stat = 1

# Security and Limits
kernel.sysrq = 1
kernel.core_uses_pid = 1
kernel.kptr_restrict = 1
kernel.yama.ptrace_scope = 1
kernel.panic = 60
kernel.panic_on_oops = 0

# Shared Memory
kernel.shmmax = 68719476736
kernel.shmall = 16777216
kernel.shmmni = 32768
kernel.msgmax = 65536
kernel.msgmnb = 65536
kernel.sem = 250 32000 100 128

# IPC
kernel.msgmni = 32768
fs.mqueue.msg_max = 8192
fs.mqueue.msgsize_max = 8192
EOF

    sysctl -p /etc/sysctl.d/10-agentic-core.conf
    log "Core kernel parameters applied" "${GREEN}"
}

# Network stack tuning
tune_network_stack() {
    log "Optimizing network stack for high-throughput APIs" "${BLUE}"

    cat > /etc/sysctl.d/20-agentic-network.conf << 'EOF'
# Network Stack Optimization for AI APIs

# Core Network Settings
net.core.somaxconn = 65535
net.core.netdev_max_backlog = 65536
net.core.netdev_budget = 600
net.core.netdev_budget_usecs = 8000
net.core.dev_weight = 64

# Socket Buffer Sizes
net.core.rmem_default = 31457280
net.core.rmem_max = 134217728
net.core.wmem_default = 31457280
net.core.wmem_max = 134217728
net.core.optmem_max = 65536

# TCP Settings
net.ipv4.tcp_rmem = 4096 87380 134217728
net.ipv4.tcp_wmem = 4096 65536 134217728
net.ipv4.tcp_mem = 786432 1048576 26777216
net.ipv4.udp_mem = 65536 131072 262144

# TCP Performance Tuning
net.ipv4.tcp_timestamps = 1
net.ipv4.tcp_window_scaling = 1
net.ipv4.tcp_sack = 1
net.ipv4.tcp_no_metrics_save = 1
net.ipv4.tcp_moderate_rcvbuf = 1
net.ipv4.tcp_congestion_control = bbr
net.ipv4.tcp_notsent_lowat = 16384
net.ipv4.tcp_mtu_probing = 1
net.ipv4.tcp_base_mss = 1024

# TCP Fast Open
net.ipv4.tcp_fastopen = 3
net.ipv4.tcp_fastopen_blackhole_timeout_sec = 0

# Connection Management
net.ipv4.tcp_fin_timeout = 15
net.ipv4.tcp_keepalive_time = 300
net.ipv4.tcp_keepalive_probes = 5
net.ipv4.tcp_keepalive_intvl = 30
net.ipv4.tcp_tw_reuse = 1
net.ipv4.tcp_max_syn_backlog = 65536
net.ipv4.tcp_max_tw_buckets = 2000000
net.ipv4.tcp_syncookies = 1
net.ipv4.tcp_synack_retries = 2
net.ipv4.tcp_syn_retries = 3

# Port Range
net.ipv4.ip_local_port_range = 1024 65535

# IP Settings
net.ipv4.ip_forward = 1
net.ipv4.conf.all.forwarding = 1
net.ipv4.conf.default.forwarding = 1
net.ipv4.conf.all.route_localnet = 1

# ARP Settings
net.ipv4.conf.all.arp_announce = 2
net.ipv4.conf.all.arp_ignore = 1
net.ipv4.conf.default.arp_announce = 2
net.ipv4.conf.default.arp_ignore = 1

# Disable IPv6 if not needed
net.ipv6.conf.all.disable_ipv6 = 0
net.ipv6.conf.default.disable_ipv6 = 0
net.ipv6.conf.lo.disable_ipv6 = 0

# Netfilter Connection Tracking
net.netfilter.nf_conntrack_max = 2000000
net.netfilter.nf_conntrack_buckets = 262144
net.netfilter.nf_conntrack_tcp_timeout_established = 86400
net.netfilter.nf_conntrack_tcp_timeout_close_wait = 60
net.netfilter.nf_conntrack_tcp_timeout_fin_wait = 60
net.netfilter.nf_conntrack_tcp_timeout_time_wait = 60

# Bridge Settings for Docker
net.bridge.bridge-nf-call-iptables = 1
net.bridge.bridge-nf-call-ip6tables = 1
net.bridge.bridge-nf-call-arptables = 1

# Queue Discipline
net.core.default_qdisc = fq
EOF

    sysctl -p /etc/sysctl.d/20-agentic-network.conf
    log "Network stack optimized" "${GREEN}"

    # Load required modules
    cat > /etc/modules-load.d/agentic-network.conf << EOF
tcp_bbr
br_netfilter
overlay
EOF

    modprobe tcp_bbr 2>/dev/null || true
    modprobe br_netfilter 2>/dev/null || true
    modprobe overlay 2>/dev/null || true

    log "Network modules loaded" "${GREEN}"
}

# File descriptor and process limits
tune_limits() {
    log "Configuring file descriptor and process limits" "${BLUE}"

    # System-wide limits
    cat > /etc/security/limits.d/99-agentic.conf << 'EOF'
# Limits for AgenticDosNode

# File Descriptors
* soft nofile 1048576
* hard nofile 1048576
root soft nofile 1048576
root hard nofile 1048576

# Processes/Threads
* soft nproc 524288
* hard nproc 524288
root soft nproc 524288
root hard nproc 524288

# Memory Locks
* soft memlock unlimited
* hard memlock unlimited
root soft memlock unlimited
root hard memlock unlimited

# Core Dumps
* soft core unlimited
* hard core unlimited

# Stack Size
* soft stack 65536
* hard stack 65536

# Real-time Priority
* soft rtprio 99
* hard rtprio 99

# Nice Priority
* soft nice -20
* hard nice -20

# Message Queues
* soft msgqueue unlimited
* hard msgqueue unlimited

# Pending Signals
* soft sigpending 1048576
* hard sigpending 1048576
EOF

    # Docker daemon limits
    mkdir -p /etc/systemd/system/docker.service.d
    cat > /etc/systemd/system/docker.service.d/limits.conf << 'EOF'
[Service]
LimitNOFILE=1048576
LimitNPROC=524288
LimitCORE=infinity
LimitMEMLOCK=infinity
TasksMax=infinity
EOF

    # Containerd limits
    mkdir -p /etc/systemd/system/containerd.service.d
    cat > /etc/systemd/system/containerd.service.d/limits.conf << 'EOF'
[Service]
LimitNOFILE=1048576
LimitNPROC=524288
LimitCORE=infinity
LimitMEMLOCK=infinity
TasksMax=infinity
EOF

    systemctl daemon-reload
    log "System limits configured" "${GREEN}"
}

# GPU-specific kernel tuning (Thanos)
tune_gpu_kernel() {
    if [ "$NODE_TYPE" != "thanos" ]; then
        log "Skipping GPU kernel tuning (not a GPU node)" "${YELLOW}"
        return
    fi

    log "Applying GPU-specific kernel optimizations" "${BLUE}"

    # NVIDIA GPU settings
    cat > /etc/modprobe.d/nvidia.conf << 'EOF'
# NVIDIA GPU Kernel Parameters
options nvidia NVreg_EnableGpuPowerManagement=1
options nvidia NVreg_RegistryDwords="PerfLevelSrc=0x3333"
options nvidia NVreg_PreserveVideoMemoryAllocations=1
options nvidia NVreg_TemporaryFilePath=/var/cache/nvidia
options nvidia-drm modeset=1
EOF

    # Create NVIDIA cache directory
    mkdir -p /var/cache/nvidia
    chmod 1777 /var/cache/nvidia

    # PCIe settings for GPU
    cat > /etc/sysctl.d/30-agentic-gpu.conf << 'EOF'
# GPU-specific optimizations
vm.nr_hugepages = 4096
vm.hugetlb_shm_group = 0
kernel.sched_rt_runtime_us = -1
EOF

    sysctl -p /etc/sysctl.d/30-agentic-gpu.conf
    log "GPU kernel parameters applied" "${GREEN}"
}

# Database-specific kernel tuning (Oracle1)
tune_database_kernel() {
    if [ "$NODE_TYPE" != "oracle1" ]; then
        log "Skipping database kernel tuning (not a database node)" "${YELLOW}"
        return
    fi

    log "Applying database-specific kernel optimizations" "${BLUE}"

    cat > /etc/sysctl.d/30-agentic-database.conf << 'EOF'
# Database-specific optimizations
# PostgreSQL optimizations
vm.nr_hugepages = 1024
vm.hugetlb_shm_group = 0
vm.dirty_background_bytes = 67108864
vm.dirty_bytes = 536870912

# I/O optimizations for databases
vm.dirty_expire_centisecs = 1500
vm.dirty_writeback_centisecs = 100
kernel.sched_min_granularity_ns = 10000000
kernel.sched_wakeup_granularity_ns = 15000000

# Checkpoint optimization
vm.laptop_mode = 0
vm.oom_kill_allocating_task = 0
EOF

    sysctl -p /etc/sysctl.d/30-agentic-database.conf
    log "Database kernel parameters applied" "${GREEN}"
}

# Container runtime optimization
tune_container_runtime() {
    log "Optimizing container runtime parameters" "${BLUE}"

    cat > /etc/sysctl.d/40-agentic-containers.conf << 'EOF'
# Container Runtime Optimization
# Namespace limits
user.max_user_namespaces = 32768
user.max_pid_namespaces = 32768
user.max_net_namespaces = 32768
user.max_ipc_namespaces = 32768
user.max_uts_namespaces = 32768
user.max_mnt_namespaces = 32768
user.max_cgroup_namespaces = 32768

# IPC for containers
kernel.msgmnb = 65536
kernel.msgmax = 65536
kernel.shmmax = 68719476736
kernel.shmall = 16777216

# File watching for containers
fs.inotify.max_user_watches = 1048576
fs.inotify.max_user_instances = 8192
fs.inotify.max_queued_events = 16384

# File handle limits
fs.file-max = 2097152
fs.nr_open = 1048576
fs.pipe-max-size = 1048576
fs.pipe-user-pages-hard = 0
fs.pipe-user-pages-soft = 16384

# EPoll limits
fs.epoll.max_user_watches = 6553600

# AIO limits
fs.aio-max-nr = 1048576
EOF

    sysctl -p /etc/sysctl.d/40-agentic-containers.conf
    log "Container runtime parameters optimized" "${GREEN}"
}

# Performance monitoring setup
setup_monitoring() {
    log "Setting up kernel performance monitoring" "${BLUE}"

    # Enable kernel performance events
    cat > /etc/sysctl.d/50-agentic-monitoring.conf << 'EOF'
# Performance Monitoring
kernel.perf_event_paranoid = -1
kernel.perf_event_max_sample_rate = 100000
kernel.perf_cpu_time_max_percent = 25
kernel.perf_event_mlock_kb = 516
EOF

    sysctl -p /etc/sysctl.d/50-agentic-monitoring.conf

    # Enable kernel tracing
    if [ -d /sys/kernel/debug/tracing ]; then
        echo 32768 > /sys/kernel/debug/tracing/buffer_size_kb 2>/dev/null || true
        log "Kernel tracing buffer configured" "${GREEN}"
    fi

    log "Performance monitoring enabled" "${GREEN}"
}

# Apply all tunings
apply_all() {
    log "Starting kernel parameter tuning for $NODE_TYPE node" "${BLUE}"

    backup_kernel_params
    tune_kernel_core
    tune_network_stack
    tune_limits
    tune_gpu_kernel
    tune_database_kernel
    tune_container_runtime
    setup_monitoring

    # Reload all sysctl settings
    sysctl --system

    log "Kernel parameter tuning completed!" "${GREEN}"

    cat << EOF

${GREEN}Kernel Tuning Summary:${NC}
- Core parameters optimized for high concurrency
- Network stack tuned for high-throughput APIs
- File descriptor limits increased to 1M
- Container runtime optimized
- Performance monitoring enabled

${YELLOW}Important Notes:${NC}
1. Some changes require a reboot to take full effect
2. Monitor system behavior for 24 hours
3. Check dmesg for any kernel warnings

${BLUE}To verify changes:${NC}
sysctl -a | grep -E "net.core|vm.|kernel."
ulimit -n
cat /proc/sys/fs/file-max

${RED}To rollback:${NC}
rm /etc/sysctl.d/*-agentic-*.conf
cp -r $BACKUP_DIR/* /etc/
sysctl --system
reboot
EOF
}

# Rollback function
rollback() {
    log "Rolling back kernel parameter changes" "${YELLOW}"

    if [ ! -d "$BACKUP_DIR" ]; then
        log "No backup found at $BACKUP_DIR" "${RED}"
        exit 1
    fi

    # Remove custom configs
    rm -f /etc/sysctl.d/*-agentic-*.conf
    rm -f /etc/security/limits.d/99-agentic.conf
    rm -f /etc/modules-load.d/agentic-*.conf
    rm -f /etc/modprobe.d/nvidia.conf
    rm -rf /etc/systemd/system/docker.service.d/limits.conf
    rm -rf /etc/systemd/system/containerd.service.d/limits.conf

    # Restore original configs if they exist
    if [ -f "$BACKUP_DIR/limits.conf.bak" ]; then
        cp "$BACKUP_DIR/limits.conf.bak" /etc/security/limits.conf
    fi

    # Reload sysctl
    sysctl --system
    systemctl daemon-reload

    log "Rollback completed. Please reboot to fully restore original settings." "${GREEN}"
}

# Main execution
main() {
    if [ "$EUID" -ne 0 ]; then
        echo "This script must be run as root (use sudo)"
        exit 1
    fi

    if [ "$NODE_TYPE" == "auto" ]; then
        NODE_TYPE=$(detect_node)
        log "Detected node type: $NODE_TYPE" "${BLUE}"
    fi

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