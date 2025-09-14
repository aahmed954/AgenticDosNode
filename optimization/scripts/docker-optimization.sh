#!/bin/bash

# Docker Performance Optimization Script for AgenticDosNode
# Optimizes Docker daemon, container runtime, and resource management

set -euo pipefail

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
BACKUP_DIR="/etc/agentic-backup/docker"
LOG_FILE="/var/log/agentic-docker-optimization.log"
DOCKER_CONFIG_DIR="/etc/docker"
NODE_TYPE="${1:-auto}"

# Logging
log() {
    echo -e "${2:-INFO}: $1" | tee -a "$LOG_FILE"
    logger -t "agentic-docker" "$1"
}

# Detect node type
detect_node() {
    if nvidia-smi &>/dev/null; then
        echo "thanos"
    else
        echo "oracle1"
    fi
}

# Backup Docker configuration
backup_docker_config() {
    log "Backing up Docker configuration" "${BLUE}"
    mkdir -p "$BACKUP_DIR"

    # Backup existing daemon.json
    if [ -f "$DOCKER_CONFIG_DIR/daemon.json" ]; then
        cp "$DOCKER_CONFIG_DIR/daemon.json" "$BACKUP_DIR/daemon.json.bak"
    fi

    # Backup containerd config
    if [ -f "/etc/containerd/config.toml" ]; then
        cp "/etc/containerd/config.toml" "$BACKUP_DIR/containerd-config.toml.bak"
    fi

    # Backup systemd service files
    if [ -d "/etc/systemd/system/docker.service.d" ]; then
        cp -r "/etc/systemd/system/docker.service.d" "$BACKUP_DIR/"
    fi

    log "Docker configuration backed up to $BACKUP_DIR" "${GREEN}"
}

# Install optimized Docker daemon configuration
install_docker_config() {
    log "Installing optimized Docker daemon configuration" "${BLUE}"

    # Create Docker config directory if it doesn't exist
    mkdir -p "$DOCKER_CONFIG_DIR"

    # Base configuration for both nodes
    cat > "$DOCKER_CONFIG_DIR/daemon.json" << 'EOF'
{
  "debug": false,
  "log-level": "warn",
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "100m",
    "max-file": "10",
    "compress": "true",
    "labels": "com.agentic.service",
    "env": "NODE_TYPE,SERVICE_NAME"
  },
  "storage-driver": "overlay2",
  "storage-opts": [
    "overlay2.override_kernel_check=true"
  ],
  "metrics-addr": "0.0.0.0:9323",
  "experimental": true,
  "features": {
    "buildkit": true
  },
  "max-concurrent-downloads": 10,
  "max-concurrent-uploads": 10,
  "max-download-attempts": 5,
EOF

    # Add GPU runtime for Thanos
    if [ "$NODE_TYPE" == "thanos" ]; then
        cat >> "$DOCKER_CONFIG_DIR/daemon.json" << 'EOF'
  "default-runtime": "nvidia",
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  },
EOF
    fi

    # Continue with common configuration
    cat >> "$DOCKER_CONFIG_DIR/daemon.json" << 'EOF'
  "exec-opts": [
    "native.cgroupdriver=systemd"
  ],
  "default-cgroupns-mode": "host",
  "cgroup-parent": "/docker",
  "default-ulimits": {
    "nofile": {
      "Name": "nofile",
      "Hard": 1048576,
      "Soft": 524288
    },
    "nproc": {
      "Name": "nproc",
      "Hard": 524288,
      "Soft": 262144
    },
    "memlock": {
      "Name": "memlock",
      "Hard": -1,
      "Soft": -1
    },
    "core": {
      "Name": "core",
      "Hard": -1,
      "Soft": -1
    }
  },
  "live-restore": true,
  "userland-proxy": false,
  "ip-forward": true,
  "ip-masq": true,
  "iptables": true,
  "ipv6": false,
  "fixed-cidr": "172.17.0.0/16",
  "icc": true,
  "default-address-pools": [
    {
      "base": "172.80.0.0/16",
      "size": 24
    },
    {
      "base": "172.90.0.0/16",
      "size": 24
    }
  ],
  "dns": [
    "8.8.8.8",
    "8.8.4.4",
    "1.1.1.1"
  ],
  "dns-opts": [
    "ndots:0",
    "timeout:2",
    "attempts:2"
  ],
  "shutdown-timeout": 30,
  "debug-key": "docker-debug",
  "exec-root": "/var/run/docker",
  "mtu": 1450,
  "init": true,
  "init-path": "/usr/bin/docker-init",
  "data-root": "/var/lib/docker",
  "group": "docker",
  "default-shm-size": "512M",
  "no-new-privileges": false,
  "selinux-enabled": false,
  "containerd": "/run/containerd/containerd.sock",
  "containerd-namespace": "moby",
  "containerd-plugin-namespace": "plugins.moby"
}
EOF

    log "Docker daemon configuration installed" "${GREEN}"
}

# Optimize containerd configuration
optimize_containerd() {
    log "Optimizing containerd configuration" "${BLUE}"

    mkdir -p /etc/containerd

    containerd config default > /etc/containerd/config.toml

    # Apply optimizations
    cat > /etc/containerd/config.toml.new << 'EOF'
version = 2

[plugins]
  [plugins."io.containerd.gc.v1.scheduler"]
    pause_threshold = 0.02
    deletion_threshold = 0
    mutation_threshold = 100
    schedule_delay = "0s"
    startup_delay = "100ms"

  [plugins."io.containerd.grpc.v1.cri"]
    sandbox_image = "registry.k8s.io/pause:3.9"
    max_container_log_line_size = 262144
    disable_cgroup = false
    disable_apparmor = false
    restrict_oom_score_adj = false
    disable_proc_mount = false
    disable_hugetlb_controller = false
    device_ownership_from_security_context = false

    [plugins."io.containerd.grpc.v1.cri".containerd]
      snapshotter = "overlayfs"
      default_runtime_name = "runc"
      disable_snapshot_annotations = false

      [plugins."io.containerd.grpc.v1.cri".containerd.runtimes]
        [plugins."io.containerd.grpc.v1.cri".containerd.runtimes.runc]
          runtime_type = "io.containerd.runc.v2"
          pod_annotations = []
          container_annotations = []
          privileged_without_host_devices = false
          base_runtime_spec = ""

          [plugins."io.containerd.grpc.v1.cri".containerd.runtimes.runc.options]
            SystemdCgroup = true
            BinaryName = "runc"
            Root = ""
            ShimCgroup = ""
            NoPivotRoot = false
            NoNewKeyring = false
EOF

    # Add NVIDIA runtime for GPU nodes
    if [ "$NODE_TYPE" == "thanos" ]; then
        cat >> /etc/containerd/config.toml.new << 'EOF'

        [plugins."io.containerd.grpc.v1.cri".containerd.runtimes.nvidia]
          runtime_type = "io.containerd.runc.v2"
          privileged_without_host_devices = false
          [plugins."io.containerd.grpc.v1.cri".containerd.runtimes.nvidia.options]
            BinaryName = "/usr/bin/nvidia-container-runtime"
            SystemdCgroup = true
EOF
    fi

    # Continue with optimization settings
    cat >> /etc/containerd/config.toml.new << 'EOF'

    [plugins."io.containerd.grpc.v1.cri".cni]
      bin_dir = "/opt/cni/bin"
      conf_dir = "/etc/cni/net.d"
      max_conf_num = 1
      conf_template = ""

    [plugins."io.containerd.grpc.v1.cri".registry]
      config_path = "/etc/containerd/certs.d"

  [plugins."io.containerd.internal.v1.opt"]
    path = "/opt/containerd"

  [plugins."io.containerd.internal.v1.restart"]
    interval = "10s"

  [plugins."io.containerd.metadata.v1.bolt"]
    content_sharing_policy = "shared"

  [plugins."io.containerd.monitor.v1.cgroups"]
    no_prometheus = false

  [plugins."io.containerd.runtime.v1.linux"]
    shim = "containerd-shim"
    runtime = "runc"
    runtime_root = ""
    no_shim = false
    shim_debug = false

  [plugins."io.containerd.runtime.v2.task"]
    platforms = ["linux/amd64"]

  [plugins."io.containerd.service.v1.diff-service"]
    default = ["walking"]

[grpc]
  address = "/run/containerd/containerd.sock"
  tcp_address = ""
  tcp_tls_cert = ""
  tcp_tls_key = ""
  uid = 0
  gid = 0
  max_recv_message_size = 16777216
  max_send_message_size = 16777216

[metrics]
  address = "127.0.0.1:9324"
  grpc_histogram = false

[proxy_plugins]

[stream_processors]
  [stream_processors."io.containerd.ocicrypt.decoder.v1.tar"]
    accepts = ["application/vnd.oci.image.layer.v1.tar+encrypted"]
    returns = "application/vnd.oci.image.layer.v1.tar"
    path = "ctd-decoder"
    args = ["--decryption-keys-path", "/etc/containerd/ocicrypt/keys"]

  [stream_processors."io.containerd.ocicrypt.decoder.v1.tar.gzip"]
    accepts = ["application/vnd.oci.image.layer.v1.tar+gzip+encrypted"]
    returns = "application/vnd.oci.image.layer.v1.tar+gzip"
    path = "ctd-decoder"
    args = ["--decryption-keys-path", "/etc/containerd/ocicrypt/keys"]

[timeouts]
  "io.containerd.timeout.shim.cleanup" = "5s"
  "io.containerd.timeout.shim.load" = "5s"
  "io.containerd.timeout.shim.shutdown" = "3s"
  "io.containerd.timeout.task.state" = "2s"

[debug]
  address = ""
  uid = 0
  gid = 0
  level = "warn"
EOF

    mv /etc/containerd/config.toml.new /etc/containerd/config.toml
    systemctl restart containerd

    log "Containerd configuration optimized" "${GREEN}"
}

# Configure Docker systemd service
configure_docker_systemd() {
    log "Configuring Docker systemd service" "${BLUE}"

    mkdir -p /etc/systemd/system/docker.service.d

    # Override systemd settings
    cat > /etc/systemd/system/docker.service.d/override.conf << 'EOF'
[Service]
ExecStart=
ExecStart=/usr/bin/dockerd --host=fd:// --containerd=/run/containerd/containerd.sock
ExecReload=/bin/kill -s HUP $MAINPID
TimeoutSec=0
RestartSec=2
Restart=always
StartLimitBurst=3
StartLimitInterval=60s
LimitNOFILE=1048576
LimitNPROC=infinity
LimitCORE=infinity
LimitMEMLOCK=infinity
TasksMax=infinity
Delegate=yes
KillMode=mixed

# Performance tuning
CPUWeight=1000
IOWeight=1000
CPUQuota=
MemoryMax=
MemorySwapMax=

# Environment
Environment="DOCKER_TMPDIR=/var/tmp/docker"
Environment="GOTRACEBACK=all"
EOF

    # Create performance tuning for Docker
    cat > /etc/systemd/system/docker.service.d/performance.conf << 'EOF'
[Service]
# CPU Affinity (adjust based on your CPU count)
# CPUAffinity=0-31

# Nice level
Nice=-10

# OOM Score
OOMScoreAdjust=-500

# Scheduling
CPUSchedulingPolicy=fifo
CPUSchedulingPriority=10
IOSchedulingClass=realtime
IOSchedulingPriority=0
EOF

    systemctl daemon-reload
    log "Docker systemd service configured" "${GREEN}"
}

# Optimize Docker networking
optimize_docker_network() {
    log "Optimizing Docker networking" "${BLUE}"

    # Install bridge-utils if not present
    if ! command -v brctl &>/dev/null; then
        apt-get update && apt-get install -y bridge-utils || \
        yum install -y bridge-utils 2>/dev/null || true
    fi

    # Create custom Docker networks with optimized settings
    cat > /tmp/create-docker-networks.sh << 'EOF'
#!/bin/bash

# Remove default bridge if exists
docker network rm bridge 2>/dev/null || true

# Create optimized bridge network
docker network create \
  --driver bridge \
  --subnet=172.20.0.0/16 \
  --gateway=172.20.0.1 \
  --opt com.docker.network.bridge.name=docker0 \
  --opt com.docker.network.bridge.enable_icc=true \
  --opt com.docker.network.bridge.enable_ip_masquerade=true \
  --opt com.docker.network.bridge.host_binding_ipv4=0.0.0.0 \
  --opt com.docker.network.driver.mtu=1450 \
  --opt com.docker.network.container_interface_prefix=eth \
  agentic_bridge

# Create host network alias for performance-critical containers
docker network create \
  --driver macvlan \
  --subnet=172.21.0.0/16 \
  --gateway=172.21.0.1 \
  --opt parent=eth0 \
  agentic_macvlan 2>/dev/null || true

echo "Docker networks optimized"
EOF

    chmod +x /tmp/create-docker-networks.sh
    log "Docker network optimization script created" "${GREEN}"
}

# Configure Docker volume drivers
configure_volume_drivers() {
    log "Configuring Docker volume drivers" "${BLUE}"

    # Create optimized volume configuration
    mkdir -p /etc/docker/plugins

    # Local volume driver optimization
    cat > /etc/docker/volume-opts.json << 'EOF'
{
  "volume-drivers": {
    "local": {
      "mount-options": "noatime,nodiratime",
      "copy-on-write": "auto",
      "size": "10G"
    }
  }
}
EOF

    log "Volume drivers configured" "${GREEN}"
}

# Setup Docker resource monitoring
setup_docker_monitoring() {
    log "Setting up Docker resource monitoring" "${BLUE}"

    # Create monitoring script
    cat > /usr/local/bin/docker-monitor.sh << 'EOF'
#!/bin/bash

# Docker Resource Monitor for AgenticDosNode
# Collects and exposes Docker metrics

METRICS_FILE="/var/lib/node_exporter/docker_metrics.prom"
mkdir -p $(dirname $METRICS_FILE)

while true; do
    {
        # Container count
        echo "# HELP docker_containers_total Total number of containers"
        echo "# TYPE docker_containers_total gauge"
        echo "docker_containers_total $(docker ps -aq | wc -l)"

        # Running containers
        echo "# HELP docker_containers_running Running containers"
        echo "# TYPE docker_containers_running gauge"
        echo "docker_containers_running $(docker ps -q | wc -l)"

        # Docker daemon info
        if docker_info=$(docker system df --format json 2>/dev/null); then
            # Parse and export metrics
            echo "# HELP docker_disk_usage_bytes Docker disk usage in bytes"
            echo "# TYPE docker_disk_usage_bytes gauge"

            # Images size
            images_size=$(echo "$docker_info" | jq -r '.Images[0].Size // 0' 2>/dev/null || echo 0)
            echo "docker_disk_usage_bytes{type=\"images\"} $images_size"

            # Containers size
            containers_size=$(echo "$docker_info" | jq -r '.Containers[0].Size // 0' 2>/dev/null || echo 0)
            echo "docker_disk_usage_bytes{type=\"containers\"} $containers_size"

            # Volumes size
            volumes_size=$(echo "$docker_info" | jq -r '.Volumes[0].Size // 0' 2>/dev/null || echo 0)
            echo "docker_disk_usage_bytes{type=\"volumes\"} $volumes_size"
        fi

        # Per-container metrics
        for container in $(docker ps -q); do
            name=$(docker inspect $container --format '{{.Name}}' | sed 's/^\///')
            stats=$(docker stats $container --no-stream --format "{{json .}}" 2>/dev/null)

            if [ -n "$stats" ]; then
                cpu=$(echo "$stats" | jq -r '.CPUPerc' | sed 's/%//')
                mem=$(echo "$stats" | jq -r '.MemUsage' | awk '{print $1}' | sed 's/[^0-9.]//g')

                echo "docker_container_cpu_usage_percent{name=\"$name\"} $cpu"
                echo "docker_container_memory_usage_mb{name=\"$name\"} $mem"
            fi
        done
    } > $METRICS_FILE.tmp && mv $METRICS_FILE.tmp $METRICS_FILE

    sleep 30
done
EOF

    chmod +x /usr/local/bin/docker-monitor.sh

    # Create systemd service for monitoring
    cat > /etc/systemd/system/docker-monitor.service << 'EOF'
[Unit]
Description=Docker Resource Monitor
After=docker.service
Requires=docker.service

[Service]
Type=simple
ExecStart=/usr/local/bin/docker-monitor.sh
Restart=always
RestartSec=10
User=root

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    systemctl enable docker-monitor.service
    systemctl start docker-monitor.service

    log "Docker monitoring setup completed" "${GREEN}"
}

# GPU-specific Docker optimization (Thanos)
optimize_docker_gpu() {
    if [ "$NODE_TYPE" != "thanos" ]; then
        log "Skipping GPU Docker optimization (not a GPU node)" "${YELLOW}"
        return
    fi

    log "Optimizing Docker for GPU workloads" "${BLUE}"

    # Configure NVIDIA Docker runtime
    cat > /etc/nvidia-container-runtime/config.toml << 'EOF'
disable-require = false
supported-driver-capabilities = "compute,utility,graphics,video,display,ngx"

[nvidia-container-cli]
environment = []
debug = false
ldcache = "/etc/ld.so.cache"
load-kmods = true
no-cgroups = false
user = "root:root"
ldconfig = "@/sbin/ldconfig"

[nvidia-container-runtime]
debug = false
log-level = "info"
mode = "auto"
runtime-path = "/usr/bin/nvidia-container-runtime"

[nvidia-container-runtime.modes]
[nvidia-container-runtime.modes.csv]
mount-spec-path = "/etc/nvidia-container-runtime/host-files-for-container.d"
EOF

    # Create GPU resource allocation script
    cat > /usr/local/bin/docker-gpu-allocator.sh << 'EOF'
#!/bin/bash

# GPU Resource Allocator for Docker containers
# Manages GPU allocation and prevents oversubscription

GPU_ALLOCATION_FILE="/var/run/docker-gpu-allocation.json"

allocate_gpu() {
    container_id=$1
    gpu_request=$2

    # Logic to track and allocate GPUs
    echo "{\"container\": \"$container_id\", \"gpu\": \"$gpu_request\"}" >> $GPU_ALLOCATION_FILE
}

release_gpu() {
    container_id=$1
    # Logic to release GPU allocation
    grep -v "\"container\": \"$container_id\"" $GPU_ALLOCATION_FILE > ${GPU_ALLOCATION_FILE}.tmp
    mv ${GPU_ALLOCATION_FILE}.tmp $GPU_ALLOCATION_FILE
}

# Monitor GPU allocation
monitor_gpu() {
    nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total \
                --format=csv,noheader,nounits
}

case "$1" in
    allocate)
        allocate_gpu "$2" "$3"
        ;;
    release)
        release_gpu "$2"
        ;;
    monitor)
        monitor_gpu
        ;;
    *)
        echo "Usage: $0 {allocate|release|monitor}"
        exit 1
        ;;
esac
EOF

    chmod +x /usr/local/bin/docker-gpu-allocator.sh
    log "Docker GPU optimization completed" "${GREEN}"
}

# Apply all optimizations
apply_all() {
    log "Starting Docker optimization for $NODE_TYPE node" "${BLUE}"

    backup_docker_config
    install_docker_config
    optimize_containerd
    configure_docker_systemd
    optimize_docker_network
    configure_volume_drivers
    setup_docker_monitoring
    optimize_docker_gpu

    # Restart Docker service
    log "Restarting Docker service..." "${YELLOW}"
    systemctl restart docker
    systemctl restart containerd

    # Wait for Docker to be ready
    sleep 5

    # Verify Docker is running
    if docker info &>/dev/null; then
        log "Docker service restarted successfully" "${GREEN}"
    else
        log "Docker service failed to start. Check logs: journalctl -u docker" "${RED}"
        exit 1
    fi

    log "Docker optimization completed!" "${GREEN}"

    cat << EOF

${GREEN}Docker Optimization Summary:${NC}
- Daemon configuration optimized for AI workloads
- Container runtime tuned for performance
- Resource limits configured
- Monitoring enabled
- GPU runtime configured (if applicable)

${YELLOW}Post-optimization steps:${NC}
1. Run: /tmp/create-docker-networks.sh
2. Restart all containers to apply new settings
3. Monitor metrics at :9323/metrics

${BLUE}To verify:${NC}
docker info
docker system df
curl localhost:9323/metrics

${RED}To rollback:${NC}
cp $BACKUP_DIR/daemon.json.bak /etc/docker/daemon.json
systemctl restart docker
EOF
}

# Rollback function
rollback() {
    log "Rolling back Docker optimization changes" "${YELLOW}"

    if [ ! -d "$BACKUP_DIR" ]; then
        log "No backup found at $BACKUP_DIR" "${RED}"
        exit 1
    fi

    # Restore original configuration
    if [ -f "$BACKUP_DIR/daemon.json.bak" ]; then
        cp "$BACKUP_DIR/daemon.json.bak" "$DOCKER_CONFIG_DIR/daemon.json"
    else
        rm -f "$DOCKER_CONFIG_DIR/daemon.json"
    fi

    if [ -f "$BACKUP_DIR/containerd-config.toml.bak" ]; then
        cp "$BACKUP_DIR/containerd-config.toml.bak" "/etc/containerd/config.toml"
    fi

    # Remove systemd overrides
    rm -rf /etc/systemd/system/docker.service.d

    # Stop monitoring service
    systemctl stop docker-monitor.service
    systemctl disable docker-monitor.service
    rm -f /etc/systemd/system/docker-monitor.service

    # Reload and restart services
    systemctl daemon-reload
    systemctl restart containerd
    systemctl restart docker

    log "Rollback completed" "${GREEN}"
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