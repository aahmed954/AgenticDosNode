#!/bin/bash

# Monitoring and Profiling Setup for AgenticDosNode
# Comprehensive performance monitoring with Prometheus, Grafana, and custom exporters

set -euo pipefail

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
LOG_FILE="/var/log/agentic-monitoring-setup.log"
MONITORING_DIR="/opt/agentic-monitoring"
GRAFANA_PORT=3000
PROMETHEUS_PORT=9090

# Logging
log() {
    echo -e "${2:-INFO}: $1" | tee -a "$LOG_FILE"
    logger -t "agentic-monitoring" "$1"
}

# Create monitoring directories
create_directories() {
    log "Creating monitoring directories" "${BLUE}"

    mkdir -p $MONITORING_DIR/{prometheus,grafana,exporters,dashboards,alerts}
    mkdir -p /var/lib/prometheus
    mkdir -p /var/lib/grafana

    log "Directories created" "${GREEN}"
}

# Setup Prometheus configuration
setup_prometheus() {
    log "Setting up Prometheus configuration" "${BLUE}"

    cat > $MONITORING_DIR/prometheus/prometheus.yml << 'EOF'
# Prometheus Configuration for AgenticDosNode
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    monitor: 'agentic-cluster'
    environment: 'production'

# Alertmanager configuration
alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - localhost:9093

# Load rules
rule_files:
  - "alerts/*.yml"

# Scrape configurations
scrape_configs:
  # System metrics - Node Exporter
  - job_name: 'node'
    static_configs:
      - targets:
        - 'thanos:9100'
        - 'oracle1:9100'
    relabel_configs:
      - source_labels: [__address__]
        regex: '([^:]+):.*'
        target_label: instance
        replacement: '${1}'

  # Docker metrics
  - job_name: 'docker'
    static_configs:
      - targets:
        - 'thanos:9323'
        - 'oracle1:9323'

  # GPU metrics - NVIDIA Exporter
  - job_name: 'nvidia-gpu'
    static_configs:
      - targets:
        - 'thanos:9835'

  # PostgreSQL metrics
  - job_name: 'postgresql'
    static_configs:
      - targets:
        - 'oracle1:9187'

  # Redis metrics
  - job_name: 'redis'
    static_configs:
      - targets:
        - 'oracle1:9121'

  # vLLM metrics
  - job_name: 'vllm'
    static_configs:
      - targets:
        - 'thanos:8000'
    metrics_path: '/metrics'

  # Embedding service metrics
  - job_name: 'embeddings'
    static_configs:
      - targets:
        - 'thanos:8001'
    metrics_path: '/metrics'

  # Qdrant metrics
  - job_name: 'qdrant'
    static_configs:
      - targets:
        - 'thanos:6333'
        - 'oracle1:6333'
    metrics_path: '/metrics'

  # n8n metrics
  - job_name: 'n8n'
    static_configs:
      - targets:
        - 'oracle1:5678'
    metrics_path: '/metrics'

  # Kong API Gateway metrics
  - job_name: 'kong'
    static_configs:
      - targets:
        - 'oracle1:8001'
    metrics_path: '/metrics'

  # Custom AI metrics
  - job_name: 'ai-metrics'
    static_configs:
      - targets:
        - 'thanos:9100'
    metrics_path: '/metrics'
    file_sd_configs:
      - files:
        - '/var/lib/node_exporter/*.prom'

  # Prometheus self-monitoring
  - job_name: 'prometheus'
    static_configs:
      - targets:
        - 'localhost:9090'

  # Grafana metrics
  - job_name: 'grafana'
    static_configs:
      - targets:
        - 'localhost:3000'
    metrics_path: '/metrics'
EOF

    log "Prometheus configuration created" "${GREEN}"
}

# Setup alert rules
setup_alert_rules() {
    log "Setting up alert rules" "${BLUE}"

    cat > $MONITORING_DIR/prometheus/alerts/system-alerts.yml << 'EOF'
groups:
  - name: system_alerts
    interval: 30s
    rules:
      # CPU alerts
      - alert: HighCPUUsage
        expr: 100 - (avg by(instance) (irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 85
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High CPU usage on {{ $labels.instance }}"
          description: "CPU usage is above 85% (current value: {{ $value }}%)"

      - alert: CriticalCPUUsage
        expr: 100 - (avg by(instance) (irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 95
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Critical CPU usage on {{ $labels.instance }}"
          description: "CPU usage is above 95% (current value: {{ $value }}%)"

      # Memory alerts
      - alert: HighMemoryUsage
        expr: (1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100 > 85
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage on {{ $labels.instance }}"
          description: "Memory usage is above 85% (current value: {{ $value }}%)"

      - alert: OutOfMemory
        expr: (1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100 > 95
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Out of memory on {{ $labels.instance }}"
          description: "Memory usage is above 95% (current value: {{ $value }}%)"

      # Disk alerts
      - alert: DiskSpaceLow
        expr: (node_filesystem_avail_bytes{fstype!~"tmpfs|fuse.lxcfs|squashfs|vfat"} / node_filesystem_size_bytes) * 100 < 15
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Low disk space on {{ $labels.instance }}"
          description: "Disk space is below 15% (current value: {{ $value }}%)"

      # GPU alerts
      - alert: GPUHighTemperature
        expr: gpu_temperature_celsius > 85
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "GPU temperature high on {{ $labels.instance }}"
          description: "GPU temperature is above 85°C (current value: {{ $value }}°C)"

      - alert: GPUMemoryFull
        expr: (gpu_memory_used_mb / gpu_memory_total_mb) * 100 > 95
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "GPU memory almost full on {{ $labels.instance }}"
          description: "GPU memory usage is above 95% (current value: {{ $value }}%)"

      # Docker alerts
      - alert: ContainerDown
        expr: up{job="docker"} == 0
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Container down on {{ $labels.instance }}"
          description: "Docker container is not responding"

      # Database alerts
      - alert: PostgreSQLDown
        expr: up{job="postgresql"} == 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "PostgreSQL is down"
          description: "PostgreSQL database is not responding"

      - alert: RedisDown
        expr: up{job="redis"} == 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Redis is down"
          description: "Redis cache is not responding"

      # AI Service alerts
      - alert: vLLMServiceDown
        expr: up{job="vllm"} == 0
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "vLLM service is down"
          description: "vLLM model serving is not responding"

      - alert: InferenceLatencyHigh
        expr: histogram_quantile(0.95, rate(vllm_request_duration_seconds_bucket[5m])) > 2
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High inference latency"
          description: "95th percentile inference latency is above 2 seconds"
EOF

    log "Alert rules created" "${GREEN}"
}

# Setup Grafana dashboards
setup_grafana_dashboards() {
    log "Setting up Grafana dashboards" "${BLUE}"

    # System Overview Dashboard
    cat > $MONITORING_DIR/dashboards/system-overview.json << 'EOF'
{
  "dashboard": {
    "title": "AgenticDosNode System Overview",
    "panels": [
      {
        "title": "CPU Usage",
        "targets": [
          {
            "expr": "100 - (avg by(instance) (irate(node_cpu_seconds_total{mode=\"idle\"}[5m])) * 100)",
            "legendFormat": "{{ instance }}"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
      },
      {
        "title": "Memory Usage",
        "targets": [
          {
            "expr": "(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100",
            "legendFormat": "{{ instance }}"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
      },
      {
        "title": "GPU Utilization",
        "targets": [
          {
            "expr": "gpu_utilization_percent",
            "legendFormat": "GPU {{ gpu }}"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8}
      },
      {
        "title": "Network I/O",
        "targets": [
          {
            "expr": "rate(node_network_receive_bytes_total[5m])",
            "legendFormat": "RX {{ instance }}"
          },
          {
            "expr": "rate(node_network_transmit_bytes_total[5m])",
            "legendFormat": "TX {{ instance }}"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8}
      }
    ]
  }
}
EOF

    # AI Performance Dashboard
    cat > $MONITORING_DIR/dashboards/ai-performance.json << 'EOF'
{
  "dashboard": {
    "title": "AI Performance Metrics",
    "panels": [
      {
        "title": "Inference Requests/sec",
        "targets": [
          {
            "expr": "rate(vllm_request_total[1m])",
            "legendFormat": "Requests/sec"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
      },
      {
        "title": "Inference Latency (p95)",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(vllm_request_duration_seconds_bucket[5m]))",
            "legendFormat": "p95 Latency"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
      },
      {
        "title": "GPU Memory Usage",
        "targets": [
          {
            "expr": "(gpu_memory_used_mb / gpu_memory_total_mb) * 100",
            "legendFormat": "GPU {{ gpu }} Memory %"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8}
      },
      {
        "title": "Model Cache Hit Rate",
        "targets": [
          {
            "expr": "rate(model_cache_hits[5m]) / (rate(model_cache_hits[5m]) + rate(model_cache_misses[5m]))",
            "legendFormat": "Cache Hit Rate"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8}
      }
    ]
  }
}
EOF

    log "Grafana dashboards created" "${GREEN}"
}

# Setup custom exporters
setup_custom_exporters() {
    log "Setting up custom exporters" "${BLUE}"

    # Create unified metrics exporter
    cat > $MONITORING_DIR/exporters/unified-exporter.py << 'EOF'
#!/usr/bin/env python3

import time
import psutil
import GPUtil
import docker
import redis
import psycopg2
import requests
from prometheus_client import start_http_server, Gauge, Counter, Histogram
from typing import Dict, List
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnifiedMetricsExporter:
    """Unified metrics exporter for AgenticDosNode"""

    def __init__(self, port: int = 9999):
        self.port = port
        self.setup_metrics()
        self.docker_client = docker.from_env()

    def setup_metrics(self):
        """Setup Prometheus metrics"""

        # System metrics
        self.cpu_usage = Gauge('agentic_cpu_usage_percent', 'CPU usage percentage', ['core'])
        self.memory_usage = Gauge('agentic_memory_usage_bytes', 'Memory usage in bytes')
        self.disk_io_read = Counter('agentic_disk_read_bytes_total', 'Total disk read bytes')
        self.disk_io_write = Counter('agentic_disk_write_bytes_total', 'Total disk write bytes')

        # GPU metrics
        self.gpu_utilization = Gauge('agentic_gpu_utilization_percent', 'GPU utilization', ['gpu_id'])
        self.gpu_memory = Gauge('agentic_gpu_memory_used_mb', 'GPU memory used', ['gpu_id'])
        self.gpu_temp = Gauge('agentic_gpu_temperature_celsius', 'GPU temperature', ['gpu_id'])

        # Container metrics
        self.container_cpu = Gauge('agentic_container_cpu_percent', 'Container CPU usage', ['container'])
        self.container_memory = Gauge('agentic_container_memory_mb', 'Container memory usage', ['container'])

        # AI service metrics
        self.inference_queue_size = Gauge('agentic_inference_queue_size', 'Inference request queue size')
        self.model_load_time = Histogram('agentic_model_load_seconds', 'Model loading time')
        self.batch_size = Gauge('agentic_batch_size', 'Current batch size')

        # Database metrics
        self.db_connections = Gauge('agentic_db_connections', 'Database connections', ['database'])
        self.db_query_time = Histogram('agentic_db_query_seconds', 'Database query time', ['query_type'])

    def collect_system_metrics(self):
        """Collect system metrics"""

        # CPU metrics
        cpu_percent = psutil.cpu_percent(percpu=True)
        for i, percent in enumerate(cpu_percent):
            self.cpu_usage.labels(core=str(i)).set(percent)

        # Memory metrics
        memory = psutil.virtual_memory()
        self.memory_usage.set(memory.used)

        # Disk I/O
        disk_io = psutil.disk_io_counters()
        if disk_io:
            self.disk_io_read._value.set(disk_io.read_bytes)
            self.disk_io_write._value.set(disk_io.write_bytes)

    def collect_gpu_metrics(self):
        """Collect GPU metrics"""

        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                self.gpu_utilization.labels(gpu_id=str(gpu.id)).set(gpu.load * 100)
                self.gpu_memory.labels(gpu_id=str(gpu.id)).set(gpu.memoryUsed)
                self.gpu_temp.labels(gpu_id=str(gpu.id)).set(gpu.temperature)
        except Exception as e:
            logger.warning(f"Failed to collect GPU metrics: {e}")

    def collect_container_metrics(self):
        """Collect Docker container metrics"""

        try:
            containers = self.docker_client.containers.list()
            for container in containers:
                stats = container.stats(stream=False)

                # Calculate CPU percentage
                cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                           stats['precpu_stats']['cpu_usage']['total_usage']
                system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                              stats['precpu_stats']['system_cpu_usage']
                cpu_percent = (cpu_delta / system_delta) * 100.0 if system_delta > 0 else 0

                self.container_cpu.labels(container=container.name).set(cpu_percent)

                # Memory usage
                mem_usage = stats['memory_stats']['usage'] / 1024 / 1024  # Convert to MB
                self.container_memory.labels(container=container.name).set(mem_usage)
        except Exception as e:
            logger.warning(f"Failed to collect container metrics: {e}")

    def collect_ai_metrics(self):
        """Collect AI service metrics"""

        try:
            # vLLM metrics
            response = requests.get('http://localhost:8000/metrics', timeout=5)
            # Parse and export relevant metrics

            # Embedding service metrics
            response = requests.get('http://localhost:8001/metrics', timeout=5)
            # Parse and export relevant metrics

        except Exception as e:
            logger.warning(f"Failed to collect AI metrics: {e}")

    def run(self):
        """Start the exporter"""

        start_http_server(self.port)
        logger.info(f"Metrics exporter started on port {self.port}")

        while True:
            try:
                self.collect_system_metrics()
                self.collect_gpu_metrics()
                self.collect_container_metrics()
                self.collect_ai_metrics()
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")

            time.sleep(10)

if __name__ == "__main__":
    exporter = UnifiedMetricsExporter()
    exporter.run()
EOF

    chmod +x $MONITORING_DIR/exporters/unified-exporter.py

    # Create systemd service for the exporter
    cat > /etc/systemd/system/agentic-exporter.service << EOF
[Unit]
Description=AgenticDosNode Metrics Exporter
After=network.target

[Service]
Type=simple
ExecStart=/usr/bin/python3 $MONITORING_DIR/exporters/unified-exporter.py
Restart=always
RestartSec=10
User=root

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    log "Custom exporters setup completed" "${GREEN}"
}

# Setup performance profiling tools
setup_profiling() {
    log "Setting up performance profiling tools" "${BLUE}"

    # Create profiling script
    cat > $MONITORING_DIR/exporters/profiler.sh << 'EOF'
#!/bin/bash

# Performance Profiling Script for AgenticDosNode

PROFILE_DIR="/var/log/agentic-profiles"
mkdir -p $PROFILE_DIR

# Function to profile CPU
profile_cpu() {
    echo "Profiling CPU for 30 seconds..."
    perf record -F 99 -a -g -- sleep 30
    perf report --stdio > $PROFILE_DIR/cpu-profile-$(date +%Y%m%d-%H%M%S).txt
}

# Function to profile memory
profile_memory() {
    echo "Profiling memory..."
    # Use valgrind for memory profiling if available
    if command -v valgrind &>/dev/null; then
        valgrind --tool=massif --massif-out-file=$PROFILE_DIR/memory-profile-$(date +%Y%m%d-%H%M%S).ms $1
    fi

    # Capture memory map
    cat /proc/meminfo > $PROFILE_DIR/meminfo-$(date +%Y%m%d-%H%M%S).txt
    ps aux --sort=-%mem | head -20 > $PROFILE_DIR/top-memory-$(date +%Y%m%d-%H%M%S).txt
}

# Function to profile I/O
profile_io() {
    echo "Profiling I/O for 30 seconds..."
    iotop -b -n 30 -d 1 > $PROFILE_DIR/io-profile-$(date +%Y%m%d-%H%M%S).txt
    iostat -x 1 30 > $PROFILE_DIR/iostat-$(date +%Y%m%d-%H%M%S).txt
}

# Function to profile network
profile_network() {
    echo "Profiling network for 30 seconds..."
    tcpdump -i any -w $PROFILE_DIR/network-$(date +%Y%m%d-%H%M%S).pcap -G 30 -W 1 &
    TCPDUMP_PID=$!
    sleep 31
    kill $TCPDUMP_PID 2>/dev/null

    # Network statistics
    ss -s > $PROFILE_DIR/socket-stats-$(date +%Y%m%d-%H%M%S).txt
    netstat -i > $PROFILE_DIR/netstat-$(date +%Y%m%d-%H%M%S).txt
}

# Function to profile GPU
profile_gpu() {
    if nvidia-smi &>/dev/null; then
        echo "Profiling GPU..."
        nvidia-smi dmon -s pucvmet -c 30 > $PROFILE_DIR/gpu-profile-$(date +%Y%m%d-%H%M%S).txt

        # CUDA profiling if nvprof is available
        if command -v nvprof &>/dev/null; then
            nvprof --print-summary --log-file $PROFILE_DIR/cuda-profile-$(date +%Y%m%d-%H%M%S).txt $1
        fi
    fi
}

# Function to generate flame graph
generate_flamegraph() {
    if [ -f perf.data ]; then
        perf script | /opt/FlameGraph/stackcollapse-perf.pl | /opt/FlameGraph/flamegraph.pl > $PROFILE_DIR/flamegraph-$(date +%Y%m%d-%H%M%S).svg
        echo "Flame graph generated"
    fi
}

# Main profiling function
main() {
    case "$1" in
        cpu)
            profile_cpu
            ;;
        memory)
            profile_memory "$2"
            ;;
        io)
            profile_io
            ;;
        network)
            profile_network
            ;;
        gpu)
            profile_gpu "$2"
            ;;
        all)
            profile_cpu &
            profile_memory &
            profile_io &
            profile_network &
            profile_gpu &
            wait
            ;;
        flamegraph)
            generate_flamegraph
            ;;
        *)
            echo "Usage: $0 {cpu|memory|io|network|gpu|all|flamegraph} [command]"
            exit 1
            ;;
    esac
}

main "$@"
EOF

    chmod +x $MONITORING_DIR/exporters/profiler.sh
    log "Profiling tools setup completed" "${GREEN}"
}

# Install monitoring stack
install_monitoring_stack() {
    log "Installing monitoring stack" "${BLUE}"

    # Create docker-compose for monitoring
    cat > $MONITORING_DIR/docker-compose.yml << EOF
version: '3.9'

services:
  prometheus:
    image: prom/prometheus:latest
    container_name: agentic-prometheus
    volumes:
      - $MONITORING_DIR/prometheus:/etc/prometheus
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/usr/share/prometheus/console_libraries'
      - '--web.console.templates=/usr/share/prometheus/consoles'
      - '--web.enable-lifecycle'
      - '--storage.tsdb.retention.time=30d'
    ports:
      - "$PROMETHEUS_PORT:9090"
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    container_name: agentic-grafana
    environment:
      - GF_SECURITY_ADMIN_USER=admin
      - GF_SECURITY_ADMIN_PASSWORD=changeme
      - GF_INSTALL_PLUGINS=redis-datasource,postgres-datasource
    volumes:
      - grafana_data:/var/lib/grafana
      - $MONITORING_DIR/dashboards:/etc/grafana/provisioning/dashboards
    ports:
      - "$GRAFANA_PORT:3000"
    restart: unless-stopped

  alertmanager:
    image: prom/alertmanager:latest
    container_name: agentic-alertmanager
    volumes:
      - $MONITORING_DIR/alertmanager:/etc/alertmanager
      - alertmanager_data:/alertmanager
    command:
      - '--config.file=/etc/alertmanager/config.yml'
      - '--storage.path=/alertmanager'
    ports:
      - "9093:9093"
    restart: unless-stopped

volumes:
  prometheus_data:
  grafana_data:
  alertmanager_data:
EOF

    log "Monitoring stack configuration created" "${GREEN}"
}

# Main setup function
main() {
    log "Starting monitoring and profiling setup" "${BLUE}"

    create_directories
    setup_prometheus
    setup_alert_rules
    setup_grafana_dashboards
    setup_custom_exporters
    setup_profiling
    install_monitoring_stack

    log "Monitoring and profiling setup completed!" "${GREEN}"

    cat << EOF

${GREEN}Monitoring Setup Complete!${NC}

${YELLOW}Access Points:${NC}
- Prometheus: http://localhost:$PROMETHEUS_PORT
- Grafana: http://localhost:$GRAFANA_PORT (admin/changeme)
- Alertmanager: http://localhost:9093

${BLUE}Start Monitoring Stack:${NC}
cd $MONITORING_DIR
docker-compose up -d

${BLUE}Start Custom Exporter:${NC}
systemctl start agentic-exporter

${BLUE}Run Performance Profiling:${NC}
$MONITORING_DIR/exporters/profiler.sh all

${GREEN}Dashboards Available:${NC}
- System Overview
- AI Performance Metrics
- Database Performance
- Container Metrics

${RED}To Remove:${NC}
cd $MONITORING_DIR
docker-compose down
systemctl stop agentic-exporter
EOF
}

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "This script must be run as root (use sudo)"
    exit 1
fi

main "$@"