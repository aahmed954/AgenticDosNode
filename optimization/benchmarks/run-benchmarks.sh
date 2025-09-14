#!/bin/bash

# Comprehensive Benchmarking Suite for AgenticDosNode
# Tests performance before and after optimizations

set -euo pipefail

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
RESULTS_DIR="/home/starlord/AgenticDosNode/optimization/results"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
RESULTS_FILE="$RESULTS_DIR/benchmark-$TIMESTAMP.json"
COMPARISON_FILE="$RESULTS_DIR/comparison-$TIMESTAMP.md"
LOG_FILE="/var/log/agentic-benchmark.log"

# Logging
log() {
    echo -e "${2:-INFO}: $1" | tee -a "$LOG_FILE"
}

# Initialize results directory
init_results() {
    mkdir -p "$RESULTS_DIR"
    log "Initializing benchmark results in $RESULTS_DIR" "${BLUE}"
}

# System Information Collection
collect_system_info() {
    log "Collecting system information" "${BLUE}"

    cat > "$RESULTS_DIR/system-info-$TIMESTAMP.txt" << EOF
=== System Information ===
Date: $(date)
Hostname: $(hostname)
Kernel: $(uname -r)
CPU: $(lscpu | grep "Model name" | cut -d: -f2 | xargs)
Cores: $(nproc)
Memory: $(free -h | awk '/^Mem:/{print $2}')
Disk: $(df -h / | awk 'NR==2{print $2}')
EOF

    if nvidia-smi &>/dev/null; then
        echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)" >> "$RESULTS_DIR/system-info-$TIMESTAMP.txt"
        echo "CUDA: $(nvidia-smi | grep "CUDA Version" | awk '{print $9}')" >> "$RESULTS_DIR/system-info-$TIMESTAMP.txt"
    fi

    log "System information collected" "${GREEN}"
}

# CPU Benchmark
benchmark_cpu() {
    log "Running CPU benchmark" "${BLUE}"

    local cpu_results=""

    # Sysbench CPU test
    if command -v sysbench &>/dev/null; then
        log "  Running sysbench CPU test..." "${YELLOW}"
        cpu_results=$(sysbench cpu --cpu-max-prime=20000 --threads=$(nproc) run | grep "events per second" | awk '{print $4}')
        echo "{\"sysbench_cpu_events_per_sec\": $cpu_results}" > "$RESULTS_DIR/cpu-benchmark-$TIMESTAMP.json"
    fi

    # Stress-ng CPU test
    if command -v stress-ng &>/dev/null; then
        log "  Running stress-ng CPU test..." "${YELLOW}"
        stress_result=$(stress-ng --cpu $(nproc) --cpu-method matrixprod --metrics-brief --timeout 30s 2>&1 | grep "cpu " | awk '{print $9}')
        echo ", \"stress_ng_bogo_ops_per_sec\": $stress_result" >> "$RESULTS_DIR/cpu-benchmark-$TIMESTAMP.json"
    fi

    # OpenSSL speed test
    log "  Running OpenSSL speed test..." "${YELLOW}"
    openssl_result=$(openssl speed -evp aes-256-cbc 2>/dev/null | grep "aes-256-cbc" | tail -1 | awk '{print $NF}')
    echo ", \"openssl_aes256_speed_kb_per_sec\": $openssl_result" >> "$RESULTS_DIR/cpu-benchmark-$TIMESTAMP.json"

    log "CPU benchmark completed" "${GREEN}"
}

# Memory Benchmark
benchmark_memory() {
    log "Running memory benchmark" "${BLUE}"

    # Sysbench memory test
    if command -v sysbench &>/dev/null; then
        log "  Running sysbench memory test..." "${YELLOW}"
        mem_speed=$(sysbench memory --memory-total-size=10G run | grep "transferred" | awk '{print $4}' | tr -d '()')
        echo "{\"sysbench_memory_speed_mb_per_sec\": \"$mem_speed\"}" > "$RESULTS_DIR/memory-benchmark-$TIMESTAMP.json"
    fi

    # Stream memory bandwidth test
    if [ -f /usr/local/bin/stream ]; then
        log "  Running STREAM memory bandwidth test..." "${YELLOW}"
        stream_result=$(/usr/local/bin/stream | grep "Copy:" | awk '{print $2}')
        echo ", \"stream_copy_mb_per_sec\": $stream_result" >> "$RESULTS_DIR/memory-benchmark-$TIMESTAMP.json"
    fi

    log "Memory benchmark completed" "${GREEN}"
}

# Disk I/O Benchmark
benchmark_disk() {
    log "Running disk I/O benchmark" "${BLUE}"

    local test_file="/tmp/benchmark-test-file"

    # FIO benchmark
    if command -v fio &>/dev/null; then
        log "  Running FIO sequential read/write test..." "${YELLOW}"

        # Sequential write
        fio --name=seq_write --ioengine=libaio --rw=write --bs=1M --size=1G \
            --numjobs=1 --runtime=30 --time_based --end_fsync=1 \
            --filename=$test_file --output-format=json > "$RESULTS_DIR/fio-write-$TIMESTAMP.json" 2>/dev/null

        # Sequential read
        fio --name=seq_read --ioengine=libaio --rw=read --bs=1M --size=1G \
            --numjobs=1 --runtime=30 --time_based \
            --filename=$test_file --output-format=json > "$RESULTS_DIR/fio-read-$TIMESTAMP.json" 2>/dev/null

        # Random IOPS
        fio --name=random_iops --ioengine=libaio --rw=randrw --bs=4k --size=100M \
            --numjobs=4 --runtime=30 --time_based --end_fsync=1 \
            --filename=$test_file --output-format=json > "$RESULTS_DIR/fio-iops-$TIMESTAMP.json" 2>/dev/null

        rm -f $test_file
    fi

    # DD benchmark
    log "  Running DD write test..." "${YELLOW}"
    dd_write=$(dd if=/dev/zero of=$test_file bs=1M count=1024 conv=fdatasync 2>&1 | grep -oP '\d+\.\d+ [MG]B/s' | tail -1)
    echo "{\"dd_write_speed\": \"$dd_write\"}" > "$RESULTS_DIR/disk-benchmark-$TIMESTAMP.json"

    log "  Running DD read test..." "${YELLOW}"
    dd_read=$(dd if=$test_file of=/dev/null bs=1M 2>&1 | grep -oP '\d+\.\d+ [MG]B/s' | tail -1)
    echo ", \"dd_read_speed\": \"$dd_read\"}" >> "$RESULTS_DIR/disk-benchmark-$TIMESTAMP.json"

    rm -f $test_file
    log "Disk I/O benchmark completed" "${GREEN}"
}

# Network Benchmark
benchmark_network() {
    log "Running network benchmark" "${BLUE}"

    # iPerf3 test (requires server)
    if command -v iperf3 &>/dev/null; then
        log "  Testing loopback network performance..." "${YELLOW}"

        # Start iperf3 server in background
        iperf3 -s -D -f m

        # Run client test
        iperf3_result=$(iperf3 -c 127.0.0.1 -t 10 -f m | grep "sender" | awk '{print $7}')
        echo "{\"iperf3_loopback_mbits_per_sec\": $iperf3_result}" > "$RESULTS_DIR/network-benchmark-$TIMESTAMP.json"

        # Kill iperf3 server
        pkill iperf3
    fi

    # Latency test
    log "  Testing network latency..." "${YELLOW}"
    ping_result=$(ping -c 100 8.8.8.8 | tail -1 | awk '{print $4}' | cut -d '/' -f 2)
    echo ", \"ping_avg_latency_ms\": $ping_result" >> "$RESULTS_DIR/network-benchmark-$TIMESTAMP.json"

    log "Network benchmark completed" "${GREEN}"
}

# GPU Benchmark
benchmark_gpu() {
    if ! nvidia-smi &>/dev/null; then
        log "No GPU detected, skipping GPU benchmark" "${YELLOW}"
        return
    fi

    log "Running GPU benchmark" "${BLUE}"

    # GPU bandwidth test
    if [ -f /usr/local/cuda/extras/demo_suite/bandwidthTest ]; then
        log "  Running CUDA bandwidth test..." "${YELLOW}"
        bandwidth_result=$(/usr/local/cuda/extras/demo_suite/bandwidthTest --csv 2>/dev/null | tail -1)
        echo "{\"cuda_bandwidth_test\": \"$bandwidth_result\"}" > "$RESULTS_DIR/gpu-benchmark-$TIMESTAMP.json"
    fi

    # GPU compute test
    if command -v gpu-burn &>/dev/null; then
        log "  Running GPU burn test for 30 seconds..." "${YELLOW}"
        gpu-burn 30 > "$RESULTS_DIR/gpu-burn-$TIMESTAMP.txt" 2>&1
    fi

    # CUDA matrix multiplication benchmark
    cat > /tmp/cuda_benchmark.py << 'EOF'
import torch
import time
import json

def benchmark_gpu():
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    device = torch.device("cuda")
    sizes = [1024, 2048, 4096, 8192]
    results = {}

    for size in sizes:
        # Create random matrices
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)

        # Warmup
        for _ in range(3):
            c = torch.matmul(a, b)
        torch.cuda.synchronize()

        # Benchmark
        start = time.time()
        for _ in range(10):
            c = torch.matmul(a, b)
        torch.cuda.synchronize()
        elapsed = time.time() - start

        # Calculate GFLOPS
        gflops = (2 * size**3 * 10) / (elapsed * 1e9)
        results[f"matmul_{size}x{size}_gflops"] = round(gflops, 2)

    return results

if __name__ == "__main__":
    results = benchmark_gpu()
    print(json.dumps(results))
EOF

    if command -v python3 &>/dev/null && python3 -c "import torch" 2>/dev/null; then
        log "  Running PyTorch GPU benchmark..." "${YELLOW}"
        python3 /tmp/cuda_benchmark.py > "$RESULTS_DIR/pytorch-gpu-$TIMESTAMP.json"
    fi

    rm -f /tmp/cuda_benchmark.py
    log "GPU benchmark completed" "${GREEN}"
}

# Docker Performance Benchmark
benchmark_docker() {
    log "Running Docker performance benchmark" "${BLUE}"

    # Container startup time
    log "  Testing container startup time..." "${YELLOW}"
    start_time=$(date +%s%N)
    docker run --rm alpine echo "test" &>/dev/null
    end_time=$(date +%s%N)
    startup_time=$(echo "scale=3; ($end_time - $start_time) / 1000000000" | bc)
    echo "{\"docker_startup_time_seconds\": $startup_time}" > "$RESULTS_DIR/docker-benchmark-$TIMESTAMP.json"

    # Image pull performance
    log "  Testing image pull performance..." "${YELLOW}"
    docker rmi alpine &>/dev/null 2>&1 || true
    start_time=$(date +%s)
    docker pull alpine &>/dev/null
    end_time=$(date +%s)
    pull_time=$((end_time - start_time))
    echo ", \"docker_image_pull_seconds\": $pull_time" >> "$RESULTS_DIR/docker-benchmark-$TIMESTAMP.json"

    log "Docker benchmark completed" "${GREEN}"
}

# Database Benchmark
benchmark_database() {
    log "Running database benchmark" "${BLUE}"

    # PostgreSQL benchmark with pgbench
    if command -v pgbench &>/dev/null; then
        log "  Running pgbench TPC-B test..." "${YELLOW}"

        # Initialize pgbench
        PGPASSWORD=${POSTGRES_PASSWORD:-changeme} pgbench -i -s 10 -U ${POSTGRES_USER:-agentic} -h localhost benchmark_db 2>/dev/null || true

        # Run benchmark
        pgbench_result=$(PGPASSWORD=${POSTGRES_PASSWORD:-changeme} pgbench -c 10 -j 2 -T 60 -U ${POSTGRES_USER:-agentic} -h localhost benchmark_db 2>/dev/null | grep "tps = " | awk '{print $3}')
        echo "{\"pgbench_tps\": $pgbench_result}" > "$RESULTS_DIR/database-benchmark-$TIMESTAMP.json"
    fi

    # Redis benchmark
    if command -v redis-benchmark &>/dev/null; then
        log "  Running Redis benchmark..." "${YELLOW}"

        redis_set=$(redis-benchmark -t set -n 100000 -q | awk '{print $2}')
        redis_get=$(redis-benchmark -t get -n 100000 -q | awk '{print $2}')
        echo ", \"redis_set_ops_per_sec\": \"$redis_set\", \"redis_get_ops_per_sec\": \"$redis_get\"" >> "$RESULTS_DIR/database-benchmark-$TIMESTAMP.json"
    fi

    log "Database benchmark completed" "${GREEN}"
}

# AI Inference Benchmark
benchmark_ai_inference() {
    log "Running AI inference benchmark" "${BLUE}"

    # Create inference benchmark script
    cat > /tmp/ai_benchmark.py << 'EOF'
import requests
import time
import json
import concurrent.futures
from typing import Dict, List

def benchmark_vllm(url: str = "http://localhost:8000", num_requests: int = 10) -> Dict:
    """Benchmark vLLM inference"""

    results = {
        "latencies": [],
        "tokens_per_second": []
    }

    prompt = "Explain quantum computing in simple terms."

    for _ in range(num_requests):
        start = time.time()
        try:
            response = requests.post(
                f"{url}/v1/completions",
                json={
                    "model": "mistralai/Mistral-7B-Instruct-v0.2",
                    "prompt": prompt,
                    "max_tokens": 100,
                    "temperature": 0.7
                },
                timeout=30
            )
            elapsed = time.time() - start
            results["latencies"].append(elapsed)

            if response.status_code == 200:
                tokens = len(response.json()["choices"][0]["text"].split())
                results["tokens_per_second"].append(tokens / elapsed)
        except Exception as e:
            print(f"Error: {e}")

    if results["latencies"]:
        return {
            "avg_latency_seconds": sum(results["latencies"]) / len(results["latencies"]),
            "p95_latency_seconds": sorted(results["latencies"])[int(len(results["latencies"]) * 0.95)],
            "avg_tokens_per_second": sum(results["tokens_per_second"]) / len(results["tokens_per_second"]) if results["tokens_per_second"] else 0
        }
    return {"error": "No successful requests"}

def benchmark_embeddings(url: str = "http://localhost:8001", num_requests: int = 100) -> Dict:
    """Benchmark embedding service"""

    texts = ["This is a test sentence for embeddings."] * 10
    latencies = []

    for _ in range(num_requests // 10):
        start = time.time()
        try:
            response = requests.post(
                f"{url}/embed",
                json={"inputs": texts},
                timeout=10
            )
            if response.status_code == 200:
                latencies.append(time.time() - start)
        except Exception as e:
            print(f"Error: {e}")

    if latencies:
        return {
            "avg_latency_seconds": sum(latencies) / len(latencies),
            "p95_latency_seconds": sorted(latencies)[int(len(latencies) * 0.95)],
            "throughput_requests_per_second": len(latencies) / sum(latencies)
        }
    return {"error": "No successful requests"}

def benchmark_concurrent_inference(url: str = "http://localhost:8000", num_workers: int = 10) -> Dict:
    """Benchmark concurrent inference"""

    def make_request():
        start = time.time()
        try:
            response = requests.post(
                f"{url}/v1/completions",
                json={
                    "model": "mistralai/Mistral-7B-Instruct-v0.2",
                    "prompt": "Hello, world!",
                    "max_tokens": 50
                },
                timeout=30
            )
            return time.time() - start if response.status_code == 200 else None
        except:
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(make_request) for _ in range(num_workers * 5)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
        latencies = [r for r in results if r is not None]

    if latencies:
        return {
            "concurrent_requests": num_workers * 5,
            "successful_requests": len(latencies),
            "avg_latency_seconds": sum(latencies) / len(latencies),
            "total_time_seconds": max(latencies),
            "throughput_requests_per_second": len(latencies) / max(latencies)
        }
    return {"error": "No successful concurrent requests"}

if __name__ == "__main__":
    results = {}

    # Benchmark vLLM
    print("Benchmarking vLLM...")
    results["vllm"] = benchmark_vllm()

    # Benchmark embeddings
    print("Benchmarking embeddings...")
    results["embeddings"] = benchmark_embeddings()

    # Benchmark concurrent inference
    print("Benchmarking concurrent inference...")
    results["concurrent"] = benchmark_concurrent_inference()

    print(json.dumps(results, indent=2))
EOF

    if command -v python3 &>/dev/null; then
        log "  Running AI inference benchmark..." "${YELLOW}"
        python3 /tmp/ai_benchmark.py > "$RESULTS_DIR/ai-inference-$TIMESTAMP.json" 2>/dev/null || echo "{\"error\": \"AI services not available\"}" > "$RESULTS_DIR/ai-inference-$TIMESTAMP.json"
    fi

    rm -f /tmp/ai_benchmark.py
    log "AI inference benchmark completed" "${GREEN}"
}

# Aggregate and analyze results
analyze_results() {
    log "Analyzing benchmark results" "${BLUE}"

    # Create analysis script
    cat > /tmp/analyze_benchmarks.py << 'EOF'
import json
import glob
import os
from datetime import datetime

def load_results(results_dir, timestamp):
    """Load all benchmark results"""
    results = {}
    pattern = f"{results_dir}/*-benchmark-{timestamp}.json"

    for file in glob.glob(pattern):
        category = os.path.basename(file).replace(f"-benchmark-{timestamp}.json", "")
        try:
            with open(file, 'r') as f:
                content = f.read()
                # Fix JSON formatting issues
                if content.startswith(","):
                    content = "{" + content[1:] + "}"
                elif not content.startswith("{"):
                    content = "{" + content + "}"
                results[category] = json.loads(content)
        except Exception as e:
            print(f"Error loading {file}: {e}")

    return results

def generate_summary(results):
    """Generate summary report"""
    summary = {
        "timestamp": datetime.now().isoformat(),
        "categories": list(results.keys()),
        "highlights": {}
    }

    # Extract key metrics
    if "cpu" in results:
        summary["highlights"]["cpu_performance"] = results["cpu"].get("sysbench_cpu_events_per_sec", "N/A")

    if "memory" in results:
        summary["highlights"]["memory_bandwidth"] = results["memory"].get("sysbench_memory_speed_mb_per_sec", "N/A")

    if "disk" in results:
        summary["highlights"]["disk_write_speed"] = results["disk"].get("dd_write_speed", "N/A")

    if "ai-inference" in results and "vllm" in results["ai-inference"]:
        summary["highlights"]["ai_inference_latency"] = results["ai-inference"]["vllm"].get("avg_latency_seconds", "N/A")

    return summary

if __name__ == "__main__":
    import sys
    results_dir = sys.argv[1]
    timestamp = sys.argv[2]

    results = load_results(results_dir, timestamp)
    summary = generate_summary(results)

    # Save complete results
    with open(f"{results_dir}/complete-results-{timestamp}.json", 'w') as f:
        json.dump({"summary": summary, "detailed": results}, f, indent=2)

    print(json.dumps(summary, indent=2))
EOF

    python3 /tmp/analyze_benchmarks.py "$RESULTS_DIR" "$TIMESTAMP" > "$RESULTS_DIR/summary-$TIMESTAMP.json"
    rm -f /tmp/analyze_benchmarks.py

    log "Results analysis completed" "${GREEN}"
}

# Compare before/after results
compare_results() {
    if [ "$#" -ne 2 ]; then
        log "Usage: compare_results <before_timestamp> <after_timestamp>" "${RED}"
        return 1
    fi

    local before_timestamp=$1
    local after_timestamp=$2

    log "Comparing results: $before_timestamp vs $after_timestamp" "${BLUE}"

    cat > /tmp/compare_benchmarks.py << 'EOF'
import json
import sys
from pathlib import Path

def load_results(results_dir, timestamp):
    file = Path(results_dir) / f"complete-results-{timestamp}.json"
    if file.exists():
        with open(file, 'r') as f:
            return json.load(f)
    return None

def calculate_improvement(before, after):
    """Calculate percentage improvement"""
    if isinstance(before, (int, float)) and isinstance(after, (int, float)):
        if before != 0:
            return ((after - before) / before) * 100
    return None

def compare_metrics(before_results, after_results):
    """Compare metrics and generate report"""
    comparison = {
        "improvements": {},
        "regressions": {},
        "unchanged": {}
    }

    # Compare detailed results
    for category in before_results.get("detailed", {}):
        if category in after_results.get("detailed", {}):
            before_metrics = before_results["detailed"][category]
            after_metrics = after_results["detailed"][category]

            for metric in before_metrics:
                if metric in after_metrics:
                    before_val = before_metrics[metric]
                    after_val = after_metrics[metric]

                    # Try to parse numeric values
                    try:
                        if isinstance(before_val, str):
                            before_num = float(before_val.split()[0])
                        else:
                            before_num = float(before_val)

                        if isinstance(after_val, str):
                            after_num = float(after_val.split()[0])
                        else:
                            after_num = float(after_val)

                        improvement = calculate_improvement(before_num, after_num)

                        if improvement:
                            key = f"{category}.{metric}"
                            if improvement > 5:
                                comparison["improvements"][key] = {
                                    "before": before_val,
                                    "after": after_val,
                                    "improvement": f"{improvement:.1f}%"
                                }
                            elif improvement < -5:
                                comparison["regressions"][key] = {
                                    "before": before_val,
                                    "after": after_val,
                                    "regression": f"{abs(improvement):.1f}%"
                                }
                            else:
                                comparison["unchanged"][key] = {
                                    "before": before_val,
                                    "after": after_val,
                                    "change": f"{improvement:.1f}%"
                                }
                    except:
                        pass

    return comparison

def generate_markdown_report(comparison, before_ts, after_ts):
    """Generate markdown comparison report"""
    report = f"""# Benchmark Comparison Report

**Before:** {before_ts}
**After:** {after_ts}

## Summary

- **Improvements:** {len(comparison['improvements'])} metrics
- **Regressions:** {len(comparison['regressions'])} metrics
- **Unchanged:** {len(comparison['unchanged'])} metrics

## Improvements 📈

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
"""

    for metric, data in comparison["improvements"].items():
        report += f"| {metric} | {data['before']} | {data['after']} | {data['improvement']} |\n"

    report += """

## Regressions 📉

| Metric | Before | After | Regression |
|--------|--------|-------|------------|
"""

    for metric, data in comparison["regressions"].items():
        report += f"| {metric} | {data['before']} | {data['after']} | -{data['regression']} |\n"

    if not comparison["regressions"]:
        report += "| No regressions detected | - | - | - |\n"

    return report

if __name__ == "__main__":
    results_dir = sys.argv[1]
    before_ts = sys.argv[2]
    after_ts = sys.argv[3]

    before = load_results(results_dir, before_ts)
    after = load_results(results_dir, after_ts)

    if before and after:
        comparison = compare_metrics(before, after)
        report = generate_markdown_report(comparison, before_ts, after_ts)
        print(report)

        # Save comparison
        with open(f"{results_dir}/comparison-{after_ts}.json", 'w') as f:
            json.dump(comparison, f, indent=2)
    else:
        print("Error: Could not load results files")
EOF

    python3 /tmp/compare_benchmarks.py "$RESULTS_DIR" "$before_timestamp" "$after_timestamp" > "$COMPARISON_FILE"
    rm -f /tmp/compare_benchmarks.py

    log "Comparison report generated: $COMPARISON_FILE" "${GREEN}"
    cat "$COMPARISON_FILE"
}

# Run all benchmarks
run_all_benchmarks() {
    log "Starting comprehensive benchmark suite" "${BLUE}"

    init_results
    collect_system_info

    # Run benchmarks with progress indication
    local benchmarks=(
        "cpu"
        "memory"
        "disk"
        "network"
        "gpu"
        "docker"
        "database"
        "ai_inference"
    )

    local total=${#benchmarks[@]}
    local current=0

    for benchmark in "${benchmarks[@]}"; do
        current=$((current + 1))
        log "[$current/$total] Running $benchmark benchmark..." "${YELLOW}"
        benchmark_$benchmark
    done

    analyze_results

    log "All benchmarks completed!" "${GREEN}"
    log "Results saved to: $RESULTS_DIR" "${GREEN}"
    log "Summary: $RESULTS_DIR/summary-$TIMESTAMP.json" "${GREEN}"
}

# Main execution
main() {
    case "${1:-run}" in
        run)
            run_all_benchmarks
            ;;
        compare)
            if [ "$#" -ne 3 ]; then
                echo "Usage: $0 compare <before_timestamp> <after_timestamp>"
                exit 1
            fi
            compare_results "$2" "$3"
            ;;
        *)
            echo "Usage: $0 {run|compare <before> <after>}"
            exit 1
            ;;
    esac
}

# Check prerequisites
check_prerequisites() {
    local missing_tools=()

    # Check for optional but recommended tools
    for tool in sysbench fio iperf3 pgbench redis-benchmark; do
        if ! command -v $tool &>/dev/null; then
            missing_tools+=($tool)
        fi
    done

    if [ ${#missing_tools[@]} -gt 0 ]; then
        log "Warning: Some benchmark tools are missing: ${missing_tools[*]}" "${YELLOW}"
        log "Install them for more comprehensive benchmarks" "${YELLOW}"
    fi
}

# Run checks before main
check_prerequisites

# Execute main function
main "$@"