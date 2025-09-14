# AgenticDosNode System Optimization Suite

Performance optimization procedures for dedicated AI machines (Thanos GPU + Oracle1 CPU) to maximize AgenticDosNode performance.

## System Architecture

- **Thanos (GPU Node)**: RTX 4090, CUDA 13.0 - Handles AI inference, embeddings, image generation
- **Oracle1 (CPU Node)**: High-performance CPU - Handles orchestration, databases, API gateway

## Optimization Categories

### 1. Hardware Optimization (`scripts/hardware-optimization.sh`)
- CPU governor settings for AI workloads
- Memory management for large AI models
- I/O scheduler optimization
- GPU memory management
- Thermal management

### 2. Kernel Tuning (`scripts/kernel-tuning.sh`)
- Network stack optimization for high-throughput APIs
- File descriptor limits
- Memory overcommit settings
- Swap configuration
- Process scheduling

### 3. Docker Optimization (`configs/docker-daemon.json`)
- Container resource management
- GPU passthrough optimization
- Volume mount optimization
- Network driver selection

### 4. Database Tuning (`scripts/database-tuning.sh`)
- PostgreSQL optimization for n8n workflows
- Redis configuration for caching
- Qdrant vector database optimization
- Connection pooling

### 5. AI-Specific Optimizations (`scripts/ai-optimization.sh`)
- CUDA memory management
- Model caching strategies
- Concurrent inference optimization
- Batch processing

### 6. Monitoring Setup (`monitoring/`)
- Performance metrics collection
- Resource usage monitoring
- Bottleneck identification
- Automated alerts

### 7. Benchmarking (`benchmarks/`)
- Before/after performance tests
- Component-specific benchmarks
- End-to-end latency tests
- Throughput measurements

## Quick Start

1. **Run baseline benchmarks**:
   ```bash
   ./optimization/run-optimization.sh --benchmark-only
   ```

2. **Apply all optimizations**:
   ```bash
   sudo ./optimization/run-optimization.sh --apply-all
   ```

3. **Apply specific optimization**:
   ```bash
   sudo ./optimization/run-optimization.sh --apply hardware
   ```

4. **Compare results**:
   ```bash
   ./optimization/run-optimization.sh --compare
   ```

## Safety Features

- All changes are reversible
- Automatic backup of original configurations
- Validation before applying changes
- Rollback capability
- Performance regression detection

## Results

Performance improvements are tracked in `results/` directory:
- Baseline metrics
- Post-optimization metrics
- Comparison reports
- System configuration snapshots

## Monitoring Dashboard

Access Grafana dashboards:
- Thanos: http://thanos:3000
- Oracle1: http://oracle1:3000

## Prerequisites

- Root/sudo access for system changes
- Docker 24.0+
- NVIDIA drivers 580+
- CUDA 13.0+
- Python 3.11+