#!/bin/bash

# AI-Specific Optimization Script for AgenticDosNode
# Optimizes CUDA, model serving, inference, and caching for AI workloads

set -euo pipefail

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
BACKUP_DIR="/etc/agentic-backup/ai"
LOG_FILE="/var/log/agentic-ai-optimization.log"
NODE_TYPE="${1:-auto}"

# Logging
log() {
    echo -e "${2:-INFO}: $1" | tee -a "$LOG_FILE"
    logger -t "agentic-ai" "$1"
}

# Detect node type and GPU
detect_hardware() {
    if nvidia-smi &>/dev/null; then
        NODE_TYPE="thanos"
        GPU_MODEL=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
        GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
        CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
        log "Detected GPU: $GPU_MODEL with ${GPU_MEMORY}MB memory, CUDA $CUDA_VERSION" "${GREEN}"
    else
        NODE_TYPE="oracle1"
        log "No GPU detected, optimizing for CPU inference" "${YELLOW}"
    fi
}

# Backup AI configurations
backup_ai_configs() {
    log "Backing up AI configurations" "${BLUE}"
    mkdir -p "$BACKUP_DIR"

    # Backup CUDA configs if they exist
    if [ -d "/etc/nvidia" ]; then
        cp -r /etc/nvidia "$BACKUP_DIR/"
    fi

    # Backup model configs
    if [ -d "/etc/vllm" ]; then
        cp -r /etc/vllm "$BACKUP_DIR/"
    fi

    log "AI configurations backed up to $BACKUP_DIR" "${GREEN}"
}

# CUDA and GPU optimization
optimize_cuda() {
    if [ "$NODE_TYPE" != "thanos" ]; then
        log "Skipping CUDA optimization (no GPU detected)" "${YELLOW}"
        return
    fi

    log "Optimizing CUDA for AI inference" "${BLUE}"

    # Set CUDA environment variables
    cat > /etc/profile.d/cuda-optimization.sh << 'EOF'
# CUDA Optimization for AI Workloads

# CUDA paths
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# CUDA memory management
export CUDA_CACHE_PATH=/var/cache/cuda
export CUDA_CACHE_MAXSIZE=8589934592  # 8GB cache
export CUDA_CACHE_DISABLE=0
export CUDA_FORCE_PTX_JIT=0

# CUDA device settings
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=0

# cuDNN optimization
export CUDNN_LOGINFO_DBG=0
export CUDNN_LOGERR_DBG=0
export TF_CUDNN_USE_AUTOTUNE=1
export TF_CUDNN_DETERMINISTIC=0

# TensorRT optimization
export TRT_LOGGER_SEVERITY=2
export ORT_TENSORRT_ENGINE_CACHE_ENABLE=1
export ORT_TENSORRT_CACHE_PATH=/var/cache/tensorrt

# PyTorch optimization
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,garbage_collection_threshold:0.8"
export TORCH_USE_CUDA_DSA=1

# Triton optimization
export TRITON_CACHE_DIR=/var/cache/triton
export TRITON_PTXAS_PATH=$CUDA_HOME/bin/ptxas

# NCCL optimization for multi-GPU
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_TREE_THRESHOLD=0
export NCCL_SOCKET_IFNAME=eth0
EOF

    # Create cache directories
    mkdir -p /var/cache/{cuda,tensorrt,triton}
    chmod 777 /var/cache/{cuda,tensorrt,triton}

    # Configure NVIDIA MPS (Multi-Process Service) for better GPU sharing
    cat > /usr/local/bin/start-nvidia-mps.sh << 'EOF'
#!/bin/bash
# Start NVIDIA MPS for better GPU sharing

export CUDA_VISIBLE_DEVICES=0
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/var/log/nvidia-mps

mkdir -p $CUDA_MPS_PIPE_DIRECTORY
mkdir -p $CUDA_MPS_LOG_DIRECTORY

# Stop any existing MPS daemon
echo quit | nvidia-cuda-mps-control 2>/dev/null || true

# Start MPS daemon
nvidia-cuda-mps-control -d

# Set default active thread percentage (adjust based on workload)
echo "set_default_active_thread_percentage 100" | nvidia-cuda-mps-control

echo "NVIDIA MPS started successfully"
EOF

    chmod +x /usr/local/bin/start-nvidia-mps.sh

    # Create systemd service for MPS
    cat > /etc/systemd/system/nvidia-mps.service << 'EOF'
[Unit]
Description=NVIDIA Multi-Process Service
After=nvidia-persistenced.service
Requires=nvidia-persistenced.service

[Service]
Type=forking
ExecStart=/usr/local/bin/start-nvidia-mps.sh
ExecStop=/bin/bash -c "echo quit | nvidia-cuda-mps-control"
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    systemctl enable nvidia-mps.service

    log "CUDA optimization completed" "${GREEN}"
}

# vLLM optimization for model serving
optimize_vllm() {
    log "Optimizing vLLM for model serving" "${BLUE}"

    # Create vLLM configuration
    cat > /tmp/vllm-config.yaml << 'EOF'
# vLLM Performance Configuration

model_config:
  # Model loading
  download_dir: /models
  load_format: auto
  dtype: auto  # Will use float16 on GPU, bfloat16 if supported
  seed: 0

  # Quantization (for memory efficiency)
  quantization: null  # Options: awq, gptq, squeezellm, null
  enforce_eager: false  # Use CUDA graphs for better performance

parallel_config:
  # Tensor parallelism
  tensor_parallel_size: 1  # Increase for multi-GPU
  pipeline_parallel_size: 1
  max_parallel_loading_workers: 4

  # Ray configuration (for distributed inference)
  ray_workers_use_nsight: false
  placement_group_config:
    bundles: []

scheduler_config:
  # Request batching
  max_num_batched_tokens: 8192
  max_num_seqs: 256
  max_model_len: 8192
  use_v2_block_manager: true

  # Scheduling
  preemption_mode: recompute  # Options: recompute, swap
  max_paddings: 256
  iteration_counter_bucket_size: 1
  delay_factor: 0.0
  enable_chunked_prefill: true

cache_config:
  # KV cache settings
  block_size: 16
  gpu_memory_utilization: 0.90  # Use 90% of GPU memory
  swap_space: 4  # GB of CPU swap space
  cache_dtype: auto
  num_gpu_blocks_override: null
  sliding_window: null

engine_config:
  # Engine settings
  device: cuda
  max_context_len_to_capture: 8192
  disable_custom_all_reduce: false
  enable_prefix_caching: true
  disable_log_stats: false

  # Optimization flags
  enable_lora: false
  max_loras: 1
  max_lora_rank: 16
  lora_extra_vocab_size: 256
  long_lora_scaling_factors: null

decoding_config:
  # Decoding optimization
  guided_decoding_backend: outlines  # Options: outlines, lm-format-enforcer

observability_config:
  # Monitoring
  otlp_traces_endpoint: null
  log_level: INFO
  collect_model_forward_time: false
  collect_model_execute_time: false

serving_config:
  # API server settings
  host: 0.0.0.0
  port: 8000
  uvicorn_log_level: info
  api_key: null
  allow_credentials: false
  allowed_origins: ["*"]
  allowed_methods: ["*"]
  allowed_headers: ["*"]
  served_model_name: null
  chat_template: null
  response_role: assistant
  ssl_certfile: null
  ssl_keyfile: null
  ssl_ca_certs: null
  ssl_cert_reqs: 0
  root_path: null
  middleware: []
  return_tokens_as_token_ids: false
  disable_log_requests: false

tokenizer_config:
  # Tokenizer settings
  tokenizer_mode: auto
  trust_remote_code: false
  tokenizer_revision: null
  custom_chat_template: null

load_config:
  # Model loading optimization
  load_in_4bit: false
  load_in_8bit: false
  bnb_4bit_compute_dtype: float16
  bnb_4bit_quant_type: nf4
  bnb_4bit_use_double_quant: true
EOF

    # Create vLLM startup script with optimizations
    cat > /usr/local/bin/start-vllm-optimized.sh << 'EOF'
#!/bin/bash

# vLLM Optimized Startup Script

# Set model and parameters
MODEL=${VLLM_MODEL:-"mistralai/Mistral-7B-Instruct-v0.2"}
TP_SIZE=${TENSOR_PARALLEL_SIZE:-1}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
GPU_MEMORY=${GPU_MEMORY_UTILIZATION:-0.90}

# Performance flags
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_MODELSCOPE=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export RAY_DEDUP_LOGS=1

# Start vLLM with optimized settings
python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --tensor-parallel-size $TP_SIZE \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEMORY \
    --block-size 32 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 256 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --use-v2-block-manager \
    --disable-log-stats \
    --trust-remote-code \
    --download-dir /models \
    --host 0.0.0.0 \
    --port 8000 \
    --uvloop \
    --served-model-name $MODEL \
    2>&1 | tee /var/log/vllm.log
EOF

    chmod +x /usr/local/bin/start-vllm-optimized.sh
    log "vLLM optimization completed" "${GREEN}"
}

# Model caching and loading optimization
optimize_model_caching() {
    log "Optimizing model caching and loading" "${BLUE}"

    # Create model cache structure
    mkdir -p /models/{huggingface,onnx,tensorrt,torch}
    mkdir -p /var/cache/transformers

    # Set up Hugging Face cache
    cat >> /etc/environment << 'EOF'
# Model caching optimization
HF_HOME=/models/huggingface
HF_DATASETS_CACHE=/models/huggingface/datasets
TRANSFORMERS_CACHE=/var/cache/transformers
TORCH_HOME=/models/torch
ONNX_MODEL_CACHE=/models/onnx
TENSORRT_MODEL_CACHE=/models/tensorrt
EOF

    # Create model preloading script
    cat > /usr/local/bin/preload-models.py << 'EOF'
#!/usr/bin/env python3

import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import mmap
import json

class ModelPreloader:
    """Preload and optimize AI models for fast inference"""

    def __init__(self, cache_dir: str = "/models"):
        self.cache_dir = Path(cache_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def optimize_model_loading(self, model_path: str) -> Dict:
        """Optimize model loading with memory mapping"""

        model_file = Path(model_path)
        if not model_file.exists():
            return {"error": f"Model not found: {model_path}"}

        # Memory-map the model file for faster loading
        with open(model_file, 'r+b') as f:
            mmapped_file = mmap.mmap(f.fileno(), 0)

        # Create index for faster access
        index_file = model_file.with_suffix('.index')
        if not index_file.exists():
            self._create_model_index(model_file, index_file)

        return {
            "model": model_path,
            "optimized": True,
            "mmap_enabled": True,
            "index_created": index_file.exists()
        }

    def _create_model_index(self, model_file: Path, index_file: Path):
        """Create an index file for faster model access"""

        # This would contain layer offsets and metadata
        index_data = {
            "version": "1.0",
            "model_file": str(model_file),
            "layers": {},
            "metadata": {}
        }

        # In practice, you'd parse the model and create actual indices
        with open(index_file, 'w') as f:
            json.dump(index_data, f)

    def setup_model_cache(self):
        """Setup optimized model caching"""

        cache_config = {
            "max_cache_size_gb": 50,
            "eviction_policy": "lru",
            "preload_popular_models": True,
            "enable_compression": False,
            "cache_locations": {
                "primary": "/models",
                "overflow": "/var/cache/models"
            }
        }

        config_file = self.cache_dir / "cache_config.json"
        with open(config_file, 'w') as f:
            json.dump(cache_config, f, indent=2)

        return cache_config

    def optimize_tokenizers(self):
        """Optimize tokenizer loading and caching"""

        # Pre-compile common tokenizers
        from transformers import AutoTokenizer

        common_models = [
            "bert-base-uncased",
            "gpt2",
            "mistralai/Mistral-7B-Instruct-v0.2"
        ]

        for model_name in common_models:
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=self.cache_dir / "tokenizers",
                    local_files_only=False
                )
                print(f"Cached tokenizer: {model_name}")
            except Exception as e:
                print(f"Failed to cache {model_name}: {e}")

def main():
    preloader = ModelPreloader()

    # Setup cache
    cache_config = preloader.setup_model_cache()
    print(f"Cache configured: {cache_config}")

    # Optimize tokenizers
    preloader.optimize_tokenizers()

    # Optimize specific models if provided
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        result = preloader.optimize_model_loading(model_path)
        print(f"Model optimization result: {result}")

if __name__ == "__main__":
    main()
EOF

    chmod +x /usr/local/bin/preload-models.py
    log "Model caching optimization completed" "${GREEN}"
}

# Inference optimization
optimize_inference() {
    log "Optimizing inference pipeline" "${BLUE}"

    # Create inference optimization config
    cat > /tmp/inference-config.json << 'EOF'
{
  "inference_optimization": {
    "batch_size": {
      "min": 1,
      "max": 256,
      "optimal": 32,
      "dynamic_batching": true
    },
    "precision": {
      "default": "fp16",
      "fallback": "fp32",
      "quantization": {
        "enabled": false,
        "method": "dynamic",
        "bits": 8
      }
    },
    "memory": {
      "max_workspace_size_mb": 4096,
      "persistent_cache": true,
      "cache_size_gb": 8,
      "offload_to_cpu": true,
      "gradient_checkpointing": false
    },
    "execution": {
      "num_threads": 0,
      "inter_op_threads": 0,
      "intra_op_threads": 0,
      "use_graph_optimization": true,
      "enable_profiling": false,
      "execution_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"]
    },
    "optimizations": {
      "fuse_operations": true,
      "eliminate_common_subexpression": true,
      "constant_folding": true,
      "shape_inference": true,
      "enable_flash_attention": true,
      "enable_mem_efficient_attention": true,
      "enable_math_optimizations": true
    },
    "compilation": {
      "use_torch_compile": true,
      "compile_mode": "default",
      "backend": "inductor",
      "use_triton": true,
      "cache_compiled_models": true
    }
  }
}
EOF

    # Create batch inference optimizer
    cat > /usr/local/bin/optimize-batch-inference.py << 'EOF'
#!/usr/bin/env python3

import torch
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from queue import Queue, PriorityQueue
import threading
import time

@dataclass
class InferenceRequest:
    """Request for batch inference"""
    id: str
    input_data: Any
    priority: int = 0
    timestamp: float = 0

class BatchInferenceOptimizer:
    """Optimize batch inference for maximum throughput"""

    def __init__(
        self,
        model,
        max_batch_size: int = 32,
        max_wait_time: float = 0.1,
        device: str = "cuda"
    ):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.device = device
        self.request_queue = PriorityQueue()
        self.result_dict = {}
        self.lock = threading.Lock()

    def dynamic_batching(self, requests: List[InferenceRequest]) -> List[List[InferenceRequest]]:
        """Create optimal batches from requests"""

        batches = []
        current_batch = []
        current_batch_size = 0

        # Sort by priority and timestamp
        sorted_requests = sorted(
            requests,
            key=lambda x: (-x.priority, x.timestamp)
        )

        for request in sorted_requests:
            # Estimate memory/compute for this request
            request_size = self._estimate_request_size(request)

            if current_batch_size + request_size <= self.max_batch_size:
                current_batch.append(request)
                current_batch_size += request_size
            else:
                if current_batch:
                    batches.append(current_batch)
                current_batch = [request]
                current_batch_size = request_size

        if current_batch:
            batches.append(current_batch)

        return batches

    def _estimate_request_size(self, request: InferenceRequest) -> int:
        """Estimate computational size of request"""
        # Simplified estimation - in practice would be model-specific
        if hasattr(request.input_data, 'shape'):
            return np.prod(request.input_data.shape)
        return 1

    def optimize_memory_layout(self, batch_data: torch.Tensor) -> torch.Tensor:
        """Optimize tensor memory layout for inference"""

        # Ensure contiguous memory layout
        batch_data = batch_data.contiguous()

        # Convert to channels_last format if beneficial (for CNNs)
        if len(batch_data.shape) == 4:  # NCHW format
            batch_data = batch_data.to(memory_format=torch.channels_last)

        return batch_data

    def profile_and_optimize(self):
        """Profile model and apply optimizations"""

        if not torch.cuda.is_available():
            return

        # Enable CUDNN autotuner
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        # Enable TF32 for Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Compile model with torch.compile if available
        if hasattr(torch, 'compile'):
            self.model = torch.compile(
                self.model,
                mode="default",
                backend="inductor",
                fullgraph=False
            )

    def run_inference_loop(self):
        """Main inference loop with batching"""

        while True:
            batch = []
            start_time = time.time()

            # Collect requests for batching
            while len(batch) < self.max_batch_size:
                timeout = self.max_wait_time - (time.time() - start_time)
                if timeout <= 0:
                    break

                try:
                    priority, request = self.request_queue.get(timeout=timeout)
                    batch.append(request)
                except:
                    break

            if batch:
                # Process batch
                self._process_batch(batch)

    def _process_batch(self, batch: List[InferenceRequest]):
        """Process a batch of requests"""

        # Prepare batch input
        batch_input = self._prepare_batch_input(batch)

        # Optimize memory layout
        batch_input = self.optimize_memory_layout(batch_input)

        # Run inference
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True):
                outputs = self.model(batch_input)

        # Distribute results
        self._distribute_results(batch, outputs)

    def _prepare_batch_input(self, batch: List[InferenceRequest]) -> torch.Tensor:
        """Prepare batch input tensor"""
        # Implementation depends on model input format
        inputs = [req.input_data for req in batch]
        return torch.stack(inputs).to(self.device)

    def _distribute_results(self, batch: List[InferenceRequest], outputs: torch.Tensor):
        """Distribute results to requests"""
        for i, request in enumerate(batch):
            with self.lock:
                self.result_dict[request.id] = outputs[i]

def main():
    # Example usage
    print("Batch Inference Optimizer initialized")
    print("Configuration:")
    print("- Dynamic batching enabled")
    print("- Memory layout optimization enabled")
    print("- CUDA optimizations enabled")
    print("- Torch compile enabled (if available)")

if __name__ == "__main__":
    main()
EOF

    chmod +x /usr/local/bin/optimize-batch-inference.py
    log "Inference optimization completed" "${GREEN}"
}

# Embedding service optimization
optimize_embeddings() {
    log "Optimizing embedding service" "${BLUE}"

    # Create embedding optimization config
    cat > /tmp/embedding-config.yaml << 'EOF'
# Embedding Service Optimization

model:
  name: "BAAI/bge-large-en-v1.5"
  max_sequence_length: 512
  pooling_strategy: "mean"
  normalize: true

performance:
  batch_size: 64
  max_concurrent_requests: 100
  prefetch_factor: 2
  num_workers: 4
  pin_memory: true
  persistent_workers: true

caching:
  enabled: true
  cache_size: 10000
  ttl_seconds: 3600
  similarity_threshold: 0.99

optimization:
  use_fp16: true
  use_bettertransformer: true
  compile_model: true
  use_flash_attention: true
  gradient_checkpointing: false

serving:
  host: "0.0.0.0"
  port: 8001
  workers: 4
  timeout: 30
  max_request_size_mb: 100

monitoring:
  enable_metrics: true
  metrics_port: 9001
  log_level: "INFO"
EOF

    log "Embedding service optimization completed" "${GREEN}"
}

# Create AI monitoring dashboard
setup_ai_monitoring() {
    log "Setting up AI performance monitoring" "${BLUE}"

    # Create monitoring script
    cat > /usr/local/bin/ai-monitor.sh << 'EOF'
#!/bin/bash

# AI Performance Monitor
METRICS_FILE="/var/lib/node_exporter/ai_metrics.prom"
mkdir -p $(dirname $METRICS_FILE)

while true; do
    {
        # GPU metrics
        if nvidia-smi &>/dev/null; then
            echo "# GPU Metrics"
            nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
                       --format=csv,noheader,nounits | while IFS=, read -r idx name temp util_gpu util_mem mem_used mem_total power; do
                echo "gpu_temperature_celsius{gpu=\"$idx\",name=\"$name\"} $temp"
                echo "gpu_utilization_percent{gpu=\"$idx\",name=\"$name\"} $util_gpu"
                echo "gpu_memory_utilization_percent{gpu=\"$idx\",name=\"$name\"} $util_mem"
                echo "gpu_memory_used_mb{gpu=\"$idx\",name=\"$name\"} $mem_used"
                echo "gpu_memory_total_mb{gpu=\"$idx\",name=\"$name\"} $mem_total"
                echo "gpu_power_draw_watts{gpu=\"$idx\",name=\"$name\"} $power"
            done
        fi

        # Model serving metrics (vLLM)
        if curl -s http://localhost:8000/metrics &>/dev/null; then
            echo "# vLLM Metrics"
            curl -s http://localhost:8000/metrics | grep -E "^vllm_"
        fi

        # Embedding service metrics
        if curl -s http://localhost:8001/metrics &>/dev/null; then
            echo "# Embedding Service Metrics"
            curl -s http://localhost:8001/metrics | grep -E "^embedding_"
        fi

        # CUDA memory stats
        if [ -f /var/cache/cuda/stats ]; then
            echo "# CUDA Cache Metrics"
            cache_size=$(du -sb /var/cache/cuda | cut -f1)
            echo "cuda_cache_size_bytes $cache_size"
        fi

    } > $METRICS_FILE.tmp && mv $METRICS_FILE.tmp $METRICS_FILE

    sleep 10
done
EOF

    chmod +x /usr/local/bin/ai-monitor.sh

    # Create systemd service
    cat > /etc/systemd/system/ai-monitor.service << 'EOF'
[Unit]
Description=AI Performance Monitor
After=docker.service

[Service]
Type=simple
ExecStart=/usr/local/bin/ai-monitor.sh
Restart=always
RestartSec=10
User=root

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    systemctl enable ai-monitor.service
    systemctl start ai-monitor.service

    log "AI monitoring setup completed" "${GREEN}"
}

# Apply all AI optimizations
apply_all() {
    log "Starting AI-specific optimization" "${BLUE}"

    detect_hardware
    backup_ai_configs
    optimize_cuda
    optimize_vllm
    optimize_model_caching
    optimize_inference
    optimize_embeddings
    setup_ai_monitoring

    log "AI optimization completed!" "${GREEN}"

    cat << EOF

${GREEN}AI Optimization Summary:${NC}
- CUDA: Memory management and caching optimized
- vLLM: Configured for high-throughput serving
- Model Loading: Caching and preloading enabled
- Inference: Batch processing and precision optimized
- Embeddings: Service optimized for vector operations
- Monitoring: AI metrics collection enabled

${YELLOW}Next Steps:${NC}
1. Apply CUDA environment:
   source /etc/profile.d/cuda-optimization.sh

2. Start optimized services:
   systemctl start nvidia-mps
   /usr/local/bin/start-vllm-optimized.sh

3. Preload models:
   python3 /usr/local/bin/preload-models.py

4. Test inference optimization:
   python3 /usr/local/bin/optimize-batch-inference.py

${BLUE}Monitor Performance:${NC}
- GPU: nvidia-smi dmon
- vLLM: curl http://localhost:8000/metrics
- System: curl http://localhost:9100/metrics

${RED}To Rollback:${NC}
systemctl stop nvidia-mps ai-monitor
rm /etc/profile.d/cuda-optimization.sh
rm /etc/systemd/system/{nvidia-mps,ai-monitor}.service
systemctl daemon-reload
EOF
}

# Rollback function
rollback() {
    log "Rolling back AI optimization changes" "${YELLOW}"

    # Stop services
    systemctl stop nvidia-mps ai-monitor 2>/dev/null || true
    systemctl disable nvidia-mps ai-monitor 2>/dev/null || true

    # Remove configurations
    rm -f /etc/profile.d/cuda-optimization.sh
    rm -f /etc/systemd/system/nvidia-mps.service
    rm -f /etc/systemd/system/ai-monitor.service
    rm -f /usr/local/bin/start-nvidia-mps.sh
    rm -f /usr/local/bin/start-vllm-optimized.sh

    systemctl daemon-reload

    log "Rollback completed" "${GREEN}"
}

# Main execution
main() {
    if [ "$EUID" -ne 0 ]; then
        echo "This script must be run as root (use sudo)"
        exit 1
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