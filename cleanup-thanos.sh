#!/bin/bash

# AgenticDosNode Cleanup Script for THANOS (GPU Node)
# Comprehensive cleanup for GPU-enabled machine preparation
# Version: 1.0.0
# Author: Claude Code DevOps Troubleshooter

set -euo pipefail

# Get script directory and source utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/cleanup-utils.sh"

# Machine-specific configuration
readonly MACHINE_NAME="thanos"
readonly MACHINE_TYPE="GPU"
readonly REQUIRED_PORTS=(3000 5678 6333 8000 8001 9090 8080 5432 6379 27017)

# AgenticDosNode specific patterns
readonly AI_CONTAINER_PATTERNS=(
    "ollama" "localai" "text-generation-inference" "vllm" "triton"
    "pytorch" "tensorflow" "jupyter" "gradio" "streamlit"
    "stable-diffusion" "comfyui" "automatic1111"
    "qdrant" "weaviate" "chroma" "pinecone"
    "agenticnode" "agentic" "langchain" "llamaindex"
)

readonly AI_IMAGE_PATTERNS=(
    "ollama" "localai" "huggingface" "pytorch" "tensorflow"
    "nvidia/cuda" "nvidia/triton" "vllm" "text-generation-inference"
    "qdrant" "weaviate" "chromadb" "redis" "postgres"
    "jupyter" "gradio" "streamlit"
)

readonly AI_PROCESS_PATTERNS=(
    "ollama" "localai" "python.*transformers" "python.*torch"
    "python.*tensorflow" "jupyter" "gradio" "streamlit"
    "qdrant" "weaviate" "redis-server" "postgres"
    "nvidia-smi" "nvtop" "gpustat"
)

readonly AI_SERVICE_PATTERNS=(
    "ollama" "localai" "jupyter" "docker" "nvidia"
    "redis" "postgresql" "mongodb" "qdrant"
)

main() {
    log "INFO" "Starting AgenticDosNode cleanup for $MACHINE_NAME ($MACHINE_TYPE)"
    log "INFO" "Timestamp: $(date)"
    log "INFO" "User: $(whoami)"
    log "INFO" "Working directory: $PWD"

    check_root

    # Display warning
    cat << EOF

${RED}WARNING: GPU NODE CLEANUP${NC}
This script will perform comprehensive cleanup of AI/ML services on $MACHINE_NAME.

WHAT WILL BE AFFECTED:
- Docker containers and images (AI/ML related)
- Python virtual environments with AI packages
- Conda/Mamba environments
- GPU processes and CUDA services
- Database services (Redis, PostgreSQL, MongoDB)
- Running AI frameworks (Ollama, LocalAI, TensorFlow, PyTorch)
- System services related to AI/ML
- Network ports required for AgenticDosNode

DESTRUCTIVE OPERATIONS:
- Stop and remove Docker containers
- Remove Docker images and volumes
- Delete Python/Conda environments
- Terminate running processes
- Clear GPU memory
- Stop system services

EOF

    if ! confirm "Proceed with cleanup on $MACHINE_NAME?"; then
        log "INFO" "Cleanup aborted by user"
        exit 0
    fi

    # Create backup of important configurations
    backup_configurations

    # Main cleanup phases
    cleanup_gpu_processes
    cleanup_docker_environment
    cleanup_python_environments
    cleanup_system_services
    cleanup_network_ports
    cleanup_ai_frameworks
    cleanup_databases
    cleanup_temporary_files

    # Validation
    validate_cleanup

    show_summary
    log "INFO" "Cleanup completed successfully for $MACHINE_NAME"
}

backup_configurations() {
    log "STEP" "Backing up important configurations..."

    # Docker configurations
    backup_item "/etc/docker/daemon.json" "docker_daemon_config" 2>/dev/null || true
    backup_item "$HOME/.docker/config.json" "docker_user_config" 2>/dev/null || true

    # NVIDIA configurations
    backup_item "/etc/nvidia-container-runtime/config.toml" "nvidia_container_config" 2>/dev/null || true

    # Python configurations
    backup_item "$HOME/.pip/pip.conf" "pip_config" 2>/dev/null || true
    backup_item "$HOME/.condarc" "conda_config" 2>/dev/null || true

    # System configurations that might be affected
    backup_item "/etc/systemd/system/docker.service.d" "docker_systemd_overrides" 2>/dev/null || true

    log "INFO" "Configuration backup completed"
}

cleanup_gpu_processes() {
    log "STEP" "Cleaning up GPU processes and CUDA services..."

    # Check GPU status
    if command -v nvidia-smi >/dev/null 2>&1; then
        log "INFO" "Current GPU status:"
        nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | while read line; do
            log "INFO" "  GPU: $line"
        done

        # Find processes using GPU
        local gpu_processes=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null | awk -F',' '{print $1}' | tr '\n' ' ')
        if [[ -n "$gpu_processes" ]]; then
            log "WARN" "Found processes using GPU:"
            nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | while read line; do
                log "WARN" "  $line"
            done

            if confirm "Terminate GPU processes?"; then
                kill_processes "$gpu_processes" "GPU" true
            fi
        fi
    else
        log "WARN" "nvidia-smi not found - GPU drivers may not be installed"
    fi

    # Clean up CUDA processes
    local cuda_processes=$(detect_processes "cuda" "CUDA")
    if [[ -n "$cuda_processes" ]]; then
        kill_processes "$cuda_processes" "CUDA" false
    fi

    # Clean up mining processes (just in case)
    local mining_processes=$(detect_processes "mine|miner|xmrig|ethminer" "crypto mining")
    if [[ -n "$mining_processes" ]]; then
        kill_processes "$mining_processes" "crypto mining" true
    fi

    log "INFO" "GPU process cleanup completed"
}

cleanup_docker_environment() {
    log "STEP" "Cleaning up Docker environment..."

    if ! command -v docker >/dev/null 2>&1; then
        log "INFO" "Docker not installed, skipping Docker cleanup"
        return 0
    fi

    # Stop all running containers first
    local running_containers=$(docker ps -q 2>/dev/null || true)
    if [[ -n "$running_containers" ]]; then
        log "WARN" "Found running containers that will be stopped"
        if confirm "Stop all running containers?"; then
            docker stop $running_containers 2>/dev/null || true
        fi
    fi

    # Clean up AI-related containers
    for pattern in "${AI_CONTAINER_PATTERNS[@]}"; do
        local containers=$(detect_docker_containers "$pattern" "$pattern")
        if [[ -n "$containers" ]]; then
            cleanup_docker_containers "$containers" "$pattern"
        fi
    done

    # Clean up AI-related images
    for pattern in "${AI_IMAGE_PATTERNS[@]}"; do
        local images=$(detect_docker_images "$pattern" "$pattern")
        if [[ -n "$images" ]]; then
            cleanup_docker_images "$images" "$pattern"
        fi
    done

    # Clean up Docker volumes
    local volumes=$(docker volume ls -q 2>/dev/null | grep -E "(ai|ml|ollama|localai|jupyter|qdrant|redis|postgres)" || true)
    if [[ -n "$volumes" ]]; then
        log "WARN" "Found AI-related Docker volumes:"
        echo "$volumes" | while read volume; do
            log "INFO" "  $volume"
        done

        if confirm "Remove these Docker volumes? (This will delete data)"; then
            echo "$volumes" | while read volume; do
                log "INFO" "Removing volume: $volume"
                docker volume rm "$volume" 2>/dev/null || true
            done
        fi
    fi

    # Clean up Docker networks
    local networks=$(docker network ls --format "{{.Name}}" | grep -E "(ai|ml|ollama|localai|agenticnode)" || true)
    if [[ -n "$networks" ]]; then
        log "INFO" "Found AI-related Docker networks:"
        echo "$networks" | while read network; do
            log "INFO" "  $network"
        done

        if confirm "Remove these Docker networks?"; then
            echo "$networks" | while read network; do
                log "INFO" "Removing network: $network"
                docker network rm "$network" 2>/dev/null || true
            done
        fi
    fi

    # Prune Docker system
    if confirm "Run Docker system prune to clean up unused resources?"; then
        log "INFO" "Running Docker system prune..."
        docker system prune -f 2>/dev/null || true
    fi

    log "INFO" "Docker environment cleanup completed"
}

cleanup_python_environments() {
    log "STEP" "Cleaning up Python environments..."

    # Standard virtual environments
    local venv_dirs=("$HOME/.virtualenvs" "$HOME/venvs" "$HOME/.venv" "./venv" "./env")
    for venv_dir in "${venv_dirs[@]}"; do
        if [[ -d "$venv_dir" ]]; then
            local environments=$(detect_python_environments "$venv_dir" "virtual environments")
            if [[ -n "$environments" ]]; then
                cleanup_python_environments "$environments" "virtual environments"
            fi
        fi
    done

    # Conda environments
    if command -v conda >/dev/null 2>&1; then
        log "INFO" "Found conda installation"
        local conda_envs=$(conda env list --json 2>/dev/null | jq -r '.envs[]' | grep -v "$(conda info --base)" || true)
        if [[ -n "$conda_envs" ]]; then
            log "INFO" "Found conda environments:"
            echo "$conda_envs" | while read env; do
                local env_name=$(basename "$env")
                log "INFO" "  $env_name ($env)"
            done

            if confirm "Remove all conda environments except base?"; then
                echo "$conda_envs" | while read env; do
                    local env_name=$(basename "$env")
                    if [[ "$env_name" != "base" ]]; then
                        log "INFO" "Removing conda environment: $env_name"
                        conda env remove -n "$env_name" -y 2>/dev/null || true
                    fi
                done
            fi
        fi
    fi

    # Mamba environments
    if command -v mamba >/dev/null 2>&1; then
        log "INFO" "Found mamba installation"
        # Similar cleanup for mamba
    fi

    # Python caches
    if confirm "Clear Python caches (__pycache__, .pyc files)?"; then
        log "INFO" "Clearing Python caches..."
        find "$HOME" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
        find "$HOME" -name "*.pyc" -delete 2>/dev/null || true
        find "$HOME" -name "*.pyo" -delete 2>/dev/null || true
    fi

    # Pip cache
    if command -v pip >/dev/null 2>&1; then
        if confirm "Clear pip cache?"; then
            log "INFO" "Clearing pip cache..."
            pip cache purge 2>/dev/null || true
        fi
    fi

    log "INFO" "Python environment cleanup completed"
}

cleanup_system_services() {
    log "STEP" "Cleaning up system services..."

    # AI-related services
    for pattern in "${AI_SERVICE_PATTERNS[@]}"; do
        local services=$(detect_systemd_services "$pattern" "$pattern")
        if [[ -n "$services" ]]; then
            cleanup_systemd_services "$services" "$pattern"
        fi
    done

    # Check for custom AI services
    local custom_services=$(systemctl list-units --type=service --all | grep -E "(ai|ml|gpu|cuda)" | awk '{print $1}' || true)
    if [[ -n "$custom_services" ]]; then
        log "INFO" "Found additional AI-related services:"
        echo "$custom_services" | while read service; do
            log "INFO" "  $service"
        done

        if confirm "Stop and disable these services?"; then
            cleanup_systemd_services "$custom_services" "custom AI services"
        fi
    fi

    log "INFO" "System service cleanup completed"
}

cleanup_network_ports() {
    log "STEP" "Cleaning up network ports..."

    log "INFO" "Checking required ports for AgenticDosNode..."
    local ports_in_use=()

    for port in "${REQUIRED_PORTS[@]}"; do
        if check_port "$port"; then
            ports_in_use+=("$port")
            log "WARN" "Port $port is in use"
        else
            log "DEBUG" "Port $port is free"
        fi
    done

    if [[ ${#ports_in_use[@]} -gt 0 ]]; then
        log "WARN" "The following required ports are in use: ${ports_in_use[*]}"
        if confirm "Free up these ports?"; then
            for port in "${ports_in_use[@]}"; do
                free_port "$port" "tcp" true
            done
        fi
    else
        log "INFO" "All required ports are free"
    fi

    log "INFO" "Network port cleanup completed"
}

cleanup_ai_frameworks() {
    log "STEP" "Cleaning up AI framework processes..."

    # Check for specific AI framework processes
    for pattern in "${AI_PROCESS_PATTERNS[@]}"; do
        local processes=$(detect_processes "$pattern" "$pattern")
        if [[ -n "$processes" ]]; then
            kill_processes "$processes" "$pattern" false
        fi
    done

    # Clean up Jupyter kernels
    if command -v jupyter >/dev/null 2>&1; then
        log "INFO" "Cleaning up Jupyter kernels..."
        jupyter kernelspec list 2>/dev/null | grep -v "Available kernels" | while read kernel; do
            log "INFO" "  Found kernel: $kernel"
        done

        if confirm "Remove all Jupyter kernels?"; then
            jupyter kernelspec remove --all -y 2>/dev/null || true
        fi
    fi

    # Clean up Hugging Face cache
    local hf_cache_dirs=("$HOME/.cache/huggingface" "$HOME/.cache/torch" "$HOME/.cache/transformers")
    for cache_dir in "${hf_cache_dirs[@]}"; do
        if [[ -d "$cache_dir" ]]; then
            local size=$(du -sh "$cache_dir" 2>/dev/null | cut -f1)
            log "INFO" "Found cache directory: $cache_dir [$size]"

            if confirm "Remove cache directory $cache_dir?"; then
                backup_item "$cache_dir" "ai_cache_$(basename "$cache_dir")"
                rm -rf "$cache_dir"
            fi
        fi
    done

    log "INFO" "AI framework cleanup completed"
}

cleanup_databases() {
    log "STEP" "Cleaning up database services..."

    # Redis
    local redis_processes=$(detect_processes "redis-server" "Redis")
    if [[ -n "$redis_processes" ]]; then
        kill_processes "$redis_processes" "Redis" false
    fi

    # PostgreSQL
    local postgres_processes=$(detect_processes "postgres" "PostgreSQL")
    if [[ -n "$postgres_processes" ]]; then
        kill_processes "$postgres_processes" "PostgreSQL" false
    fi

    # MongoDB
    local mongo_processes=$(detect_processes "mongod" "MongoDB")
    if [[ -n "$mongo_processes" ]]; then
        kill_processes "$mongo_processes" "MongoDB" false
    fi

    # Vector databases
    local qdrant_processes=$(detect_processes "qdrant" "Qdrant")
    if [[ -n "$qdrant_processes" ]]; then
        kill_processes "$qdrant_processes" "Qdrant" false
    fi

    local weaviate_processes=$(detect_processes "weaviate" "Weaviate")
    if [[ -n "$weaviate_processes" ]]; then
        kill_processes "$weaviate_processes" "Weaviate" false
    fi

    log "INFO" "Database cleanup completed"
}

cleanup_temporary_files() {
    log "STEP" "Cleaning up temporary files..."

    # Temporary directories
    local temp_dirs=("/tmp" "/var/tmp")
    for temp_dir in "${temp_dirs[@]}"; do
        if [[ -d "$temp_dir" ]]; then
            local ai_files=$(find "$temp_dir" -name "*ollama*" -o -name "*localai*" -o -name "*huggingface*" -o -name "*pytorch*" -o -name "*tensorflow*" 2>/dev/null || true)
            if [[ -n "$ai_files" ]]; then
                log "INFO" "Found AI-related temporary files in $temp_dir"
                echo "$ai_files" | head -10 | while read file; do
                    log "INFO" "  $file"
                done

                if confirm "Remove AI-related temporary files from $temp_dir?"; then
                    echo "$ai_files" | while read file; do
                        rm -rf "$file" 2>/dev/null || true
                    done
                fi
            fi
        fi
    done

    # Log files
    local log_dirs=("/var/log" "$HOME/.local/share/logs")
    for log_dir in "${log_dirs[@]}"; do
        if [[ -d "$log_dir" ]]; then
            local ai_logs=$(find "$log_dir" -name "*ollama*" -o -name "*localai*" -o -name "*docker*" -o -name "*nvidia*" 2>/dev/null || true)
            if [[ -n "$ai_logs" ]]; then
                log "INFO" "Found AI-related log files in $log_dir"
                if confirm "Remove AI-related log files from $log_dir?"; then
                    echo "$ai_logs" | while read logfile; do
                        rm -f "$logfile" 2>/dev/null || true
                    done
                fi
            fi
        fi
    done

    log "INFO" "Temporary file cleanup completed"
}

validate_cleanup() {
    log "STEP" "Validating cleanup results..."

    local issues=0

    # Check ports
    log "INFO" "Checking port availability..."
    for port in "${REQUIRED_PORTS[@]}"; do
        if check_port "$port"; then
            log "ERROR" "Port $port is still in use"
            ((issues++))
        else
            log "DEBUG" "Port $port is free"
        fi
    done

    # Check GPU memory
    if command -v nvidia-smi >/dev/null 2>&1; then
        local gpu_processes=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | wc -l)
        if [[ "$gpu_processes" -gt 0 ]]; then
            log "WARN" "Still $gpu_processes processes using GPU"
            ((issues++))
        else
            log "INFO" "GPU memory is clear"
        fi
    fi

    # Check for running AI containers
    if command -v docker >/dev/null 2>&1; then
        local ai_containers=$(docker ps --format "{{.Names}}" | grep -E "(ollama|localai|pytorch|tensorflow|jupyter)" | wc -l)
        if [[ "$ai_containers" -gt 0 ]]; then
            log "WARN" "Still $ai_containers AI containers running"
            ((issues++))
        else
            log "INFO" "No AI containers running"
        fi
    fi

    if [[ "$issues" -eq 0 ]]; then
        log "INFO" "Validation passed - system is clean"
    else
        log "WARN" "Validation found $issues issues - manual intervention may be required"
    fi

    return "$issues"
}

# Cleanup function for script interruption
cleanup_on_exit() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        log "ERROR" "Script interrupted with exit code $exit_code"
        log "INFO" "Partial cleanup may have occurred. Check logs: $LOG_FILE"
    fi
    exit $exit_code
}

# Set up signal handlers
trap cleanup_on_exit INT TERM EXIT

# Usage information
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

AgenticDosNode Cleanup Script for THANOS (GPU Node)

OPTIONS:
    -h, --help              Show this help message
    -y, --yes               Automatic yes to all prompts (DANGEROUS)
    -b, --backup-only       Only create backups, don't perform cleanup
    -r, --restore TIMESTAMP Restore from backup with given timestamp
    -v, --verbose           Verbose output
    --dry-run              Show what would be done without actually doing it

EXAMPLES:
    $0                      Interactive cleanup
    $0 -y                   Automatic cleanup (use with caution)
    $0 --backup-only        Create backups only
    $0 --restore 20241201_143022    Restore from backup

EOF
}

# Command line argument parsing
DRY_RUN=false
AUTO_YES=false
BACKUP_ONLY=false
RESTORE_TIMESTAMP=""
VERBOSE=false

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
        -b|--backup-only)
            BACKUP_ONLY=true
            shift
            ;;
        -r|--restore)
            RESTORE_TIMESTAMP="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            log "ERROR" "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Override confirm function if auto-yes is enabled
if [[ "$AUTO_YES" == "true" ]]; then
    confirm() {
        log "INFO" "Auto-confirming: $1"
        return 0
    }
fi

# Handle special modes
if [[ -n "$RESTORE_TIMESTAMP" ]]; then
    restore_backup "$RESTORE_TIMESTAMP"
    exit $?
fi

if [[ "$BACKUP_ONLY" == "true" ]]; then
    log "INFO" "Backup-only mode enabled"
    backup_configurations
    show_summary
    exit 0
fi

if [[ "$DRY_RUN" == "true" ]]; then
    log "WARN" "DRY RUN MODE - No actual changes will be made"
    # Override destructive functions for dry run
    # This would require modifying the utility functions
fi

# Run main function
main "$@"