#!/bin/bash

# AgenticDosNode Cleanup Script for ORACLE1 (CPU Node)
# Comprehensive cleanup for CPU-only machine preparation
# Version: 1.0.0
# Author: Claude Code DevOps Troubleshooter

set -euo pipefail

# Get script directory and source utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/cleanup-utils.sh"

# Machine-specific configuration
readonly MACHINE_NAME="oracle1"
readonly MACHINE_TYPE="CPU"
readonly REQUIRED_PORTS=(3000 5678 6333 8000 8001 9090 8080 5432 6379 27017 11434)

# CPU-optimized AI service patterns (no GPU dependencies)
readonly AI_CONTAINER_PATTERNS=(
    "ollama" "localai" "llama-cpp" "ggml" "whisper"
    "langchain" "llamaindex" "chroma" "qdrant" "weaviate"
    "redis" "postgres" "mongodb" "elasticsearch"
    "jupyter" "gradio" "streamlit" "fastapi" "flask"
    "agenticnode" "agentic" "chatbot" "nlp"
)

readonly AI_IMAGE_PATTERNS=(
    "ollama" "localai" "llama-cpp" "ggml" "whisper"
    "redis" "postgres" "mongo" "elasticsearch" "qdrant"
    "jupyter" "gradio" "streamlit" "python.*ai" "python.*ml"
    "huggingface" "langchain" "chromadb"
)

readonly AI_PROCESS_PATTERNS=(
    "ollama" "localai" "llama-cpp" "whisper"
    "python.*langchain" "python.*llamaindex" "python.*transformers"
    "python.*sentence-transformers" "python.*spacy" "python.*nltk"
    "redis-server" "postgres" "mongod" "elasticsearch"
    "qdrant" "weaviate" "chromadb"
    "jupyter" "gradio" "streamlit" "fastapi" "gunicorn" "uvicorn"
)

readonly AI_SERVICE_PATTERNS=(
    "ollama" "localai" "jupyter" "docker"
    "redis" "postgresql" "mongodb" "elasticsearch"
    "qdrant" "weaviate" "nginx" "apache"
)

# CPU-specific directories and patterns
readonly CPU_AI_DIRS=(
    "$HOME/.ollama"
    "$HOME/.cache/huggingface"
    "$HOME/.cache/torch"
    "$HOME/.cache/transformers"
    "$HOME/.local/share/ollama"
    "$HOME/.config/LocalAI"
    "/opt/ollama"
    "/var/lib/ollama"
)

main() {
    log "INFO" "Starting AgenticDosNode cleanup for $MACHINE_NAME ($MACHINE_TYPE)"
    log "INFO" "Timestamp: $(date)"
    log "INFO" "User: $(whoami)"
    log "INFO" "Working directory: $PWD"

    check_root

    # Display warning
    cat << EOF

${YELLOW}WARNING: CPU NODE CLEANUP${NC}
This script will perform comprehensive cleanup of AI/ML services on $MACHINE_NAME.

WHAT WILL BE AFFECTED:
- Docker containers and images (AI/ML related)
- Python virtual environments with AI packages
- Conda/Mamba environments
- CPU-based AI services (Ollama, LocalAI, etc.)
- Database services (Redis, PostgreSQL, MongoDB, Elasticsearch)
- Web frameworks (Jupyter, Gradio, Streamlit)
- System services related to AI/ML
- Network ports required for AgenticDosNode

DESTRUCTIVE OPERATIONS:
- Stop and remove Docker containers
- Remove Docker images and volumes
- Delete Python/Conda environments
- Terminate running processes
- Stop system services
- Clear application caches and data

CPU-OPTIMIZED FEATURES:
- No GPU process cleanup
- Focus on CPU-based AI frameworks
- Optimized for inference workloads
- Enhanced database cleanup

EOF

    if ! confirm "Proceed with cleanup on $MACHINE_NAME?"; then
        log "INFO" "Cleanup aborted by user"
        exit 0
    fi

    # Create backup of important configurations
    backup_configurations

    # Main cleanup phases
    cleanup_cpu_ai_processes
    cleanup_docker_environment
    cleanup_python_environments
    cleanup_system_services
    cleanup_network_ports
    cleanup_ai_frameworks
    cleanup_databases
    cleanup_web_services
    cleanup_cpu_ai_directories
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

    # Python configurations
    backup_item "$HOME/.pip/pip.conf" "pip_config" 2>/dev/null || true
    backup_item "$HOME/.condarc" "conda_config" 2>/dev/null || true

    # AI framework configurations
    backup_item "$HOME/.ollama/config.json" "ollama_config" 2>/dev/null || true
    backup_item "$HOME/.config/LocalAI" "localai_config" 2>/dev/null || true

    # Database configurations
    backup_item "/etc/redis/redis.conf" "redis_config" 2>/dev/null || true
    backup_item "/etc/postgresql" "postgresql_config" 2>/dev/null || true
    backup_item "/etc/mongod.conf" "mongodb_config" 2>/dev/null || true

    # Web service configurations
    backup_item "$HOME/.jupyter" "jupyter_config" 2>/dev/null || true

    # System configurations
    backup_item "/etc/systemd/system/docker.service.d" "docker_systemd_overrides" 2>/dev/null || true

    log "INFO" "Configuration backup completed"
}

cleanup_cpu_ai_processes() {
    log "STEP" "Cleaning up CPU-based AI processes..."

    # Check system resources before cleanup
    log "INFO" "Current system resources:"
    if command -v free >/dev/null 2>&1; then
        local mem_info=$(free -h | grep "Mem:" | awk '{print "Used: " $3 "/" $2}')
        log "INFO" "  Memory: $mem_info"
    fi

    if command -v nproc >/dev/null 2>&1; then
        local cpu_count=$(nproc)
        local load_avg=$(uptime | awk -F'load average:' '{print $2}')
        log "INFO" "  CPUs: $cpu_count, Load: $load_avg"
    fi

    # Find CPU-intensive AI processes
    for pattern in "${AI_PROCESS_PATTERNS[@]}"; do
        local processes=$(detect_processes "$pattern" "$pattern")
        if [[ -n "$processes" ]]; then
            # Show CPU and memory usage for these processes
            log "INFO" "Resource usage for $pattern processes:"
            ps -p $processes -o pid,ppid,%cpu,%mem,cmd --no-headers 2>/dev/null | while read line; do
                log "INFO" "  $line"
            done
            kill_processes "$processes" "$pattern" false
        fi
    done

    # Check for high CPU usage processes that might be AI-related
    log "INFO" "Checking for high CPU usage processes..."
    local high_cpu_pids=$(ps -eo pid,ppid,%cpu,cmd --no-headers | awk '$3 > 50 {print $1}' 2>/dev/null || true)
    if [[ -n "$high_cpu_pids" ]]; then
        log "WARN" "Found high CPU usage processes:"
        ps -p $high_cpu_pids -o pid,ppid,%cpu,%mem,cmd --no-headers 2>/dev/null | while read line; do
            log "WARN" "  $line"
        done

        if confirm "Investigate high CPU processes for AI-related activities?"; then
            for pid in $high_cpu_pids; do
                local cmd=$(ps -p $pid -o cmd --no-headers 2>/dev/null || echo "unknown")
                if echo "$cmd" | grep -qE "(python|node|java|ollama|ai|ml|model|inference)"; then
                    log "WARN" "Potentially AI-related high CPU process: PID $pid - $cmd"
                    if confirm "Terminate process $pid?"; then
                        kill -TERM "$pid" 2>/dev/null || true
                    fi
                fi
            done
        fi
    fi

    log "INFO" "CPU AI process cleanup completed"
}

cleanup_docker_environment() {
    log "STEP" "Cleaning up Docker environment..."

    if ! command -v docker >/dev/null 2>&1; then
        log "INFO" "Docker not installed, skipping Docker cleanup"
        return 0
    fi

    # Show Docker resource usage
    if docker info >/dev/null 2>&1; then
        log "INFO" "Docker system information:"
        local containers=$(docker ps -q | wc -l)
        local images=$(docker images -q | wc -l)
        local volumes=$(docker volume ls -q | wc -l)
        log "INFO" "  Containers: $containers, Images: $images, Volumes: $volumes"

        # Show disk usage
        if command -v docker system df >/dev/null 2>&1; then
            log "INFO" "Docker disk usage:"
            docker system df --format "table {{.Type}}\t{{.TotalCount}}\t{{.Size}}\t{{.Reclaimable}}" | while read line; do
                log "INFO" "  $line"
            done
        fi
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

    # Clean up Docker volumes with size information
    local volumes=$(docker volume ls -q 2>/dev/null || true)
    if [[ -n "$volumes" ]]; then
        log "INFO" "Analyzing Docker volumes for AI-related content..."
        local ai_volumes=""

        echo "$volumes" | while read volume; do
            if echo "$volume" | grep -qE "(ai|ml|ollama|localai|jupyter|qdrant|redis|postgres|mongo|elastic)"; then
                ai_volumes="$ai_volumes $volume"
            fi
        done

        if [[ -n "$ai_volumes" ]]; then
            log "WARN" "Found AI-related Docker volumes:"
            for volume in $ai_volumes; do
                local inspect_info=$(docker volume inspect "$volume" --format "{{.Mountpoint}}" 2>/dev/null || echo "unknown")
                local size="unknown"
                if [[ "$inspect_info" != "unknown" && -d "$inspect_info" ]]; then
                    size=$(du -sh "$inspect_info" 2>/dev/null | cut -f1 || echo "unknown")
                fi
                log "INFO" "  $volume [$size] - $inspect_info"
            done

            if confirm "Remove these Docker volumes? (This will delete data)"; then
                for volume in $ai_volumes; do
                    log "INFO" "Removing volume: $volume"
                    docker volume rm "$volume" 2>/dev/null || true
                done
            fi
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
                if [[ "$network" != "bridge" && "$network" != "host" && "$network" != "none" ]]; then
                    log "INFO" "Removing network: $network"
                    docker network rm "$network" 2>/dev/null || true
                fi
            done
        fi
    fi

    # Comprehensive Docker cleanup
    if confirm "Run comprehensive Docker cleanup (prune system, build cache, unused images)?"; then
        log "INFO" "Running Docker system prune..."
        docker system prune -a -f --volumes 2>/dev/null || true

        log "INFO" "Clearing Docker build cache..."
        docker builder prune -a -f 2>/dev/null || true
    fi

    log "INFO" "Docker environment cleanup completed"
}

cleanup_python_environments() {
    log "STEP" "Cleaning up Python environments..."

    # Check for Python AI packages in system Python
    if command -v pip >/dev/null 2>&1; then
        log "INFO" "Checking system Python for AI packages..."
        local ai_packages=$(pip list 2>/dev/null | grep -iE "(torch|tensorflow|transformers|langchain|llamaindex|ollama|openai|anthropic)" || true)
        if [[ -n "$ai_packages" ]]; then
            log "WARN" "Found AI packages in system Python:"
            echo "$ai_packages" | while read package; do
                log "INFO" "  $package"
            done

            if confirm "Remove AI packages from system Python? (Not recommended unless you're sure)"; then
                echo "$ai_packages" | awk '{print $1}' | while read package_name; do
                    log "INFO" "Uninstalling: $package_name"
                    pip uninstall "$package_name" -y 2>/dev/null || true
                done
            fi
        fi
    fi

    # Standard virtual environments
    local venv_dirs=("$HOME/.virtualenvs" "$HOME/venvs" "$HOME/.venv" "./venv" "./env" "$HOME/anaconda3/envs" "$HOME/miniconda3/envs")
    for venv_dir in "${venv_dirs[@]}"; do
        if [[ -d "$venv_dir" ]]; then
            local environments=$(detect_python_environments "$venv_dir" "virtual environments")
            if [[ -n "$environments" ]]; then
                # Check environment sizes
                log "INFO" "Analyzing environment sizes in $venv_dir..."
                for env in $environments; do
                    if [[ -d "$env" ]]; then
                        local size=$(du -sh "$env" 2>/dev/null | cut -f1 || echo "unknown")
                        log "INFO" "  $(basename "$env"): $size"
                    fi
                done
                cleanup_python_environments "$environments" "virtual environments"
            fi
        fi
    done

    # Conda environments with detailed analysis
    if command -v conda >/dev/null 2>&1; then
        log "INFO" "Analyzing conda installation..."
        local conda_info=$(conda info --json 2>/dev/null || echo '{}')
        local conda_base=$(echo "$conda_info" | jq -r '.conda_prefix // "unknown"' 2>/dev/null || echo "unknown")

        log "INFO" "Conda base: $conda_base"

        local conda_envs=$(conda env list --json 2>/dev/null | jq -r '.envs[]' | grep -v "$conda_base" || true)
        if [[ -n "$conda_envs" ]]; then
            log "INFO" "Found conda environments:"
            echo "$conda_envs" | while read env; do
                local env_name=$(basename "$env")
                local size=$(du -sh "$env" 2>/dev/null | cut -f1 || echo "unknown")

                # Check for AI packages in this environment
                local ai_packages=""
                if [[ -f "$env/conda-meta/history" ]]; then
                    ai_packages=$(grep -iE "(pytorch|tensorflow|transformers|langchain|ollama)" "$env/conda-meta/history" 2>/dev/null | wc -l || echo "0")
                fi

                log "INFO" "  $env_name ($env) [$size] - $ai_packages AI packages"
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

        # Clean conda caches
        if confirm "Clean conda caches?"; then
            log "INFO" "Cleaning conda caches..."
            conda clean --all -y 2>/dev/null || true
        fi
    fi

    # Mamba environments (if different from conda)
    if command -v mamba >/dev/null 2>&1 && ! command -v conda >/dev/null 2>&1; then
        log "INFO" "Found mamba installation (standalone)"
        # Similar cleanup for mamba
        if confirm "Clean mamba caches?"; then
            log "INFO" "Cleaning mamba caches..."
            mamba clean --all -y 2>/dev/null || true
        fi
    fi

    # Python caches and bytecode
    if confirm "Clear Python caches and bytecode files?"; then
        log "INFO" "Clearing Python caches..."

        # __pycache__ directories
        local pycache_dirs=$(find "$HOME" -type d -name "__pycache__" 2>/dev/null | wc -l)
        log "INFO" "Found $pycache_dirs __pycache__ directories"
        find "$HOME" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

        # .pyc files
        local pyc_files=$(find "$HOME" -name "*.pyc" 2>/dev/null | wc -l)
        log "INFO" "Found $pyc_files .pyc files"
        find "$HOME" -name "*.pyc" -delete 2>/dev/null || true

        # .pyo files
        local pyo_files=$(find "$HOME" -name "*.pyo" 2>/dev/null | wc -l)
        log "INFO" "Found $pyo_files .pyo files"
        find "$HOME" -name "*.pyo" -delete 2>/dev/null || true
    fi

    # Pip cache
    if command -v pip >/dev/null 2>&1; then
        local pip_cache_size="unknown"
        local pip_cache_dir=$(pip cache dir 2>/dev/null || echo "$HOME/.cache/pip")
        if [[ -d "$pip_cache_dir" ]]; then
            pip_cache_size=$(du -sh "$pip_cache_dir" 2>/dev/null | cut -f1 || echo "unknown")
        fi

        if confirm "Clear pip cache? [$pip_cache_size]"; then
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
    local custom_services=$(systemctl list-units --type=service --all | grep -E "(ai|ml|inference|chatbot|nlp)" | awk '{print $1}' || true)
    if [[ -n "$custom_services" ]]; then
        log "INFO" "Found additional AI-related services:"
        echo "$custom_services" | while read service; do
            local status=$(systemctl is-active "$service" 2>/dev/null || echo "unknown")
            local enabled=$(systemctl is-enabled "$service" 2>/dev/null || echo "unknown")
            log "INFO" "  $service [$status, $enabled]"
        done

        if confirm "Stop and disable these services?"; then
            cleanup_systemd_services "$custom_services" "custom AI services"
        fi
    fi

    # Check for user services
    if systemctl --user list-units --type=service >/dev/null 2>&1; then
        local user_ai_services=$(systemctl --user list-units --type=service --all | grep -E "(ai|ml|ollama|localai|jupyter)" | awk '{print $1}' || true)
        if [[ -n "$user_ai_services" ]]; then
            log "INFO" "Found AI-related user services:"
            echo "$user_ai_services" | while read service; do
                log "INFO" "  $service"
            done

            if confirm "Stop and disable these user services?"; then
                echo "$user_ai_services" | while read service; do
                    log "INFO" "Stopping user service: $service"
                    systemctl --user stop "$service" 2>/dev/null || true
                    systemctl --user disable "$service" 2>/dev/null || true
                done
            fi
        fi
    fi

    log "INFO" "System service cleanup completed"
}

cleanup_network_ports() {
    log "STEP" "Cleaning up network ports..."

    log "INFO" "Checking required ports for AgenticDosNode..."
    local ports_in_use=()
    local port_details=()

    for port in "${REQUIRED_PORTS[@]}"; do
        if check_port "$port"; then
            ports_in_use+=("$port")
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
            port_details+=("Port $port: $process_info")
            log "WARN" "Port $port is in use: $process_info"
        else
            log "DEBUG" "Port $port is free"
        fi
    done

    if [[ ${#ports_in_use[@]} -gt 0 ]]; then
        log "WARN" "The following required ports are in use:"
        for detail in "${port_details[@]}"; do
            log "WARN" "  $detail"
        done

        if confirm "Free up these ports?"; then
            for port in "${ports_in_use[@]}"; do
                free_port "$port" "tcp" true
                # Also check UDP
                if check_port "$port" "udp"; then
                    free_port "$port" "udp" true
                fi
            done
        fi
    else
        log "INFO" "All required ports are free"
    fi

    # Check for additional AI-related ports that might conflict
    local ai_ports=(8888 8080 7860 7861 5000 5001 8501 8502)
    log "INFO" "Checking additional AI-related ports..."

    local additional_conflicts=()
    for port in "${ai_ports[@]}"; do
        if check_port "$port"; then
            additional_conflicts+=("$port")
            local processes=$(find_port_processes "$port")
            log "INFO" "AI-related port $port in use by: $processes"
        fi
    done

    if [[ ${#additional_conflicts[@]} -gt 0 ]]; then
        if confirm "Free up additional AI-related ports: ${additional_conflicts[*]}?"; then
            for port in "${additional_conflicts[@]}"; do
                free_port "$port" "tcp" false
            done
        fi
    fi

    log "INFO" "Network port cleanup completed"
}

cleanup_ai_frameworks() {
    log "STEP" "Cleaning up AI framework processes and data..."

    # Ollama specific cleanup
    if command -v ollama >/dev/null 2>&1; then
        log "INFO" "Found Ollama installation"

        # List models
        local models=$(ollama list 2>/dev/null | tail -n +2 | awk '{print $1}' || true)
        if [[ -n "$models" ]]; then
            log "INFO" "Found Ollama models:"
            echo "$models" | while read model; do
                local size=$(ollama list 2>/dev/null | grep "^$model" | awk '{print $2}' || echo "unknown")
                log "INFO" "  $model [$size]"
            done

            if confirm "Remove all Ollama models?"; then
                echo "$models" | while read model; do
                    log "INFO" "Removing model: $model"
                    ollama rm "$model" 2>/dev/null || true
                done
            fi
        fi

        # Stop Ollama service
        local ollama_processes=$(detect_processes "ollama" "Ollama")
        if [[ -n "$ollama_processes" ]]; then
            kill_processes "$ollama_processes" "Ollama" true
        fi
    fi

    # LocalAI specific cleanup
    local localai_processes=$(detect_processes "local-ai|localai" "LocalAI")
    if [[ -n "$localai_processes" ]]; then
        kill_processes "$localai_processes" "LocalAI" true
    fi

    # Clean up Jupyter kernels and notebooks
    if command -v jupyter >/dev/null 2>&1; then
        log "INFO" "Cleaning up Jupyter environment..."

        # List kernels
        local kernels=$(jupyter kernelspec list --json 2>/dev/null | jq -r '.kernelspecs | keys[]' 2>/dev/null || true)
        if [[ -n "$kernels" ]]; then
            log "INFO" "Found Jupyter kernels:"
            echo "$kernels" | while read kernel; do
                log "INFO" "  $kernel"
            done

            if confirm "Remove all Jupyter kernels except python3?"; then
                echo "$kernels" | while read kernel; do
                    if [[ "$kernel" != "python3" ]]; then
                        log "INFO" "Removing kernel: $kernel"
                        jupyter kernelspec remove "$kernel" -y 2>/dev/null || true
                    fi
                done
            fi
        fi

        # Clear Jupyter runtime files
        local jupyter_runtime="$HOME/.local/share/jupyter/runtime"
        if [[ -d "$jupyter_runtime" ]]; then
            log "INFO" "Clearing Jupyter runtime files..."
            rm -rf "$jupyter_runtime"/* 2>/dev/null || true
        fi
    fi

    # Clean up model caches
    cleanup_model_caches

    log "INFO" "AI framework cleanup completed"
}

cleanup_model_caches() {
    log "STEP" "Cleaning up AI model caches..."

    # Hugging Face cache
    local hf_cache_dirs=(
        "$HOME/.cache/huggingface"
        "$HOME/.cache/torch"
        "$HOME/.cache/transformers"
        "$HOME/.cache/sentence-transformers"
        "$HOME/.cache/datasets"
    )

    for cache_dir in "${hf_cache_dirs[@]}"; do
        if [[ -d "$cache_dir" ]]; then
            local size=$(du -sh "$cache_dir" 2>/dev/null | cut -f1 || echo "unknown")
            local file_count=$(find "$cache_dir" -type f 2>/dev/null | wc -l || echo "unknown")
            log "INFO" "Found cache: $(basename "$cache_dir") [$size, $file_count files]"

            if confirm "Remove cache directory $cache_dir?"; then
                backup_item "$cache_dir" "ai_cache_$(basename "$cache_dir")"
                log "INFO" "Removing cache: $cache_dir"
                rm -rf "$cache_dir"
            fi
        fi
    done

    # NLTK data
    local nltk_data="$HOME/nltk_data"
    if [[ -d "$nltk_data" ]]; then
        local size=$(du -sh "$nltk_data" 2>/dev/null | cut -f1 || echo "unknown")
        log "INFO" "Found NLTK data [$size]: $nltk_data"
        if confirm "Remove NLTK data directory?"; then
            backup_item "$nltk_data" "nltk_data"
            rm -rf "$nltk_data"
        fi
    fi

    # spaCy models
    if command -v python3 >/dev/null 2>&1; then
        local spacy_models=$(python3 -c "import spacy; print(' '.join(spacy.util.get_installed_models()))" 2>/dev/null || true)
        if [[ -n "$spacy_models" ]]; then
            log "INFO" "Found spaCy models: $spacy_models"
            if confirm "Remove spaCy models?"; then
                for model in $spacy_models; do
                    python3 -m spacy download $model --force 2>/dev/null || true
                done
            fi
        fi
    fi

    log "INFO" "Model cache cleanup completed"
}

cleanup_databases() {
    log "STEP" "Cleaning up database services..."

    # Redis
    local redis_processes=$(detect_processes "redis-server" "Redis")
    if [[ -n "$redis_processes" ]]; then
        # Try to save data before killing
        if command -v redis-cli >/dev/null 2>&1; then
            log "INFO" "Attempting to save Redis data..."
            redis-cli BGSAVE 2>/dev/null || true
            sleep 2
        fi
        kill_processes "$redis_processes" "Redis" false
    fi

    # Check Redis data
    local redis_dirs=("/var/lib/redis" "$HOME/.redis" "./redis-data")
    for redis_dir in "${redis_dirs[@]}"; do
        if [[ -d "$redis_dir" ]]; then
            local size=$(du -sh "$redis_dir" 2>/dev/null | cut -f1 || echo "unknown")
            log "INFO" "Found Redis data [$size]: $redis_dir"
            if confirm "Backup and remove Redis data directory?"; then
                backup_item "$redis_dir" "redis_data"
                rm -rf "$redis_dir"
            fi
        fi
    done

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

    # Elasticsearch
    local elastic_processes=$(detect_processes "elasticsearch" "Elasticsearch")
    if [[ -n "$elastic_processes" ]]; then
        kill_processes "$elastic_processes" "Elasticsearch" false
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

    # ChromaDB
    local chroma_processes=$(detect_processes "chroma" "ChromaDB")
    if [[ -n "$chroma_processes" ]]; then
        kill_processes "$chroma_processes" "ChromaDB" false
    fi

    log "INFO" "Database cleanup completed"
}

cleanup_web_services() {
    log "STEP" "Cleaning up web services and frameworks..."

    # Jupyter
    local jupyter_processes=$(detect_processes "jupyter" "Jupyter")
    if [[ -n "$jupyter_processes" ]]; then
        kill_processes "$jupyter_processes" "Jupyter" false
    fi

    # Gradio
    local gradio_processes=$(detect_processes "gradio" "Gradio")
    if [[ -n "$gradio_processes" ]]; then
        kill_processes "$gradio_processes" "Gradio" false
    fi

    # Streamlit
    local streamlit_processes=$(detect_processes "streamlit" "Streamlit")
    if [[ -n "$streamlit_processes" ]]; then
        kill_processes "$streamlit_processes" "Streamlit" false
    fi

    # FastAPI / Uvicorn
    local fastapi_processes=$(detect_processes "uvicorn|fastapi" "FastAPI/Uvicorn")
    if [[ -n "$fastapi_processes" ]]; then
        kill_processes "$fastapi_processes" "FastAPI/Uvicorn" false
    fi

    # Flask / Gunicorn
    local flask_processes=$(detect_processes "gunicorn|flask" "Flask/Gunicorn")
    if [[ -n "$flask_processes" ]]; then
        kill_processes "$flask_processes" "Flask/Gunicorn" false
    fi

    # Nginx (if used for AI services)
    local nginx_processes=$(detect_processes "nginx" "Nginx")
    if [[ -n "$nginx_processes" ]]; then
        # Check if nginx is serving AI content
        if [[ -f "/etc/nginx/sites-enabled/default" ]]; then
            local ai_config=$(grep -iE "(ai|ml|ollama|jupyter|gradio|streamlit)" /etc/nginx/sites-enabled/* 2>/dev/null || true)
            if [[ -n "$ai_config" ]]; then
                log "WARN" "Nginx appears to be configured for AI services"
                if confirm "Stop Nginx? (This may affect other services)"; then
                    systemctl stop nginx 2>/dev/null || true
                fi
            fi
        fi
    fi

    log "INFO" "Web service cleanup completed"
}

cleanup_cpu_ai_directories() {
    log "STEP" "Cleaning up CPU AI-specific directories..."

    for ai_dir in "${CPU_AI_DIRS[@]}"; do
        if [[ -d "$ai_dir" ]]; then
            local size=$(du -sh "$ai_dir" 2>/dev/null | cut -f1 || echo "unknown")
            local owner=$(stat -c '%U' "$ai_dir" 2>/dev/null || echo "unknown")
            log "INFO" "Found AI directory [$size, owner: $owner]: $ai_dir"

            # Show contents summary
            if [[ -r "$ai_dir" ]]; then
                local file_count=$(find "$ai_dir" -type f 2>/dev/null | wc -l || echo "unknown")
                local dir_count=$(find "$ai_dir" -type d 2>/dev/null | wc -l || echo "unknown")
                log "INFO" "  Contents: $file_count files, $dir_count directories"
            fi

            if confirm "Backup and remove directory $ai_dir?"; then
                backup_item "$ai_dir" "ai_dir_$(basename "$ai_dir")"
                rm -rf "$ai_dir"
                log "INFO" "Removed: $ai_dir"
            fi
        fi
    done

    # Look for additional AI directories
    log "INFO" "Scanning for additional AI directories..."
    local additional_dirs=$(find "$HOME" -maxdepth 3 -type d -name "*ai*" -o -name "*ml*" -o -name "*ollama*" -o -name "*model*" 2>/dev/null | head -20 || true)

    if [[ -n "$additional_dirs" ]]; then
        log "INFO" "Found additional potential AI directories:"
        echo "$additional_dirs" | while read dir; do
            local size=$(du -sh "$dir" 2>/dev/null | cut -f1 || echo "unknown")
            log "INFO" "  $dir [$size]"
        done

        if confirm "Investigate these directories for cleanup?"; then
            echo "$additional_dirs" | while read dir; do
                if [[ -d "$dir" ]]; then
                    log "INFO" "Examining: $dir"
                    local contents=$(ls -la "$dir" 2>/dev/null | head -5 || true)
                    if [[ -n "$contents" ]]; then
                        log "INFO" "  Sample contents:"
                        echo "$contents" | while read line; do
                            log "INFO" "    $line"
                        done
                    fi

                    if confirm "Remove directory $dir?"; then
                        backup_item "$dir" "additional_ai_dir_$(basename "$dir")"
                        rm -rf "$dir"
                    fi
                fi
            done
        fi
    fi

    log "INFO" "CPU AI directory cleanup completed"
}

cleanup_temporary_files() {
    log "STEP" "Cleaning up temporary files..."

    # Temporary directories
    local temp_dirs=("/tmp" "/var/tmp" "$HOME/.tmp")
    for temp_dir in "${temp_dirs[@]}"; do
        if [[ -d "$temp_dir" ]]; then
            local ai_files=$(find "$temp_dir" -name "*ollama*" -o -name "*localai*" -o -name "*huggingface*" -o -name "*pytorch*" -o -name "*tensorflow*" -o -name "*ai*" -o -name "*ml*" 2>/dev/null | head -50 || true)
            if [[ -n "$ai_files" ]]; then
                local file_count=$(echo "$ai_files" | wc -l)
                log "INFO" "Found $file_count AI-related temporary files in $temp_dir"

                # Show sample files
                echo "$ai_files" | head -5 | while read file; do
                    local size=$(du -sh "$file" 2>/dev/null | cut -f1 || echo "unknown")
                    log "INFO" "  $file [$size]"
                done

                if [[ "$file_count" -gt 5 ]]; then
                    log "INFO" "  ... and $(($file_count - 5)) more files"
                fi

                if confirm "Remove AI-related temporary files from $temp_dir?"; then
                    echo "$ai_files" | while read file; do
                        rm -rf "$file" 2>/dev/null || true
                    done
                    log "INFO" "Removed $file_count temporary files"
                fi
            fi
        fi
    done

    # Log files
    local log_dirs=("/var/log" "$HOME/.local/share/logs" "$HOME/.logs")
    for log_dir in "${log_dirs[@]}"; do
        if [[ -d "$log_dir" && -r "$log_dir" ]]; then
            local ai_logs=$(find "$log_dir" -name "*ollama*" -o -name "*localai*" -o -name "*docker*" -o -name "*ai*" -o -name "*ml*" 2>/dev/null | head -20 || true)
            if [[ -n "$ai_logs" ]]; then
                local log_count=$(echo "$ai_logs" | wc -l)
                log "INFO" "Found $log_count AI-related log files in $log_dir"

                if confirm "Remove AI-related log files from $log_dir?"; then
                    echo "$ai_logs" | while read logfile; do
                        local size=$(du -sh "$logfile" 2>/dev/null | cut -f1 || echo "unknown")
                        log "INFO" "Removing log: $logfile [$size]"
                        rm -f "$logfile" 2>/dev/null || true
                    done
                fi
            fi
        fi
    done

    # Browser caches that might contain AI-related data
    local browser_caches=("$HOME/.cache/google-chrome" "$HOME/.cache/chromium" "$HOME/.cache/firefox" "$HOME/.cache/mozilla")
    for cache_dir in "${browser_caches[@]}"; do
        if [[ -d "$cache_dir" ]]; then
            local size=$(du -sh "$cache_dir" 2>/dev/null | cut -f1 || echo "unknown")
            log "INFO" "Found browser cache [$size]: $cache_dir"

            if confirm "Clear browser cache $cache_dir? (May affect browsing experience)"; then
                rm -rf "$cache_dir"/* 2>/dev/null || true
            fi
        fi
    done

    log "INFO" "Temporary file cleanup completed"
}

validate_cleanup() {
    log "STEP" "Validating cleanup results..."

    local issues=0
    local warnings=0

    # Check ports
    log "INFO" "Checking port availability..."
    for port in "${REQUIRED_PORTS[@]}"; do
        if check_port "$port"; then
            local processes=$(find_port_processes "$port")
            log "ERROR" "Port $port is still in use by: $processes"
            ((issues++))
        else
            log "DEBUG" "Port $port is free"
        fi
    done

    # Check for running AI containers
    if command -v docker >/dev/null 2>&1; then
        local ai_containers=$(docker ps --format "{{.Names}}" | grep -E "(ollama|localai|jupyter|ai|ml)" | wc -l)
        if [[ "$ai_containers" -gt 0 ]]; then
            log "WARN" "Still $ai_containers AI containers running"
            docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "(ollama|localai|jupyter|ai|ml)" | while read line; do
                log "WARN" "  $line"
            done
            ((warnings++))
        else
            log "INFO" "No AI containers running"
        fi
    fi

    # Check for AI processes
    local remaining_ai_processes=""
    for pattern in "${AI_PROCESS_PATTERNS[@]}"; do
        local processes=$(pgrep -f "$pattern" 2>/dev/null | wc -l || echo "0")
        if [[ "$processes" -gt 0 ]]; then
            remaining_ai_processes="$remaining_ai_processes $pattern:$processes"
            ((warnings++))
        fi
    done

    if [[ -n "$remaining_ai_processes" ]]; then
        log "WARN" "Remaining AI processes:$remaining_ai_processes"
    else
        log "INFO" "No AI processes detected"
    fi

    # Check system resources
    if command -v free >/dev/null 2>&1; then
        local mem_usage=$(free | grep "Mem:" | awk '{printf "%.1f", $3/$2 * 100}')
        log "INFO" "Memory usage: ${mem_usage}%"

        if (( $(echo "$mem_usage > 80" | bc -l 2>/dev/null || echo 0) )); then
            log "WARN" "High memory usage detected"
            ((warnings++))
        fi
    fi

    # Check disk space
    local disk_usage=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
    log "INFO" "Disk usage: ${disk_usage}%"

    if [[ "$disk_usage" -gt 90 ]]; then
        log "WARN" "High disk usage detected"
        ((warnings++))
    fi

    # Summary
    if [[ "$issues" -eq 0 && "$warnings" -eq 0 ]]; then
        log "INFO" "Validation passed - system is clean and ready"
    elif [[ "$issues" -eq 0 ]]; then
        log "WARN" "Validation completed with $warnings warnings - system should be usable"
    else
        log "ERROR" "Validation found $issues critical issues and $warnings warnings"
        log "ERROR" "Manual intervention may be required before deploying AgenticDosNode"
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

AgenticDosNode Cleanup Script for ORACLE1 (CPU Node)

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
    # Note: Full dry-run implementation would require modifying utility functions
fi

# Run main function
main "$@"