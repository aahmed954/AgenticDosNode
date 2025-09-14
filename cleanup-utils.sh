#!/bin/bash

# Shared utility functions for AgenticDosNode cleanup scripts
# Version: 1.0.0
# Author: Claude Code DevOps Troubleshooter

set -euo pipefail

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m' # No Color

# Global variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/cleanup-logs"
BACKUP_DIR="${SCRIPT_DIR}/cleanup-backups"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="${LOG_DIR}/cleanup_${TIMESTAMP}.log"

# Create directories
mkdir -p "${LOG_DIR}" "${BACKUP_DIR}"

# Logging functions
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    case "$level" in
        "INFO")  echo -e "${GREEN}[INFO]${NC} $message" | tee -a "$LOG_FILE" ;;
        "WARN")  echo -e "${YELLOW}[WARN]${NC} $message" | tee -a "$LOG_FILE" ;;
        "ERROR") echo -e "${RED}[ERROR]${NC} $message" | tee -a "$LOG_FILE" ;;
        "DEBUG") echo -e "${BLUE}[DEBUG]${NC} $message" | tee -a "$LOG_FILE" ;;
        "STEP")  echo -e "${CYAN}[STEP]${NC} $message" | tee -a "$LOG_FILE" ;;
        *)       echo "[$timestamp] $level: $message" | tee -a "$LOG_FILE" ;;
    esac
}

# Confirmation prompt
confirm() {
    local message="$1"
    local default="${2:-n}"
    local response

    if [[ "$default" == "y" ]]; then
        read -p "${message} [Y/n]: " response
        response=${response:-y}
    else
        read -p "${message} [y/N]: " response
        response=${response:-n}
    fi

    case "$response" in
        [Yy]|[Yy][Ee][Ss]) return 0 ;;
        *) return 1 ;;
    esac
}

# Check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        log "WARN" "Running as root. Some operations may be more destructive."
        if ! confirm "Continue as root?"; then
            log "ERROR" "Aborted by user"
            exit 1
        fi
    fi
}

# Backup a file or directory
backup_item() {
    local item="$1"
    local backup_name="$2"

    if [[ -e "$item" ]]; then
        local backup_path="${BACKUP_DIR}/${backup_name}_${TIMESTAMP}"
        log "INFO" "Backing up $item to $backup_path"

        if [[ -d "$item" ]]; then
            cp -r "$item" "$backup_path"
        else
            cp "$item" "$backup_path"
        fi

        echo "$item:$backup_path" >> "${BACKUP_DIR}/restore_map_${TIMESTAMP}.txt"
        return 0
    else
        log "DEBUG" "Item does not exist: $item"
        return 1
    fi
}

# Detect running processes by pattern
detect_processes() {
    local pattern="$1"
    local description="$2"

    log "STEP" "Detecting $description processes..."

    local pids=$(pgrep -f "$pattern" 2>/dev/null || true)
    if [[ -n "$pids" ]]; then
        log "INFO" "Found $description processes:"
        ps -p $pids -o pid,ppid,cmd --no-headers | while read line; do
            log "INFO" "  $line"
        done
        echo "$pids"
    else
        log "DEBUG" "No $description processes found"
    fi
}

# Kill processes safely
kill_processes() {
    local pids="$1"
    local description="$2"
    local force="${3:-false}"

    if [[ -z "$pids" ]]; then
        return 0
    fi

    log "WARN" "About to terminate $description processes: $pids"
    if ! confirm "Terminate these processes?"; then
        log "INFO" "Skipping process termination"
        return 0
    fi

    # Try SIGTERM first
    for pid in $pids; do
        if kill -0 "$pid" 2>/dev/null; then
            log "INFO" "Sending SIGTERM to process $pid"
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done

    # Wait a bit
    sleep 5

    # Check which processes are still running
    local remaining_pids=""
    for pid in $pids; do
        if kill -0 "$pid" 2>/dev/null; then
            remaining_pids="$remaining_pids $pid"
        fi
    done

    # Force kill if necessary
    if [[ -n "$remaining_pids" && "$force" == "true" ]]; then
        log "WARN" "Force killing remaining processes: $remaining_pids"
        if confirm "Force kill remaining processes?"; then
            for pid in $remaining_pids; do
                if kill -0 "$pid" 2>/dev/null; then
                    log "INFO" "Sending SIGKILL to process $pid"
                    kill -KILL "$pid" 2>/dev/null || true
                fi
            done
        fi
    fi
}

# Detect Docker containers
detect_docker_containers() {
    local pattern="$1"
    local description="$2"

    if ! command -v docker >/dev/null 2>&1; then
        log "DEBUG" "Docker not installed"
        return 0
    fi

    log "STEP" "Detecting $description Docker containers..."

    local containers=$(docker ps -a --filter "name=$pattern" --format "{{.Names}}" 2>/dev/null || true)
    if [[ -n "$containers" ]]; then
        log "INFO" "Found $description containers:"
        echo "$containers" | while read container; do
            local status=$(docker inspect --format='{{.State.Status}}' "$container" 2>/dev/null || echo "unknown")
            log "INFO" "  $container [$status]"
        done
        echo "$containers"
    else
        log "DEBUG" "No $description containers found matching pattern: $pattern"
    fi
}

# Stop and remove Docker containers
cleanup_docker_containers() {
    local containers="$1"
    local description="$2"

    if [[ -z "$containers" ]]; then
        return 0
    fi

    log "WARN" "About to stop and remove $description containers"
    if ! confirm "Stop and remove these containers?"; then
        log "INFO" "Skipping container cleanup"
        return 0
    fi

    echo "$containers" | while read container; do
        if [[ -n "$container" ]]; then
            log "INFO" "Stopping container: $container"
            docker stop "$container" 2>/dev/null || true

            log "INFO" "Removing container: $container"
            docker rm "$container" 2>/dev/null || true
        fi
    done
}

# Detect Docker images
detect_docker_images() {
    local pattern="$1"
    local description="$2"

    if ! command -v docker >/dev/null 2>&1; then
        return 0
    fi

    log "STEP" "Detecting $description Docker images..."

    local images=$(docker images --format "{{.Repository}}:{{.Tag}}" | grep -i "$pattern" 2>/dev/null || true)
    if [[ -n "$images" ]]; then
        log "INFO" "Found $description images:"
        echo "$images" | while read image; do
            local size=$(docker images --format "{{.Repository}}:{{.Tag}} {{.Size}}" | grep "^$image " | awk '{print $2}')
            log "INFO" "  $image [$size]"
        done
        echo "$images"
    else
        log "DEBUG" "No $description images found matching pattern: $pattern"
    fi
}

# Remove Docker images
cleanup_docker_images() {
    local images="$1"
    local description="$2"

    if [[ -z "$images" ]]; then
        return 0
    fi

    log "WARN" "About to remove $description Docker images"
    if ! confirm "Remove these images?"; then
        log "INFO" "Skipping image cleanup"
        return 0
    fi

    echo "$images" | while read image; do
        if [[ -n "$image" ]]; then
            log "INFO" "Removing image: $image"
            docker rmi "$image" 2>/dev/null || true
        fi
    done
}

# Check port usage
check_port() {
    local port="$1"
    local protocol="${2:-tcp}"

    if command -v ss >/dev/null 2>&1; then
        ss -ln${protocol:0:1} | grep -q ":$port " && return 0 || return 1
    elif command -v netstat >/dev/null 2>&1; then
        netstat -ln${protocol:0:1} | grep -q ":$port " && return 0 || return 1
    else
        log "WARN" "Neither ss nor netstat available for port checking"
        return 1
    fi
}

# Find processes using a port
find_port_processes() {
    local port="$1"
    local protocol="${2:-tcp}"

    if command -v ss >/dev/null 2>&1; then
        ss -lnp${protocol:0:1} | grep ":$port " | awk '{print $7}' | cut -d',' -f2 | cut -d'=' -f2 2>/dev/null || true
    elif command -v lsof >/dev/null 2>&1; then
        lsof -ti ${protocol}:$port 2>/dev/null || true
    else
        log "WARN" "Neither ss nor lsof available for finding port processes"
    fi
}

# Free up a port
free_port() {
    local port="$1"
    local protocol="${2:-tcp}"
    local force="${3:-false}"

    if ! check_port "$port" "$protocol"; then
        log "DEBUG" "Port $port/$protocol is already free"
        return 0
    fi

    log "STEP" "Freeing port $port/$protocol"
    local pids=$(find_port_processes "$port" "$protocol")

    if [[ -n "$pids" ]]; then
        log "INFO" "Found processes using port $port/$protocol:"
        for pid in $pids; do
            if kill -0 "$pid" 2>/dev/null; then
                local cmd=$(ps -p "$pid" -o cmd --no-headers 2>/dev/null || echo "unknown")
                log "INFO" "  PID $pid: $cmd"
            fi
        done

        kill_processes "$pids" "port $port/$protocol" "$force"
    else
        log "WARN" "Port $port/$protocol is in use but couldn't identify processes"
    fi
}

# Detect systemd services
detect_systemd_services() {
    local pattern="$1"
    local description="$2"

    if ! command -v systemctl >/dev/null 2>&1; then
        log "DEBUG" "systemctl not available"
        return 0
    fi

    log "STEP" "Detecting $description systemd services..."

    local services=$(systemctl list-units --type=service --all | grep "$pattern" | awk '{print $1}' 2>/dev/null || true)
    if [[ -n "$services" ]]; then
        log "INFO" "Found $description services:"
        echo "$services" | while read service; do
            local status=$(systemctl is-active "$service" 2>/dev/null || echo "unknown")
            local enabled=$(systemctl is-enabled "$service" 2>/dev/null || echo "unknown")
            log "INFO" "  $service [$status, $enabled]"
        done
        echo "$services"
    else
        log "DEBUG" "No $description services found matching pattern: $pattern"
    fi
}

# Stop and disable systemd services
cleanup_systemd_services() {
    local services="$1"
    local description="$2"

    if [[ -z "$services" ]]; then
        return 0
    fi

    log "WARN" "About to stop and disable $description services"
    if ! confirm "Stop and disable these services?"; then
        log "INFO" "Skipping service cleanup"
        return 0
    fi

    echo "$services" | while read service; do
        if [[ -n "$service" ]]; then
            log "INFO" "Stopping service: $service"
            systemctl stop "$service" 2>/dev/null || true

            log "INFO" "Disabling service: $service"
            systemctl disable "$service" 2>/dev/null || true
        fi
    done
}

# Detect Python environments
detect_python_environments() {
    local base_dir="$1"
    local description="$2"

    if [[ ! -d "$base_dir" ]]; then
        log "DEBUG" "Directory does not exist: $base_dir"
        return 0
    fi

    log "STEP" "Detecting $description in $base_dir..."

    local environments=""

    # Virtual environments
    find "$base_dir" -name "pyvenv.cfg" -type f 2>/dev/null | while read cfg; do
        local env_dir=$(dirname "$cfg")
        environments="$environments $env_dir"
    done

    # Conda environments
    if [[ -d "$base_dir" ]]; then
        find "$base_dir" -name "conda-meta" -type d 2>/dev/null | while read meta; do
            local env_dir=$(dirname "$meta")
            environments="$environments $env_dir"
        done
    fi

    if [[ -n "$environments" ]]; then
        log "INFO" "Found $description environments:"
        for env in $environments; do
            log "INFO" "  $env"
        done
        echo "$environments"
    else
        log "DEBUG" "No $description environments found in $base_dir"
    fi
}

# Clean up Python environments
cleanup_python_environments() {
    local environments="$1"
    local description="$2"

    if [[ -z "$environments" ]]; then
        return 0
    fi

    log "WARN" "About to remove $description environments"
    if ! confirm "Remove these environments?"; then
        log "INFO" "Skipping environment cleanup"
        return 0
    fi

    for env in $environments; do
        if [[ -d "$env" && -n "$env" ]]; then
            backup_item "$env" "python_env_$(basename "$env")"
            log "INFO" "Removing environment: $env"
            rm -rf "$env"
        fi
    done
}

# Display summary
show_summary() {
    log "STEP" "Cleanup Summary"
    log "INFO" "Log file: $LOG_FILE"
    log "INFO" "Backup directory: $BACKUP_DIR"

    if [[ -f "${BACKUP_DIR}/restore_map_${TIMESTAMP}.txt" ]]; then
        log "INFO" "Restore map: ${BACKUP_DIR}/restore_map_${TIMESTAMP}.txt"
        log "INFO" "Backed up items:"
        cat "${BACKUP_DIR}/restore_map_${TIMESTAMP}.txt" | while IFS=: read original backup; do
            log "INFO" "  $original -> $backup"
        done
    fi
}

# Restore from backup
restore_backup() {
    local timestamp="$1"
    local restore_map="${BACKUP_DIR}/restore_map_${timestamp}.txt"

    if [[ ! -f "$restore_map" ]]; then
        log "ERROR" "Restore map not found: $restore_map"
        return 1
    fi

    log "WARN" "About to restore from backup timestamp: $timestamp"
    if ! confirm "This will overwrite current files. Continue?"; then
        log "INFO" "Restore aborted"
        return 1
    fi

    while IFS=: read original backup; do
        if [[ -e "$backup" ]]; then
            log "INFO" "Restoring $backup -> $original"
            if [[ -d "$backup" ]]; then
                rm -rf "$original" 2>/dev/null || true
                cp -r "$backup" "$original"
            else
                cp "$backup" "$original"
            fi
        else
            log "ERROR" "Backup not found: $backup"
        fi
    done < "$restore_map"
}

# Export functions for use in other scripts
export -f log confirm check_root backup_item
export -f detect_processes kill_processes
export -f detect_docker_containers cleanup_docker_containers
export -f detect_docker_images cleanup_docker_images
export -f check_port find_port_processes free_port
export -f detect_systemd_services cleanup_systemd_services
export -f detect_python_environments cleanup_python_environments
export -f show_summary restore_backup

log "INFO" "Cleanup utilities loaded successfully"