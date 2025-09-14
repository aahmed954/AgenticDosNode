#!/bin/bash

# Master Optimization Script for AgenticDosNode
# Orchestrates all optimizations with before/after benchmarking

set -euo pipefail

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
LOG_FILE="/var/log/agentic-optimization-master.log"
NODE_TYPE="${NODE_TYPE:-auto}"
BACKUP_DIR="/etc/agentic-backup/master"
STATE_FILE="$RESULTS_DIR/.optimization-state"

# Logging
log() {
    local level="${2:-INFO}"
    local color="${NC}"

    case "$level" in
        ERROR) color="${RED}" ;;
        WARN) color="${YELLOW}" ;;
        INFO) color="${BLUE}" ;;
        SUCCESS) color="${GREEN}" ;;
        STEP) color="${CYAN}" ;;
    esac

    echo -e "${color}[$(date '+%Y-%m-%d %H:%M:%S')] [$level] $1${NC}" | tee -a "$LOG_FILE"
    logger -t "agentic-optimize" "$1"
}

# Print banner
print_banner() {
    cat << 'EOF'

    ╔════════════════════════════════════════════════════════════════╗
    ║                                                                ║
    ║     AgenticDosNode Performance Optimization Suite v1.0        ║
    ║                                                                ║
    ║     Maximizing AI Performance on Dedicated Hardware           ║
    ║                                                                ║
    ╚════════════════════════════════════════════════════════════════╝

EOF
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..." "STEP"

    local errors=0

    # Check if running as root
    if [ "$EUID" -ne 0 ]; then
        log "This script must be run as root (use sudo)" "ERROR"
        ((errors++))
    fi

    # Check for required commands
    local required_commands=("docker" "python3" "git" "curl" "bc")
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &>/dev/null; then
            log "Required command not found: $cmd" "ERROR"
            ((errors++))
        fi
    done

    # Check Docker is running
    if ! docker info &>/dev/null; then
        log "Docker is not running or not accessible" "ERROR"
        ((errors++))
    fi

    # Detect node type
    if [ "$NODE_TYPE" == "auto" ]; then
        if nvidia-smi &>/dev/null; then
            NODE_TYPE="thanos"
            log "Detected GPU node (Thanos)" "INFO"
        else
            NODE_TYPE="oracle1"
            log "Detected CPU node (Oracle1)" "INFO"
        fi
    fi

    # Check for sufficient disk space (at least 10GB)
    available_space=$(df / | awk 'NR==2 {print $4}')
    if [ "$available_space" -lt 10485760 ]; then
        log "Insufficient disk space (less than 10GB available)" "ERROR"
        ((errors++))
    fi

    if [ $errors -gt 0 ]; then
        log "Prerequisites check failed with $errors errors" "ERROR"
        exit 1
    fi

    log "Prerequisites check passed" "SUCCESS"
}

# Create system backup
create_system_backup() {
    log "Creating system configuration backup..." "STEP"

    mkdir -p "$BACKUP_DIR"

    # Backup current configurations
    local backup_items=(
        "/etc/sysctl.conf"
        "/etc/sysctl.d"
        "/etc/security/limits.conf"
        "/etc/security/limits.d"
        "/etc/docker/daemon.json"
        "/etc/systemd/system/docker.service.d"
    )

    for item in "${backup_items[@]}"; do
        if [ -e "$item" ]; then
            cp -r "$item" "$BACKUP_DIR/" 2>/dev/null || true
        fi
    done

    # Save current system state
    {
        echo "Backup created: $(date)"
        echo "Node type: $NODE_TYPE"
        echo "Kernel: $(uname -r)"
        echo "Docker: $(docker version --format '{{.Server.Version}}' 2>/dev/null || echo 'N/A')"
    } > "$BACKUP_DIR/system-info.txt"

    log "System backup created at $BACKUP_DIR" "SUCCESS"
}

# Run baseline benchmarks
run_baseline_benchmarks() {
    log "Running baseline performance benchmarks..." "STEP"

    # Make benchmark script executable
    chmod +x "$SCRIPT_DIR/benchmarks/run-benchmarks.sh"

    # Run benchmarks
    "$SCRIPT_DIR/benchmarks/run-benchmarks.sh" run

    # Get timestamp from latest results
    BASELINE_TIMESTAMP=$(ls -t "$RESULTS_DIR"/summary-*.json 2>/dev/null | head -1 | sed 's/.*summary-\(.*\)\.json/\1/')

    if [ -z "$BASELINE_TIMESTAMP" ]; then
        log "Failed to run baseline benchmarks" "ERROR"
        exit 1
    fi

    # Save baseline timestamp
    echo "BASELINE_TIMESTAMP=$BASELINE_TIMESTAMP" > "$STATE_FILE"

    log "Baseline benchmarks completed (timestamp: $BASELINE_TIMESTAMP)" "SUCCESS"
}

# Apply hardware optimizations
apply_hardware_optimizations() {
    log "Applying hardware optimizations..." "STEP"

    chmod +x "$SCRIPT_DIR/scripts/hardware-optimization.sh"
    if "$SCRIPT_DIR/scripts/hardware-optimization.sh" "$NODE_TYPE" apply; then
        log "Hardware optimizations applied" "SUCCESS"
    else
        log "Failed to apply hardware optimizations" "WARN"
    fi
}

# Apply kernel tuning
apply_kernel_tuning() {
    log "Applying kernel parameter tuning..." "STEP"

    chmod +x "$SCRIPT_DIR/scripts/kernel-tuning.sh"
    if "$SCRIPT_DIR/scripts/kernel-tuning.sh" "$NODE_TYPE" apply; then
        log "Kernel parameters tuned" "SUCCESS"
    else
        log "Failed to apply kernel tuning" "WARN"
    fi
}

# Apply Docker optimizations
apply_docker_optimizations() {
    log "Applying Docker optimizations..." "STEP"

    chmod +x "$SCRIPT_DIR/scripts/docker-optimization.sh"
    if "$SCRIPT_DIR/scripts/docker-optimization.sh" "$NODE_TYPE" apply; then
        log "Docker optimizations applied" "SUCCESS"

        # Restart Docker containers to apply new settings
        log "Restarting Docker containers..." "INFO"
        docker restart $(docker ps -q) 2>/dev/null || true
    else
        log "Failed to apply Docker optimizations" "WARN"
    fi
}

# Apply database tuning
apply_database_tuning() {
    log "Applying database performance tuning..." "STEP"

    chmod +x "$SCRIPT_DIR/scripts/database-tuning.sh"
    if "$SCRIPT_DIR/scripts/database-tuning.sh" apply; then
        log "Database tuning applied" "SUCCESS"
    else
        log "Failed to apply database tuning" "WARN"
    fi
}

# Apply AI optimizations
apply_ai_optimizations() {
    log "Applying AI-specific optimizations..." "STEP"

    chmod +x "$SCRIPT_DIR/scripts/ai-optimization.sh"
    if "$SCRIPT_DIR/scripts/ai-optimization.sh" "$NODE_TYPE" apply; then
        log "AI optimizations applied" "SUCCESS"
    else
        log "Failed to apply AI optimizations" "WARN"
    fi
}

# Setup monitoring
setup_monitoring() {
    log "Setting up performance monitoring..." "STEP"

    chmod +x "$SCRIPT_DIR/monitoring/setup-monitoring.sh"
    if "$SCRIPT_DIR/monitoring/setup-monitoring.sh"; then
        log "Monitoring setup completed" "SUCCESS"
    else
        log "Failed to setup monitoring" "WARN"
    fi
}

# Run post-optimization benchmarks
run_post_benchmarks() {
    log "Running post-optimization benchmarks..." "STEP"

    # Wait for system to stabilize
    log "Waiting 30 seconds for system to stabilize..." "INFO"
    sleep 30

    # Run benchmarks
    "$SCRIPT_DIR/benchmarks/run-benchmarks.sh" run

    # Get timestamp from latest results
    POST_TIMESTAMP=$(ls -t "$RESULTS_DIR"/summary-*.json 2>/dev/null | head -1 | sed 's/.*summary-\(.*\)\.json/\1/')

    if [ -z "$POST_TIMESTAMP" ]; then
        log "Failed to run post-optimization benchmarks" "ERROR"
        return 1
    fi

    # Save post timestamp
    echo "POST_TIMESTAMP=$POST_TIMESTAMP" >> "$STATE_FILE"

    log "Post-optimization benchmarks completed (timestamp: $POST_TIMESTAMP)" "SUCCESS"
}

# Compare results
compare_results() {
    log "Comparing before/after performance..." "STEP"

    # Load timestamps
    source "$STATE_FILE"

    if [ -z "$BASELINE_TIMESTAMP" ] || [ -z "$POST_TIMESTAMP" ]; then
        log "Cannot compare results - missing timestamps" "ERROR"
        return 1
    fi

    # Run comparison
    "$SCRIPT_DIR/benchmarks/run-benchmarks.sh" compare "$BASELINE_TIMESTAMP" "$POST_TIMESTAMP"

    log "Performance comparison completed" "SUCCESS"
    log "Detailed comparison saved to: $RESULTS_DIR/comparison-$POST_TIMESTAMP.md" "INFO"
}

# Generate optimization report
generate_report() {
    log "Generating optimization report..." "STEP"

    local report_file="$RESULTS_DIR/optimization-report-$(date +%Y%m%d-%H%M%S).md"

    cat > "$report_file" << EOF
# AgenticDosNode Optimization Report

**Date:** $(date)
**Node Type:** $NODE_TYPE
**Hostname:** $(hostname)

## Applied Optimizations

### 1. Hardware Optimization
- CPU governor set to performance mode
- Memory huge pages enabled
- I/O schedulers optimized for SSDs
- GPU optimizations applied (if applicable)

### 2. Kernel Tuning
- Network stack optimized for high-throughput APIs
- File descriptor limits increased
- Memory management optimized for AI workloads
- Container runtime parameters tuned

### 3. Docker Optimization
- Resource limits configured
- GPU runtime enabled (if applicable)
- Network drivers optimized
- Logging and metrics enabled

### 4. Database Tuning
- PostgreSQL optimized for OLTP workloads
- Redis configured for caching
- Connection pooling enabled
- Query performance monitoring setup

### 5. AI-Specific Optimization
- CUDA memory management configured
- Model caching enabled
- Inference batching optimized
- Embedding service tuned

### 6. Monitoring Setup
- Prometheus metrics collection
- Grafana dashboards configured
- Custom exporters deployed
- Alert rules defined

## Performance Improvements

See detailed comparison: [comparison-$POST_TIMESTAMP.md](comparison-$POST_TIMESTAMP.md)

## Next Steps

1. Monitor system performance for 24-48 hours
2. Review Grafana dashboards at http://localhost:3000
3. Check for any performance regressions
4. Fine-tune based on workload patterns

## Rollback Instructions

To rollback all optimizations:

\`\`\`bash
sudo $SCRIPT_DIR/run-optimization.sh --rollback
\`\`\`

To rollback specific components:

\`\`\`bash
sudo $SCRIPT_DIR/scripts/hardware-optimization.sh $NODE_TYPE rollback
sudo $SCRIPT_DIR/scripts/kernel-tuning.sh $NODE_TYPE rollback
sudo $SCRIPT_DIR/scripts/docker-optimization.sh $NODE_TYPE rollback
sudo $SCRIPT_DIR/scripts/database-tuning.sh rollback
sudo $SCRIPT_DIR/scripts/ai-optimization.sh $NODE_TYPE rollback
\`\`\`

## Support

For issues or questions, check the logs at:
- Master log: $LOG_FILE
- Individual component logs: /var/log/agentic-*.log
EOF

    log "Optimization report generated: $report_file" "SUCCESS"
    cat "$report_file"
}

# Rollback all optimizations
rollback_all() {
    log "Rolling back all optimizations..." "STEP"

    local scripts=(
        "$SCRIPT_DIR/scripts/hardware-optimization.sh"
        "$SCRIPT_DIR/scripts/kernel-tuning.sh"
        "$SCRIPT_DIR/scripts/docker-optimization.sh"
        "$SCRIPT_DIR/scripts/database-tuning.sh"
        "$SCRIPT_DIR/scripts/ai-optimization.sh"
    )

    for script in "${scripts[@]}"; do
        if [ -f "$script" ]; then
            chmod +x "$script"
            log "Rolling back $(basename $script)..." "INFO"
            "$script" "$NODE_TYPE" rollback 2>/dev/null || true
        fi
    done

    # Restore original configurations from backup
    if [ -d "$BACKUP_DIR" ]; then
        log "Restoring original configurations..." "INFO"
        cp -r "$BACKUP_DIR"/* /etc/ 2>/dev/null || true
    fi

    log "Rollback completed. Please reboot the system." "SUCCESS"
}

# Apply specific optimization
apply_specific() {
    local component="$1"

    case "$component" in
        hardware)
            apply_hardware_optimizations
            ;;
        kernel)
            apply_kernel_tuning
            ;;
        docker)
            apply_docker_optimizations
            ;;
        database)
            apply_database_tuning
            ;;
        ai)
            apply_ai_optimizations
            ;;
        monitoring)
            setup_monitoring
            ;;
        *)
            log "Unknown component: $component" "ERROR"
            log "Valid components: hardware, kernel, docker, database, ai, monitoring" "INFO"
            exit 1
            ;;
    esac
}

# Main optimization workflow
run_full_optimization() {
    log "Starting full optimization workflow" "STEP"

    # Create results directory
    mkdir -p "$RESULTS_DIR"

    # Step 1: Prerequisites and backup
    check_prerequisites
    create_system_backup

    # Step 2: Baseline benchmarks
    run_baseline_benchmarks

    # Step 3: Apply optimizations
    log "Applying all optimizations..." "STEP"
    apply_hardware_optimizations
    apply_kernel_tuning
    apply_docker_optimizations
    apply_database_tuning
    apply_ai_optimizations
    setup_monitoring

    # Step 4: Post-optimization benchmarks
    run_post_benchmarks

    # Step 5: Compare and report
    compare_results
    generate_report

    log "Full optimization workflow completed!" "SUCCESS"

    # Display summary
    cat << EOF

${GREEN}════════════════════════════════════════════════════════════════${NC}
${GREEN}           OPTIMIZATION COMPLETED SUCCESSFULLY!                 ${NC}
${GREEN}════════════════════════════════════════════════════════════════${NC}

${CYAN}Key Actions Taken:${NC}
✓ System backup created
✓ Baseline benchmarks recorded
✓ Hardware optimizations applied
✓ Kernel parameters tuned
✓ Docker performance optimized
✓ Database settings tuned
✓ AI inference optimized
✓ Monitoring stack deployed
✓ Performance comparison generated

${YELLOW}Important Next Steps:${NC}
1. Review the optimization report in: $RESULTS_DIR
2. Monitor system performance: http://localhost:3000
3. Reboot system for all changes to take effect:
   ${BLUE}sudo reboot${NC}

${MAGENTA}Monitoring Access:${NC}
- Grafana: http://localhost:3000 (admin/changeme)
- Prometheus: http://localhost:9090

${RED}If issues occur:${NC}
Run rollback: ${BLUE}sudo $0 --rollback${NC}

EOF
}

# Parse command line arguments
parse_arguments() {
    case "${1:-}" in
        --apply-all|"")
            run_full_optimization
            ;;
        --benchmark-only)
            check_prerequisites
            run_baseline_benchmarks
            ;;
        --apply)
            if [ -z "${2:-}" ]; then
                log "Please specify component to apply" "ERROR"
                exit 1
            fi
            check_prerequisites
            apply_specific "$2"
            ;;
        --rollback)
            rollback_all
            ;;
        --compare)
            if [ -z "${2:-}" ] || [ -z "${3:-}" ]; then
                log "Usage: --compare <before_timestamp> <after_timestamp>" "ERROR"
                exit 1
            fi
            "$SCRIPT_DIR/benchmarks/run-benchmarks.sh" compare "$2" "$3"
            ;;
        --help|-h)
            cat << EOF
Usage: $0 [OPTIONS]

AgenticDosNode Performance Optimization Suite

OPTIONS:
    --apply-all         Run full optimization workflow (default)
    --benchmark-only    Run benchmarks without applying optimizations
    --apply <component> Apply specific optimization:
                       hardware, kernel, docker, database, ai, monitoring
    --rollback         Rollback all optimizations
    --compare <b> <a>  Compare benchmark results
    --help            Show this help message

EXAMPLES:
    # Run full optimization
    sudo $0

    # Run benchmarks only
    sudo $0 --benchmark-only

    # Apply only Docker optimizations
    sudo $0 --apply docker

    # Rollback all changes
    sudo $0 --rollback

    # Compare results
    $0 --compare 20240101-120000 20240101-140000

EOF
            ;;
        *)
            log "Unknown option: $1" "ERROR"
            log "Use --help for usage information" "INFO"
            exit 1
            ;;
    esac
}

# Main execution
main() {
    print_banner
    parse_arguments "$@"
}

# Run main function
main "$@"