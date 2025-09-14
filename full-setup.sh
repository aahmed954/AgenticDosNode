#!/bin/bash

# AgenticDosNode Full Setup Workflow
# Automated cleanup, preparation, and validation
# Version: 1.0.0
# Author: Claude Code DevOps Troubleshooter

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MACHINE_TYPE="${1:-auto}"
AUTO_MODE="${2:-interactive}"

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m' # No Color

log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    case "$level" in
        "INFO")  echo -e "${GREEN}[INFO]${NC} $message" ;;
        "WARN")  echo -e "${YELLOW}[WARN]${NC} $message" ;;
        "ERROR") echo -e "${RED}[ERROR]${NC} $message" ;;
        "STEP")  echo -e "${CYAN}[STEP]${NC} $message" ;;
    esac
}

detect_machine_type() {
    if [[ "$MACHINE_TYPE" != "auto" ]]; then
        return 0
    fi

    log "INFO" "Auto-detecting machine type..."

    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
        MACHINE_TYPE="GPU"
        log "INFO" "Detected GPU-capable machine"
    else
        MACHINE_TYPE="CPU"
        log "INFO" "Detected CPU-only machine"
    fi
}

run_cleanup() {
    log "STEP" "Running cleanup phase..."

    local cleanup_script=""
    case "$MACHINE_TYPE" in
        "GPU")
            cleanup_script="cleanup-thanos.sh"
            ;;
        "CPU")
            cleanup_script="cleanup-oracle1.sh"
            ;;
        *)
            log "ERROR" "Unknown machine type: $MACHINE_TYPE"
            return 1
            ;;
    esac

    local cleanup_args=""
    if [[ "$AUTO_MODE" == "auto" ]]; then
        cleanup_args="-y"
    fi

    log "INFO" "Running $cleanup_script..."
    if [[ $EUID -eq 0 ]]; then
        "${SCRIPT_DIR}/${cleanup_script}" $cleanup_args
    else
        sudo "${SCRIPT_DIR}/${cleanup_script}" $cleanup_args
    fi

    return $?
}

run_preparation() {
    log "STEP" "Running environment preparation..."

    local prep_args=""
    if [[ "$AUTO_MODE" == "auto" ]]; then
        prep_args="-y"
    fi

    log "INFO" "Running prepare-environment.sh..."
    if [[ $EUID -eq 0 ]]; then
        "${SCRIPT_DIR}/prepare-environment.sh" $prep_args
    else
        sudo "${SCRIPT_DIR}/prepare-environment.sh" $prep_args
    fi

    return $?
}

run_validation() {
    log "STEP" "Running system validation..."

    log "INFO" "Running validate-clean-state.sh..."
    "${SCRIPT_DIR}/validate-clean-state.sh" -t "$MACHINE_TYPE"

    return $?
}

show_summary() {
    local exit_code=$1

    log "STEP" "Setup Summary"

    cat << EOF

${CYAN}========================================
AGENTICNODE SETUP SUMMARY
========================================${NC}

Machine Type: $MACHINE_TYPE
Mode: $AUTO_MODE

EOF

    case $exit_code in
        0)
            cat << EOF
${GREEN}✅ SUCCESS: System is ready for AgenticDosNode deployment${NC}

Next Steps:
1. Deploy AgenticDosNode using docker-compose
2. Configure application-specific settings
3. Start services: systemctl start agenticnode
4. Access the interface at http://your-server:3000

EOF
            ;;
        1)
            cat << EOF
${RED}❌ FAILED: Critical issues prevent deployment${NC}

Actions Required:
1. Review error messages above
2. Address critical issues
3. Re-run the setup script
4. Check logs in cleanup-logs/ directory

EOF
            ;;
        2)
            cat << EOF
${YELLOW}⚠️  WARNINGS: System ready but with cautions${NC}

Recommendations:
1. Review warnings in validation report
2. Consider addressing performance optimizations
3. Monitor system during initial deployment
4. Proceed with deployment if acceptable

EOF
            ;;
        *)
            cat << EOF
${RED}❓ UNKNOWN: Setup completed with unexpected status${NC}

Please review the output above and logs for details.

EOF
            ;;
    esac

    cat << EOF
Log files are available in:
- cleanup-logs/ - Cleanup and preparation logs
- cleanup-backups/ - Configuration backups

For detailed validation results, see:
- cleanup-logs/validation_report_*.txt
- cleanup-logs/validation_report_*.json

EOF
}

main() {
    log "INFO" "Starting AgenticDosNode full setup workflow"
    log "INFO" "Timestamp: $(date)"
    log "INFO" "User: $(whoami)"
    log "INFO" "Working directory: $PWD"

    # Check script location
    if [[ ! -f "${SCRIPT_DIR}/cleanup-utils.sh" ]]; then
        log "ERROR" "Required script cleanup-utils.sh not found in $SCRIPT_DIR"
        exit 1
    fi

    # Detect machine type
    detect_machine_type

    cat << EOF

${BLUE}========================================
AGENTICNODE FULL SETUP WORKFLOW
========================================${NC}

Machine Type: $MACHINE_TYPE
Mode: $AUTO_MODE

This workflow will:
1. 🧹 Clean up existing AI/ML services and conflicting processes
2. 🔧 Prepare the environment with required packages and configuration
3. ✅ Validate that the system is ready for AgenticDosNode deployment

${YELLOW}WARNING: This will make significant changes to your system!${NC}

EOF

    if [[ "$AUTO_MODE" != "auto" ]]; then
        read -p "Do you want to continue? [y/N]: " -r response
        case "$response" in
            [Yy]|[Yy][Ee][Ss]) ;;
            *)
                log "INFO" "Setup aborted by user"
                exit 0
                ;;
        esac
    else
        log "INFO" "Running in automatic mode"
    fi

    local overall_success=true
    local final_exit_code=0

    # Phase 1: Cleanup
    log "STEP" "Phase 1/3: System Cleanup"
    if ! run_cleanup; then
        log "ERROR" "Cleanup phase failed"
        overall_success=false
        final_exit_code=1
    else
        log "INFO" "Cleanup phase completed successfully"
    fi

    # Phase 2: Preparation (only if cleanup succeeded)
    if [[ "$overall_success" == "true" ]]; then
        log "STEP" "Phase 2/3: Environment Preparation"
        if ! run_preparation; then
            log "ERROR" "Preparation phase failed"
            overall_success=false
            final_exit_code=1
        else
            log "INFO" "Preparation phase completed successfully"
        fi
    else
        log "WARN" "Skipping preparation due to cleanup failure"
    fi

    # Phase 3: Validation (always run for diagnostic purposes)
    log "STEP" "Phase 3/3: System Validation"
    local validation_result=0
    run_validation || validation_result=$?

    case $validation_result in
        0)
            log "INFO" "Validation passed - system is ready"
            if [[ "$final_exit_code" -eq 0 ]]; then
                final_exit_code=0
            fi
            ;;
        1)
            log "ERROR" "Validation failed - critical issues found"
            final_exit_code=1
            ;;
        2)
            log "WARN" "Validation completed with warnings"
            if [[ "$final_exit_code" -eq 0 ]]; then
                final_exit_code=2
            fi
            ;;
    esac

    # Show summary
    show_summary $final_exit_code

    exit $final_exit_code
}

usage() {
    cat << EOF
Usage: $0 [MACHINE_TYPE] [MODE]

AgenticDosNode Full Setup Workflow

PARAMETERS:
    MACHINE_TYPE    Machine type: GPU, CPU, or auto [default: auto]
    MODE           Execution mode: interactive or auto [default: interactive]

EXAMPLES:
    $0                    Auto-detect machine type, interactive mode
    $0 GPU               GPU machine, interactive mode
    $0 CPU auto          CPU machine, automatic mode
    $0 auto auto         Auto-detect machine, automatic mode

MODES:
    interactive         Prompt for confirmations (safer)
    auto               Automatic execution with default answers (faster)

EOF
}

# Argument validation
case "${1:-auto}" in
    "GPU"|"CPU"|"auto")
        MACHINE_TYPE="$1"
        ;;
    "-h"|"--help")
        usage
        exit 0
        ;;
    *)
        log "ERROR" "Invalid machine type: $1"
        usage
        exit 1
        ;;
esac

case "${2:-interactive}" in
    "interactive"|"auto")
        AUTO_MODE="$2"
        ;;
    *)
        log "ERROR" "Invalid mode: $2"
        usage
        exit 1
        ;;
esac

# Change to script directory
cd "$SCRIPT_DIR"

# Run main workflow
main "$@"