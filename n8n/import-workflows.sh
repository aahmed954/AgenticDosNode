#!/bin/bash

# N8N Workflow Import Helper Script
# Automates the import of workflow files into n8n via API

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
N8N_URL="${N8N_URL:-http://localhost:5678}"
WORKFLOWS_DIR="${SCRIPT_DIR}/workflows"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if n8n is accessible
check_n8n_accessibility() {
    log "Checking n8n accessibility at ${N8N_URL}..."

    if ! curl -s "${N8N_URL}/healthz" > /dev/null 2>&1; then
        error "n8n is not accessible at ${N8N_URL}"
        echo ""
        echo "Please ensure:"
        echo "1. n8n is running (./deploy-n8n.sh deploy)"
        echo "2. URL is correct (set N8N_URL environment variable if different)"
        echo "3. No firewall blocking the connection"
        exit 1
    fi

    success "n8n is accessible"
}

# Get authentication cookie
get_auth_cookie() {
    log "Authenticating with n8n..."

    # Source environment for credentials
    if [[ -f "${SCRIPT_DIR}/docker/.env" ]]; then
        source "${SCRIPT_DIR}/docker/.env"
    fi

    local username="${N8N_BASIC_AUTH_USER:-admin}"
    local password="${N8N_BASIC_AUTH_PASSWORD:-}"

    if [[ -z "$password" ]]; then
        read -s -p "Enter n8n password for user '$username': " password
        echo
    fi

    # Get authentication cookie
    COOKIE_JAR=$(mktemp)

    if ! curl -s -c "$COOKIE_JAR" -X POST \
        -H "Content-Type: application/json" \
        -d "{\"email\":\"$username\",\"password\":\"$password\"}" \
        "${N8N_URL}/rest/login" > /dev/null 2>&1; then
        error "Authentication failed"
        rm -f "$COOKIE_JAR"
        exit 1
    fi

    success "Authentication successful"
}

# Import a single workflow
import_workflow() {
    local workflow_file="$1"
    local workflow_name=$(basename "$workflow_file" .json)

    log "Importing workflow: $workflow_name"

    # Read workflow content
    local workflow_content
    workflow_content=$(cat "$workflow_file")

    # Import workflow via API
    local response
    response=$(curl -s -b "$COOKIE_JAR" -X POST \
        -H "Content-Type: application/json" \
        -d "$workflow_content" \
        "${N8N_URL}/rest/workflows" 2>/dev/null)

    if echo "$response" | grep -q '"id"'; then
        local workflow_id
        workflow_id=$(echo "$response" | grep -o '"id":"[^"]*"' | cut -d'"' -f4 | head -1)
        success "Imported $workflow_name (ID: $workflow_id)"
        return 0
    else
        error "Failed to import $workflow_name"
        if echo "$response" | grep -q "already exists"; then
            warning "Workflow with this name already exists"
        fi
        return 1
    fi
}

# List available workflows
list_workflows() {
    log "Available workflows in ${WORKFLOWS_DIR}:"
    echo ""

    if [[ ! -d "$WORKFLOWS_DIR" ]]; then
        error "Workflows directory not found: $WORKFLOWS_DIR"
        return 1
    fi

    local count=0
    for workflow in "$WORKFLOWS_DIR"/*.json; do
        if [[ -f "$workflow" ]]; then
            count=$((count + 1))
            local name=$(basename "$workflow" .json)
            local description=""

            # Try to extract description from workflow
            if grep -q '"name":' "$workflow"; then
                description=$(grep '"name":' "$workflow" | head -1 | sed 's/.*"name": *"\([^"]*\)".*/\1/')
            fi

            printf "%2d. %-30s %s\n" $count "$name" "$description"
        fi
    done

    if [[ $count -eq 0 ]]; then
        warning "No workflow files found in $WORKFLOWS_DIR"
        return 1
    fi

    echo ""
    echo "Total workflows available: $count"
}

# Import all workflows
import_all_workflows() {
    log "Starting bulk workflow import..."

    local success_count=0
    local total_count=0

    for workflow_file in "$WORKFLOWS_DIR"/*.json; do
        if [[ -f "$workflow_file" ]]; then
            total_count=$((total_count + 1))
            if import_workflow "$workflow_file"; then
                success_count=$((success_count + 1))
            fi
            echo ""
        fi
    done

    echo "=================================="
    echo "Import Summary:"
    echo "  Total workflows: $total_count"
    echo "  Successfully imported: $success_count"
    echo "  Failed: $((total_count - success_count))"
    echo "=================================="

    if [[ $success_count -eq $total_count ]]; then
        success "All workflows imported successfully!"
    elif [[ $success_count -gt 0 ]]; then
        warning "Some workflows imported with issues"
    else
        error "No workflows were imported successfully"
    fi
}

# Interactive workflow selection
select_workflows() {
    log "Interactive workflow selection..."
    list_workflows

    echo ""
    echo "Select workflows to import:"
    echo "  - Enter numbers separated by spaces (e.g., 1 3 5)"
    echo "  - Enter 'all' to import all workflows"
    echo "  - Enter 'quit' to exit"
    echo ""

    read -p "Your selection: " selection

    if [[ "$selection" == "quit" ]]; then
        log "Import cancelled by user"
        exit 0
    elif [[ "$selection" == "all" ]]; then
        import_all_workflows
    else
        local success_count=0
        local total_count=0

        for num in $selection; do
            if [[ "$num" =~ ^[0-9]+$ ]]; then
                local workflow_files=("$WORKFLOWS_DIR"/*.json)
                local index=$((num - 1))

                if [[ $index -ge 0 && $index -lt ${#workflow_files[@]} ]]; then
                    total_count=$((total_count + 1))
                    if import_workflow "${workflow_files[$index]}"; then
                        success_count=$((success_count + 1))
                    fi
                else
                    warning "Invalid selection: $num"
                fi
            else
                warning "Invalid input: $num (expected number)"
            fi
        done

        if [[ $total_count -gt 0 ]]; then
            echo ""
            echo "Selected import summary: $success_count/$total_count successful"
        fi
    fi
}

# Show workflow status
show_workflow_status() {
    log "Fetching current workflows from n8n..."

    local response
    response=$(curl -s -b "$COOKIE_JAR" "${N8N_URL}/rest/workflows" 2>/dev/null)

    if [[ -z "$response" ]]; then
        error "Failed to fetch workflows from n8n"
        return 1
    fi

    echo ""
    echo "Current workflows in n8n:"
    echo "========================="

    # Parse and display workflows (simplified JSON parsing)
    echo "$response" | grep -o '"name":"[^"]*"' | sed 's/"name":"//; s/"$//' | sort | nl -w2 -s". "

    echo ""
    local count=$(echo "$response" | grep -o '"name":"[^"]*"' | wc -l)
    echo "Total active workflows: $count"
}

# Cleanup
cleanup() {
    if [[ -f "$COOKIE_JAR" ]]; then
        rm -f "$COOKIE_JAR"
    fi
}

# Main function
main() {
    echo ""
    echo "🔄 N8N Workflow Import Tool"
    echo "==========================="
    echo ""

    # Set cleanup trap
    trap cleanup EXIT

    case "${1:-interactive}" in
        "list")
            list_workflows
            ;;
        "all")
            check_n8n_accessibility
            get_auth_cookie
            import_all_workflows
            ;;
        "status")
            check_n8n_accessibility
            get_auth_cookie
            show_workflow_status
            ;;
        "interactive"|"")
            check_n8n_accessibility
            get_auth_cookie
            select_workflows
            ;;
        *)
            echo "Usage: $0 {interactive|all|list|status}"
            echo ""
            echo "Commands:"
            echo "  interactive - Select specific workflows to import (default)"
            echo "  all         - Import all available workflows"
            echo "  list        - List available workflow files"
            echo "  status      - Show current workflows in n8n"
            echo ""
            echo "Environment Variables:"
            echo "  N8N_URL     - n8n instance URL (default: http://localhost:5678)"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"