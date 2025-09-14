#!/bin/bash

# AgenticDosNode Deployment Validation Script
# Comprehensive end-to-end testing of the agentic AI stack

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="${PROJECT_ROOT}/validation.log"

# Test results tracking
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
TEST_RESULTS=()

# Logging functions
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${GREEN}[PASS] $1${NC}" | tee -a "$LOG_FILE"
    ((PASSED_TESTS++))
}

fail() {
    echo -e "${RED}[FAIL] $1${NC}" | tee -a "$LOG_FILE"
    ((FAILED_TESTS++))
}

warning() {
    echo -e "${YELLOW}[WARN] $1${NC}" | tee -a "$LOG_FILE"
}

# Test execution wrapper
run_test() {
    local test_name="$1"
    local test_function="$2"

    ((TOTAL_TESTS++))
    log "Running test: $test_name"

    if $test_function; then
        success "$test_name"
        TEST_RESULTS+=("✅ $test_name")
    else
        fail "$test_name"
        TEST_RESULTS+=("❌ $test_name")
    fi
}

# Network connectivity tests
test_tailscale_connectivity() {
    if command -v tailscale &> /dev/null; then
        if tailscale status &> /dev/null; then
            return 0
        fi
    fi
    return 1
}

test_inter_node_connectivity() {
    # Test connectivity between nodes if in multi-node setup
    local nodes=("thanos" "oracle1")

    for node in "${nodes[@]}"; do
        if command -v tailscale &> /dev/null; then
            local node_ip=$(tailscale status | grep "$node" | awk '{print $1}' || true)
            if [[ -n "$node_ip" ]] && ping -c 1 -W 2 "$node_ip" &> /dev/null; then
                log "Node $node ($node_ip) is reachable"
            else
                return 1
            fi
        fi
    done

    return 0
}

# Service health tests
test_service_health() {
    local service_name="$1"
    local service_url="$2"
    local expected_status="${3:-200}"

    local response_code
    response_code=$(curl -s -o /dev/null -w "%{http_code}" "$service_url" || echo "000")

    if [[ "$response_code" -eq "$expected_status" ]]; then
        return 0
    else
        log "Service $service_name returned status $response_code (expected $expected_status)"
        return 1
    fi
}

test_demo_app() {
    test_service_health "Demo App" "http://localhost:3000" "200"
}

test_orchestrator() {
    test_service_health "Orchestrator" "http://localhost:8000/health" "200"
}

test_n8n() {
    test_service_health "n8n" "http://localhost:5678" "200"
}

test_qdrant() {
    test_service_health "Qdrant" "http://localhost:6333/collections" "200"
}

test_claude_proxy() {
    test_service_health "Claude Proxy" "http://localhost:8001/health" "200"
}

test_prometheus() {
    test_service_health "Prometheus" "http://localhost:9090/-/healthy" "200"
}

test_grafana() {
    test_service_health "Grafana" "http://localhost:3001/api/health" "200"
}

# AI functionality tests
test_ai_chat_basic() {
    local response
    response=$(curl -s -X POST http://localhost:8000/execute \
        -H "Content-Type: application/json" \
        -d '{"task": "Hello, this is a test message", "mode": "auto"}' | \
        jq -r '.response' 2>/dev/null || echo "")

    if [[ -n "$response" ]] && [[ "$response" != "null" ]]; then
        log "AI chat test response: ${response:0:100}..."
        return 0
    fi

    return 1
}

test_model_routing() {
    # Test different model routing scenarios
    local models=("auto" "simple" "complex")

    for mode in "${models[@]}"; do
        local response
        response=$(curl -s -X POST http://localhost:8000/execute \
            -H "Content-Type: application/json" \
            -d "{\"task\": \"What is 2+2?\", \"mode\": \"$mode\"}" | \
            jq -r '.metadata.model_used' 2>/dev/null || echo "")

        if [[ -n "$response" ]] && [[ "$response" != "null" ]]; then
            log "Model routing ($mode): $response"
        else
            return 1
        fi
    done

    return 0
}

test_vector_database() {
    # Create a test collection
    local collection_name="test_validation"

    # Create collection
    curl -s -X PUT "http://localhost:6333/collections/$collection_name" \
        -H "Content-Type: application/json" \
        -d '{"vectors": {"size": 384, "distance": "Cosine"}}' > /dev/null

    # Check collection exists
    local collections
    collections=$(curl -s "http://localhost:6333/collections" | jq -r '.result.collections[].name' 2>/dev/null || echo "")

    if echo "$collections" | grep -q "$collection_name"; then
        # Clean up
        curl -s -X DELETE "http://localhost:6333/collections/$collection_name" > /dev/null
        return 0
    fi

    return 1
}

test_rag_pipeline() {
    # Test document ingestion and search
    local test_text="This is a test document for RAG validation. It contains information about agentic AI systems."

    # Create temporary test file
    echo "$test_text" > /tmp/test_doc.txt

    # Test document upload
    local response
    response=$(curl -s -X POST http://localhost:8000/rag/ingest \
        -F "file=@/tmp/test_doc.txt" | \
        jq -r '.document_id' 2>/dev/null || echo "")

    if [[ -n "$response" ]] && [[ "$response" != "null" ]]; then
        log "Document ingested with ID: $response"

        # Test search
        local search_response
        search_response=$(curl -s -X POST http://localhost:8000/rag/search \
            -H "Content-Type: application/json" \
            -d '{"query": "agentic AI systems", "limit": 5}' | \
            jq -r '.results | length' 2>/dev/null || echo "0")

        # Clean up
        rm -f /tmp/test_doc.txt

        if [[ "$search_response" -gt 0 ]]; then
            return 0
        fi
    fi

    # Clean up
    rm -f /tmp/test_doc.txt
    return 1
}

# Cost optimization tests
test_cost_tracking() {
    local response
    response=$(curl -s "http://localhost:8000/costs" | \
        jq -r '.total_cost_usd' 2>/dev/null || echo "")

    if [[ -n "$response" ]] && [[ "$response" =~ ^[0-9]+\.?[0-9]*$ ]]; then
        log "Cost tracking active: $response USD"
        return 0
    fi

    return 1
}

test_model_statistics() {
    local response
    response=$(curl -s "http://localhost:8000/models/stats" | \
        jq -r '.models | length' 2>/dev/null || echo "0")

    if [[ "$response" -gt 0 ]]; then
        log "Model statistics available for $response models"
        return 0
    fi

    return 1
}

# Security tests
test_security_headers() {
    local headers
    headers=$(curl -s -I "http://localhost:3000" | grep -i "security\|x-\|strict\|content-security" || echo "")

    if [[ -n "$headers" ]]; then
        log "Security headers detected"
        return 0
    fi

    return 1
}

test_authentication() {
    # Test that protected endpoints require authentication
    local response_code
    response_code=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:8001/admin" || echo "000")

    # Should return 401 (Unauthorized) or 403 (Forbidden)
    if [[ "$response_code" -eq 401 ]] || [[ "$response_code" -eq 403 ]]; then
        return 0
    fi

    return 1
}

# Automation tests
test_n8n_workflows() {
    # Check if workflows are loaded
    local workflow_count
    workflow_count=$(curl -s "http://localhost:5678/api/v1/workflows" \
        -H "Accept: application/json" 2>/dev/null | \
        jq -r '.data | length' 2>/dev/null || echo "0")

    if [[ "$workflow_count" -gt 0 ]]; then
        log "n8n has $workflow_count workflows loaded"
        return 0
    fi

    return 1
}

test_webhook_endpoints() {
    # Test that webhook endpoints are responding
    local webhooks=("health-check" "github-integration" "cost-alert")

    for webhook in "${webhooks[@]}"; do
        local response_code
        response_code=$(curl -s -o /dev/null -w "%{http_code}" \
            "http://localhost:5678/webhook/$webhook" || echo "000")

        if [[ "$response_code" -ne 404 ]]; then
            log "Webhook $webhook is available"
        else
            return 1
        fi
    done

    return 0
}

# Performance tests
test_response_times() {
    local url="http://localhost:8000/health"
    local response_time
    response_time=$(curl -o /dev/null -s -w "%{time_total}" "$url" || echo "10.0")

    # Response should be under 2 seconds
    if (( $(echo "$response_time < 2.0" | bc -l) )); then
        log "Response time: ${response_time}s"
        return 0
    fi

    return 1
}

test_concurrent_requests() {
    # Test system handles concurrent requests
    local pids=()

    for i in {1..5}; do
        (curl -s "http://localhost:8000/health" > /dev/null) &
        pids+=($!)
    done

    # Wait for all requests to complete
    local failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            ((failed++))
        fi
    done

    if [[ $failed -eq 0 ]]; then
        return 0
    fi

    return 1
}

# Integration tests
test_end_to_end_workflow() {
    # Test complete workflow: upload document → search → AI analysis
    local test_doc="AI agents are autonomous systems that can perform tasks without human intervention."
    echo "$test_doc" > /tmp/e2e_test.txt

    # Upload document
    local doc_id
    doc_id=$(curl -s -X POST http://localhost:8000/rag/ingest \
        -F "file=@/tmp/e2e_test.txt" | \
        jq -r '.document_id' 2>/dev/null || echo "")

    if [[ -n "$doc_id" ]] && [[ "$doc_id" != "null" ]]; then
        # Search document
        local search_results
        search_results=$(curl -s -X POST http://localhost:8000/rag/search \
            -H "Content-Type: application/json" \
            -d '{"query": "autonomous systems", "limit": 1}')

        # Analyze with AI
        local analysis
        analysis=$(curl -s -X POST http://localhost:8000/execute \
            -H "Content-Type: application/json" \
            -d '{"task": "Analyze the concept of AI agents", "mode": "auto"}' | \
            jq -r '.response' 2>/dev/null || echo "")

        # Clean up
        rm -f /tmp/e2e_test.txt

        if [[ -n "$analysis" ]] && [[ "$analysis" != "null" ]]; then
            log "End-to-end workflow completed successfully"
            return 0
        fi
    fi

    # Clean up
    rm -f /tmp/e2e_test.txt
    return 1
}

# Resource utilization tests
test_resource_usage() {
    # Check that services aren't consuming excessive resources
    local cpu_usage
    cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | awk -F'%' '{print $1}')

    local memory_usage
    memory_usage=$(free | grep Mem | awk '{printf("%.1f", $3/$2 * 100.0)}')

    log "CPU usage: ${cpu_usage}%, Memory usage: ${memory_usage}%"

    # Fail if CPU > 80% or Memory > 90%
    if (( $(echo "$cpu_usage < 80" | bc -l) )) && (( $(echo "$memory_usage < 90" | bc -l) )); then
        return 0
    fi

    return 1
}

# Generate test report
generate_report() {
    echo -e "\n${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║                    VALIDATION REPORT                        ║${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}\n"

    echo -e "${BLUE}Test Summary:${NC}"
    echo -e "  Total Tests: ${TOTAL_TESTS}"
    echo -e "  Passed: ${GREEN}${PASSED_TESTS}${NC}"
    echo -e "  Failed: ${RED}${FAILED_TESTS}${NC}"
    echo -e "  Success Rate: $(( PASSED_TESTS * 100 / TOTAL_TESTS ))%\n"

    echo -e "${BLUE}Test Results:${NC}"
    for result in "${TEST_RESULTS[@]}"; do
        echo "  $result"
    done

    echo -e "\n${BLUE}Logs:${NC} $LOG_FILE\n"

    if [[ $FAILED_TESTS -eq 0 ]]; then
        echo -e "${GREEN}🎉 All tests passed! Your AgenticDosNode deployment is fully operational.${NC}\n"
        return 0
    else
        echo -e "${RED}⚠️  Some tests failed. Please check the logs and fix issues before production use.${NC}\n"
        return 1
    fi
}

# Main validation function
main() {
    log "Starting AgenticDosNode deployment validation..."
    echo -e "${BLUE}Running comprehensive validation tests...${NC}\n"

    # Network tests
    run_test "Tailscale Connectivity" test_tailscale_connectivity
    run_test "Inter-node Connectivity" test_inter_node_connectivity

    # Service health tests
    run_test "Demo Application Health" test_demo_app
    run_test "Orchestrator Health" test_orchestrator
    run_test "n8n Health" test_n8n
    run_test "Qdrant Health" test_qdrant
    run_test "Claude Proxy Health" test_claude_proxy
    run_test "Prometheus Health" test_prometheus
    run_test "Grafana Health" test_grafana

    # AI functionality tests
    run_test "Basic AI Chat" test_ai_chat_basic
    run_test "Model Routing" test_model_routing
    run_test "Vector Database" test_vector_database
    run_test "RAG Pipeline" test_rag_pipeline

    # Cost optimization tests
    run_test "Cost Tracking" test_cost_tracking
    run_test "Model Statistics" test_model_statistics

    # Security tests
    run_test "Security Headers" test_security_headers
    run_test "Authentication" test_authentication

    # Automation tests
    run_test "n8n Workflows" test_n8n_workflows
    run_test "Webhook Endpoints" test_webhook_endpoints

    # Performance tests
    run_test "Response Times" test_response_times
    run_test "Concurrent Requests" test_concurrent_requests

    # Integration tests
    run_test "End-to-End Workflow" test_end_to_end_workflow

    # Resource tests
    run_test "Resource Usage" test_resource_usage

    # Generate final report
    generate_report
}

# Script execution
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi