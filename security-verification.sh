#!/bin/bash
# security-verification.sh
# Post-cleanup security verification script for AgenticDosNode

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "================================================"
echo "   AgenticDosNode Security Verification        "
echo "================================================"
echo "Timestamp: $(date -Iseconds)"
echo ""

# Initialize counters
PASS_COUNT=0
FAIL_COUNT=0
WARN_COUNT=0

# Verification results file
RESULTS_FILE="/var/log/agenticdos-verification-$(date +%Y%m%d-%H%M%S).log"

# Function to check and report status
check_status() {
    local check_name="$1"
    local command="$2"
    local expected="$3"

    echo -n "Checking $check_name... "

    if eval "$command" 2>/dev/null; then
        echo -e "${GREEN}[PASS]${NC}"
        echo "[PASS] $check_name" >> "$RESULTS_FILE"
        ((PASS_COUNT++))
    else
        echo -e "${RED}[FAIL]${NC}"
        echo "[FAIL] $check_name - Expected: $expected" >> "$RESULTS_FILE"
        ((FAIL_COUNT++))
    fi
}

# Function for warning checks
check_warning() {
    local check_name="$1"
    local command="$2"

    echo -n "Checking $check_name... "

    if eval "$command" 2>/dev/null; then
        echo -e "${GREEN}[OK]${NC}"
        echo "[OK] $check_name" >> "$RESULTS_FILE"
        ((PASS_COUNT++))
    else
        echo -e "${YELLOW}[WARN]${NC}"
        echo "[WARN] $check_name - Review recommended" >> "$RESULTS_FILE"
        ((WARN_COUNT++))
    fi
}

echo "=== System Security Checks ===" | tee -a "$RESULTS_FILE"
echo ""

# 1. Firewall Status
check_status "UFW Firewall Status" "ufw status | grep -q 'Status: active'" "Firewall should be active"

# 2. SSH Configuration
check_status "SSH Root Login Disabled" "grep -q '^PermitRootLogin no' /etc/ssh/sshd_config || grep -q '^PermitRootLogin prohibit-password' /etc/ssh/sshd_config" "Root login should be disabled"
check_status "SSH Password Auth" "grep -q '^PasswordAuthentication no' /etc/ssh/sshd_config" "Password authentication should be disabled"
check_status "SSH Protocol 2" "! grep -q '^Protocol 1' /etc/ssh/sshd_config" "Only SSH Protocol 2 should be allowed"

# 3. Audit System
check_status "Auditd Service" "systemctl is-active --quiet auditd" "Audit daemon should be running"
check_status "Audit Rules Loaded" "auditctl -l | grep -q 'execve'" "Audit rules should be loaded"

# 4. Network Security
check_status "IPv4 Forwarding Disabled" "[ $(sysctl -n net.ipv4.ip_forward) -eq 0 ]" "IP forwarding should be disabled"
check_status "SYN Cookies Enabled" "[ $(sysctl -n net.ipv4.tcp_syncookies) -eq 1 ]" "SYN cookies should be enabled"
check_status "ICMP Redirects Disabled" "[ $(sysctl -n net.ipv4.conf.all.accept_redirects) -eq 0 ]" "ICMP redirects should be disabled"

# 5. File Permissions
check_status "Shadow File Permissions" "[ $(stat -c %a /etc/shadow) -eq 640 ]" "Shadow file should have 640 permissions"
check_status "SSH Config Permissions" "[ $(stat -c %a /etc/ssh/sshd_config) -eq 600 ]" "SSH config should have 600 permissions"

# 6. Password Policies
check_status "Password Quality Config" "[ -f /etc/security/pwquality.conf ]" "Password quality should be configured"
check_status "Password Minimum Length" "grep -q 'minlen = 14' /etc/security/pwquality.conf" "Minimum password length should be 14"

# 7. Automatic Updates
check_status "Unattended Upgrades" "[ -f /etc/apt/apt.conf.d/50unattended-upgrades ]" "Automatic security updates should be configured"

# 8. User Accounts
check_warning "No Empty Passwords" "! awk -F: '($2 == \"\" || $2 == \"!\") {print $1}' /etc/shadow | grep -v '^$'"
check_warning "No UID 0 Besides Root" "[ $(awk -F: '$3 == 0 {print $1}' /etc/passwd | wc -l) -eq 1 ]"

# 9. Service Hardening
echo ""
echo "=== Service Status Checks ===" | tee -a "$RESULTS_FILE"

UNNECESSARY_SERVICES=(
    "avahi-daemon"
    "cups"
    "bluetooth"
    "nfs-client"
    "rpcbind"
)

for service in "${UNNECESSARY_SERVICES[@]}"; do
    check_warning "$service disabled" "! systemctl is-enabled --quiet $service"
done

# 10. Kernel Modules
echo ""
echo "=== Kernel Module Checks ===" | tee -a "$RESULTS_FILE"

DISABLED_MODULES=(
    "cramfs"
    "freevxfs"
    "jffs2"
    "hfs"
    "hfsplus"
    "udf"
)

for module in "${DISABLED_MODULES[@]}"; do
    check_warning "$module module disabled" "grep -q 'install $module /bin/true' /etc/modprobe.d/*.conf"
done

# 11. Check for sensitive files
echo ""
echo "=== Sensitive Data Checks ===" | tee -a "$RESULTS_FILE"

check_status "No .env files in /home" "! find /home -name '*.env' 2>/dev/null | grep -q '.'" "No .env files should exist"
check_status "No API key files" "! find /home -iname '*api*key*' 2>/dev/null | grep -q '.'" "No API key files should remain"
check_status "No model files" "! find /home -name '*.gguf' -o -name '*.safetensors' 2>/dev/null | grep -q '.'" "No AI model files should remain"

# 12. Compliance Checks
echo ""
echo "=== Compliance Checks ===" | tee -a "$RESULTS_FILE"

# Run Lynis if available
if command -v lynis &>/dev/null; then
    echo "Running Lynis security scan..."
    LYNIS_SCORE=$(lynis audit system --quick 2>/dev/null | grep "Hardening index" | awk '{print $NF}' || echo "0")
    echo "Lynis Hardening Score: $LYNIS_SCORE" | tee -a "$RESULTS_FILE"

    if [ "${LYNIS_SCORE%\[*}" -ge 70 ]; then
        echo -e "${GREEN}[PASS]${NC} Lynis score is acceptable (>= 70)"
        ((PASS_COUNT++))
    else
        echo -e "${YELLOW}[WARN]${NC} Lynis score could be improved"
        ((WARN_COUNT++))
    fi
fi

# 13. Backup Verification
echo ""
echo "=== Backup Verification ===" | tee -a "$RESULTS_FILE"

check_warning "Backup directory exists" "[ -d /secure-backup ]"
check_warning "Encryption key exists" "[ -f /root/.backup-encryption.key ]"

# 14. Log Configuration
echo ""
echo "=== Logging Configuration ===" | tee -a "$RESULTS_FILE"

check_status "Rsyslog running" "systemctl is-active --quiet rsyslog" "System logging should be active"
check_status "Log rotation configured" "[ -f /etc/logrotate.d/rsyslog ]" "Log rotation should be configured"

# Generate Summary Report
echo ""
echo "================================================" | tee -a "$RESULTS_FILE"
echo "           Verification Summary                 " | tee -a "$RESULTS_FILE"
echo "================================================" | tee -a "$RESULTS_FILE"
echo -e "Passed Checks: ${GREEN}$PASS_COUNT${NC}" | tee -a "$RESULTS_FILE"
echo -e "Failed Checks: ${RED}$FAIL_COUNT${NC}" | tee -a "$RESULTS_FILE"
echo -e "Warnings: ${YELLOW}$WARN_COUNT${NC}" | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"

# Calculate security score
TOTAL_CHECKS=$((PASS_COUNT + FAIL_COUNT + WARN_COUNT))
if [ $TOTAL_CHECKS -gt 0 ]; then
    SECURITY_SCORE=$((PASS_COUNT * 100 / TOTAL_CHECKS))
    echo "Security Score: $SECURITY_SCORE%" | tee -a "$RESULTS_FILE"

    if [ $SECURITY_SCORE -ge 90 ]; then
        echo -e "${GREEN}System security is EXCELLENT${NC}" | tee -a "$RESULTS_FILE"
    elif [ $SECURITY_SCORE -ge 70 ]; then
        echo -e "${GREEN}System security is GOOD${NC}" | tee -a "$RESULTS_FILE"
    elif [ $SECURITY_SCORE -ge 50 ]; then
        echo -e "${YELLOW}System security needs IMPROVEMENT${NC}" | tee -a "$RESULTS_FILE"
    else
        echo -e "${RED}System security is POOR - immediate action required${NC}" | tee -a "$RESULTS_FILE"
    fi
fi

echo ""
echo "Detailed results saved to: $RESULTS_FILE"

# Exit with appropriate code
if [ $FAIL_COUNT -gt 0 ]; then
    exit 1
elif [ $WARN_COUNT -gt 0 ]; then
    exit 2
else
    exit 0
fi