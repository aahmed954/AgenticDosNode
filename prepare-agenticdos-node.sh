#!/bin/bash
# prepare-agenticdos-node.sh
# Quick deployment script for AgenticDosNode with security focus

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="/var/log/agenticdos-preparation"
BACKUP_DIR="/secure-backup"

# Print banner
print_banner() {
    echo -e "${BLUE}"
    echo "================================================"
    echo "     AgenticDosNode Secure Preparation         "
    echo "================================================"
    echo -e "${NC}"
}

# Print section header
print_section() {
    echo ""
    echo -e "${YELLOW}=== $1 ===${NC}"
}

# Print status
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# Check if running as root
check_root() {
    if [ "$EUID" -ne 0 ]; then
        print_error "This script must be run as root"
        exit 1
    fi
}

# Create necessary directories
setup_directories() {
    print_section "Setting up directories"

    mkdir -p "$LOG_DIR"
    mkdir -p "$BACKUP_DIR"
    mkdir -p /etc/agenticdos-cleanup

    print_status "Directories created"
}

# Install required packages
install_dependencies() {
    print_section "Installing security dependencies"

    apt-get update

    # Core security tools
    PACKAGES=(
        "auditd"
        "aide"
        "rkhunter"
        "lynis"
        "fail2ban"
        "ufw"
        "cryptsetup"
        "secure-delete"
        "apparmor-utils"
        "libpam-pwquality"
        "unattended-upgrades"
        "jq"
        "net-tools"
    )

    for package in "${PACKAGES[@]}"; do
        if dpkg -l | grep -q "^ii.*$package"; then
            echo "  $package already installed"
        else
            echo "  Installing $package..."
            apt-get install -y "$package" >/dev/null 2>&1
        fi
    done

    print_status "Security dependencies installed"
}

# Interactive configuration
interactive_config() {
    print_section "Configuration Options"

    # Backup configuration
    read -p "Enable encrypted backups? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        ENABLE_BACKUP=true
        print_status "Encrypted backups will be configured"
    else
        ENABLE_BACKUP=false
        print_warning "Skipping backup configuration"
    fi

    # Tailscale configuration
    read -p "Configure Tailscale VPN? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        ENABLE_TAILSCALE=true
        print_status "Tailscale will be configured"
    else
        ENABLE_TAILSCALE=false
        print_warning "Skipping Tailscale configuration"
    fi

    # Aggressive cleanup
    read -p "Perform aggressive cleanup? (removes all user data) (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        AGGRESSIVE_CLEANUP=true
        print_warning "Aggressive cleanup enabled - ALL user data will be removed"
    else
        AGGRESSIVE_CLEANUP=false
        print_status "Conservative cleanup mode selected"
    fi
}

# Run security audit
run_security_audit() {
    print_section "Running Security Audit"

    local audit_script="$SCRIPT_DIR/security-audit.sh"
    if [ -f "$audit_script" ]; then
        bash "$audit_script"
        print_status "Security audit completed"
    else
        print_warning "Security audit script not found"
    fi
}

# Perform secure cleanup
perform_cleanup() {
    print_section "Performing Secure Cleanup"

    # Create backup if enabled
    if [ "$ENABLE_BACKUP" = true ]; then
        print_status "Creating encrypted backup..."
        bash "$SCRIPT_DIR/secure-backup.sh" || print_warning "Backup failed"
    fi

    # Remove sensitive data
    if [ "$AGGRESSIVE_CLEANUP" = true ]; then
        print_status "Performing aggressive cleanup..."
        bash "$SCRIPT_DIR/secure-delete.sh" || print_warning "Some files could not be deleted"
    else
        print_status "Performing conservative cleanup..."
        # Only remove obvious sensitive files
        find /home -name "*.env" -delete 2>/dev/null || true
        find /home -name "*api*key*" -delete 2>/dev/null || true
    fi

    print_status "Cleanup completed"
}

# Configure security
configure_security() {
    print_section "Configuring Security"

    # Network security
    print_status "Configuring firewall..."
    ufw --force reset >/dev/null 2>&1
    ufw default deny incoming >/dev/null 2>&1
    ufw default allow outgoing >/dev/null 2>&1
    ufw allow 22/tcp comment 'SSH' >/dev/null 2>&1
    ufw --force enable >/dev/null 2>&1

    # SSH hardening
    print_status "Hardening SSH..."
    cp /etc/ssh/sshd_config /etc/ssh/sshd_config.backup
    cat >> /etc/ssh/sshd_config << 'EOF'

# AgenticDosNode Security Configuration
PermitRootLogin prohibit-password
PasswordAuthentication no
PubkeyAuthentication yes
X11Forwarding no
ClientAliveInterval 300
ClientAliveCountMax 2
MaxAuthTries 3
MaxSessions 2
EOF
    systemctl restart sshd

    # Configure automatic updates
    print_status "Configuring automatic security updates..."
    dpkg-reconfigure -plow unattended-upgrades >/dev/null 2>&1

    # Configure audit system
    print_status "Configuring audit system..."
    systemctl enable auditd >/dev/null 2>&1
    systemctl start auditd >/dev/null 2>&1

    print_status "Security configuration completed"
}

# Setup Tailscale if requested
setup_tailscale() {
    if [ "$ENABLE_TAILSCALE" = true ]; then
        print_section "Setting up Tailscale"

        if ! command -v tailscale &>/dev/null; then
            curl -fsSL https://tailscale.com/install.sh | sh
        fi

        print_status "Tailscale installed"
        print_warning "Run 'tailscale up' to authenticate and connect"
    fi
}

# Run verification
run_verification() {
    print_section "Running Security Verification"

    local verify_script="$SCRIPT_DIR/security-verification.sh"
    if [ -f "$verify_script" ]; then
        bash "$verify_script" || true
    else
        print_warning "Verification script not found"
    fi
}

# Generate final report
generate_report() {
    print_section "Generating Final Report"

    local report_file="$LOG_DIR/preparation-report-$(date +%Y%m%d-%H%M%S).txt"

    cat > "$report_file" << EOF
AgenticDosNode Preparation Report
==================================
Date: $(date)
Hostname: $(hostname -f)
OS: $(lsb_release -d | cut -f2)
Kernel: $(uname -r)

Configuration Options:
- Encrypted Backup: $ENABLE_BACKUP
- Tailscale VPN: $ENABLE_TAILSCALE
- Aggressive Cleanup: $AGGRESSIVE_CLEANUP

Security Status:
- Firewall: $(ufw status | grep Status | awk '{print $2}')
- SSH: Hardened
- Audit: $(systemctl is-active auditd)
- Auto-updates: Configured

Services Status:
$(systemctl list-units --state=running --type=service | grep -E "ssh|ufw|auditd|fail2ban" || echo "Core services check failed")

Network Listeners:
$(ss -tulpn | grep LISTEN | head -10)

Next Steps:
1. Review security verification results
2. Configure application-specific settings
3. Set up monitoring and alerting
4. Schedule regular security audits
5. Document any custom configurations

Report Location: $report_file
Log Directory: $LOG_DIR
EOF

    print_status "Report saved to: $report_file"
}

# Main execution flow
main() {
    print_banner

    # Pre-flight checks
    check_root

    # Setup
    setup_directories
    install_dependencies

    # Interactive configuration
    interactive_config

    # Execute preparation steps
    run_security_audit
    perform_cleanup
    configure_security
    setup_tailscale

    # Verification
    run_verification

    # Reporting
    generate_report

    # Final summary
    print_section "Preparation Complete"
    echo -e "${GREEN}"
    echo "================================================"
    echo "   AgenticDosNode is ready for deployment!     "
    echo "================================================"
    echo -e "${NC}"
    echo ""
    echo "Important reminders:"
    echo "  1. Test SSH access before closing current session"
    echo "  2. Review the security verification results"
    echo "  3. Configure application-specific settings"
    echo "  4. Set up monitoring and alerting"
    echo "  5. Document any custom configurations"
    echo ""
    echo "Security logs: $LOG_DIR"
    echo "Backup location: $BACKUP_DIR"
    echo ""
}

# Trap errors
trap 'print_error "An error occurred. Check logs at $LOG_DIR"' ERR

# Run main function
main "$@"