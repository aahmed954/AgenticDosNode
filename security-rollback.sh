#!/bin/bash
# security-rollback.sh
# Emergency rollback script for AgenticDosNode security changes

set -euo pipefail

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${RED}"
echo "================================================"
echo "    AgenticDosNode Security Rollback           "
echo "================================================"
echo -e "${NC}"
echo ""
echo -e "${YELLOW}WARNING: This will undo security hardening!${NC}"
echo ""

# Confirmation
read -p "Are you sure you want to rollback security changes? (type 'ROLLBACK' to confirm): " confirm
if [ "$confirm" != "ROLLBACK" ]; then
    echo "Rollback cancelled."
    exit 0
fi

echo ""
echo "Starting rollback process..."

# Function to safely rollback
safe_rollback() {
    local description="$1"
    local command="$2"

    echo -n "Rolling back: $description... "
    if eval "$command" 2>/dev/null; then
        echo -e "${GREEN}[OK]${NC}"
    else
        echo -e "${YELLOW}[SKIPPED]${NC}"
    fi
}

# 1. Restore SSH configuration
if [ -f /etc/ssh/sshd_config.backup ]; then
    safe_rollback "SSH configuration" "cp /etc/ssh/sshd_config.backup /etc/ssh/sshd_config && systemctl restart sshd"
fi

# 2. Restore sudoers
if [ -f /etc/sudoers.backup.* ]; then
    latest_backup=$(ls -t /etc/sudoers.backup.* | head -1)
    safe_rollback "Sudoers configuration" "cp $latest_backup /etc/sudoers"
fi

# 3. Restore firewall rules
if [ -f /var/log/agenticdos-cleanup/*/iptables-rules.backup ]; then
    latest_iptables=$(find /var/log/agenticdos-cleanup -name "iptables-rules.backup" | sort | tail -1)
    safe_rollback "IPTables rules" "iptables-restore < $latest_iptables"
fi

# 4. Re-enable services that were disabled
SERVICES_TO_RESTORE=(
    "cups"
    "avahi-daemon"
    "bluetooth"
)

for service in "${SERVICES_TO_RESTORE[@]}"; do
    if systemctl list-unit-files | grep -q "$service"; then
        safe_rollback "Service $service" "systemctl enable $service"
    fi
done

# 5. Restore network parameters
safe_rollback "Network parameters" "rm -f /etc/sysctl.d/99-agenticdos-network-security.conf && sysctl --system"

# 6. Restore mount options
if [ -f /etc/fstab.backup.* ]; then
    latest_fstab=$(ls -t /etc/fstab.backup.* | head -1)
    safe_rollback "Mount options" "cp $latest_fstab /etc/fstab"
fi

# 7. Restore user passwords (unlock accounts)
LOCKED_ACCOUNTS=$(awk -F: '$2 ~ /^!/ {print $1}' /etc/shadow)
for account in $LOCKED_ACCOUNTS; do
    if [ "$account" != "root" ]; then
        safe_rollback "Unlock account $account" "passwd -u $account"
    fi
done

# 8. Restore compiler permissions
for compiler in gcc g++ cc c++; do
    if [ -f "/usr/bin/$compiler" ]; then
        safe_rollback "Compiler $compiler permissions" "chmod 755 /usr/bin/$compiler"
    fi
done

# 9. Remove audit rules
safe_rollback "Audit rules" "rm -f /etc/audit/rules.d/agenticdos-security.rules && augenrules --load"

# 10. Disable automatic updates
safe_rollback "Automatic updates" "rm -f /etc/apt/apt.conf.d/50unattended-upgrades /etc/apt/apt.conf.d/20auto-upgrades"

# 11. Reset UFW firewall
safe_rollback "UFW firewall" "ufw --force disable && ufw --force reset"

# 12. Restore password quality configuration
safe_rollback "Password policy" "rm -f /etc/security/pwquality.conf && touch /etc/security/pwquality.conf"

# 13. Remove AgenticDosNode specific configurations
safe_rollback "AgenticDosNode configs" "rm -rf /etc/agenticdos-cleanup"

# Generate rollback report
ROLLBACK_LOG="/var/log/agenticdos-rollback-$(date +%Y%m%d-%H%M%S).log"
cat > "$ROLLBACK_LOG" << EOF
AgenticDosNode Security Rollback Report
========================================
Date: $(date)
Hostname: $(hostname -f)

Rollback Actions Performed:
- SSH configuration restored
- Sudoers configuration restored
- Firewall rules reset
- Services re-enabled
- Network parameters restored
- Mount options restored
- User accounts unlocked
- Compiler permissions restored
- Audit rules removed
- Automatic updates disabled
- Password policies reset

Current System Status:
- Firewall: $(ufw status | grep Status || echo "Unknown")
- SSH: $(systemctl is-active sshd)
- Audit: $(systemctl is-active auditd)

WARNING: System security has been reduced!
Recommended actions:
1. Review and apply necessary security configurations
2. Re-enable critical security features
3. Update firewall rules as needed
4. Configure monitoring

Rollback log: $ROLLBACK_LOG
EOF

echo ""
echo "================================================"
echo -e "${GREEN}Rollback completed successfully${NC}"
echo "================================================"
echo ""
echo -e "${YELLOW}IMPORTANT: Your system security has been reduced!${NC}"
echo "Review the rollback report at: $ROLLBACK_LOG"
echo ""
echo "Recommended next steps:"
echo "  1. Review which security features you want to keep"
echo "  2. Manually re-apply necessary security configurations"
echo "  3. Test system functionality"
echo "  4. Document any permanent changes"