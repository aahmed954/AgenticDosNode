#!/bin/bash
# deploy-security.sh - Main deployment script for security hardening

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   log_error "This script must be run as root"
   exit 1
fi

log_info "Starting AI Security Hardening Deployment"

# Create security directories
log_info "Creating security directory structure..."
mkdir -p /opt/ai-security/{configs,scripts,certs,logs,backups}
mkdir -p /etc/ai-security
mkdir -p /var/log/ai-{conversations,audit}
mkdir -p /opt/ai-jail
mkdir -p /forensics

# Set proper permissions
chmod 700 /opt/ai-security
chmod 700 /etc/ai-security
chmod 750 /var/log/ai-conversations
chmod 750 /var/log/ai-audit
chmod 700 /forensics

# Install required packages
log_info "Installing security packages..."
apt-get update
apt-get install -y \
    ufw \
    fail2ban \
    auditd \
    apparmor \
    apparmor-utils \
    libpam-google-authenticator \
    openssl \
    gnupg \
    rkhunter \
    chkrootkit \
    lynis \
    clamav \
    clamav-daemon \
    aide \
    python3-pip \
    jq \
    net-tools \
    iptables-persistent

# Install Python security packages
pip3 install \
    cryptography \
    bcrypt \
    pyotp \
    qrcode \
    hvac \
    redis \
    flask \
    jwt \
    structlog \
    elasticsearch \
    kafka-python

# Configure system security
log_info "Configuring system security parameters..."

# Kernel hardening
cat > /etc/sysctl.d/99-ai-security.conf << 'EOF'
# Network security
net.ipv4.tcp_syncookies = 1
net.ipv4.ip_forward = 0
net.ipv6.conf.all.forwarding = 0
net.ipv4.conf.all.send_redirects = 0
net.ipv4.conf.default.send_redirects = 0
net.ipv4.conf.all.accept_source_route = 0
net.ipv4.conf.default.accept_source_route = 0
net.ipv6.conf.all.accept_source_route = 0
net.ipv6.conf.default.accept_source_route = 0
net.ipv4.conf.all.accept_redirects = 0
net.ipv4.conf.default.accept_redirects = 0
net.ipv6.conf.all.accept_redirects = 0
net.ipv6.conf.default.accept_redirects = 0
net.ipv4.conf.all.secure_redirects = 0
net.ipv4.conf.default.secure_redirects = 0
net.ipv4.conf.all.log_martians = 1
net.ipv4.conf.default.log_martians = 1
net.ipv4.conf.all.rp_filter = 1
net.ipv4.conf.default.rp_filter = 1
net.ipv4.icmp_echo_ignore_broadcasts = 1
net.ipv4.icmp_ignore_bogus_error_responses = 1
net.ipv6.conf.all.accept_ra = 0
net.ipv6.conf.default.accept_ra = 0

# File system security
fs.suid_dumpable = 0
fs.protected_hardlinks = 1
fs.protected_symlinks = 1

# Process security
kernel.randomize_va_space = 2
kernel.yama.ptrace_scope = 1
kernel.core_uses_pid = 1
kernel.kptr_restrict = 2
kernel.dmesg_restrict = 1
kernel.printk = 3 3 3 3
kernel.unprivileged_bpf_disabled = 1
kernel.unprivileged_userns_clone = 0

# Resource limits
kernel.pid_max = 65536
vm.max_map_count = 262144
EOF

sysctl -p /etc/sysctl.d/99-ai-security.conf

# Configure auditd
log_info "Configuring audit system..."
cat > /etc/audit/rules.d/ai-security.rules << 'EOF'
# Delete all rules
-D

# Buffer Size
-b 8192

# Failure Mode
-f 1

# Monitor authentication events
-w /var/log/faillog -p wa -k auth_failures
-w /var/log/lastlog -p wa -k logins
-w /var/log/tallylog -p wa -k logins

# Monitor user/group changes
-w /etc/group -p wa -k identity
-w /etc/passwd -p wa -k identity
-w /etc/gshadow -p wa -k identity
-w /etc/shadow -p wa -k identity
-w /etc/security/opasswd -p wa -k identity

# Monitor sudo commands
-w /usr/bin/sudo -p x -k privileged
-a always,exit -F arch=b64 -S execve -F euid=0 -F key=root-commands

# Monitor AI-specific directories
-w /opt/ai -p wa -k ai_modifications
-w /opt/ai-security -p wa -k security_modifications
-w /var/log/ai-conversations -p rwa -k conversation_access
-w /etc/ai-security -p wa -k config_changes

# Monitor network connections
-a always,exit -F arch=b64 -S socket -F a0=2 -k network
-a always,exit -F arch=b64 -S connect -k network

# Monitor container escapes
-a always,exit -F arch=b64 -S mount -F auid>=1000 -F auid!=4294967295 -k container_mount
-a always,exit -F arch=b64 -S ptrace -k container_escape

# Make configuration immutable
-e 2
EOF

systemctl restart auditd

# Configure fail2ban
log_info "Configuring fail2ban..."
cat > /etc/fail2ban/jail.local << 'EOF'
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 5
destemail = security@domain.com
action = %(action_mwl)s

[sshd]
enabled = true
port = 22
filter = sshd
logpath = /var/log/auth.log
maxretry = 3

[ai-auth]
enabled = true
port = 8080,443
filter = ai-auth
logpath = /var/log/ai-audit/auth.log
maxretry = 5
bantime = 7200

[ai-api]
enabled = true
port = 8080,443
filter = ai-api
logpath = /var/log/ai-audit/api.log
maxretry = 10
findtime = 60
bantime = 3600
EOF

# Create fail2ban filters
cat > /etc/fail2ban/filter.d/ai-auth.conf << 'EOF'
[Definition]
failregex = ^.*Failed authentication attempt.*from <HOST>.*$
            ^.*Invalid credentials.*from <HOST>.*$
            ^.*Account locked.*from <HOST>.*$
ignoreregex =
EOF

cat > /etc/fail2ban/filter.d/ai-api.conf << 'EOF'
[Definition]
failregex = ^.*Unauthorized API access.*from <HOST>.*$
            ^.*Rate limit exceeded.*from <HOST>.*$
            ^.*Invalid API key.*from <HOST>.*$
ignoreregex =
EOF

systemctl restart fail2ban

# Setup UFW firewall
log_info "Configuring UFW firewall..."
ufw --force reset
ufw default deny incoming
ufw default allow outgoing
ufw default deny routed

# Allow Tailscale
ufw allow in on tailscale0
ufw limit in on tailscale0 to any port 22 proto tcp

# Enable logging
ufw logging on
ufw logging high

# Enable firewall
ufw --force enable

# Create AI execution user
log_info "Creating AI execution user..."
useradd -M -s /bin/false aiexec || true
usermod -L aiexec

# Setup chroot jail
log_info "Setting up chroot jail..."
bash -c "$(cat << 'CHROOT_SCRIPT'
JAIL_PATH="/opt/ai-jail"

# Create jail structure
mkdir -p $JAIL_PATH/{bin,lib,lib64,usr,tmp,dev,proc,sys,sandbox}
mkdir -p $JAIL_PATH/usr/{bin,lib}

# Copy essential binaries
BINARIES="bash sh python3 ls cat echo"
for BIN in $BINARIES; do
    if command -v $BIN > /dev/null; then
        cp $(which $BIN) $JAIL_PATH/bin/ 2>/dev/null || true
        # Copy library dependencies
        ldd $(which $BIN) 2>/dev/null | grep -o '/lib.*\.[0-9]' | while read lib; do
            cp --parents "$lib" "$JAIL_PATH" 2>/dev/null || true
        done
    fi
done

# Setup minimal /dev
mknod -m 666 $JAIL_PATH/dev/null c 1 3 2>/dev/null || true
mknod -m 666 $JAIL_PATH/dev/zero c 1 5 2>/dev/null || true
mknod -m 666 $JAIL_PATH/dev/random c 1 8 2>/dev/null || true
mknod -m 666 $JAIL_PATH/dev/urandom c 1 9 2>/dev/null || true

# Set permissions
chown -R root:root $JAIL_PATH
chmod 755 $JAIL_PATH
chmod 1777 $JAIL_PATH/tmp
chown aiexec:aiexec $JAIL_PATH/sandbox
chmod 700 $JAIL_PATH/sandbox
CHROOT_SCRIPT
)"

# Generate encryption keys
log_info "Generating encryption keys..."
openssl rand -hex 32 > /etc/ai-security/env.key
openssl rand -hex 32 > /etc/ai-security/master.key
openssl rand -hex 32 > /etc/ai-security/backup.key
openssl genrsa -out /etc/ai-security/backup-sign.key 4096

# Set key permissions
chmod 600 /etc/ai-security/*.key

# Setup cgroups for resource limitation
log_info "Setting up cgroups..."
if ! grep -q "cgroup2" /proc/filesystems; then
    log_warn "cgroups v2 not available, skipping cgroup setup"
else
    # Create AI cgroup
    cgcreate -g memory,cpu,pids:/ai_sandbox 2>/dev/null || true
fi

# Configure AppArmor profiles
log_info "Configuring AppArmor profiles..."
cat > /etc/apparmor.d/ai-executor << 'EOF'
#include <tunables/global>

profile ai-executor flags=(attach_disconnected,mediate_deleted) {
  #include <abstractions/base>

  network inet tcp,
  network inet udp,
  network inet icmp,
  network netlink raw,

  /opt/ai-jail/ r,
  /opt/ai-jail/** r,
  /opt/ai-jail/sandbox/ r,
  /opt/ai-jail/sandbox/** rw,
  /opt/ai-jail/tmp/ r,
  /opt/ai-jail/tmp/** rw,

  /proc/sys/kernel/ngroups_max r,
  /sys/devices/system/cpu/ r,
  /sys/devices/system/cpu/** r,

  deny /proc/kcore rwx,
  deny /proc/*/mem rwx,
  deny /sys/kernel/security/** rwx,

  deny /bin/mount x,
  deny /bin/umount x,
  deny /usr/bin/sudo x,

  signal (send) peer=ai-executor,
}
EOF

apparmor_parser -r /etc/apparmor.d/ai-executor || true

# Create security monitoring scripts
log_info "Creating security monitoring scripts..."
cat > /opt/ai-security/scripts/security-monitor.sh << 'EOF'
#!/bin/bash
# Security monitoring script

while true; do
    # Check for suspicious processes
    ps aux | grep -E "(nc|netcat|socat|nmap)" | grep -v grep && \
        echo "[ALERT] Suspicious network tool detected" >> /var/log/ai-audit/alerts.log

    # Check for privilege escalation
    find / -type f -perm -4000 -ls 2>/dev/null | \
        diff /opt/ai-security/baseline/suid.list - | \
        grep ">" && echo "[ALERT] New SUID file detected" >> /var/log/ai-audit/alerts.log

    # Check for container escapes
    ps aux | grep -E "nsenter|docker.*exec.*privileged" | grep -v grep && \
        echo "[ALERT] Possible container escape attempt" >> /var/log/ai-audit/alerts.log

    # Monitor outbound connections
    netstat -tunp | grep ESTABLISHED | awk '{print $5}' | \
        grep -vE "(127.0.0.1|::1|tailscale)" | \
        while read conn; do
            echo "[INFO] Outbound connection: $conn" >> /var/log/ai-audit/connections.log
        done

    sleep 60
done
EOF

chmod +x /opt/ai-security/scripts/security-monitor.sh

# Create baseline for monitoring
log_info "Creating security baseline..."
mkdir -p /opt/ai-security/baseline
find / -type f -perm -4000 -ls 2>/dev/null > /opt/ai-security/baseline/suid.list
rpm -qa --qf '%{NAME}-%{VERSION}-%{RELEASE}\n' 2>/dev/null > /opt/ai-security/baseline/packages.list || \
    dpkg -l | awk '{print $2"-"$3}' > /opt/ai-security/baseline/packages.list

# Setup log rotation
log_info "Configuring log rotation..."
cat > /etc/logrotate.d/ai-security << 'EOF'
/var/log/ai-conversations/*.log
/var/log/ai-conversations/*.jsonl
/var/log/ai-audit/*.log
/var/log/ai-audit/*.jsonl
{
    daily
    rotate 90
    compress
    delaycompress
    notifempty
    create 0640 root root
    sharedscripts
    postrotate
        # Encrypt rotated logs
        for file in /var/log/ai-*/*.gz; do
            if [ -f "$file" ]; then
                openssl enc -aes-256-cbc -salt -in "$file" -out "${file}.enc" -pass file:/etc/ai-security/master.key
                rm "$file"
            fi
        done
    endscript
}
EOF

# Create systemd service for security monitor
log_info "Creating systemd services..."
cat > /etc/systemd/system/ai-security-monitor.service << 'EOF'
[Unit]
Description=AI Security Monitor
After=network.target

[Service]
Type=simple
User=root
ExecStart=/opt/ai-security/scripts/security-monitor.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable ai-security-monitor.service
systemctl start ai-security-monitor.service

# Configure AIDE (Advanced Intrusion Detection Environment)
log_info "Configuring AIDE..."
cat > /etc/aide/aide.conf.d/ai-security << 'EOF'
# AI System monitoring
/opt/ai R+b+sha256
/opt/ai-security R+b+sha256
/etc/ai-security R+b+sha256
/usr/local/bin/ai-* R+b+sha256+x
EOF

# Initialize AIDE database
aideinit -y -f

# Setup cron jobs
log_info "Setting up cron jobs..."
cat > /etc/cron.d/ai-security << 'EOF'
# API key rotation (monthly)
0 2 1 * * root /opt/ai-security/scripts/rotate-keys.sh

# Security scan (daily)
0 3 * * * root /usr/bin/lynis audit system --quiet

# AIDE check (daily)
0 4 * * * root /usr/bin/aide --check

# Backup (daily)
0 5 * * * root /opt/ai-security/scripts/backup.sh

# ClamAV update and scan (daily)
0 6 * * * root /usr/bin/freshclam && /usr/bin/clamscan -r /opt/ai --quiet
EOF

# Final security checks
log_info "Running final security checks..."

# Disable unnecessary services
systemctl disable bluetooth.service 2>/dev/null || true
systemctl disable cups.service 2>/dev/null || true
systemctl disable avahi-daemon.service 2>/dev/null || true

# Set secure permissions on sensitive files
chmod 600 /etc/shadow
chmod 600 /etc/gshadow
chmod 644 /etc/passwd
chmod 644 /etc/group

# Remove unnecessary packages
apt-get autoremove -y
apt-get autoclean

# Create completion marker
touch /opt/ai-security/.deployment-complete
date > /opt/ai-security/.deployment-date

log_info "Security hardening deployment completed successfully!"
log_warn "Please review and customize the configuration files in /opt/ai-security/configs/"
log_warn "Remember to:"
echo "  1. Configure Tailscale ACLs according to your network"
echo "  2. Update email addresses in configuration files"
echo "  3. Set up HashiCorp Vault for production secret management"
echo "  4. Configure monitoring endpoints (Prometheus, Grafana)"
echo "  5. Test backup and recovery procedures"
echo "  6. Run a security audit with: lynis audit system"