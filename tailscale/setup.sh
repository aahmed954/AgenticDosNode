#!/bin/bash

# Tailscale Network Setup Script
# This script configures Tailscale on both nodes

set -e

echo "=== Tailscale Network Setup ==="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
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
if [[ $EUID -ne 0 ]]; then
   print_error "This script must be run as root"
   exit 1
fi

# Function to install Tailscale
install_tailscale() {
    local node_name=$1

    print_status "Installing Tailscale on $node_name..."

    # Check if Tailscale is already installed
    if command -v tailscale &> /dev/null; then
        print_warning "Tailscale already installed on $node_name"
        return 0
    fi

    # Install Tailscale
    curl -fsSL https://tailscale.com/install.sh | sh

    print_status "Tailscale installed successfully on $node_name"
}

# Function to configure Tailscale
configure_tailscale() {
    local node_name=$1
    local authkey=$2
    local routes=$3

    print_status "Configuring Tailscale on $node_name..."

    # Start Tailscale with configuration
    tailscale up \
        --authkey="$authkey" \
        --hostname="$node_name" \
        --advertise-routes="$routes" \
        --accept-routes \
        --ssh

    print_status "Tailscale configured successfully on $node_name"
}

# Function to setup firewall rules
setup_firewall() {
    local node_name=$1

    print_status "Setting up firewall rules on $node_name..."

    # Enable IP forwarding
    echo "net.ipv4.ip_forward=1" >> /etc/sysctl.conf
    echo "net.ipv6.conf.all.forwarding=1" >> /etc/sysctl.conf
    sysctl -p

    # UFW rules (if UFW is installed)
    if command -v ufw &> /dev/null; then
        # Allow Tailscale
        ufw allow in on tailscale0
        ufw allow 41641/udp

        # Allow Docker networks
        ufw allow from 172.20.0.0/24
        ufw allow from 172.21.0.0/24

        print_status "UFW rules configured"
    fi

    # iptables rules for Docker integration
    iptables -I DOCKER-USER -i tailscale0 -j ACCEPT
    iptables -I DOCKER-USER -o tailscale0 -j ACCEPT

    # Save iptables rules
    if command -v netfilter-persistent &> /dev/null; then
        netfilter-persistent save
    fi

    print_status "Firewall rules configured successfully"
}

# Function to create systemd service for persistence
create_systemd_service() {
    print_status "Creating systemd service for Tailscale persistence..."

    cat > /etc/systemd/system/tailscale-routes.service << EOF
[Unit]
Description=Tailscale Routes Configuration
After=tailscaled.service
Wants=tailscaled.service

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/bin/tailscale set --advertise-routes=\${ROUTES}
ExecStart=/usr/bin/tailscale set --accept-routes

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    systemctl enable tailscale-routes.service

    print_status "Systemd service created"
}

# Main setup function
main() {
    # Check for required environment variables
    if [ -z "$TAILSCALE_AUTHKEY" ]; then
        print_error "TAILSCALE_AUTHKEY environment variable not set"
        echo "Please set: export TAILSCALE_AUTHKEY=tskey-xxx"
        exit 1
    fi

    # Detect which node we're on
    HOSTNAME=$(hostname)

    case "$HOSTNAME" in
        "thanos")
            NODE_NAME="thanos"
            ROUTES="172.20.0.0/24"
            ;;
        "oracle1")
            NODE_NAME="oracle1"
            ROUTES="172.21.0.0/24"
            ;;
        *)
            print_error "Unknown hostname: $HOSTNAME"
            echo "This script should be run on either 'thanos' or 'oracle1'"
            exit 1
            ;;
    esac

    print_status "Setting up Tailscale for node: $NODE_NAME"

    # Install Tailscale
    install_tailscale "$NODE_NAME"

    # Configure Tailscale
    configure_tailscale "$NODE_NAME" "$TAILSCALE_AUTHKEY" "$ROUTES"

    # Setup firewall
    setup_firewall "$NODE_NAME"

    # Create systemd service
    create_systemd_service

    # Verify connection
    print_status "Verifying Tailscale connection..."
    tailscale status

    print_status "=== Tailscale setup completed successfully ==="
    echo ""
    echo "Next steps:"
    echo "1. Run this script on the other node"
    echo "2. Apply ACL rules: tailscale acl set ./tailscale-acl.json"
    echo "3. Test connectivity: tailscale ping <other-node>"
    echo "4. Start Docker services: docker-compose up -d"
}

# Run main function
main "$@"