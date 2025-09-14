# ARM64 Docker Deployment Guide for Oracle Ampere CPU

## Overview

This deployment has been fully optimized for Oracle Cloud Infrastructure (OCI) ARM64 Ampere instances. All Docker images have been verified and configured for ARM64 compatibility.

## Key ARM64 Compatibility Fixes Applied

### 1. Kong API Gateway
- **Issue**: Kong 3.4-alpine has no ARM64 manifest
- **Solution**: Upgraded to `kong:3.7-ubuntu` which has full ARM64 support
- **Note**: Alpine images discontinued since Kong v3.4

### 2. Platform Specifications
- Added `platform: linux/arm64` to all services
- Configured multi-platform builds for custom services
- Verified all third-party images support ARM64

### 3. Restic Backup Service
- **Original**: `restic/restic:latest` (limited ARM64 support)
- **Fixed**: `instrumentisto/restic:latest` (better ARM64 support)

### 4. Service Compatibility Status

| Service | Image | ARM64 Status | Notes |
|---------|-------|--------------|-------|
| PostgreSQL | postgres:15-alpine | ✅ Full Support | Optimized for ARM64 |
| Redis | redis:7-alpine | ✅ Full Support | Native ARM64 build |
| n8n | n8nio/n8n:latest | ✅ Full Support | Multi-arch image |
| Kong | kong:3.7-ubuntu | ✅ Full Support | Ubuntu-based for ARM64 |
| Qdrant | qdrant/qdrant:latest | ✅ Full Support | Native ARM64 support |
| Prometheus | prom/prometheus:latest | ✅ Full Support | Multi-arch image |
| Grafana | grafana/grafana:latest | ✅ Full Support | Multi-arch image |
| Node Exporter | prom/node-exporter:latest | ✅ Full Support | Multi-arch image |
| Tailscale | tailscale/tailscale:latest | ✅ Full Support | Native ARM64 |
| Restic | instrumentisto/restic:latest | ✅ Full Support | ARM64 optimized |

## Oracle Ampere CPU Optimizations

### System Requirements
- **Minimum**: A1.Flex with 4 OCPUs and 24GB RAM
- **Recommended**: A1.Flex with 8 OCPUs and 48GB RAM
- **Storage**: Block volumes with high IOPS for databases

### Applied Optimizations
1. **PostgreSQL**: Configured for ARM64 with appropriate buffer sizes
2. **Redis**: Optimized memory policies for ARM architecture
3. **Kong**: Nginx worker processes set to auto-detect CPU cores
4. **Container Runtime**: BuildKit enabled for faster builds

## Deployment Instructions

### 1. Prerequisites
```bash
# Verify ARM64 architecture
uname -m  # Should output: aarch64 or arm64

# Install Docker and Docker Compose
sudo apt update
sudo apt install -y docker.io docker-compose

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker
```

### 2. Quick Start
```bash
# Clone or navigate to the oracle1 directory
cd /path/to/oracle1

# Verify ARM64 compatibility
./verify-arm64.sh

# Start the stack with ARM64 optimizations
./start-arm64.sh
```

### 3. Manual Deployment
```bash
# Copy ARM64 optimized environment
cp .env.arm64 .env

# Edit .env with your specific values
vim .env

# Initialize Kong database
docker-compose up -d postgres
sleep 10
docker run --rm \
  --platform linux/arm64 \
  --network oracle1_agentic_net \
  -e KONG_DATABASE=postgres \
  -e KONG_PG_HOST=postgres \
  -e KONG_PG_DATABASE=kong \
  -e KONG_PG_USER=agentic \
  -e KONG_PG_PASSWORD=changeme \
  kong:3.7-ubuntu kong migrations bootstrap

# Start all services
docker-compose up -d
```

### 4. Verification
```bash
# Check all services are running
docker-compose ps

# Verify ARM64 images are being used
docker-compose images

# Check service logs
docker-compose logs -f [service-name]
```

## Oracle Cloud Specific Configuration

### Network Security List Rules
Add the following ingress rules to your VCN security list:

| Service | Port | Protocol | Source |
|---------|------|----------|--------|
| PostgreSQL | 5432 | TCP | Your IP/CIDR |
| Redis | 6379 | TCP | Internal only |
| n8n | 5678 | TCP | 0.0.0.0/0 |
| Kong Gateway | 8000 | TCP | 0.0.0.0/0 |
| Kong SSL | 8443 | TCP | 0.0.0.0/0 |
| Kong Admin | 8001 | TCP | Your IP/CIDR |
| LangGraph | 8080 | TCP | 0.0.0.0/0 |
| Claude Proxy | 8081 | TCP | Internal only |
| Qdrant | 6333 | TCP | Internal only |
| Prometheus | 9090 | TCP | Your IP/CIDR |
| Grafana | 3000 | TCP | 0.0.0.0/0 |

### Oracle Linux 8/9 Firewall
```bash
# Open required ports
sudo firewall-cmd --permanent --add-port=5678/tcp
sudo firewall-cmd --permanent --add-port=8000/tcp
sudo firewall-cmd --permanent --add-port=8443/tcp
sudo firewall-cmd --permanent --add-port=8080/tcp
sudo firewall-cmd --permanent --add-port=3000/tcp
sudo firewall-cmd --reload
```

### Performance Tuning
```bash
# Enable hugepages for PostgreSQL (as root)
echo 'vm.nr_hugepages=512' >> /etc/sysctl.conf
sysctl -p

# Set CPU governor to performance
sudo cpupower frequency-set -g performance

# Increase file descriptor limits
echo '* soft nofile 65536' >> /etc/security/limits.conf
echo '* hard nofile 65536' >> /etc/security/limits.conf
```

## Troubleshooting

### Issue: "no matching manifest for linux/arm64"
**Solution**: The image doesn't support ARM64. Check the compatibility table above or use the suggested alternatives.

### Issue: Kong fails to start
**Solution**: Ensure Kong database migrations have been run:
```bash
docker-compose exec kong kong migrations bootstrap
docker-compose restart kong
```

### Issue: Services running slowly
**Solution**: Check resource allocation:
```bash
# Monitor resource usage
docker stats

# Increase memory limits in docker-compose.yml if needed
```

### Issue: Cannot connect to services
**Solution**: Verify Oracle Cloud security lists and local firewall rules are configured correctly.

## Monitoring

Access monitoring dashboards:
- **Grafana**: http://your-oracle-ip:3000 (admin/changeme)
- **Prometheus**: http://your-oracle-ip:9090

## Backup Strategy

The Restic service is configured for automated backups to S3:
```bash
# Manual backup
docker-compose run --rm restic backup /data

# List snapshots
docker-compose run --rm restic snapshots

# Restore from backup
docker-compose run --rm restic restore latest --target /restore
```

## Support

For issues specific to ARM64 deployment:
1. Run `./verify-arm64.sh` to check image compatibility
2. Check service logs: `docker-compose logs [service-name]`
3. Verify architecture: `docker exec [container] uname -m`

## Performance Benchmarks

Expected performance on Oracle A1.Flex (4 OCPU, 24GB RAM):
- PostgreSQL: ~5000 TPS (pgbench)
- Redis: ~100k ops/sec
- Kong: ~10k req/sec (with caching)
- n8n: 100+ concurrent workflows

## Security Considerations

1. Change all default passwords in `.env`
2. Use Tailscale for secure internal networking
3. Enable Kong authentication plugins
4. Configure PostgreSQL SSL
5. Implement regular backup rotation
6. Monitor with Prometheus/Grafana alerts

## Updates and Maintenance

```bash
# Update all images to latest ARM64 versions
docker-compose pull

# Restart services with zero downtime
docker-compose up -d --no-deps --build [service-name]

# Clean up old images
docker image prune -a
```