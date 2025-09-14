# AgenticDosNode Machine Cleanup Scripts

Comprehensive cleanup and preparation scripts for dedicated machines (thanos and oracle1) before AgenticDosNode deployment.

## Scripts Overview

### 1. `cleanup-utils.sh`
Shared utility functions used by all other scripts. Contains logging, backup, process management, and validation functions.

### 2. `cleanup-thanos.sh` (GPU Node)
Comprehensive cleanup script for the **thanos** GPU-enabled machine:
- GPU process cleanup and CUDA service management
- NVIDIA-specific Docker image removal
- GPU memory clearing
- Hardware-accelerated AI framework cleanup
- Mining process detection and removal

### 3. `cleanup-oracle1.sh` (CPU Node)
Comprehensive cleanup script for the **oracle1** CPU-only machine:
- CPU-optimized AI framework cleanup
- Database service management
- Web service cleanup (Jupyter, Gradio, Streamlit)
- Enhanced Python environment analysis
- Model cache management

### 4. `prepare-environment.sh`
Common environment preparation for both machines:
- Docker and Docker Compose installation
- System user creation (`agenticnode`)
- Directory structure setup
- System optimization (limits, kernel parameters)
- Security configuration
- Monitoring tools installation

### 5. `validate-clean-state.sh`
Comprehensive validation script to verify system readiness:
- System resource validation
- Network port availability
- Process state verification
- Configuration validation
- Machine-specific checks (GPU/CPU)

## Usage Instructions

### Phase 1: Cleanup

**For thanos (GPU node):**
```bash
sudo ./cleanup-thanos.sh
```

**For oracle1 (CPU node):**
```bash
sudo ./cleanup-oracle1.sh
```

**Options available:**
- `-y, --yes`: Auto-confirm all prompts (DANGEROUS)
- `-b, --backup-only`: Create backups without cleanup
- `-r, --restore TIMESTAMP`: Restore from backup
- `--dry-run`: Show what would be done

### Phase 2: Environment Preparation

**On both machines:**
```bash
sudo ./prepare-environment.sh
```

**Options available:**
- `-y, --yes`: Auto-confirm all prompts
- `--skip-updates`: Skip system package updates
- `--skip-security`: Skip security configuration
- `--allowed-ips IPS`: Restrict access to specific IPs
- `--install-dir DIR`: Custom installation directory
- `--data-dir DIR`: Custom data directory

### Phase 3: Validation

**On both machines:**
```bash
./validate-clean-state.sh
```

**Options available:**
- `-t, --type TYPE`: Machine type (GPU|CPU|auto)
- `--json-output`: JSON format results
- `--report-only`: Generate report without console output

## Recommended Workflow

### Complete Cleanup and Setup

1. **Run cleanup on each machine:**
   ```bash
   # On thanos:
   sudo ./cleanup-thanos.sh

   # On oracle1:
   sudo ./cleanup-oracle1.sh
   ```

2. **Prepare environment on both machines:**
   ```bash
   sudo ./prepare-environment.sh
   ```

3. **Validate readiness:**
   ```bash
   ./validate-clean-state.sh
   ```

### Safe Cleanup with Backups

1. **Create backups first:**
   ```bash
   sudo ./cleanup-thanos.sh --backup-only
   ```

2. **Review what would be cleaned:**
   ```bash
   sudo ./cleanup-thanos.sh --dry-run
   ```

3. **Perform interactive cleanup:**
   ```bash
   sudo ./cleanup-thanos.sh
   ```

## Safety Features

### Backup System
- Automatic configuration backups before changes
- Restore capability with timestamps
- Backup location: `./cleanup-backups/`

### Interactive Confirmations
- Prompts before destructive operations
- Detailed information about what will be affected
- Option to skip individual cleanup phases

### Comprehensive Logging
- All operations logged with timestamps
- Log location: `./cleanup-logs/`
- Separate logs for each script run

### Rollback Capability
```bash
# List available backups
ls -la cleanup-backups/

# Restore from specific timestamp
sudo ./cleanup-thanos.sh --restore 20241201_143022
```

## Port Management

The scripts will free up these required ports for AgenticDosNode:

**Critical Ports:**
- 3000: Frontend interface
- 8000: API server
- 8001: Admin interface

**Service Ports:**
- 5678: Debug interface
- 6333: Qdrant vector database
- 9090: Prometheus metrics
- 5432: PostgreSQL
- 6379: Redis
- 27017: MongoDB
- 11434: Ollama API

## System Requirements

### Minimum Requirements
- **CPU:** 2+ cores
- **RAM:** 4GB minimum, 8GB recommended
- **Disk:** 20GB minimum, 50GB recommended
- **OS:** Ubuntu 20.04+ or compatible

### GPU Node (thanos) Additional
- **GPU:** NVIDIA GPU with recent drivers
- **CUDA:** Compatible CUDA installation
- **GPU Memory:** 4GB+ recommended

## Troubleshooting

### Common Issues

1. **Port conflicts:**
   ```bash
   # Check what's using a port
   sudo ss -tulpn | grep :8000

   # Force free a port
   sudo ./cleanup-thanos.sh  # Will handle port cleanup
   ```

2. **Docker issues:**
   ```bash
   # Check Docker status
   sudo systemctl status docker

   # Reset Docker completely
   sudo ./cleanup-thanos.sh  # Will clean Docker environment
   ```

3. **Permission errors:**
   ```bash
   # Ensure scripts are executable
   chmod +x *.sh

   # Run with sudo for system modifications
   sudo ./cleanup-thanos.sh
   ```

4. **Validation failures:**
   ```bash
   # Run detailed validation
   ./validate-clean-state.sh -v

   # Check specific issues
   ./validate-clean-state.sh --json-output
   ```

### Recovery Procedures

**If cleanup goes wrong:**
1. Check logs in `./cleanup-logs/`
2. Use restore function with timestamp
3. Manual cleanup of remaining issues

**If preparation fails:**
1. Review preparation logs
2. Fix specific issues identified
3. Re-run preparation script

**If validation fails:**
1. Address critical issues first
2. Review warnings for optimization
3. Re-run validation until clean

## File Structure

```
AgenticDosNode/
├── cleanup-utils.sh           # Shared utility functions
├── cleanup-thanos.sh          # GPU node cleanup
├── cleanup-oracle1.sh         # CPU node cleanup
├── prepare-environment.sh     # Environment preparation
├── validate-clean-state.sh    # System validation
├── CLEANUP_README.md          # This file
├── cleanup-logs/              # Operation logs
│   ├── cleanup_20241201_143022.log
│   └── validation_20241201_143500.log
└── cleanup-backups/           # Configuration backups
    ├── docker_daemon_config_20241201_143022
    ├── restore_map_20241201_143022.txt
    └── ...
```

## Script Features Summary

### cleanup-utils.sh
- Logging and output formatting
- Process detection and management
- Docker container/image management
- Port management and freeing
- Backup and restore functionality
- System service management

### Machine-Specific Scripts
- **GPU (thanos):** GPU memory management, CUDA cleanup, mining detection
- **CPU (oracle1):** Enhanced database cleanup, web service management, model cache cleanup

### prepare-environment.sh
- Docker installation and configuration
- System user and permission setup
- Performance optimization
- Security hardening
- Monitoring tool installation

### validate-clean-state.sh
- Resource requirement verification
- Port availability checking
- Service state validation
- Performance optimization verification
- Machine-specific capability testing

## Exit Codes

- **0:** Success
- **1:** Critical failure (not ready for deployment)
- **2:** Warnings present (proceed with caution)

## Support

For issues with these scripts:
1. Check the comprehensive logs in `cleanup-logs/`
2. Review the validation report for specific problems
3. Use the backup/restore functionality for recovery
4. Run scripts in verbose mode for detailed output

**Remember:** Always run cleanup scripts on dedicated machines only. These scripts are designed for complete system cleanup before AgenticDosNode deployment.