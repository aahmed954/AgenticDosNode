# AgenticDosNode Security Cleanup and Preparation Suite

## Overview

This comprehensive security suite provides enterprise-grade procedures for safely cleaning and preparing dedicated machines for AgenticDosNode deployment. The procedures implement defense-in-depth security principles, ensure compliance with major frameworks (GDPR, CIS, NIST), and maintain complete audit trails throughout the process.

## Quick Start

For rapid deployment with interactive configuration:

```bash
sudo bash prepare-agenticdos-node.sh
```

For complete security-focused cleanup:

```bash
sudo bash master-cleanup.sh
```

## File Structure

```
AgenticDosNode/
├── security-cleanup-procedures.md   # Comprehensive documentation
├── prepare-agenticdos-node.sh      # Quick interactive preparation
├── master-cleanup.sh                # Full cleanup orchestration
├── security-verification.sh        # Post-cleanup verification
├── security-rollback.sh           # Emergency rollback script
└── SECURITY-README.md             # This file
```

## Security Features

### 1. Multi-Layer Security
- **Network Security**: Firewall configuration, port hardening, network isolation
- **Access Control**: SSH hardening, sudo restrictions, user account management
- **Data Protection**: Encrypted backups, secure deletion, API key removal
- **System Hardening**: Kernel parameters, mount options, service minimization
- **Monitoring**: Comprehensive audit trails, log aggregation, anomaly detection

### 2. Compliance Framework Support
- **GDPR**: Data minimization, encryption, audit trails, breach notification
- **CIS Benchmarks**: Ubuntu Linux security baseline implementation
- **NIST Cybersecurity**: Framework controls and risk management
- **PCI-DSS**: Security controls for payment card data (if applicable)
- **SOC 2**: Security, availability, and confidentiality controls

### 3. Security Controls

#### Authentication & Authorization
- SSH key-based authentication only
- Multi-factor authentication ready
- Password complexity enforcement (14+ characters)
- Account lockout after 5 failed attempts
- Principle of least privilege

#### Network Security
- UFW firewall with deny-by-default
- Rate limiting on SSH connections
- Disabled unnecessary network services
- Hardened kernel network parameters
- Tailscale VPN integration support

#### Data Security
- AES-256 encryption for backups
- Secure deletion with 3-pass overwrite
- Automatic removal of sensitive files
- Swap space clearing
- Model file secure deletion

#### System Monitoring
- Auditd with comprehensive rules
- Real-time security event monitoring
- Failed authentication tracking
- System call auditing
- File integrity monitoring

## Usage Instructions

### 1. Quick Preparation (Recommended)

Run the interactive preparation script:

```bash
sudo bash prepare-agenticdos-node.sh
```

This will:
- Install security dependencies
- Perform basic cleanup
- Configure essential security
- Run verification checks
- Generate compliance reports

### 2. Full Security Cleanup

For comprehensive cleanup with all security features:

```bash
# First, review what will be cleaned
sudo bash security-audit.sh

# Run the complete cleanup
sudo bash master-cleanup.sh

# Verify the results
sudo bash security-verification.sh
```

### 3. Individual Components

Run specific security phases as needed:

```bash
# Phase 1: Security Assessment
sudo bash security-audit.sh

# Phase 2: Secure Backup
sudo bash secure-backup.sh

# Phase 3: Secure Data Deletion
sudo bash secure-delete.sh

# Phase 4: User Management
sudo bash user-cleanup.sh

# Phase 5: Network Security
sudo bash network-security.sh

# Phase 6: System Hardening
sudo bash filesystem-security.sh

# Phase 7: Audit Configuration
sudo bash audit-configuration.sh

# Phase 8: Compliance Check
sudo bash compliance-check.sh
```

### 4. Verification

After cleanup, verify security posture:

```bash
sudo bash security-verification.sh
```

This provides:
- Security score (0-100%)
- Pass/Fail status for each check
- Compliance verification
- Recommendations for improvements

### 5. Emergency Rollback

If issues occur, rollback security changes:

```bash
sudo bash security-rollback.sh
```

**WARNING**: This reduces security posture. Use only in emergencies.

## Security Verification Checks

The verification script performs 40+ security checks including:

### Critical Checks (Must Pass)
- Firewall enabled and configured
- SSH properly hardened
- Audit system running
- Password policies enforced
- Automatic updates configured

### Important Checks (Should Pass)
- Unnecessary services disabled
- Network parameters hardened
- File permissions secured
- Kernel modules disabled
- User accounts audited

### Compliance Checks
- Lynis hardening score (target: 70+)
- CIS benchmark compliance
- GDPR requirements
- Security baseline verification

## Backup and Recovery

### Encrypted Backups

Backups are created with AES-256 encryption:

```bash
# Backup location
/secure-backup/agenticdos-YYYYMMDD-HHMMSS/

# Encryption key location
/root/.backup-encryption.key

# To decrypt a backup:
openssl enc -d -aes-256-cbc -pass file:/root/.backup-encryption.key \
  -in backup.tar.gz.enc | tar xzf -
```

### Data Recovery

If you need to recover deleted data:

1. Check encrypted backups first
2. Review audit logs for deletion records
3. Use data recovery tools if within time window
4. Contact system administrator for assistance

## Logging and Auditing

All security operations are logged to:

```
/var/log/agenticdos-cleanup/      # Cleanup logs
/var/log/agenticdos-preparation/  # Preparation logs
/var/log/audit/                   # System audit logs
/var/log/auth.log                 # Authentication logs
/var/log/sudo.log                 # Sudo command logs
```

### Log Retention

- Audit logs: 30 days (compressed)
- Security logs: 90 days
- Cleanup logs: Permanent (for compliance)

## Network Access

After security configuration:

### Open Ports
- **22/TCP**: SSH (rate-limited)
- All other ports: Denied by default

### Adding Services

To open additional ports:

```bash
# Example: Open port 443 for HTTPS
sudo ufw allow 443/tcp comment 'HTTPS'

# Example: Open port for specific IP
sudo ufw allow from 192.168.1.100 to any port 3000
```

## User Management

### Default Configuration
- Root login: Disabled (SSH)
- Password authentication: Disabled
- Service accounts: Locked
- Sudo: Logged and restricted

### Adding Users

```bash
# Create new user
sudo adduser newuser

# Add to sudo group (if needed)
sudo usermod -aG sudo newuser

# Set up SSH key
sudo -u newuser mkdir -p /home/newuser/.ssh
sudo -u newuser nano /home/newuser/.ssh/authorized_keys
# Paste public key
sudo chmod 600 /home/newuser/.ssh/authorized_keys
```

## Troubleshooting

### Common Issues

#### Cannot SSH After Hardening
1. Ensure you have SSH key configured
2. Check firewall: `sudo ufw status`
3. Review SSH config: `sudo nano /etc/ssh/sshd_config`
4. Check logs: `sudo tail -f /var/log/auth.log`

#### Services Not Starting
1. Check service status: `sudo systemctl status <service>`
2. Review logs: `sudo journalctl -xe`
3. Verify dependencies: `sudo systemctl list-dependencies <service>`

#### Verification Failures
1. Review specific failure in verification log
2. Apply recommended fixes
3. Re-run verification
4. Consider rollback if critical

### Getting Help

1. Review logs in `/var/log/agenticdos-cleanup/`
2. Check security verification results
3. Consult compliance reports
4. Contact security team if needed

## Maintenance

### Daily Tasks
- Monitor audit logs
- Check failed authentication attempts
- Verify backup completion

### Weekly Tasks
- Run security verification
- Review user access logs
- Update threat intelligence

### Monthly Tasks
- Apply security patches
- Audit user permissions
- Update documentation
- Run compliance checks

## Security Best Practices

1. **Never disable the firewall** without explicit authorization
2. **Always test SSH access** before closing current session
3. **Keep encryption keys secure** and backed up offline
4. **Document all changes** for audit purposes
5. **Regular security audits** using the verification script
6. **Monitor logs actively** for suspicious activity
7. **Update regularly** but test updates first
8. **Principle of least privilege** for all access
9. **Defense in depth** with multiple security layers
10. **Incident response plan** ready and tested

## Compliance Notes

This security suite helps meet requirements for:

- **GDPR Article 32**: Technical and organizational measures
- **CIS Controls**: Prioritized security actions
- **NIST 800-53**: Security and privacy controls
- **ISO 27001**: Information security management
- **PCI-DSS**: Payment card security (if applicable)

## Support and Contribution

For issues, improvements, or questions:

1. Review existing documentation
2. Check verification results
3. Consult security team
4. Submit issues with full logs
5. Follow security disclosure policy

## License and Warranty

These security procedures are provided as-is without warranty. Always:
- Test in non-production first
- Maintain backups
- Document changes
- Follow organizational policies
- Comply with regulations

## Version History

- v1.0.0: Initial release with comprehensive security procedures
- Includes: Cleanup, hardening, verification, rollback
- Compliance: GDPR, CIS, NIST frameworks
- Platform: Ubuntu Linux (tested on 22.04 LTS)

---

**Remember**: Security is a continuous process, not a one-time configuration. Regular audits, updates, and monitoring are essential for maintaining a secure AgenticDosNode deployment.