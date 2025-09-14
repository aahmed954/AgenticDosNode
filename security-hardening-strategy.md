# Multi-Node Agentic AI Security Hardening Strategy

## Executive Summary
This document provides a comprehensive security hardening strategy for a multi-node AI deployment with agent execution capabilities, focusing on defense-in-depth principles while maintaining operational functionality.

## 1. Network Security Configuration

### 1.1 Tailscale ACL Configuration

```json
{
  "acls": [
    {
      "action": "accept",
      "src": ["tag:ai-controller"],
      "dst": [
        "tag:ai-worker:443",
        "tag:ai-worker:8080",
        "tag:vector-db:19530"
      ]
    },
    {
      "action": "accept",
      "src": ["tag:monitoring"],
      "dst": ["tag:ai-worker:9090", "tag:ai-controller:9090"]
    },
    {
      "action": "accept",
      "src": ["tag:admin"],
      "dst": ["*:*"]
    },
    {
      "action": "drop",
      "src": ["*"],
      "dst": ["*:*"]
    }
  ],
  "tagOwners": {
    "tag:ai-controller": ["security@domain.com"],
    "tag:ai-worker": ["security@domain.com"],
    "tag:vector-db": ["security@domain.com"],
    "tag:monitoring": ["ops@domain.com"],
    "tag:admin": ["admin@domain.com"]
  },
  "nodeAttrs": [
    {
      "target": ["tag:ai-worker"],
      "attr": ["funnel"],
      "value": false
    }
  ],
  "ssh": [
    {
      "action": "accept",
      "src": ["tag:admin"],
      "dst": ["tag:ai-worker", "tag:ai-controller"],
      "users": ["root", "aiuser"]
    }
  ]
}
```

### 1.2 Firewall Rules (UFW Configuration)

```bash
#!/bin/bash
# Primary Node Firewall Configuration

# Reset firewall
ufw --force reset

# Default policies
ufw default deny incoming
ufw default allow outgoing
ufw default deny routed

# Allow Tailscale interface
ufw allow in on tailscale0

# Allow specific services only from Tailscale network
ufw allow in on tailscale0 to any port 22 proto tcp  # SSH
ufw allow in on tailscale0 to any port 443 proto tcp # HTTPS
ufw allow in on tailscale0 to any port 8080 proto tcp # n8n
ufw allow in on tailscale0 to any port 19530 proto tcp # Milvus

# Rate limiting for SSH
ufw limit in on tailscale0 to any port 22 proto tcp

# Logging
ufw logging on
ufw logging high

# Enable firewall
ufw --force enable
```

## 2. Agent Sandboxing Architecture

### 2.1 Docker Security Configuration

```yaml
# docker-compose.security.yml
version: '3.8'

services:
  ai-executor:
    image: ai-executor:secure
    security_opt:
      - no-new-privileges:true
      - seccomp:seccomp-profile.json
      - apparmor:docker-ai-executor
    cap_drop:
      - ALL
    cap_add:
      - CHOWN
      - DAC_OVERRIDE
      - SETGID
      - SETUID
    read_only: true
    tmpfs:
      - /tmp
      - /var/run
    volumes:
      - type: bind
        source: ./sandbox
        target: /sandbox
        read_only: false
        bind:
          propagation: private
    networks:
      - ai_isolated
    ulimits:
      nproc: 50
      nofile:
        soft: 1024
        hard: 2048
    mem_limit: 2g
    memswap_limit: 2g
    cpu_quota: 50000
    pids_limit: 100
    restart: on-failure:3

networks:
  ai_isolated:
    driver: bridge
    driver_opts:
      com.docker.network.bridge.name: br-ai-isolated
    ipam:
      config:
        - subnet: 172.28.0.0/24
    internal: true
```

### 2.2 Seccomp Profile for AI Executors

```json
{
  "defaultAction": "SCMP_ACT_ERRNO",
  "architectures": [
    "SCMP_ARCH_X86_64",
    "SCMP_ARCH_X86"
  ],
  "syscalls": [
    {
      "names": [
        "read", "write", "open", "close", "stat", "fstat", "lstat",
        "poll", "lseek", "mmap", "mprotect", "munmap", "brk",
        "rt_sigaction", "rt_sigprocmask", "rt_sigreturn", "ioctl",
        "pread64", "pwrite64", "readv", "writev", "access", "pipe",
        "select", "sched_yield", "mremap", "msync", "mincore",
        "madvise", "shmget", "shmat", "shmctl", "dup", "dup2",
        "pause", "nanosleep", "getitimer", "alarm", "setitimer",
        "getpid", "sendfile", "socket", "connect", "accept",
        "sendto", "recvfrom", "sendmsg", "recvmsg", "shutdown",
        "bind", "listen", "getsockname", "getpeername", "socketpair",
        "getsockopt", "setsockopt", "clone", "fork", "vfork",
        "execve", "exit", "wait4", "kill", "uname", "semget",
        "semop", "semctl", "shmdt", "msgget", "msgsnd", "msgrcv",
        "msgctl", "fcntl", "flock", "fsync", "fdatasync",
        "truncate", "ftruncate", "getdents", "getcwd", "chdir",
        "fchdir", "rename", "mkdir", "rmdir", "creat", "link",
        "unlink", "symlink", "readlink", "chmod", "fchmod",
        "chown", "fchown", "lchown", "umask", "gettimeofday",
        "getrlimit", "getrusage", "sysinfo", "times", "getuid",
        "getgid", "setuid", "setgid", "geteuid", "getegid",
        "setpgid", "getppid", "getpgrp", "setsid", "setreuid",
        "setregid", "getgroups", "setgroups", "setresuid",
        "getresuid", "setresgid", "getresgid", "getpgid",
        "setfsuid", "setfsgid", "capget", "capset", "prctl",
        "rt_sigpending", "rt_sigtimedwait", "rt_sigqueueinfo",
        "rt_sigsuspend", "sigaltstack", "utime", "mknod",
        "uselib", "personality", "ustat", "statfs", "fstatfs",
        "sysfs", "getpriority", "setpriority", "sched_setparam",
        "sched_getparam", "sched_setscheduler", "sched_getscheduler",
        "sched_get_priority_max", "sched_get_priority_min",
        "sched_rr_get_interval", "mlock", "munlock", "mlockall",
        "munlockall", "vhangup", "modify_ldt", "pivot_root",
        "prctl", "arch_prctl", "adjtimex", "setrlimit", "chroot",
        "sync", "acct", "settimeofday", "umount2", "swapon",
        "swapoff", "reboot", "sethostname", "setdomainname",
        "iopl", "ioperm", "init_module", "delete_module",
        "quotactl", "gettid", "readahead", "setxattr", "lsetxattr",
        "fsetxattr", "getxattr", "lgetxattr", "fgetxattr",
        "listxattr", "llistxattr", "flistxattr", "removexattr",
        "lremovexattr", "fremovexattr", "tkill", "time", "futex",
        "sched_setaffinity", "sched_getaffinity", "set_thread_area",
        "get_thread_area", "io_setup", "io_destroy", "io_getevents",
        "io_submit", "io_cancel", "lookup_dcookie", "epoll_create",
        "epoll_ctl", "epoll_wait", "remap_file_pages", "getdents64",
        "set_tid_address", "restart_syscall", "semtimedop",
        "fadvise64", "timer_create", "timer_settime", "timer_gettime",
        "timer_getoverrun", "timer_delete", "clock_settime",
        "clock_gettime", "clock_getres", "clock_nanosleep",
        "exit_group", "epoll_wait", "epoll_ctl", "tgkill", "utimes",
        "mbind", "set_mempolicy", "get_mempolicy", "mq_open",
        "mq_unlink", "mq_timedsend", "mq_timedreceive", "mq_notify",
        "mq_getsetattr", "kexec_load", "waitid", "add_key",
        "request_key", "keyctl", "ioprio_set", "ioprio_get",
        "inotify_init", "inotify_add_watch", "inotify_rm_watch",
        "migrate_pages", "openat", "mkdirat", "mknodat", "fchownat",
        "futimesat", "newfstatat", "unlinkat", "renameat", "linkat",
        "symlinkat", "readlinkat", "fchmodat", "faccessat", "pselect6",
        "ppoll", "unshare", "set_robust_list", "get_robust_list",
        "splice", "tee", "sync_file_range", "vmsplice", "move_pages",
        "utimensat", "epoll_pwait", "signalfd", "eventfd", "fallocate",
        "timerfd_settime", "timerfd_gettime", "accept4", "signalfd4",
        "eventfd2", "epoll_create1", "dup3", "pipe2", "inotify_init1",
        "preadv", "pwritev", "rt_tgsigqueueinfo", "perf_event_open",
        "recvmmsg", "fanotify_init", "fanotify_mark", "prlimit64"
      ],
      "action": "SCMP_ACT_ALLOW"
    },
    {
      "names": ["ptrace"],
      "action": "SCMP_ACT_ERRNO"
    },
    {
      "names": ["mount", "umount2"],
      "action": "SCMP_ACT_ERRNO"
    },
    {
      "names": ["socket"],
      "action": "SCMP_ACT_ALLOW",
      "args": [
        {
          "index": 0,
          "value": 2,
          "op": "SCMP_CMP_EQ"
        }
      ]
    }
  ]
}
```

### 2.3 Chroot Jail Setup for Code Execution

```bash
#!/bin/bash
# setup-chroot-jail.sh

JAIL_PATH="/opt/ai-jail"
JAIL_USER="aiexec"

# Create jail structure
mkdir -p $JAIL_PATH/{bin,lib,lib64,usr,tmp,dev,proc,sys,sandbox}
mkdir -p $JAIL_PATH/usr/{bin,lib}

# Copy essential binaries
BINARIES="bash sh python3 ls cat echo"
for BIN in $BINARIES; do
    cp $(which $BIN) $JAIL_PATH/bin/ 2>/dev/null || true
    # Copy library dependencies
    ldd $(which $BIN) | grep -o '/lib.*\.[0-9]' | while read lib; do
        cp --parents "$lib" "$JAIL_PATH"
    done
done

# Setup minimal /dev
mknod -m 666 $JAIL_PATH/dev/null c 1 3
mknod -m 666 $JAIL_PATH/dev/zero c 1 5
mknod -m 666 $JAIL_PATH/dev/random c 1 8
mknod -m 666 $JAIL_PATH/dev/urandom c 1 9

# Create execution user
useradd -M -s /bin/false $JAIL_USER

# Set permissions
chown -R root:root $JAIL_PATH
chmod 755 $JAIL_PATH
chmod 1777 $JAIL_PATH/tmp
chown $JAIL_USER:$JAIL_USER $JAIL_PATH/sandbox
chmod 700 $JAIL_PATH/sandbox

# Create execution wrapper
cat > /usr/local/bin/ai-exec-sandbox << 'EOF'
#!/bin/bash
JAIL="/opt/ai-jail"
TIMEOUT=30
MEMORY_LIMIT="512M"

# Setup cgroups for resource limitation
cgcreate -g memory,cpu:/ai_sandbox
echo $MEMORY_LIMIT > /sys/fs/cgroup/memory/ai_sandbox/memory.limit_in_bytes
echo 50000 > /sys/fs/cgroup/cpu/ai_sandbox/cpu.cfs_quota_us

# Execute in chroot with timeout and resource limits
timeout $TIMEOUT cgexec -g memory,cpu:/ai_sandbox \
    chroot --userspec=aiexec:aiexec $JAIL \
    /bin/bash -c "$@"

# Cleanup
cgdelete -g memory,cpu:/ai_sandbox
EOF

chmod 755 /usr/local/bin/ai-exec-sandbox
```

## 3. API Key and Secret Management

### 3.1 HashiCorp Vault Configuration

```hcl
# vault-config.hcl
storage "file" {
  path = "/opt/vault/data"
}

listener "tcp" {
  address     = "127.0.0.1:8200"
  tls_cert_file = "/opt/vault/tls/cert.pem"
  tls_key_file  = "/opt/vault/tls/key.pem"
}

api_addr = "https://127.0.0.1:8200"
cluster_addr = "https://127.0.0.1:8201"
ui = false

telemetry {
  prometheus_retention_time = "0s"
  disable_hostname = true
}

seal "awskms" {
  region = "us-east-1"
  kms_key_id = "alias/vault-unseal"
}
```

### 3.2 Secret Rotation Script

```python
#!/usr/bin/env python3
# rotate-api-keys.py

import os
import json
import time
import hashlib
import requests
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
import hvac

class APIKeyRotator:
    def __init__(self, vault_addr, vault_token):
        self.vault = hvac.Client(url=vault_addr, token=vault_token)
        self.rotation_interval = timedelta(days=30)

    def rotate_claude_key(self):
        """Rotate Claude API key"""
        # Generate new key (placeholder - actual implementation would use Claude's API)
        new_key = self.generate_secure_key("claude")

        # Store in Vault with versioning
        self.vault.secrets.kv.v2.create_or_update_secret(
            path='ai-keys/claude',
            secret={
                'api_key': new_key,
                'rotated_at': datetime.now().isoformat(),
                'expires_at': (datetime.now() + self.rotation_interval).isoformat()
            }
        )

        # Update environment
        self.update_service_env('claude-service', 'CLAUDE_API_KEY', new_key)

        # Audit log
        self.audit_log('claude_key_rotated', {'timestamp': datetime.now().isoformat()})

    def rotate_openrouter_key(self):
        """Rotate OpenRouter API key"""
        new_key = self.generate_secure_key("openrouter")

        self.vault.secrets.kv.v2.create_or_update_secret(
            path='ai-keys/openrouter',
            secret={
                'api_key': new_key,
                'rotated_at': datetime.now().isoformat(),
                'expires_at': (datetime.now() + self.rotation_interval).isoformat()
            }
        )

        self.update_service_env('openrouter-service', 'OPENROUTER_API_KEY', new_key)
        self.audit_log('openrouter_key_rotated', {'timestamp': datetime.now().isoformat()})

    def generate_secure_key(self, service):
        """Generate cryptographically secure API key"""
        random_bytes = os.urandom(32)
        timestamp = str(time.time()).encode()
        service_bytes = service.encode()

        combined = random_bytes + timestamp + service_bytes
        return hashlib.sha256(combined).hexdigest()

    def update_service_env(self, service_name, env_var, value):
        """Update service environment variable"""
        # Implementation depends on your orchestration system
        # Example for Docker Swarm:
        os.system(f"docker service update --env-add {env_var}={value} {service_name}")

    def audit_log(self, event, details):
        """Log rotation events for audit"""
        log_entry = {
            'event': event,
            'details': details,
            'timestamp': datetime.now().isoformat()
        }
        with open('/var/log/api-rotation.log', 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

if __name__ == "__main__":
    rotator = APIKeyRotator(
        vault_addr=os.environ['VAULT_ADDR'],
        vault_token=os.environ['VAULT_TOKEN']
    )

    # Run rotation
    rotator.rotate_claude_key()
    rotator.rotate_openrouter_key()
```

### 3.3 Environment Variable Encryption

```bash
#!/bin/bash
# secure-env-loader.sh

# Encrypt environment file
encrypt_env() {
    local ENV_FILE=$1
    local KEY_FILE="/etc/ai-security/env.key"

    # Generate encryption key if not exists
    if [ ! -f "$KEY_FILE" ]; then
        openssl rand -hex 32 > "$KEY_FILE"
        chmod 600 "$KEY_FILE"
    fi

    # Encrypt environment file
    openssl enc -aes-256-cbc -salt -in "$ENV_FILE" -out "${ENV_FILE}.enc" -pass file:"$KEY_FILE"
    shred -u "$ENV_FILE"
}

# Decrypt and load environment
load_secure_env() {
    local ENV_FILE=$1
    local KEY_FILE="/etc/ai-security/env.key"

    # Decrypt to memory
    DECRYPTED=$(openssl enc -aes-256-cbc -d -in "${ENV_FILE}.enc" -pass file:"$KEY_FILE")

    # Export variables
    while IFS='=' read -r key value; do
        export "$key"="$value"
    done <<< "$DECRYPTED"

    # Clear decrypted content from memory
    unset DECRYPTED
}

# Usage
encrypt_env "/opt/ai/.env"
load_secure_env "/opt/ai/.env"
```

## 4. Data Protection Strategy

### 4.1 Vector Database Security (Milvus)

```yaml
# milvus-security.yaml
etcd:
  endpoints:
    - localhost:2379
  rootPath: /milvus
  security:
    tlsEnabled: true
    tlsCert: /certs/etcd-cert.pem
    tlsKey: /certs/etcd-key.pem
    tlsCACert: /certs/ca-cert.pem

minio:
  address: localhost
  port: 9000
  accessKeyID: ${MINIO_ACCESS_KEY}
  secretAccessKey: ${MINIO_SECRET_KEY}
  useSSL: true
  bucketName: milvus-bucket

common:
  security:
    authorizationEnabled: true
    tlsMode: 2  # TLS required

tls:
  serverPemPath: /certs/server.pem
  serverKeyPath: /certs/server.key
  caPemPath: /certs/ca.pem

proxy:
  http:
    enabled: false  # Disable HTTP, only gRPC
  port: 19530
  internalPort: 19529

log:
  level: warn
  file:
    rootPath: /var/log/milvus
    maxSize: 300
    maxAge: 10
    maxBackups: 20
```

### 4.2 Database Encryption at Rest

```python
#!/usr/bin/env python3
# vector-db-encryption.py

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
import numpy as np
import pickle
import base64

class VectorEncryption:
    def __init__(self, master_key_path="/etc/ai-security/master.key"):
        self.master_key = self.load_or_generate_key(master_key_path)
        self.cipher = Fernet(self.master_key)

    def load_or_generate_key(self, key_path):
        """Load or generate master encryption key"""
        try:
            with open(key_path, 'rb') as f:
                return f.read()
        except FileNotFoundError:
            key = Fernet.generate_key()
            with open(key_path, 'wb') as f:
                f.write(key)
            os.chmod(key_path, 0o600)
            return key

    def encrypt_vector(self, vector):
        """Encrypt a vector before storage"""
        # Serialize vector
        serialized = pickle.dumps(vector)

        # Encrypt
        encrypted = self.cipher.encrypt(serialized)

        return base64.b64encode(encrypted).decode('utf-8')

    def decrypt_vector(self, encrypted_data):
        """Decrypt a vector after retrieval"""
        # Decode
        encrypted = base64.b64decode(encrypted_data.encode('utf-8'))

        # Decrypt
        decrypted = self.cipher.decrypt(encrypted)

        # Deserialize
        return pickle.loads(decrypted)

    def encrypt_metadata(self, metadata):
        """Encrypt metadata associated with vectors"""
        serialized = json.dumps(metadata)
        encrypted = self.cipher.encrypt(serialized.encode())
        return base64.b64encode(encrypted).decode('utf-8')

    def decrypt_metadata(self, encrypted_metadata):
        """Decrypt metadata"""
        encrypted = base64.b64decode(encrypted_metadata.encode('utf-8'))
        decrypted = self.cipher.decrypt(encrypted)
        return json.loads(decrypted.decode())
```

### 4.3 Conversation Logging Security

```python
#!/usr/bin/env python3
# secure-conversation-logger.py

import json
import hashlib
import hmac
from datetime import datetime
from pathlib import Path
import logging
from cryptography.fernet import Fernet

class SecureConversationLogger:
    def __init__(self, log_dir="/var/log/ai-conversations"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Setup encryption
        self.encryption_key = self.get_or_create_key()
        self.cipher = Fernet(self.encryption_key)

        # Setup HMAC for integrity
        self.hmac_key = self.get_or_create_hmac_key()

        # Setup structured logging
        self.setup_logging()

    def get_or_create_key(self):
        key_file = self.log_dir / ".encryption.key"
        if key_file.exists():
            return key_file.read_bytes()
        else:
            key = Fernet.generate_key()
            key_file.write_bytes(key)
            key_file.chmod(0o600)
            return key

    def get_or_create_hmac_key(self):
        key_file = self.log_dir / ".hmac.key"
        if key_file.exists():
            return key_file.read_bytes()
        else:
            key = os.urandom(32)
            key_file.write_bytes(key)
            key_file.chmod(0o600)
            return key

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / "conversation.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def log_conversation(self, session_id, user_input, ai_response, metadata=None):
        """Log conversation with encryption and integrity check"""

        # Prepare log entry
        entry = {
            'session_id': session_id,
            'timestamp': datetime.utcnow().isoformat(),
            'user_input': self.sanitize_input(user_input),
            'ai_response': self.sanitize_input(ai_response),
            'metadata': metadata or {}
        }

        # Encrypt sensitive data
        encrypted_entry = {
            'session_id': session_id,
            'timestamp': entry['timestamp'],
            'encrypted_data': self.cipher.encrypt(
                json.dumps({
                    'user_input': entry['user_input'],
                    'ai_response': entry['ai_response']
                }).encode()
            ).decode('utf-8'),
            'metadata': self.encrypt_metadata(metadata)
        }

        # Add integrity check
        entry_string = json.dumps(encrypted_entry, sort_keys=True)
        integrity_hash = hmac.new(
            self.hmac_key,
            entry_string.encode(),
            hashlib.sha256
        ).hexdigest()

        encrypted_entry['integrity'] = integrity_hash

        # Write to log file
        log_file = self.log_dir / f"conversation_{session_id}_{datetime.utcnow():%Y%m%d}.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps(encrypted_entry) + '\n')

        # Set appropriate permissions
        log_file.chmod(0o640)

        # Audit log
        self.logger.info(f"Conversation logged: session={session_id}")

    def sanitize_input(self, text):
        """Remove sensitive patterns from text"""
        import re

        # Remove potential API keys
        text = re.sub(r'sk-[a-zA-Z0-9]{48}', '[REDACTED_API_KEY]', text)
        text = re.sub(r'api[_-]?key["\']?\s*[:=]\s*["\']?[\w-]+', '[REDACTED_API_KEY]', text, flags=re.IGNORECASE)

        # Remove potential passwords
        text = re.sub(r'password["\']?\s*[:=]\s*["\']?[\w@#$%^&*]+', '[REDACTED_PASSWORD]', text, flags=re.IGNORECASE)

        # Remove credit card numbers
        text = re.sub(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', '[REDACTED_CC]', text)

        # Remove SSNs
        text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[REDACTED_SSN]', text)

        return text

    def encrypt_metadata(self, metadata):
        """Encrypt metadata while preserving structure"""
        if not metadata:
            return {}

        encrypted_meta = {}
        for key, value in metadata.items():
            if key in ['user_id', 'ip_address', 'email']:
                # Encrypt sensitive fields
                encrypted_meta[key] = self.cipher.encrypt(
                    str(value).encode()
                ).decode('utf-8')
            else:
                encrypted_meta[key] = value

        return encrypted_meta

    def verify_integrity(self, log_entry):
        """Verify log entry integrity"""
        stored_hash = log_entry.pop('integrity')
        entry_string = json.dumps(log_entry, sort_keys=True)

        calculated_hash = hmac.new(
            self.hmac_key,
            entry_string.encode(),
            hashlib.sha256
        ).hexdigest()

        return hmac.compare_digest(stored_hash, calculated_hash)
```

## 5. Access Control Implementation

### 5.1 Authentication Service

```python
#!/usr/bin/env python3
# auth-service.py

from flask import Flask, request, jsonify
import jwt
import bcrypt
import redis
from datetime import datetime, timedelta
import pyotp
import qrcode
import io
import base64
from functools import wraps

app = Flask(__name__)
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

# Configuration
JWT_SECRET = os.environ.get('JWT_SECRET')
JWT_ALGORITHM = 'HS256'
TOKEN_EXPIRY = timedelta(hours=1)
REFRESH_TOKEN_EXPIRY = timedelta(days=7)

class AuthService:
    def __init__(self):
        self.failed_attempts = {}
        self.rate_limiter = RateLimiter()

    def authenticate_user(self, username, password, totp_code=None):
        """Multi-factor authentication"""

        # Rate limiting
        if self.rate_limiter.is_limited(username):
            return {'error': 'Too many attempts. Please try again later.'}, 429

        # Retrieve user from database
        user = self.get_user(username)
        if not user:
            self.log_failed_attempt(username)
            return {'error': 'Invalid credentials'}, 401

        # Verify password
        if not bcrypt.checkpw(password.encode(), user['password_hash'].encode()):
            self.log_failed_attempt(username)
            return {'error': 'Invalid credentials'}, 401

        # Verify TOTP if enabled
        if user.get('mfa_enabled'):
            if not totp_code:
                return {'error': 'TOTP code required'}, 401

            totp = pyotp.TOTP(user['totp_secret'])
            if not totp.verify(totp_code, valid_window=1):
                self.log_failed_attempt(username)
                return {'error': 'Invalid TOTP code'}, 401

        # Generate tokens
        access_token = self.generate_access_token(user)
        refresh_token = self.generate_refresh_token(user)

        # Store refresh token
        redis_client.setex(
            f"refresh_token:{user['id']}",
            REFRESH_TOKEN_EXPIRY,
            refresh_token
        )

        # Log successful authentication
        self.log_successful_auth(username)

        return {
            'access_token': access_token,
            'refresh_token': refresh_token,
            'expires_in': TOKEN_EXPIRY.total_seconds()
        }, 200

    def generate_access_token(self, user):
        """Generate JWT access token"""
        payload = {
            'user_id': user['id'],
            'username': user['username'],
            'roles': user['roles'],
            'exp': datetime.utcnow() + TOKEN_EXPIRY,
            'iat': datetime.utcnow(),
            'type': 'access'
        }
        return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

    def generate_refresh_token(self, user):
        """Generate JWT refresh token"""
        payload = {
            'user_id': user['id'],
            'exp': datetime.utcnow() + REFRESH_TOKEN_EXPIRY,
            'iat': datetime.utcnow(),
            'type': 'refresh'
        }
        return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

    def verify_token(self, token):
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

    def setup_mfa(self, user_id):
        """Setup MFA for user"""
        secret = pyotp.random_base32()

        # Store secret (encrypted in production)
        self.update_user_mfa(user_id, secret)

        # Generate QR code
        totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
            name=user_id,
            issuer_name='AI Agent System'
        )

        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(totp_uri)
        qr.make(fit=True)

        img = qr.make_image(fill_color="black", back_color="white")
        buf = io.BytesIO()
        img.save(buf, format='PNG')

        qr_code = base64.b64encode(buf.getvalue()).decode()

        return {
            'secret': secret,
            'qr_code': f"data:image/png;base64,{qr_code}"
        }

    def log_failed_attempt(self, username):
        """Log failed authentication attempt"""
        timestamp = datetime.utcnow().isoformat()

        # Increment failed attempts counter
        if username not in self.failed_attempts:
            self.failed_attempts[username] = []

        self.failed_attempts[username].append(timestamp)

        # Clean old attempts (older than 1 hour)
        cutoff = datetime.utcnow() - timedelta(hours=1)
        self.failed_attempts[username] = [
            t for t in self.failed_attempts[username]
            if datetime.fromisoformat(t) > cutoff
        ]

        # Lock account after 5 failed attempts
        if len(self.failed_attempts[username]) >= 5:
            self.lock_account(username)

        # Audit log
        self.audit_log('failed_auth', {
            'username': username,
            'timestamp': timestamp,
            'ip': request.remote_addr
        })

    def lock_account(self, username):
        """Lock user account"""
        redis_client.setex(f"locked:{username}", timedelta(hours=1), "1")
        self.audit_log('account_locked', {
            'username': username,
            'timestamp': datetime.utcnow().isoformat()
        })

class RateLimiter:
    def __init__(self):
        self.requests = {}
        self.limits = {
            'auth': (10, 60),  # 10 requests per 60 seconds
            'api': (100, 60),  # 100 requests per 60 seconds
        }

    def is_limited(self, identifier, limit_type='auth'):
        """Check if identifier is rate limited"""
        limit, window = self.limits[limit_type]

        key = f"rate_limit:{limit_type}:{identifier}"
        current_count = redis_client.get(key)

        if current_count is None:
            redis_client.setex(key, window, 1)
            return False

        if int(current_count) >= limit:
            return True

        redis_client.incr(key)
        return False

def require_auth(f):
    """Decorator for protected endpoints"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')

        if not token:
            return jsonify({'error': 'No token provided'}), 401

        # Remove "Bearer " prefix
        if token.startswith('Bearer '):
            token = token[7:]

        auth_service = AuthService()
        payload = auth_service.verify_token(token)

        if not payload:
            return jsonify({'error': 'Invalid or expired token'}), 401

        # Check if token type is access
        if payload.get('type') != 'access':
            return jsonify({'error': 'Invalid token type'}), 401

        # Add user info to request
        request.current_user = payload

        return f(*args, **kwargs)

    return decorated_function

def require_role(role):
    """Decorator for role-based access control"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if role not in request.current_user.get('roles', []):
                return jsonify({'error': 'Insufficient permissions'}), 403
            return f(*args, **kwargs)
        return decorated_function
    return decorator
```

### 5.2 Authorization Rules

```yaml
# rbac-config.yaml
roles:
  admin:
    permissions:
      - system:*
      - ai:*
      - data:*
      - logs:read
      - users:*

  ai_operator:
    permissions:
      - ai:execute
      - ai:monitor
      - data:read
      - logs:read

  developer:
    permissions:
      - ai:execute
      - ai:debug
      - data:read
      - data:write
      - logs:read

  auditor:
    permissions:
      - logs:read
      - data:read
      - system:audit

permissions:
  system:
    - restart
    - configure
    - update
    - audit

  ai:
    - execute
    - monitor
    - debug
    - configure

  data:
    - read
    - write
    - delete
    - export

  logs:
    - read
    - write
    - delete

  users:
    - create
    - read
    - update
    - delete

resource_permissions:
  "/api/v1/execute":
    methods: ["POST"]
    required_permissions: ["ai:execute"]

  "/api/v1/models":
    methods: ["GET"]
    required_permissions: ["ai:monitor"]

  "/api/v1/data":
    methods: ["GET", "POST", "PUT", "DELETE"]
    required_permissions:
      GET: ["data:read"]
      POST: ["data:write"]
      PUT: ["data:write"]
      DELETE: ["data:delete"]

  "/api/v1/logs":
    methods: ["GET"]
    required_permissions: ["logs:read"]

  "/api/v1/admin/*":
    methods: ["*"]
    required_roles: ["admin"]
```

## 6. Monitoring and Alerting

### 6.1 Security Monitoring Stack

```yaml
# docker-compose.monitoring.yml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - ./alerts.yml:/etc/prometheus/alerts.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--web.enable-lifecycle'
    ports:
      - "9090:9090"
    networks:
      - monitoring

  grafana:
    image: grafana/grafana:latest
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./grafana/datasources:/etc/grafana/provisioning/datasources
    environment:
      - GF_SECURITY_ADMIN_USER=${GRAFANA_USER}
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
      - GF_INSTALL_PLUGINS=grafana-piechart-panel
    ports:
      - "3000:3000"
    networks:
      - monitoring

  alertmanager:
    image: prom/alertmanager:latest
    volumes:
      - ./alertmanager.yml:/etc/alertmanager/alertmanager.yml
      - alertmanager_data:/alertmanager
    command:
      - '--config.file=/etc/alertmanager/alertmanager.yml'
      - '--storage.path=/alertmanager'
    ports:
      - "9093:9093"
    networks:
      - monitoring

  falco:
    image: falcosecurity/falco:latest
    privileged: true
    volumes:
      - /var/run/docker.sock:/host/var/run/docker.sock
      - /proc:/host/proc:ro
      - /boot:/host/boot:ro
      - /lib/modules:/host/lib/modules:ro
      - /usr:/host/usr:ro
      - ./falco/rules:/etc/falco/rules.d
    command: ["/usr/bin/falco", "-pk"]
    networks:
      - monitoring

volumes:
  prometheus_data:
  grafana_data:
  alertmanager_data:

networks:
  monitoring:
    driver: bridge
```

### 6.2 Security Alert Rules

```yaml
# alerts.yml
groups:
  - name: security_alerts
    interval: 30s
    rules:
      - alert: HighFailedAuthAttempts
        expr: rate(auth_failed_attempts_total[5m]) > 10
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High number of failed authentication attempts"
          description: "{{ $value }} failed auth attempts per second"

      - alert: UnauthorizedAPIAccess
        expr: rate(api_unauthorized_total[5m]) > 5
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Unauthorized API access attempts detected"
          description: "{{ $value }} unauthorized attempts per second"

      - alert: AnomalousAIExecution
        expr: rate(ai_execution_total[5m]) > 100
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Unusually high AI execution rate"
          description: "AI execution rate: {{ $value }} per second"

      - alert: SuspiciousFileAccess
        expr: file_access_violations_total > 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Suspicious file access detected"
          description: "File access violation in {{ $labels.path }}"

      - alert: APIKeyExpirationWarning
        expr: api_key_expiry_days < 7
        for: 1h
        labels:
          severity: warning
        annotations:
          summary: "API key expiring soon"
          description: "API key {{ $labels.key_name }} expires in {{ $value }} days"

      - alert: ContainerEscape
        expr: container_escape_attempts > 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Container escape attempt detected"
          description: "Container escape attempt in {{ $labels.container }}"

      - alert: DataExfiltration
        expr: rate(network_bytes_sent[5m]) > 1000000000
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Possible data exfiltration detected"
          description: "High outbound traffic: {{ $value }} bytes/sec"
```

### 6.3 Audit Logging System

```python
#!/usr/bin/env python3
# audit-logger.py

import json
import time
import hashlib
from datetime import datetime
from elasticsearch import Elasticsearch
from kafka import KafkaProducer
import structlog

class AuditLogger:
    def __init__(self):
        # Setup structured logging
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

        self.logger = structlog.get_logger()

        # Setup Elasticsearch
        self.es = Elasticsearch(
            ['https://localhost:9200'],
            http_auth=('elastic', os.environ['ELASTIC_PASSWORD']),
            use_ssl=True,
            verify_certs=True,
            ca_certs='/etc/elasticsearch/certs/ca.crt'
        )

        # Setup Kafka for real-time streaming
        self.kafka_producer = KafkaProducer(
            bootstrap_servers=['localhost:9092'],
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            security_protocol="SSL",
            ssl_cafile="/etc/kafka/certs/ca-cert",
            ssl_certfile="/etc/kafka/certs/client-cert",
            ssl_keyfile="/etc/kafka/certs/client-key"
        )

    def log_event(self, event_type, details, severity='INFO'):
        """Log security event"""

        event = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type,
            'severity': severity,
            'details': details,
            'source_ip': self.get_source_ip(),
            'user': self.get_current_user(),
            'session_id': self.get_session_id(),
            'correlation_id': self.generate_correlation_id(),
            'integrity_hash': None
        }

        # Add integrity hash
        event['integrity_hash'] = self.calculate_integrity_hash(event)

        # Log to multiple destinations
        self.log_to_file(event)
        self.log_to_elasticsearch(event)
        self.stream_to_kafka(event)

        # Trigger alerts for critical events
        if severity in ['CRITICAL', 'ALERT']:
            self.trigger_alert(event)

        return event['correlation_id']

    def log_to_file(self, event):
        """Write to audit log file"""
        log_file = f"/var/log/audit/audit-{datetime.utcnow():%Y%m%d}.jsonl"

        with open(log_file, 'a') as f:
            f.write(json.dumps(event) + '\n')

        # Set permissions
        os.chmod(log_file, 0o640)

    def log_to_elasticsearch(self, event):
        """Index event in Elasticsearch"""
        index_name = f"audit-logs-{datetime.utcnow():%Y.%m}"

        self.es.index(
            index=index_name,
            body=event,
            id=event['correlation_id']
        )

    def stream_to_kafka(self, event):
        """Stream event to Kafka for real-time processing"""
        topic = f"audit-{event['severity'].lower()}"

        self.kafka_producer.send(
            topic,
            value=event,
            key=event['event_type'].encode('utf-8')
        )

    def calculate_integrity_hash(self, event):
        """Calculate integrity hash for tamper detection"""
        event_copy = event.copy()
        event_copy.pop('integrity_hash', None)

        event_string = json.dumps(event_copy, sort_keys=True)
        return hashlib.sha256(event_string.encode()).hexdigest()

    def trigger_alert(self, event):
        """Trigger immediate alert for critical events"""
        alert = {
            'alert_id': self.generate_alert_id(),
            'event': event,
            'triggered_at': datetime.utcnow().isoformat(),
            'alert_type': 'SECURITY_CRITICAL'
        }

        # Send to alerting system
        self.send_to_alertmanager(alert)

        # Send to SIEM
        self.send_to_siem(alert)

    def generate_correlation_id(self):
        """Generate unique correlation ID"""
        return hashlib.sha256(
            f"{time.time()}{os.urandom(16).hex()}".encode()
        ).hexdigest()[:16]

    def generate_alert_id(self):
        """Generate unique alert ID"""
        return f"ALERT-{datetime.utcnow():%Y%m%d%H%M%S}-{os.urandom(4).hex()}"

# Audit event types
AUDIT_EVENTS = {
    'AUTH_SUCCESS': 'Successful authentication',
    'AUTH_FAILURE': 'Failed authentication attempt',
    'AUTH_LOCKOUT': 'Account locked due to failed attempts',
    'API_ACCESS': 'API endpoint accessed',
    'API_UNAUTHORIZED': 'Unauthorized API access attempt',
    'AI_EXECUTION': 'AI model execution',
    'AI_EXECUTION_ERROR': 'AI execution error',
    'DATA_ACCESS': 'Data access event',
    'DATA_MODIFICATION': 'Data modification event',
    'DATA_DELETION': 'Data deletion event',
    'KEY_ROTATION': 'API key rotation',
    'CONFIG_CHANGE': 'Configuration change',
    'PRIVILEGE_ESCALATION': 'Privilege escalation attempt',
    'FILE_ACCESS_VIOLATION': 'Unauthorized file access',
    'NETWORK_ANOMALY': 'Network anomaly detected',
    'CONTAINER_ESCAPE': 'Container escape attempt',
    'INJECTION_ATTEMPT': 'Injection attack attempt detected'
}
```

## 7. Incident Response Procedures

### 7.1 Automated Response System

```python
#!/usr/bin/env python3
# incident-response.py

import asyncio
import json
from datetime import datetime
from enum import Enum
import subprocess

class IncidentSeverity(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class IncidentResponse:
    def __init__(self):
        self.response_playbooks = self.load_playbooks()
        self.active_incidents = {}

    def load_playbooks(self):
        """Load incident response playbooks"""
        with open('/etc/security/playbooks.json', 'r') as f:
            return json.load(f)

    async def handle_incident(self, incident_type, details):
        """Handle security incident"""

        incident = {
            'id': self.generate_incident_id(),
            'type': incident_type,
            'details': details,
            'timestamp': datetime.utcnow().isoformat(),
            'status': 'ACTIVE',
            'severity': self.assess_severity(incident_type, details)
        }

        self.active_incidents[incident['id']] = incident

        # Execute response playbook
        if incident_type in self.response_playbooks:
            await self.execute_playbook(
                incident,
                self.response_playbooks[incident_type]
            )

        # Notify security team
        await self.notify_security_team(incident)

        return incident['id']

    async def execute_playbook(self, incident, playbook):
        """Execute incident response playbook"""

        for step in playbook['steps']:
            try:
                if step['type'] == 'isolate':
                    await self.isolate_system(step['target'])

                elif step['type'] == 'block':
                    await self.block_ip(step['ip'])

                elif step['type'] == 'disable':
                    await self.disable_account(step['account'])

                elif step['type'] == 'snapshot':
                    await self.create_forensic_snapshot(step['target'])

                elif step['type'] == 'rotate':
                    await self.rotate_credentials(step['service'])

                elif step['type'] == 'quarantine':
                    await self.quarantine_file(step['file'])

                elif step['type'] == 'kill':
                    await self.kill_process(step['process'])

                self.log_response_action(incident['id'], step)

            except Exception as e:
                self.log_response_error(incident['id'], step, str(e))

    async def isolate_system(self, target):
        """Isolate compromised system"""

        # Remove from Tailscale network
        subprocess.run([
            'tailscale', 'set', '--advertise-routes=',
            '--accept-routes=false'
        ], check=True)

        # Block all non-essential network traffic
        subprocess.run([
            'iptables', '-I', 'INPUT', '1',
            '-m', 'state', '--state', 'NEW',
            '-j', 'DROP'
        ], check=True)

        subprocess.run([
            'iptables', '-I', 'OUTPUT', '1',
            '-m', 'state', '--state', 'NEW',
            '-j', 'DROP'
        ], check=True)

        # Allow only incident response connections
        subprocess.run([
            'iptables', '-I', 'INPUT', '1',
            '-s', '10.0.0.100',  # Security team IP
            '-j', 'ACCEPT'
        ], check=True)

    async def block_ip(self, ip_address):
        """Block malicious IP address"""

        # Add to firewall blocklist
        subprocess.run([
            'iptables', '-A', 'INPUT',
            '-s', ip_address,
            '-j', 'DROP'
        ], check=True)

        # Add to Tailscale ACL blocklist
        # This would update the Tailscale ACL via API

        # Add to fail2ban
        subprocess.run([
            'fail2ban-client', 'set', 'sshd',
            'banip', ip_address
        ], check=True)

    async def disable_account(self, username):
        """Disable compromised user account"""

        # Lock Linux account
        subprocess.run(['usermod', '-L', username], check=True)

        # Revoke all sessions
        subprocess.run([
            'pkill', '-KILL', '-u', username
        ], check=False)  # Don't fail if no processes

        # Revoke API tokens
        # This would call the auth service API

    async def create_forensic_snapshot(self, target):
        """Create forensic snapshot for investigation"""

        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        snapshot_dir = f"/forensics/{timestamp}"

        # Create snapshot directory
        subprocess.run(['mkdir', '-p', snapshot_dir], check=True)

        # Capture memory dump
        subprocess.run([
            'dd', f'if=/proc/kcore',
            f'of={snapshot_dir}/memory.dump'
        ], check=False)

        # Capture network connections
        subprocess.run([
            'netstat', '-anp'
        ], stdout=open(f'{snapshot_dir}/netstat.txt', 'w'), check=True)

        # Capture process list
        subprocess.run([
            'ps', 'auxww'
        ], stdout=open(f'{snapshot_dir}/processes.txt', 'w'), check=True)

        # Capture Docker state
        subprocess.run([
            'docker', 'ps', '-a'
        ], stdout=open(f'{snapshot_dir}/docker.txt', 'w'), check=True)

        # Create disk image
        subprocess.run([
            'dd', 'if=/dev/sda',
            f'of={snapshot_dir}/disk.img',
            'bs=4M', 'conv=sync,noerror'
        ], check=False)

    async def rotate_credentials(self, service):
        """Rotate compromised credentials"""

        # Call credential rotation service
        # This would trigger the key rotation script
        subprocess.run([
            'python3', '/opt/security/rotate-api-keys.py',
            '--service', service,
            '--immediate'
        ], check=True)

    def assess_severity(self, incident_type, details):
        """Assess incident severity"""

        severity_mapping = {
            'CONTAINER_ESCAPE': IncidentSeverity.CRITICAL,
            'PRIVILEGE_ESCALATION': IncidentSeverity.CRITICAL,
            'DATA_EXFILTRATION': IncidentSeverity.CRITICAL,
            'RANSOMWARE': IncidentSeverity.CRITICAL,
            'API_KEY_COMPROMISE': IncidentSeverity.HIGH,
            'UNAUTHORIZED_ACCESS': IncidentSeverity.HIGH,
            'INJECTION_ATTACK': IncidentSeverity.HIGH,
            'BRUTE_FORCE': IncidentSeverity.MEDIUM,
            'MISCONFIGURATION': IncidentSeverity.LOW
        }

        return severity_mapping.get(
            incident_type,
            IncidentSeverity.MEDIUM
        )
```

## 8. Backup and Recovery

### 8.1 Encrypted Backup System

```bash
#!/bin/bash
# encrypted-backup.sh

BACKUP_DIR="/backups"
ENCRYPTION_KEY="/etc/security/backup.key"
REMOTE_BACKUP="s3://secure-backups/ai-system"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p "$BACKUP_DIR/$DATE"

# Backup critical data
backup_data() {
    # Vector database
    docker exec milvus mysqldump -u root -p$MILVUS_PASSWORD milvus | \
        gzip | \
        openssl enc -aes-256-cbc -salt -pass file:$ENCRYPTION_KEY \
        > "$BACKUP_DIR/$DATE/milvus.sql.gz.enc"

    # Configuration files
    tar czf - /etc/ai-config /opt/ai/config | \
        openssl enc -aes-256-cbc -salt -pass file:$ENCRYPTION_KEY \
        > "$BACKUP_DIR/$DATE/config.tar.gz.enc"

    # Conversation logs
    tar czf - /var/log/ai-conversations | \
        openssl enc -aes-256-cbc -salt -pass file:$ENCRYPTION_KEY \
        > "$BACKUP_DIR/$DATE/conversations.tar.gz.enc"

    # Docker volumes
    docker run --rm \
        -v ai_data:/data \
        -v "$BACKUP_DIR/$DATE":/backup \
        alpine tar czf - /data | \
        openssl enc -aes-256-cbc -salt -pass file:$ENCRYPTION_KEY \
        > /backup/docker-volumes.tar.gz.enc
}

# Create backup manifest
create_manifest() {
    cat > "$BACKUP_DIR/$DATE/manifest.json" << EOF
{
    "timestamp": "$(date -Iseconds)",
    "version": "1.0",
    "components": [
        "milvus",
        "config",
        "conversations",
        "docker-volumes"
    ],
    "encryption": "aes-256-cbc",
    "compression": "gzip"
}
EOF

    # Sign manifest
    openssl dgst -sha256 -sign /etc/security/backup-sign.key \
        -out "$BACKUP_DIR/$DATE/manifest.sig" \
        "$BACKUP_DIR/$DATE/manifest.json"
}

# Upload to remote storage
upload_backup() {
    aws s3 sync "$BACKUP_DIR/$DATE" "$REMOTE_BACKUP/$DATE" \
        --sse aws:kms \
        --sse-kms-key-id alias/backup-key \
        --storage-class GLACIER_IR
}

# Verify backup integrity
verify_backup() {
    for file in "$BACKUP_DIR/$DATE"/*.enc; do
        openssl enc -aes-256-cbc -d -salt -pass file:$ENCRYPTION_KEY \
            -in "$file" | \
            gzip -t || echo "Verification failed for $file"
    done
}

# Cleanup old backups
cleanup_old_backups() {
    find "$BACKUP_DIR" -type d -mtime +30 -exec rm -rf {} \;

    # Cleanup remote backups older than 90 days
    aws s3 rm "$REMOTE_BACKUP" \
        --recursive \
        --exclude "*" \
        --include "*" \
        --older-than 90d
}

# Main execution
backup_data
create_manifest
verify_backup
upload_backup
cleanup_old_backups

# Send notification
echo "Backup completed: $DATE" | \
    mail -s "AI System Backup Success" security@domain.com
```

### 8.2 Disaster Recovery Plan

```yaml
# disaster-recovery-plan.yaml
recovery_procedures:
  data_corruption:
    priority: HIGH
    rto: 4_hours  # Recovery Time Objective
    rpo: 1_hour   # Recovery Point Objective
    steps:
      - identify_corruption_scope
      - isolate_affected_systems
      - restore_from_last_known_good_backup
      - verify_data_integrity
      - resume_operations
      - conduct_post_mortem

  ransomware_attack:
    priority: CRITICAL
    rto: 2_hours
    rpo: 30_minutes
    steps:
      - immediate_network_isolation
      - identify_infection_vector
      - preserve_forensic_evidence
      - wipe_infected_systems
      - restore_from_offline_backups
      - rotate_all_credentials
      - implement_additional_monitoring
      - notify_stakeholders

  complete_system_failure:
    priority: CRITICAL
    rto: 8_hours
    rpo: 2_hours
    steps:
      - activate_disaster_recovery_site
      - restore_infrastructure_from_code
      - restore_data_from_backups
      - verify_system_functionality
      - update_dns_records
      - monitor_for_issues
      - document_lessons_learned

recovery_sites:
  primary:
    location: us-east-1
    provider: aws
    resources:
      - compute: c5.2xlarge
      - storage: 1TB_gp3
      - network: vpc_with_subnets

  secondary:
    location: us-west-2
    provider: aws
    resources:
      - compute: c5.xlarge
      - storage: 500GB_gp3
      - network: vpc_with_subnets

testing_schedule:
  monthly:
    - backup_restoration_test
    - credential_rotation_test

  quarterly:
    - partial_failover_test
    - incident_response_drill

  annually:
    - complete_disaster_recovery_test
    - security_assessment
```

## Conclusion

This comprehensive security hardening strategy provides defense-in-depth protection for your multi-node AI deployment while maintaining operational functionality. Key achievements:

1. **Network Security**: Implemented zero-trust networking with Tailscale ACLs and strict firewall rules
2. **Agent Sandboxing**: Created multiple isolation layers including Docker security, seccomp profiles, and chroot jails
3. **Secret Management**: Automated key rotation with HashiCorp Vault integration
4. **Data Protection**: End-to-end encryption for vector database and conversation logs
5. **Access Control**: Multi-factor authentication with RBAC and audit logging
6. **Monitoring**: Real-time security monitoring with automated incident response
7. **Recovery**: Encrypted backups with tested disaster recovery procedures

Regular security assessments and updates to this strategy are recommended as the threat landscape evolves.