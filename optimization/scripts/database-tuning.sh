#!/bin/bash

# Database Performance Tuning Script for AgenticDosNode
# Optimizes PostgreSQL, Redis, and Qdrant for AI workloads

set -euo pipefail

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
BACKUP_DIR="/etc/agentic-backup/databases"
LOG_FILE="/var/log/agentic-database-tuning.log"
TOTAL_MEM=$(free -b | awk '/^Mem:/{print $2}')
TOTAL_MEM_GB=$((TOTAL_MEM / 1024 / 1024 / 1024))
CPU_CORES=$(nproc)

# Logging
log() {
    echo -e "${2:-INFO}: $1" | tee -a "$LOG_FILE"
    logger -t "agentic-db" "$1"
}

# Backup database configurations
backup_configs() {
    log "Backing up database configurations" "${BLUE}"
    mkdir -p "$BACKUP_DIR"

    # Backup PostgreSQL config if exists
    if [ -f "/etc/postgresql/postgresql.conf" ]; then
        cp /etc/postgresql/postgresql.conf "$BACKUP_DIR/postgresql.conf.bak"
    fi

    # Backup Redis config if exists
    if [ -f "/etc/redis/redis.conf" ]; then
        cp /etc/redis/redis.conf "$BACKUP_DIR/redis.conf.bak"
    fi

    log "Database configurations backed up to $BACKUP_DIR" "${GREEN}"
}

# PostgreSQL optimization for n8n workflows
optimize_postgresql() {
    log "Optimizing PostgreSQL for n8n and LangGraph workflows" "${BLUE}"

    # Calculate optimal PostgreSQL settings based on available memory
    SHARED_BUFFERS=$((TOTAL_MEM_GB / 4))GB  # 25% of RAM
    EFFECTIVE_CACHE=$((TOTAL_MEM_GB * 3 / 4))GB  # 75% of RAM
    MAINTENANCE_MEM=$((TOTAL_MEM_GB / 16))GB  # 6.25% of RAM
    WORK_MEM=$((TOTAL_MEM / 1024 / 1024 / 100))  # ~1% of RAM / 100 connections

    cat > /tmp/postgresql-optimization.conf << EOF
# PostgreSQL Performance Optimization for AgenticDosNode
# Generated on $(date)
# System: ${TOTAL_MEM_GB}GB RAM, ${CPU_CORES} CPU cores

#------------------------------------------------------------------------------
# CONNECTIONS AND AUTHENTICATION
#------------------------------------------------------------------------------
max_connections = 200
superuser_reserved_connections = 5

#------------------------------------------------------------------------------
# MEMORY
#------------------------------------------------------------------------------
shared_buffers = ${SHARED_BUFFERS}
effective_cache_size = ${EFFECTIVE_CACHE}
maintenance_work_mem = ${MAINTENANCE_MEM}
work_mem = ${WORK_MEM}MB
huge_pages = try
temp_buffers = 32MB
max_prepared_transactions = 100
track_activity_query_size = 2048

#------------------------------------------------------------------------------
# DISK
#------------------------------------------------------------------------------
random_page_cost = 1.1  # SSD optimization
effective_io_concurrency = 200  # SSD optimization
max_worker_processes = ${CPU_CORES}
max_parallel_workers_per_gather = $((CPU_CORES / 2))
max_parallel_workers = ${CPU_CORES}
max_parallel_maintenance_workers = $((CPU_CORES / 2))

#------------------------------------------------------------------------------
# WAL (Write-Ahead Logging)
#------------------------------------------------------------------------------
wal_level = replica
wal_buffers = 16MB
wal_compression = on
wal_log_hints = on
wal_writer_delay = 200ms
checkpoint_segments = 32
checkpoint_completion_target = 0.9
max_wal_size = 4GB
min_wal_size = 1GB
archive_mode = on
archive_command = '/bin/true'

#------------------------------------------------------------------------------
# QUERY TUNING
#------------------------------------------------------------------------------
enable_partitionwise_join = on
enable_partitionwise_aggregate = on
jit = on
jit_above_cost = 100000
jit_inline_above_cost = 500000
jit_optimize_above_cost = 500000

#------------------------------------------------------------------------------
# REPORTING AND LOGGING
#------------------------------------------------------------------------------
log_destination = 'csvlog'
logging_collector = on
log_directory = '/var/log/postgresql'
log_filename = 'postgresql-%Y-%m-%d_%H%M%S.log'
log_rotation_age = 1d
log_rotation_size = 100MB
log_min_duration_statement = 100  # Log slow queries > 100ms
log_checkpoints = on
log_connections = on
log_disconnections = on
log_lock_waits = on
log_temp_files = 0
log_autovacuum_min_duration = 0
log_line_prefix = '%t [%p]: [%l-1] user=%u,db=%d,app=%a,client=%h '

#------------------------------------------------------------------------------
# STATISTICS
#------------------------------------------------------------------------------
track_activities = on
track_counts = on
track_io_timing = on
track_functions = all
stats_temp_directory = '/var/run/postgresql/stats_temp'

#------------------------------------------------------------------------------
# AUTOVACUUM
#------------------------------------------------------------------------------
autovacuum = on
autovacuum_max_workers = 4
autovacuum_naptime = 10s
autovacuum_vacuum_threshold = 50
autovacuum_analyze_threshold = 50
autovacuum_vacuum_scale_factor = 0.1
autovacuum_analyze_scale_factor = 0.05
autovacuum_vacuum_cost_delay = 2ms
autovacuum_vacuum_cost_limit = 1000

#------------------------------------------------------------------------------
# LOCK MANAGEMENT
#------------------------------------------------------------------------------
deadlock_timeout = 1s
max_locks_per_transaction = 256
max_pred_locks_per_transaction = 256
max_pred_locks_per_relation = -2
max_pred_locks_per_page = 2

#------------------------------------------------------------------------------
# CONNECTION POOLING (for pgbouncer compatibility)
#------------------------------------------------------------------------------
session_preload_libraries = 'auto_explain,pg_stat_statements'
shared_preload_libraries = 'pg_stat_statements'
pg_stat_statements.max = 10000
pg_stat_statements.track = all
auto_explain.log_min_duration = '100ms'
auto_explain.log_analyze = true
auto_explain.log_buffers = true

#------------------------------------------------------------------------------
# CUSTOM SETTINGS FOR AI WORKLOADS
#------------------------------------------------------------------------------
# Optimize for OLTP with occasional OLAP queries
default_statistics_target = 100
constraint_exclusion = partition
cursor_tuple_fraction = 0.1
from_collapse_limit = 8
join_collapse_limit = 8
force_parallel_mode = off
EOF

    log "PostgreSQL optimization configuration created" "${GREEN}"

    # Create connection pooling configuration for pgbouncer
    cat > /tmp/pgbouncer.ini << EOF
[databases]
n8n = host=localhost port=5432 dbname=n8n
langgraph = host=localhost port=5432 dbname=langgraph
kong = host=localhost port=5432 dbname=kong

[pgbouncer]
listen_addr = *
listen_port = 6432
auth_type = md5
auth_file = /etc/pgbouncer/userlist.txt
admin_users = postgres
stats_users = postgres, monitor

# Pool settings
pool_mode = transaction
max_client_conn = 1000
default_pool_size = 25
reserve_pool_size = 5
reserve_pool_timeout = 3
max_db_connections = 100
max_user_connections = 100

# Timeouts
server_lifetime = 3600
server_idle_timeout = 600
server_connect_timeout = 15
server_login_retry = 15
query_timeout = 0
query_wait_timeout = 120
client_idle_timeout = 0
client_login_timeout = 60

# Low-level tuning
pkt_buf = 4096
max_packet_size = 2147483647
listen_backlog = 128
sbuf_loopcnt = 5
suspend_timeout = 10
tcp_defer_accept = 45
tcp_socket_buffer = 0
tcp_keepalive = 1
tcp_keepcnt = 9
tcp_keepidle = 7200
tcp_keepintvl = 75

# Logging
log_connections = 1
log_disconnections = 1
log_pooler_errors = 1
stats_period = 60
verbose = 0

# Security
server_tls_sslmode = prefer
server_tls_protocols = secure
server_tls_ciphers = fast
EOF

    log "PgBouncer configuration created" "${GREEN}"
}

# Redis optimization for caching and sessions
optimize_redis() {
    log "Optimizing Redis for caching and session management" "${BLUE}"

    # Calculate Redis memory limit (10% of total RAM for caching)
    REDIS_MAXMEM=$((TOTAL_MEM_GB / 10))gb

    cat > /tmp/redis-optimization.conf << EOF
# Redis Performance Optimization for AgenticDosNode
# Generated on $(date)

#------------------------------------------------------------------------------
# NETWORK
#------------------------------------------------------------------------------
bind 0.0.0.0
protected-mode yes
port 6379
tcp-backlog 511
tcp-keepalive 300
timeout 0

#------------------------------------------------------------------------------
# GENERAL
#------------------------------------------------------------------------------
daemonize no
supervised systemd
pidfile /var/run/redis/redis-server.pid
loglevel notice
logfile /var/log/redis/redis-server.log
databases 16

#------------------------------------------------------------------------------
# SNAPSHOTTING
#------------------------------------------------------------------------------
save 900 1
save 300 10
save 60 10000
stop-writes-on-bgsave-error yes
rdbcompression yes
rdbchecksum yes
dbfilename dump.rdb
dir /var/lib/redis

#------------------------------------------------------------------------------
# REPLICATION
#------------------------------------------------------------------------------
replica-read-only yes
repl-diskless-sync no
repl-diskless-sync-delay 5
repl-ping-replica-period 10
repl-timeout 60
repl-disable-tcp-nodelay no
repl-backlog-size 64mb
repl-backlog-ttl 3600

#------------------------------------------------------------------------------
# MEMORY MANAGEMENT
#------------------------------------------------------------------------------
maxmemory ${REDIS_MAXMEM}
maxmemory-policy allkeys-lru
maxmemory-samples 5
lazyfree-lazy-eviction yes
lazyfree-lazy-expire yes
lazyfree-lazy-server-del yes
replica-lazy-flush yes
lazyfree-lazy-user-del yes

#------------------------------------------------------------------------------
# APPEND ONLY MODE
#------------------------------------------------------------------------------
appendonly yes
appendfilename "appendonly.aof"
appendfsync everysec
no-appendfsync-on-rewrite no
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size 64mb
aof-load-truncated yes
aof-use-rdb-preamble yes

#------------------------------------------------------------------------------
# LUA SCRIPTING
#------------------------------------------------------------------------------
lua-time-limit 5000

#------------------------------------------------------------------------------
# REDIS CLUSTER
#------------------------------------------------------------------------------
# cluster-enabled no
# cluster-config-file nodes-6379.conf
# cluster-node-timeout 15000

#------------------------------------------------------------------------------
# SLOW LOG
#------------------------------------------------------------------------------
slowlog-log-slower-than 10000
slowlog-max-len 128

#------------------------------------------------------------------------------
# LATENCY MONITOR
#------------------------------------------------------------------------------
latency-monitor-threshold 100

#------------------------------------------------------------------------------
# EVENT NOTIFICATION
#------------------------------------------------------------------------------
notify-keyspace-events ""

#------------------------------------------------------------------------------
# ADVANCED CONFIG
#------------------------------------------------------------------------------
hash-max-ziplist-entries 512
hash-max-ziplist-value 64
list-max-ziplist-size -2
list-compress-depth 0
set-max-intset-entries 512
zset-max-ziplist-entries 128
zset-max-ziplist-value 64
hll-sparse-max-bytes 3000
stream-node-max-bytes 4096
stream-node-max-entries 100
activerehashing yes
client-output-buffer-limit normal 0 0 0
client-output-buffer-limit replica 256mb 64mb 60
client-output-buffer-limit pubsub 32mb 8mb 60
hz 10
dynamic-hz yes
aof-rewrite-incremental-fsync yes
rdb-save-incremental-fsync yes

#------------------------------------------------------------------------------
# THREADED I/O
#------------------------------------------------------------------------------
io-threads ${CPU_CORES}
io-threads-do-reads yes

#------------------------------------------------------------------------------
# KERNEL TRANSPARENT HUGEPAGE
#------------------------------------------------------------------------------
disable-thp yes

#------------------------------------------------------------------------------
# ACTIVE DEFRAGMENTATION
#------------------------------------------------------------------------------
activedefrag yes
active-defrag-ignore-bytes 100mb
active-defrag-threshold-lower 10
active-defrag-threshold-upper 100
active-defrag-cycle-min 5
active-defrag-cycle-max 75
active-defrag-max-scan-fields 1000
jemalloc-bg-thread yes
EOF

    log "Redis optimization configuration created" "${GREEN}"
}

# Qdrant vector database optimization
optimize_qdrant() {
    log "Optimizing Qdrant vector database for embeddings" "${BLUE}"

    cat > /tmp/qdrant-config.yaml << EOF
# Qdrant Performance Optimization for AgenticDosNode
# Generated on $(date)

service:
  host: 0.0.0.0
  http_port: 6333
  grpc_port: 6334
  max_request_size_mb: 256
  max_workers: ${CPU_CORES}
  enable_cors: true
  enable_tls: false

storage:
  storage_path: /qdrant/storage
  snapshots_path: /qdrant/snapshots
  on_disk_payload: true
  performance:
    optimizers_config:
      deleted_threshold: 0.2
      vacuum_min_vector_number: 1000
      default_segment_number: 4
      max_segment_size_kb: 200000
      memmap_threshold_kb: 50000
      indexing_threshold: 20000
      flush_interval_sec: 5
      max_optimization_threads: ${CPU_CORES}
    hnsw_index:
      m: 16
      ef_construct: 200
      full_scan_threshold: 10000
      max_indexing_threads: ${CPU_CORES}
      on_disk: false
      payload_m: null
      skip_initial_index_building: false
    memmap_enabled: true
    memmap_threshold_kb: 100000
    disk_read_buffer_size: 1048576
    disk_write_buffer_size: 1048576

cluster:
  enabled: false
  p2p:
    port: 6335
  consensus:
    tick_period_ms: 100
    bootstrap_timeout_sec: 30
    max_message_queue_size: 100

service_config:
  max_request_size_mb: 256
  max_workers: ${CPU_CORES}
  enable_static_content: true

telemetry:
  telemetry_disabled: true
  anonymize: true

log_level: INFO

# Performance tips:
# 1. Use batch operations for better throughput
# 2. Optimize vector dimensions (smaller is faster)
# 3. Use appropriate distance metrics (Cosine is generally fastest)
# 4. Enable on-disk storage for large collections
# 5. Tune HNSW parameters based on accuracy requirements
EOF

    log "Qdrant optimization configuration created" "${GREEN}"

    # Create Qdrant collection optimization script
    cat > /tmp/optimize-qdrant-collections.py << 'EOF'
#!/usr/bin/env python3

import requests
import json
from typing import Dict, Any

QDRANT_URL = "http://localhost:6333"

def optimize_collection(collection_name: str, vector_size: int) -> Dict[str, Any]:
    """Optimize a Qdrant collection for performance"""

    optimization_config = {
        "vectors": {
            "size": vector_size,
            "distance": "Cosine"
        },
        "optimizers_config": {
            "deleted_threshold": 0.2,
            "vacuum_min_vector_number": 1000,
            "default_segment_number": 4,
            "max_segment_size": 200000,
            "memmap_threshold": 50000,
            "indexing_threshold": 20000,
            "flush_interval_sec": 5
        },
        "hnsw_config": {
            "m": 16,
            "ef_construct": 200,
            "full_scan_threshold": 10000,
            "max_indexing_threads": 0,  # Use all available
            "on_disk": False
        },
        "wal_config": {
            "wal_capacity_mb": 32,
            "wal_segments_ahead": 0
        },
        "quantization_config": {
            "scalar": {
                "type": "int8",
                "quantile": 0.99,
                "always_ram": True
            }
        }
    }

    # Update collection configuration
    response = requests.put(
        f"{QDRANT_URL}/collections/{collection_name}",
        json=optimization_config
    )

    return response.json()

def create_indexes(collection_name: str):
    """Create optimized indexes for common operations"""

    # Create payload index for metadata filtering
    index_config = {
        "field_name": "metadata.source",
        "field_schema": "keyword"
    }

    response = requests.put(
        f"{QDRANT_URL}/collections/{collection_name}/index",
        json=index_config
    )

    return response.json()

def main():
    # Get all collections
    collections_response = requests.get(f"{QDRANT_URL}/collections")
    collections = collections_response.json().get("result", {}).get("collections", [])

    for collection in collections:
        name = collection["name"]
        print(f"Optimizing collection: {name}")

        # Get collection info
        info_response = requests.get(f"{QDRANT_URL}/collections/{name}")
        info = info_response.json().get("result", {})

        vector_size = info.get("config", {}).get("params", {}).get("vectors", {}).get("size", 768)

        # Apply optimizations
        result = optimize_collection(name, vector_size)
        print(f"  Optimization result: {result}")

        # Create indexes
        index_result = create_indexes(name)
        print(f"  Index creation result: {index_result}")

if __name__ == "__main__":
    main()
EOF

    chmod +x /tmp/optimize-qdrant-collections.py
    log "Qdrant collection optimization script created" "${GREEN}"
}

# Create database monitoring setup
setup_database_monitoring() {
    log "Setting up database performance monitoring" "${BLUE}"

    # PostgreSQL monitoring queries
    cat > /tmp/postgresql-monitoring.sql << 'EOF'
-- Create monitoring schema
CREATE SCHEMA IF NOT EXISTS monitoring;

-- Query performance view
CREATE OR REPLACE VIEW monitoring.slow_queries AS
SELECT
    query,
    calls,
    total_exec_time,
    mean_exec_time,
    stddev_exec_time,
    rows,
    100.0 * shared_blks_hit / nullif(shared_blks_hit + shared_blks_read, 0) AS hit_percent
FROM pg_stat_statements
WHERE query NOT LIKE '%pg_stat_statements%'
ORDER BY mean_exec_time DESC
LIMIT 20;

-- Table bloat monitoring
CREATE OR REPLACE VIEW monitoring.table_bloat AS
SELECT
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size,
    CASE WHEN pg_total_relation_size(schemaname||'.'||tablename) > 0
         THEN round(100 * n_dead_tup / pg_total_relation_size(schemaname||'.'||tablename))
         ELSE 0
    END AS dead_tuple_percent
FROM pg_stat_user_tables
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- Connection monitoring
CREATE OR REPLACE VIEW monitoring.connection_stats AS
SELECT
    datname,
    numbackends,
    xact_commit,
    xact_rollback,
    blks_read,
    blks_hit,
    tup_returned,
    tup_fetched,
    tup_inserted,
    tup_updated,
    tup_deleted
FROM pg_stat_database
WHERE datname NOT IN ('template0', 'template1', 'postgres');

-- Index usage monitoring
CREATE OR REPLACE VIEW monitoring.index_usage AS
SELECT
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch,
    pg_size_pretty(pg_relation_size(indexrelid)) AS index_size
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;

-- Lock monitoring
CREATE OR REPLACE VIEW monitoring.lock_monitor AS
SELECT
    blocked_locks.pid AS blocked_pid,
    blocked_activity.usename AS blocked_user,
    blocking_locks.pid AS blocking_pid,
    blocking_activity.usename AS blocking_user,
    blocked_activity.query AS blocked_statement,
    blocking_activity.query AS blocking_statement
FROM pg_catalog.pg_locks blocked_locks
JOIN pg_catalog.pg_stat_activity blocked_activity ON blocked_activity.pid = blocked_locks.pid
JOIN pg_catalog.pg_locks blocking_locks
    ON blocking_locks.locktype = blocked_locks.locktype
    AND blocking_locks.database IS NOT DISTINCT FROM blocked_locks.database
    AND blocking_locks.relation IS NOT DISTINCT FROM blocked_locks.relation
    AND blocking_locks.page IS NOT DISTINCT FROM blocked_locks.page
    AND blocking_locks.tuple IS NOT DISTINCT FROM blocked_locks.tuple
    AND blocking_locks.virtualxid IS NOT DISTINCT FROM blocked_locks.virtualxid
    AND blocking_locks.transactionid IS NOT DISTINCT FROM blocked_locks.transactionid
    AND blocking_locks.classid IS NOT DISTINCT FROM blocked_locks.classid
    AND blocking_locks.objid IS NOT DISTINCT FROM blocked_locks.objid
    AND blocking_locks.objsubid IS NOT DISTINCT FROM blocked_locks.objsubid
    AND blocking_locks.pid != blocked_locks.pid
JOIN pg_catalog.pg_stat_activity blocking_activity ON blocking_activity.pid = blocking_locks.pid
WHERE NOT blocked_locks.granted;

-- Grant access to monitoring user
GRANT USAGE ON SCHEMA monitoring TO monitor;
GRANT SELECT ON ALL TABLES IN SCHEMA monitoring TO monitor;
EOF

    # Redis monitoring script
    cat > /usr/local/bin/redis-monitor.sh << 'EOF'
#!/bin/bash

# Redis Performance Monitor
REDIS_CLI="redis-cli"
METRICS_FILE="/var/lib/node_exporter/redis_metrics.prom"

while true; do
    {
        # Get Redis info
        info=$($REDIS_CLI INFO)

        # Extract and format metrics
        echo "# Redis Performance Metrics"
        echo "# TYPE redis_connected_clients gauge"
        echo "redis_connected_clients $(echo "$info" | grep connected_clients: | cut -d: -f2)"

        echo "# TYPE redis_used_memory_bytes gauge"
        echo "redis_used_memory_bytes $(echo "$info" | grep used_memory: | cut -d: -f2)"

        echo "# TYPE redis_ops_per_sec gauge"
        echo "redis_ops_per_sec $(echo "$info" | grep instantaneous_ops_per_sec: | cut -d: -f2)"

        echo "# TYPE redis_hit_rate gauge"
        hits=$(echo "$info" | grep keyspace_hits: | cut -d: -f2)
        misses=$(echo "$info" | grep keyspace_misses: | cut -d: -f2)
        if [ "$hits" -gt 0 ] && [ "$misses" -gt 0 ]; then
            hit_rate=$(echo "scale=2; $hits / ($hits + $misses)" | bc)
            echo "redis_hit_rate $hit_rate"
        fi

        # Get slow log
        slow_queries=$($REDIS_CLI SLOWLOG LEN | tail -1)
        echo "# TYPE redis_slow_queries_total counter"
        echo "redis_slow_queries_total $slow_queries"

    } > $METRICS_FILE.tmp && mv $METRICS_FILE.tmp $METRICS_FILE

    sleep 30
done
EOF

    chmod +x /usr/local/bin/redis-monitor.sh
    log "Database monitoring setup completed" "${GREEN}"
}

# Apply all database optimizations
apply_all() {
    log "Starting database optimization" "${BLUE}"

    backup_configs
    optimize_postgresql
    optimize_redis
    optimize_qdrant
    setup_database_monitoring

    log "Database optimization completed!" "${GREEN}"

    cat << EOF

${GREEN}Database Optimization Summary:${NC}
- PostgreSQL: Tuned for ${TOTAL_MEM_GB}GB RAM, ${CPU_CORES} cores
- Redis: Configured with ${REDIS_MAXMEM} max memory
- Qdrant: Optimized for vector operations
- Monitoring: Performance tracking enabled

${YELLOW}Next Steps:${NC}
1. Apply PostgreSQL config:
   cp /tmp/postgresql-optimization.conf /etc/postgresql/postgresql.conf
   systemctl restart postgresql

2. Apply Redis config:
   cp /tmp/redis-optimization.conf /etc/redis/redis.conf
   systemctl restart redis

3. Apply Qdrant config:
   cp /tmp/qdrant-config.yaml /qdrant/config/config.yaml
   docker restart qdrant

4. Setup connection pooling:
   cp /tmp/pgbouncer.ini /etc/pgbouncer/pgbouncer.ini
   systemctl restart pgbouncer

5. Run optimization scripts:
   python3 /tmp/optimize-qdrant-collections.py
   psql -U postgres < /tmp/postgresql-monitoring.sql

${BLUE}Monitor Performance:${NC}
- PostgreSQL: psql -c "SELECT * FROM monitoring.slow_queries;"
- Redis: redis-cli INFO stats
- Qdrant: curl http://localhost:6333/metrics

${RED}To Rollback:${NC}
cp $BACKUP_DIR/*.bak /etc/
systemctl restart postgresql redis
EOF
}

# Rollback function
rollback() {
    log "Rolling back database optimization changes" "${YELLOW}"

    if [ ! -d "$BACKUP_DIR" ]; then
        log "No backup found at $BACKUP_DIR" "${RED}"
        exit 1
    fi

    # Restore original configurations
    if [ -f "$BACKUP_DIR/postgresql.conf.bak" ]; then
        cp "$BACKUP_DIR/postgresql.conf.bak" /etc/postgresql/postgresql.conf
        systemctl restart postgresql
    fi

    if [ -f "$BACKUP_DIR/redis.conf.bak" ]; then
        cp "$BACKUP_DIR/redis.conf.bak" /etc/redis/redis.conf
        systemctl restart redis
    fi

    log "Rollback completed" "${GREEN}"
}

# Main execution
main() {
    if [ "$EUID" -ne 0 ]; then
        echo "This script must be run as root (use sudo)"
        exit 1
    fi

    case "${1:-apply}" in
        apply)
            apply_all
            ;;
        rollback)
            rollback
            ;;
        *)
            echo "Usage: $0 [apply|rollback]"
            exit 1
            ;;
    esac
}

main "$@"