-- Create additional databases for Oracle1 services
CREATE DATABASE n8n;
CREATE DATABASE langgraph;
CREATE DATABASE metrics;

-- Create users and grant permissions
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'n8n_user') THEN
        CREATE USER n8n_user WITH PASSWORD 'n8n123!';
    END IF;

    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'langgraph_user') THEN
        CREATE USER langgraph_user WITH PASSWORD 'langgraph123!';
    END IF;

    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'metrics_user') THEN
        CREATE USER metrics_user WITH PASSWORD 'metrics123!';
    END IF;
END
$$;

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE n8n TO n8n_user;
GRANT ALL PRIVILEGES ON DATABASE langgraph TO langgraph_user;
GRANT ALL PRIVILEGES ON DATABASE metrics TO metrics_user;

-- Create extensions
\c n8n;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

\c langgraph;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
CREATE EXTENSION IF NOT EXISTS "vector";

\c metrics;
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Set up basic monitoring tables
\c agentic_db;
CREATE TABLE IF NOT EXISTS deployment_status (
    id SERIAL PRIMARY KEY,
    service_name VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL,
    last_updated TIMESTAMP DEFAULT NOW(),
    node_name VARCHAR(20) DEFAULT 'oracle1'
);

-- Insert initial status
INSERT INTO deployment_status (service_name, status) VALUES
('postgres', 'healthy'),
('redis', 'starting'),
('qdrant', 'starting'),
('n8n', 'starting'),
('prometheus', 'starting'),
('grafana', 'starting')
ON CONFLICT (service_name) DO UPDATE SET
    status = EXCLUDED.status,
    last_updated = NOW();