-- PostgreSQL initialization script for n8n
-- Create additional databases and users for different services

-- Create user for application access
CREATE USER n8n_user WITH PASSWORD 'n8n_user_password';

-- Grant necessary permissions
GRANT CONNECT ON DATABASE n8n TO n8n_user;
GRANT USAGE ON SCHEMA public TO n8n_user;
GRANT CREATE ON SCHEMA public TO n8n_user;

-- Create extensions if needed
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create additional tables for workflow metadata
CREATE TABLE IF NOT EXISTS workflow_metrics (
    id SERIAL PRIMARY KEY,
    workflow_id VARCHAR(255) NOT NULL,
    execution_id VARCHAR(255) NOT NULL,
    status VARCHAR(50) NOT NULL,
    execution_time INTEGER,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ai_interaction_logs (
    id SERIAL PRIMARY KEY,
    workflow_id VARCHAR(255) NOT NULL,
    ai_service VARCHAR(100) NOT NULL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cost_estimate DECIMAL(10,4),
    response_time INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS webhook_logs (
    id SERIAL PRIMARY KEY,
    webhook_name VARCHAR(255) NOT NULL,
    source_ip INET,
    payload JSONB,
    headers JSONB,
    processed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Grant permissions on new tables
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO n8n_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO n8n_user;

-- Create indexes for better performance
CREATE INDEX idx_workflow_metrics_workflow_id ON workflow_metrics(workflow_id);
CREATE INDEX idx_workflow_metrics_created_at ON workflow_metrics(created_at);
CREATE INDEX idx_ai_logs_workflow_id ON ai_interaction_logs(workflow_id);
CREATE INDEX idx_ai_logs_created_at ON ai_interaction_logs(created_at);
CREATE INDEX idx_webhook_logs_created_at ON webhook_logs(created_at);
CREATE INDEX idx_webhook_logs_processed ON webhook_logs(processed);