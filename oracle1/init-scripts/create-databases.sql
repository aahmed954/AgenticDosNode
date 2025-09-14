-- Database creation script for Oracle1 AgenticDosNode
-- Creates multiple databases for different services

-- Create databases if they don't exist
SELECT 'CREATE DATABASE n8n OWNER agentic'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'n8n')\gexec

SELECT 'CREATE DATABASE langgraph OWNER agentic'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'langgraph')\gexec

SELECT 'CREATE DATABASE kong OWNER agentic'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'kong')\gexec

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE n8n TO agentic;
GRANT ALL PRIVILEGES ON DATABASE langgraph TO agentic;
GRANT ALL PRIVILEGES ON DATABASE kong TO agentic;