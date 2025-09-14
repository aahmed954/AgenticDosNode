# N8N Automation Setup Guide - Agentic AI Stack

This comprehensive guide will help you set up n8n automation workflows for your agentic AI stack, providing enterprise-grade automation for AI services, infrastructure monitoring, and third-party integrations.

## 🏗️ Architecture Overview

The n8n automation suite includes:

- **Core AI Workflows**: Daily briefings, code review automation, document processing, multi-agent conversations
- **Infrastructure Automation**: Health monitoring, cost tracking, security monitoring, backup automation
- **Integration Workflows**: GitHub integration, email/Slack notifications, external API integrations
- **Advanced Features**: Error handling, conditional logic, webhook triggers, scheduled executions

## 📋 Prerequisites

### System Requirements
- Docker and Docker Compose
- At least 4GB RAM
- 20GB free disk space
- Linux/macOS/Windows with WSL2

### Required API Keys & Tokens
- Claude API key (Anthropic)
- OpenAI API key (optional, for multi-agent routing)
- GitHub Personal Access Token
- Slack Bot Token or Webhook URL
- News API key
- OpenWeatherMap API key
- AWS credentials (for backup to S3)

## 🚀 Quick Start Installation

### 1. Clone and Setup

```bash
cd /home/starlord/AgenticDosNode/n8n
```

### 2. Configure Environment Variables

Copy and customize the environment file:

```bash
cp docker/.env.n8n docker/.env
```

Edit the `.env` file with your actual credentials:

```env
# N8N Configuration
N8N_BASIC_AUTH_USER=admin
N8N_BASIC_AUTH_PASSWORD=your_secure_password_here
N8N_HOST=localhost
N8N_WEBHOOK_URL=http://localhost:5678/

# AI Service Integration
CLAUDE_API_KEY=sk-ant-your-claude-api-key-here
CLAUDE_PROXY_URL=http://ccproxy:3000
OPENAI_API_KEY=sk-your-openai-key-here

# Database Passwords
N8N_POSTGRES_PASSWORD=secure_db_password
REDIS_PASSWORD=secure_redis_password

# External APIs
GITHUB_TOKEN=ghp_your_github_token_here
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK
NEWS_API_KEY=your_news_api_key_here
WEATHER_API_KEY=your_openweather_api_key_here

# Email Configuration
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password_here

# AWS Backup Configuration
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
BACKUP_S3_BUCKET=your-n8n-backups-bucket
```

### 3. Start the Services

```bash
# Start the complete n8n automation stack
docker-compose -f docker/docker-compose.n8n.yml --env-file docker/.env up -d

# Check service status
docker-compose -f docker/docker-compose.n8n.yml ps
```

### 4. Access n8n Interface

Open your browser and navigate to:
- **n8n Interface**: http://localhost:5678
- **Login**: Use the credentials from your `.env` file

## 📁 Workflow Import Process

### Import All Workflows

1. **Access n8n**: Go to http://localhost:5678 and log in
2. **Import Workflows**:
   - Click "Import from File" or use the keyboard shortcut `Ctrl+O`
   - Import each workflow file from the `/workflows` directory:
     - `01-daily-briefing-workflow.json`
     - `02-code-review-automation.json`
     - `03-document-processing-rag.json`
     - `04-multi-agent-conversations.json`
     - `05-health-monitoring.json`
     - `06-cost-tracking-alerts.json`
     - `07-security-monitoring.json`
     - `08-backup-automation.json`
     - `09-github-integrations.json`
     - `10-external-api-integrations.json`

### Configure Credentials

For each workflow, you'll need to set up credentials:

#### 1. Claude API Credentials
- **Name**: `claude-api-creds`
- **Type**: HTTP Bearer Auth
- **Token**: Your Claude API key

#### 2. GitHub API Credentials
- **Name**: `github-api-creds`
- **Type**: GitHub API
- **Access Token**: Your GitHub Personal Access Token

#### 3. PostgreSQL Credentials
- **Name**: `postgres-creds`
- **Type**: Postgres
- **Host**: `postgres-n8n`
- **Database**: `n8n`
- **User**: `n8n`
- **Password**: Value from `N8N_POSTGRES_PASSWORD`

#### 4. SMTP Credentials
- **Name**: `smtp-creds`
- **Type**: SMTP
- **Host**: Value from `SMTP_HOST`
- **Port**: Value from `SMTP_PORT`
- **User**: Value from `SMTP_USER`
- **Password**: Value from `SMTP_PASSWORD`

#### 5. Redis Credentials
- **Name**: `redis-creds`
- **Type**: Redis
- **Host**: `redis-n8n`
- **Port**: `6379`
- **Password**: Value from `REDIS_PASSWORD`

## 🔧 Database Schema Setup

The workflows require specific database tables. Run this SQL to create them:

```sql
-- Connect to your PostgreSQL database and run:

-- Table for workflow execution metrics
CREATE TABLE IF NOT EXISTS workflow_metrics (
    id SERIAL PRIMARY KEY,
    workflow_id VARCHAR(255) NOT NULL,
    execution_id VARCHAR(255) NOT NULL,
    status VARCHAR(50) NOT NULL,
    execution_time INTEGER,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table for AI interaction logging
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

-- Table for webhook request logging
CREATE TABLE IF NOT EXISTS webhook_logs (
    id SERIAL PRIMARY KEY,
    webhook_name VARCHAR(255) NOT NULL,
    source_ip INET,
    payload JSONB,
    headers JSONB,
    processed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table for health check results
CREATE TABLE IF NOT EXISTS health_checks (
    id SERIAL PRIMARY KEY,
    check_id VARCHAR(255) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    overall_status VARCHAR(50) NOT NULL,
    health_score INTEGER,
    total_services INTEGER,
    healthy_services INTEGER,
    unhealthy_services INTEGER,
    critical_failures INTEGER,
    avg_response_time INTEGER,
    details JSONB
);

-- Table for cost tracking
CREATE TABLE IF NOT EXISTS cost_tracking (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    daily_cost DECIMAL(10,4),
    weekly_cost DECIMAL(10,4),
    monthly_cost DECIMAL(10,4),
    daily_budget_percent INTEGER,
    weekly_budget_percent INTEGER,
    monthly_budget_percent INTEGER,
    alert_count INTEGER,
    top_service VARCHAR(255),
    top_service_cost DECIMAL(10,4),
    anomaly_detected BOOLEAN DEFAULT FALSE
);

-- Table for security alerts
CREATE TABLE IF NOT EXISTS security_alerts (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    security_status VARCHAR(50) NOT NULL,
    security_score INTEGER,
    critical_issues INTEGER,
    high_severity_issues INTEGER,
    total_issues INTEGER,
    threats_detected INTEGER,
    warnings_raised INTEGER,
    alert_details JSONB
);

-- Table for backup logs
CREATE TABLE IF NOT EXISTS backup_logs (
    id SERIAL PRIMARY KEY,
    backup_id VARCHAR(255) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    status VARCHAR(50) NOT NULL,
    success_rate INTEGER,
    total_items INTEGER,
    successful_items INTEGER,
    failed_items INTEGER,
    backup_size VARCHAR(50),
    duration_ms BIGINT,
    remote_location TEXT,
    completed_at TIMESTAMP
);

-- Table for GitHub issue analysis
CREATE TABLE IF NOT EXISTS github_issue_analysis (
    id SERIAL PRIMARY KEY,
    issue_number INTEGER NOT NULL,
    repository VARCHAR(255) NOT NULL,
    title TEXT NOT NULL,
    author VARCHAR(255),
    issue_type VARCHAR(100),
    priority VARCHAR(50),
    complexity VARCHAR(50),
    category VARCHAR(100),
    effort_estimate INTEGER,
    suggested_labels JSONB,
    auto_labeled BOOLEAN DEFAULT FALSE,
    auto_responded BOOLEAN DEFAULT FALSE,
    needs_triaging BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP
);

-- Table for document processing metadata
CREATE TABLE IF NOT EXISTS document_metadata (
    id SERIAL PRIMARY KEY,
    processing_id VARCHAR(255) NOT NULL,
    file_name VARCHAR(255) NOT NULL,
    file_type VARCHAR(50),
    file_size INTEGER,
    word_count INTEGER,
    chunk_count INTEGER,
    summary TEXT,
    tags JSONB,
    relevance_score INTEGER,
    processed_at TIMESTAMP,
    user_id VARCHAR(255),
    category VARCHAR(100),
    status VARCHAR(50)
);

-- Table for conversation logs
CREATE TABLE IF NOT EXISTS conversation_logs (
    id SERIAL PRIMARY KEY,
    conversation_id VARCHAR(255) NOT NULL,
    user_id VARCHAR(255),
    message TEXT NOT NULL,
    response TEXT NOT NULL,
    model_used VARCHAR(100),
    input_tokens INTEGER,
    output_tokens INTEGER,
    estimated_cost DECIMAL(10,6),
    rag_enabled BOOLEAN DEFAULT FALSE,
    response_time INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table for external API request logs
CREATE TABLE IF NOT EXISTS external_api_requests (
    id SERIAL PRIMARY KEY,
    request_id VARCHAR(255) NOT NULL,
    service VARCHAR(100) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    success BOOLEAN,
    processing_time_ms INTEGER,
    data_summary TEXT
);

-- Create indexes for better performance
CREATE INDEX idx_workflow_metrics_workflow_id ON workflow_metrics(workflow_id);
CREATE INDEX idx_workflow_metrics_created_at ON workflow_metrics(created_at);
CREATE INDEX idx_ai_logs_workflow_id ON ai_interaction_logs(workflow_id);
CREATE INDEX idx_ai_logs_created_at ON ai_interaction_logs(created_at);
CREATE INDEX idx_webhook_logs_created_at ON webhook_logs(created_at);
CREATE INDEX idx_health_checks_timestamp ON health_checks(timestamp);
CREATE INDEX idx_cost_tracking_timestamp ON cost_tracking(timestamp);
CREATE INDEX idx_security_alerts_timestamp ON security_alerts(timestamp);
CREATE INDEX idx_conversation_logs_created_at ON conversation_logs(created_at);
```

## 🔗 Webhook Configuration

### GitHub Webhook Setup

1. **Go to Repository Settings**: Navigate to your GitHub repository settings
2. **Add Webhook**:
   - **Payload URL**: `http://your-domain.com:8080/webhook/github`
   - **Content Type**: `application/json`
   - **Events**: Select "Issues", "Pull requests", "Push"
   - **Secret**: Set a secure secret and add it to your environment variables

### Document Upload Webhook

- **Endpoint**: `http://localhost:5678/webhook/document-upload`
- **Method**: POST
- **Content-Type**: `multipart/form-data`
- **Supported formats**: PDF, DOCX, TXT, MD, HTML

### Multi-Agent Chat API

- **Endpoint**: `http://localhost:5678/webhook/ai-chat`
- **Method**: POST
- **Body**: JSON with message, preferences, and context

Example request:
```json
{
  "message": "Explain quantum computing",
  "model": "auto",
  "useRAG": true,
  "temperature": 0.7,
  "maxTokens": 1000,
  "userId": "user123",
  "conversationId": "conv456"
}
```

## 🔒 Security Configuration

### Environment Security
- Use strong passwords for all services
- Rotate API keys regularly
- Enable HTTPS in production
- Configure firewall rules appropriately

### n8n Security Settings
- Enable basic authentication
- Configure webhook authentication
- Set up IP restrictions if needed
- Use secure environment variable injection

### Backup Security
- Encrypt backups before uploading to S3
- Use IAM roles with minimal required permissions
- Enable S3 bucket versioning and lifecycle policies
- Monitor backup access logs

## 📊 Monitoring and Maintenance

### Health Monitoring
The health monitoring workflow runs every 5 minutes and checks:
- n8n service availability
- Claude proxy health
- Vector database connectivity
- PostgreSQL and Redis status
- All workflow execution status

### Cost Tracking
- Monitors AI API usage and costs
- Sends alerts when budget thresholds are exceeded
- Provides daily/weekly/monthly usage reports
- Tracks cost per workflow and user

### Security Monitoring
- Analyzes webhook request patterns
- Detects suspicious IP activity
- Monitors workflow failure rates
- Provides automated threat response

### Backup Automation
- Daily automated backups of all databases
- File system backups of n8n configurations
- Automated upload to S3 with retention policies
- Backup integrity verification

## 🔧 Customization and Extension

### Adding New Workflows
1. Create new workflow in n8n interface
2. Export as JSON file
3. Save to `/workflows` directory
4. Update documentation
5. Add any required database schema changes

### Custom Node Development
- Follow n8n community node development guidelines
- Create custom nodes for specific integrations
- Package and distribute internal tools

### Scaling Considerations
- Use n8n queue mode for high-volume processing
- Implement horizontal scaling with Redis
- Consider separating workflows by function
- Monitor resource usage and optimize accordingly

## 🆘 Troubleshooting

### Common Issues

#### Workflow Execution Failures
```bash
# Check n8n logs
docker-compose -f docker/docker-compose.n8n.yml logs n8n

# Check PostgreSQL connectivity
docker-compose -f docker/docker-compose.n8n.yml exec postgres-n8n psql -U n8n -c "\\l"
```

#### API Integration Issues
- Verify API keys in environment variables
- Check rate limiting and quota usage
- Review webhook authentication setup
- Monitor external service status

#### Database Connection Problems
```bash
# Restart database services
docker-compose -f docker/docker-compose.n8n.yml restart postgres-n8n redis-n8n

# Check database logs
docker-compose -f docker/docker-compose.n8n.yml logs postgres-n8n
```

### Getting Help
- Check n8n community documentation
- Review workflow execution logs in n8n interface
- Monitor system resources and container health
- Consult the troubleshooting section of each workflow

## 🎯 Next Steps

1. **Test Workflows**: Execute each workflow manually to verify functionality
2. **Customize Settings**: Adjust schedules, thresholds, and notification preferences
3. **Monitor Performance**: Review execution logs and optimize slow-running workflows
4. **Scale Infrastructure**: Add more resources as workflow volume increases
5. **Extend Functionality**: Add new workflows for additional use cases

## 📝 Workflow Summary

| Workflow | Purpose | Schedule | Dependencies |
|----------|---------|----------|--------------|
| Daily Briefing | News aggregation & AI analysis | 7 AM weekdays | News API, Claude |
| Code Review | Automated PR analysis | On GitHub webhook | GitHub API, Claude |
| Document Processing | RAG embedding & summarization | On upload webhook | Claude, Vector DB |
| Multi-Agent Chat | AI model routing & conversation | On demand | Claude, OpenAI |
| Health Monitoring | Infrastructure health checks | Every 5 minutes | All services |
| Cost Tracking | API usage & budget monitoring | Every 6 hours | Database logs |
| Security Monitoring | Threat detection & response | Every 15 minutes | Access logs |
| Backup Automation | Data backup & archival | Daily 2 AM | S3, All databases |
| GitHub Integration | Issue management & repo health | On webhooks + 6 hours | GitHub API |
| External APIs | Weather, crypto, tech updates | 8 AM weekdays | Multiple APIs |

This comprehensive automation suite provides enterprise-grade workflow automation for your agentic AI stack, ensuring reliable operations, monitoring, and integration across all your services.