# 🤖 N8N Automation Suite for Agentic AI Stack

A comprehensive n8n automation solution providing enterprise-grade workflow automation for AI services, infrastructure monitoring, and third-party integrations.

## 🌟 Features

### 🧠 Core AI Workflows
- **Daily Briefing**: Automated news aggregation with Claude AI analysis
- **Code Review**: Intelligent PR analysis and automated feedback
- **Document Processing**: RAG embedding pipeline with summarization
- **Multi-Agent Chat**: Smart routing between AI models (Claude, GPT-4, etc.)

### 🏗️ Infrastructure Automation
- **Health Monitoring**: Real-time service health checks and alerts
- **Cost Tracking**: API usage monitoring with budget alerts
- **Security Monitoring**: Threat detection and automated response
- **Backup Automation**: Automated database and file backups with S3 integration

### 🔗 Integration Workflows
- **GitHub Integration**: Issue management and repository health monitoring
- **Notification System**: Email and Slack notifications with rich formatting
- **External APIs**: Weather, cryptocurrency, and tech update integrations
- **Webhook Handling**: Secure webhook processing with rate limiting

### 🚀 Advanced Features
- **Error Handling**: Comprehensive error handling with automatic retries
- **Conditional Logic**: Smart branching and decision-making workflows
- **Scheduled Executions**: Flexible cron-based scheduling
- **Data Transformation**: Advanced data processing and formatting
- **Security**: Authentication, rate limiting, and access control

## 📁 Directory Structure

```
n8n/
├── docker/                    # Docker configuration
│   ├── docker-compose.n8n.yml
│   ├── .env.n8n (template)
│   └── .env (your config)
├── workflows/                 # N8N workflow definitions
│   ├── 01-daily-briefing-workflow.json
│   ├── 02-code-review-automation.json
│   ├── 03-document-processing-rag.json
│   ├── 04-multi-agent-conversations.json
│   ├── 05-health-monitoring.json
│   ├── 06-cost-tracking-alerts.json
│   ├── 07-security-monitoring.json
│   ├── 08-backup-automation.json
│   ├── 09-github-integrations.json
│   └── 10-external-api-integrations.json
├── config/                    # Configuration files
│   ├── postgres-init.sql
│   ├── qdrant.yaml
│   └── nginx.conf
├── credentials/              # Credential templates
├── deploy-n8n.sh           # Deployment script
├── import-workflows.sh      # Workflow import helper
├── N8N_SETUP_GUIDE.md      # Detailed setup guide
└── README.md               # This file
```

## 🚀 Quick Start

### 1. Prerequisites
- Docker & Docker Compose installed
- 4GB+ RAM available
- 20GB+ disk space
- Required API keys (see setup guide)

### 2. Deploy the Stack

```bash
# Clone and navigate to the n8n directory
cd /home/starlord/AgenticDosNode/n8n

# Run the deployment script
./deploy-n8n.sh deploy
```

### 3. Configure Environment

Edit the `.env` file with your credentials:

```bash
# Edit environment variables
nano docker/.env

# Required: Set these at minimum
N8N_BASIC_AUTH_USER=admin
N8N_BASIC_AUTH_PASSWORD=your_secure_password
CLAUDE_API_KEY=sk-ant-your-claude-key
GITHUB_TOKEN=ghp_your-github-token
SLACK_WEBHOOK_URL=https://hooks.slack.com/your/webhook
```

### 4. Import Workflows

```bash
# Import all workflows automatically
./import-workflows.sh all

# Or select specific workflows interactively
./import-workflows.sh interactive
```

### 5. Access n8n

Open your browser to: **http://localhost:5678**

Login with your configured credentials and start using your automation workflows!

## 📊 Workflow Overview

| Workflow | Trigger | Purpose | Key Features |
|----------|---------|---------|--------------|
| **Daily Briefing** | 7 AM weekdays | News analysis & morning updates | Claude AI analysis, multi-source aggregation |
| **Code Review** | GitHub PR webhooks | Automated code review | AI-powered analysis, auto-labeling, team notifications |
| **Document Processing** | File upload webhook | RAG pipeline processing | Text extraction, vector embedding, AI summarization |
| **Multi-Agent Chat** | API webhook | Intelligent AI routing | Model selection, RAG integration, cost optimization |
| **Health Monitoring** | Every 5 minutes | Infrastructure monitoring | Service health checks, automated alerts |
| **Cost Tracking** | Every 6 hours | Budget monitoring | Usage analysis, budget alerts, trend detection |
| **Security Monitoring** | Every 15 minutes | Threat detection | Anomaly detection, auto-remediation |
| **Backup Automation** | Daily 2 AM | Data protection | Database backups, S3 upload, retention management |
| **GitHub Integration** | Webhooks + scheduled | Repository management | Issue analysis, health reporting |
| **External APIs** | 8 AM weekdays | Data aggregation | Weather, crypto, tech updates with AI briefing |

## 🔧 Configuration

### Environment Variables

Key environment variables to configure:

```env
# N8N Core
N8N_BASIC_AUTH_USER=admin
N8N_BASIC_AUTH_PASSWORD=secure_password
N8N_HOST=localhost
N8N_WEBHOOK_URL=http://localhost:5678/

# AI Services
CLAUDE_API_KEY=sk-ant-your-key
CLAUDE_PROXY_URL=http://ccproxy:3000
OPENAI_API_KEY=sk-your-openai-key

# Integrations
GITHUB_TOKEN=ghp_your-token
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
NEWS_API_KEY=your-news-api-key
WEATHER_API_KEY=your-weather-api-key

# Infrastructure
POSTGRES_PASSWORD=secure_db_password
REDIS_PASSWORD=secure_redis_password
AWS_ACCESS_KEY_ID=your-aws-key
AWS_SECRET_ACCESS_KEY=your-aws-secret
BACKUP_S3_BUCKET=your-backup-bucket
```

### Webhook Endpoints

The automation suite exposes these webhook endpoints:

- **GitHub Webhooks**: `http://localhost:8080/webhook/github`
- **Document Upload**: `http://localhost:5678/webhook/document-upload`
- **AI Chat API**: `http://localhost:5678/webhook/ai-chat`
- **External Data**: `http://localhost:5678/webhook/external-data-request`

## 📖 Documentation

- **[N8N_SETUP_GUIDE.md](./N8N_SETUP_GUIDE.md)**: Complete setup and configuration guide
- **Workflow Documentation**: Each workflow file contains detailed node documentation
- **API Documentation**: Webhook endpoints and request/response formats
- **Troubleshooting Guide**: Common issues and solutions

## 🛠️ Management Commands

```bash
# Deployment Management
./deploy-n8n.sh deploy     # Deploy the complete stack
./deploy-n8n.sh stop       # Stop all services
./deploy-n8n.sh restart    # Restart services
./deploy-n8n.sh status     # Show service status
./deploy-n8n.sh logs       # View service logs
./deploy-n8n.sh cleanup    # Remove everything (destructive!)

# Workflow Management
./import-workflows.sh all         # Import all workflows
./import-workflows.sh interactive # Select specific workflows
./import-workflows.sh list        # List available workflows
./import-workflows.sh status      # Show imported workflows
```

## 🔒 Security Features

- **Authentication**: Basic auth for n8n interface
- **Webhook Security**: Signature verification for GitHub webhooks
- **Rate Limiting**: Built-in rate limiting for webhook endpoints
- **Access Control**: IP-based restrictions (configurable)
- **Data Encryption**: Encrypted environment variables and secrets
- **Backup Security**: Encrypted backups with retention policies

## 📈 Monitoring & Observability

### Built-in Monitoring
- **Health Dashboards**: Real-time service health monitoring
- **Cost Analytics**: Detailed AI API usage and cost tracking
- **Security Alerts**: Automated threat detection and response
- **Performance Metrics**: Workflow execution times and success rates

### Integration with Existing Tools
- **Prometheus**: Metrics export for monitoring
- **Grafana**: Custom dashboards for visualization
- **Slack**: Real-time alerts and notifications
- **Email**: Critical alert notifications

## 🔄 Workflow Customization

### Adding New Workflows
1. Create workflow in n8n interface
2. Export as JSON file
3. Save to `/workflows` directory
4. Update documentation
5. Test and deploy

### Modifying Existing Workflows
1. Import workflow to n8n
2. Make modifications in the interface
3. Test thoroughly
4. Export and save updated JSON
5. Document changes

### Custom Integrations
- **Custom Nodes**: Develop custom n8n nodes for specific needs
- **External Services**: Add new API integrations
- **Notification Channels**: Extend notification capabilities
- **Data Sources**: Connect additional data sources

## 🚨 Troubleshooting

### Common Issues

#### Services Not Starting
```bash
# Check service logs
./deploy-n8n.sh logs

# Verify port availability
netstat -tlnp | grep ':5678\|:6333\|:8080'

# Check Docker resources
docker system df
```

#### Workflow Execution Failures
```bash
# Access n8n logs
docker-compose -f docker/docker-compose.n8n.yml logs n8n

# Check database connectivity
docker-compose -f docker/docker-compose.n8n.yml exec postgres-n8n psql -U n8n -c "\l"
```

#### API Integration Issues
- Verify API keys in `.env` file
- Check rate limits and quotas
- Review webhook authentication
- Monitor external service status

### Getting Help

1. **Check Documentation**: Review setup guide and workflow docs
2. **Examine Logs**: Use `./deploy-n8n.sh logs` for detailed logs
3. **Verify Configuration**: Ensure all environment variables are set
4. **Test Connectivity**: Verify external API access and database connections
5. **Community Support**: Consult n8n community forums and documentation

## 🎯 Use Cases

### Enterprise Automation
- **DevOps Workflows**: Automated deployment pipelines and infrastructure management
- **Content Management**: Document processing and knowledge base automation
- **Customer Support**: Automated ticket routing and response systems
- **Business Intelligence**: Data aggregation and reporting automation

### AI-Powered Operations
- **Intelligent Monitoring**: AI-driven anomaly detection and response
- **Smart Notifications**: Context-aware alerting with AI analysis
- **Automated Analysis**: AI-powered log analysis and troubleshooting
- **Predictive Maintenance**: AI-driven infrastructure optimization

### Development Productivity
- **Code Quality**: Automated code review and quality assurance
- **Issue Management**: Intelligent issue triaging and assignment
- **Documentation**: Automated documentation generation and updates
- **Team Communication**: Smart notification and collaboration workflows

## 🔮 Roadmap

### Planned Features
- **Advanced AI Models**: Integration with additional AI providers
- **Enhanced Security**: Advanced threat detection and response
- **Mobile Support**: Mobile-optimized dashboards and notifications
- **Advanced Analytics**: Machine learning-powered insights
- **Multi-tenancy**: Support for multiple organizations/teams

### Community Contributions
- **Custom Nodes**: Community-contributed n8n nodes
- **Workflow Templates**: Pre-built workflows for common use cases
- **Integration Modules**: Additional service integrations
- **Documentation**: Enhanced guides and tutorials

---

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines and feel free to submit pull requests, report issues, or suggest new features.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **n8n Community**: For the amazing automation platform
- **Anthropic**: For Claude AI integration
- **Open Source Contributors**: For all the tools and libraries used

---

**🚀 Ready to automate your agentic AI stack? Get started with `./deploy-n8n.sh deploy`!**