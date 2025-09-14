const express = require('express');
const http = require('http');
const socketIo = require('socket.io');
const path = require('path');
const cors = require('cors');
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');
const morgan = require('morgan');
const multer = require('multer');
const axios = require('axios');

const app = express();
const server = http.createServer(app);
const io = socketIo(server, {
  cors: {
    origin: "*",
    methods: ["GET", "POST"]
  }
});

// Middleware
app.use(helmet());
app.use(cors());
app.use(morgan('combined'));
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true }));

// Rate limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // limit each IP to 100 requests per windowMs
  message: 'Too many requests from this IP, please try again later.'
});
app.use('/api', limiter);

// Static files
app.use(express.static(path.join(__dirname, '../public')));

// File upload configuration
const upload = multer({
  dest: 'uploads/',
  limits: {
    fileSize: 10 * 1024 * 1024, // 10MB limit
  }
});

// Configuration
const CONFIG = {
  ORCHESTRATOR_URL: process.env.ORCHESTRATOR_URL || 'http://localhost:8000',
  N8N_URL: process.env.N8N_URL || 'http://localhost:5678',
  QDRANT_URL: process.env.QDRANT_URL || 'http://localhost:6333',
  CLAUDE_PROXY_URL: process.env.CLAUDE_PROXY_URL || 'http://localhost:8001',
  MONITORING_URL: process.env.MONITORING_URL || 'http://localhost:9090'
};

// Routes

// Health check
app.get('/api/health', async (req, res) => {
  try {
    const services = [];

    // Check orchestrator
    try {
      await axios.get(`${CONFIG.ORCHESTRATOR_URL}/health`, { timeout: 2000 });
      services.push({ name: 'orchestrator', status: 'healthy' });
    } catch (error) {
      services.push({ name: 'orchestrator', status: 'unhealthy', error: error.message });
    }

    // Check vector database
    try {
      await axios.get(`${CONFIG.QDRANT_URL}/collections`, { timeout: 2000 });
      services.push({ name: 'vector_db', status: 'healthy' });
    } catch (error) {
      services.push({ name: 'vector_db', status: 'unhealthy', error: error.message });
    }

    // Check Claude proxy
    try {
      await axios.get(`${CONFIG.CLAUDE_PROXY_URL}/health`, { timeout: 2000 });
      services.push({ name: 'claude_proxy', status: 'healthy' });
    } catch (error) {
      services.push({ name: 'claude_proxy', status: 'unhealthy', error: error.message });
    }

    const allHealthy = services.every(s => s.status === 'healthy');

    res.status(allHealthy ? 200 : 503).json({
      status: allHealthy ? 'healthy' : 'degraded',
      services,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    res.status(500).json({
      status: 'error',
      message: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

// Chat endpoint
app.post('/api/chat', async (req, res) => {
  try {
    const { message, mode = 'auto', stream = false } = req.body;

    if (!message) {
      return res.status(400).json({ error: 'Message is required' });
    }

    if (stream) {
      res.writeHead(200, {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
      });

      // Stream response from orchestrator
      const response = await axios.post(`${CONFIG.ORCHESTRATOR_URL}/execute`, {
        task: message,
        mode,
        stream: true
      }, {
        responseType: 'stream'
      });

      response.data.on('data', (chunk) => {
        res.write(`data: ${chunk}\n\n`);
      });

      response.data.on('end', () => {
        res.write('data: [DONE]\n\n');
        res.end();
      });
    } else {
      const response = await axios.post(`${CONFIG.ORCHESTRATOR_URL}/execute`, {
        task: message,
        mode
      });

      res.json(response.data);
    }
  } catch (error) {
    console.error('Chat error:', error);
    res.status(500).json({
      error: 'Failed to process request',
      details: error.message
    });
  }
});

// Document upload and processing
app.post('/api/upload', upload.single('document'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No file uploaded' });
    }

    const formData = new FormData();
    const fs = require('fs');
    formData.append('file', fs.createReadStream(req.file.path), req.file.originalname);

    // Process with RAG pipeline
    const response = await axios.post(`${CONFIG.ORCHESTRATOR_URL}/rag/ingest`, formData, {
      headers: {
        ...formData.getHeaders()
      }
    });

    // Clean up temporary file
    fs.unlinkSync(req.file.path);

    res.json({
      success: true,
      document_id: response.data.document_id,
      summary: response.data.summary,
      chunks: response.data.chunks
    });
  } catch (error) {
    console.error('Upload error:', error);
    res.status(500).json({
      error: 'Failed to process document',
      details: error.message
    });
  }
});

// Search documents
app.post('/api/search', async (req, res) => {
  try {
    const { query, limit = 10 } = req.body;

    if (!query) {
      return res.status(400).json({ error: 'Query is required' });
    }

    const response = await axios.post(`${CONFIG.ORCHESTRATOR_URL}/rag/search`, {
      query,
      limit
    });

    res.json(response.data);
  } catch (error) {
    console.error('Search error:', error);
    res.status(500).json({
      error: 'Failed to search documents',
      details: error.message
    });
  }
});

// Cost monitoring
app.get('/api/costs', async (req, res) => {
  try {
    const response = await axios.get(`${CONFIG.ORCHESTRATOR_URL}/costs`);
    res.json(response.data);
  } catch (error) {
    console.error('Cost monitoring error:', error);
    res.status(500).json({
      error: 'Failed to fetch cost data',
      details: error.message
    });
  }
});

// Model statistics
app.get('/api/models/stats', async (req, res) => {
  try {
    const response = await axios.get(`${CONFIG.ORCHESTRATOR_URL}/models/stats`);
    res.json(response.data);
  } catch (error) {
    console.error('Model stats error:', error);
    res.status(500).json({
      error: 'Failed to fetch model statistics',
      details: error.message
    });
  }
});

// Trigger n8n workflow
app.post('/api/workflows/:workflowId/trigger', async (req, res) => {
  try {
    const { workflowId } = req.params;
    const payload = req.body;

    const response = await axios.post(
      `${CONFIG.N8N_URL}/webhook/${workflowId}`,
      payload
    );

    res.json({
      success: true,
      execution_id: response.data.executionId,
      data: response.data
    });
  } catch (error) {
    console.error('Workflow trigger error:', error);
    res.status(500).json({
      error: 'Failed to trigger workflow',
      details: error.message
    });
  }
});

// WebSocket connections for real-time updates
io.on('connection', (socket) => {
  console.log('Client connected:', socket.id);

  socket.on('join-room', (room) => {
    socket.join(room);
    console.log(`Socket ${socket.id} joined room ${room}`);
  });

  socket.on('chat-message', async (data) => {
    try {
      const { message, mode = 'auto' } = data;

      // Emit typing indicator
      socket.emit('typing', true);

      // Process with orchestrator
      const response = await axios.post(`${CONFIG.ORCHESTRATOR_URL}/execute`, {
        task: message,
        mode
      });

      // Emit response
      socket.emit('chat-response', {
        response: response.data.response,
        metadata: response.data.metadata,
        timestamp: new Date().toISOString()
      });

      socket.emit('typing', false);
    } catch (error) {
      console.error('WebSocket chat error:', error);
      socket.emit('error', {
        message: 'Failed to process message',
        error: error.message
      });
      socket.emit('typing', false);
    }
  });

  socket.on('disconnect', () => {
    console.log('Client disconnected:', socket.id);
  });
});

// Serve React app for all other routes
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, '../public/index.html'));
});

// Error handling middleware
app.use((error, req, res, next) => {
  console.error('Unhandled error:', error);
  res.status(500).json({
    error: 'Internal server error',
    message: error.message
  });
});

const PORT = process.env.PORT || 3000;
server.listen(PORT, '0.0.0.0', () => {
  console.log(`Demo server running on port ${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/api/health`);
});

module.exports = { app, server };