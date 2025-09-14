"""
LangGraph Multi-Agent Orchestration System
"""

import os
import asyncio
from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass
import json
import logging

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_community.llms import VLLMOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
import redis.asyncio as redis
from prometheus_client import Counter, Histogram, generate_latest
import uvicorn

# Configure logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

# Metrics
request_counter = Counter("langgraph_requests_total", "Total requests", ["agent", "status"])
latency_histogram = Histogram("langgraph_latency_seconds", "Request latency", ["agent"])

# Initialize FastAPI
app = FastAPI(title="LangGraph Agent Orchestration API", version="1.0.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
class Config:
    DATABASE_URL = os.getenv("DATABASE_URL")
    REDIS_URL = os.getenv("REDIS_URL")
    CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://100.64.1.1:8000/v1")
    QDRANT_URL = os.getenv("QDRANT_URL", "http://100.64.1.1:6333")
    EMBEDDING_SERVICE_URL = os.getenv("EMBEDDING_SERVICE_URL", "http://100.64.1.1:8001")
    API_KEY = os.getenv("API_KEY", "changeme")

config = Config()

# Initialize Redis
redis_client = redis.from_url(config.REDIS_URL, decode_responses=True)

# Initialize LLMs
claude_opus = ChatAnthropic(
    model="claude-3-opus-20240229",
    anthropic_api_key=config.CLAUDE_API_KEY,
    max_tokens=4096
)

claude_sonnet = ChatAnthropic(
    model="claude-3-sonnet-20240229",
    anthropic_api_key=config.CLAUDE_API_KEY,
    max_tokens=4096
)

vllm_local = VLLMOpenAI(
    openai_api_key="EMPTY",
    openai_api_base=config.VLLM_BASE_URL,
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    max_tokens=2048
)

# Initialize Vector Store
qdrant_client = QdrantClient(url=config.QDRANT_URL)
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# Agent Types
class AgentType(str, Enum):
    RESEARCHER = "researcher"
    CODER = "coder"
    ANALYZER = "analyzer"
    PLANNER = "planner"
    EXECUTOR = "executor"

# State Management
@dataclass
class AgentState:
    messages: List[Dict[str, Any]]
    current_agent: Optional[str]
    context: Dict[str, Any]
    results: List[Dict[str, Any]]
    next_action: Optional[str]
    metadata: Dict[str, Any]

# Request/Response Models
class AgentRequest(BaseModel):
    agent: AgentType
    input: str
    context: Optional[Dict[str, Any]] = {}
    config: Optional[Dict[str, Any]] = {}

class AgentResponse(BaseModel):
    agent: str
    output: str
    context: Dict[str, Any]
    metadata: Dict[str, Any]

# Agent Implementations
class ResearchAgent:
    """Agent for research and information gathering"""

    def __init__(self, llm, vector_store):
        self.llm = llm
        self.vector_store = vector_store

    async def run(self, state: AgentState) -> AgentState:
        query = state.messages[-1]["content"]

        # Search vector store
        docs = self.vector_store.similarity_search(query, k=5)
        context = "\n".join([doc.page_content for doc in docs])

        # Generate response
        prompt = f"Based on the following context, answer the query:\n\nContext: {context}\n\nQuery: {query}"
        response = await self.llm.ainvoke(prompt)

        state.results.append({
            "agent": "researcher",
            "output": response.content,
            "sources": [doc.metadata for doc in docs]
        })

        return state

class CoderAgent:
    """Agent for code generation and debugging"""

    def __init__(self, llm):
        self.llm = llm

    async def run(self, state: AgentState) -> AgentState:
        task = state.messages[-1]["content"]

        prompt = f"""You are an expert programmer. Generate code for the following task:

Task: {task}

Provide clean, well-commented code with error handling."""

        response = await self.llm.ainvoke(prompt)

        state.results.append({
            "agent": "coder",
            "output": response.content,
            "language": state.context.get("language", "python")
        })

        return state

class AnalyzerAgent:
    """Agent for data analysis and insights"""

    def __init__(self, llm):
        self.llm = llm

    async def run(self, state: AgentState) -> AgentState:
        data = state.context.get("data", "")
        query = state.messages[-1]["content"]

        prompt = f"""Analyze the following data and provide insights:

Data: {data}

Analysis Request: {query}

Provide detailed analysis with key findings and recommendations."""

        response = await self.llm.ainvoke(prompt)

        state.results.append({
            "agent": "analyzer",
            "output": response.content,
            "analysis_type": state.context.get("analysis_type", "general")
        })

        return state

class PlannerAgent:
    """Agent for task planning and orchestration"""

    def __init__(self, llm):
        self.llm = llm

    async def run(self, state: AgentState) -> AgentState:
        goal = state.messages[-1]["content"]

        prompt = f"""Create a detailed plan to achieve the following goal:

Goal: {goal}

Break down the goal into specific, actionable steps. For each step, identify:
1. The action to take
2. The agent best suited for the task
3. Dependencies and prerequisites
4. Success criteria

Format as JSON."""

        response = await self.llm.ainvoke(prompt)

        try:
            plan = json.loads(response.content)
        except:
            plan = {"steps": [{"action": response.content, "agent": "executor"}]}

        state.results.append({
            "agent": "planner",
            "output": response.content,
            "plan": plan
        })

        # Set next action based on plan
        if plan.get("steps"):
            state.next_action = plan["steps"][0].get("agent", "executor")

        return state

class ExecutorAgent:
    """Agent for executing planned actions"""

    def __init__(self, llm):
        self.llm = llm

    async def run(self, state: AgentState) -> AgentState:
        action = state.messages[-1]["content"]
        plan = state.context.get("plan", {})

        prompt = f"""Execute the following action:

Action: {action}

Context: {json.dumps(plan, indent=2)}

Provide the execution result and any relevant output."""

        response = await self.llm.ainvoke(prompt)

        state.results.append({
            "agent": "executor",
            "output": response.content,
            "status": "completed"
        })

        return state

# Agent Orchestrator
class AgentOrchestrator:
    """Main orchestrator for managing multi-agent workflows"""

    def __init__(self):
        # Initialize vector store
        self.vector_store = Qdrant(
            client=qdrant_client,
            collection_name="knowledge_base",
            embeddings=embeddings
        )

        # Initialize agents with appropriate LLMs
        self.agents = {
            AgentType.RESEARCHER: ResearchAgent(claude_sonnet, self.vector_store),
            AgentType.CODER: CoderAgent(claude_sonnet),
            AgentType.ANALYZER: AnalyzerAgent(claude_sonnet),
            AgentType.PLANNER: PlannerAgent(claude_opus),
            AgentType.EXECUTOR: ExecutorAgent(vllm_local)
        }

        # Build workflow graph
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """Build the agent workflow graph"""
        workflow = StateGraph(AgentState)

        # Add nodes for each agent
        for agent_type, agent in self.agents.items():
            workflow.add_node(agent_type.value, agent.run)

        # Add routing logic
        def route_next(state: AgentState) -> str:
            if state.next_action:
                return state.next_action
            return END

        # Add edges
        workflow.add_edge("planner", route_next)
        workflow.add_edge("researcher", END)
        workflow.add_edge("coder", END)
        workflow.add_edge("analyzer", END)
        workflow.add_edge("executor", END)

        # Set entry point
        workflow.set_entry_point("planner")

        return workflow.compile()

    async def run_agent(self, request: AgentRequest) -> AgentResponse:
        """Run a specific agent with the given input"""
        # Initialize state
        state = AgentState(
            messages=[{"role": "user", "content": request.input}],
            current_agent=request.agent.value,
            context=request.context or {},
            results=[],
            next_action=None,
            metadata={"config": request.config}
        )

        # Get the appropriate agent
        agent = self.agents.get(request.agent)
        if not agent:
            raise ValueError(f"Unknown agent: {request.agent}")

        # Run the agent
        with latency_histogram.labels(agent=request.agent.value).time():
            state = await agent.run(state)

        # Extract results
        result = state.results[-1] if state.results else {"output": "No output"}

        request_counter.labels(agent=request.agent.value, status="success").inc()

        return AgentResponse(
            agent=request.agent.value,
            output=result.get("output", ""),
            context=state.context,
            metadata=state.metadata
        )

    async def run_workflow(self, goal: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run a complete multi-agent workflow"""
        # Initialize state
        state = AgentState(
            messages=[{"role": "user", "content": goal}],
            current_agent="planner",
            context={},
            results=[],
            next_action=None,
            metadata={"config": config or {}}
        )

        # Run the workflow
        final_state = await self.workflow.ainvoke(state)

        return {
            "goal": goal,
            "results": final_state.results,
            "context": final_state.context,
            "metadata": final_state.metadata
        }

# Initialize orchestrator
orchestrator = AgentOrchestrator()

# API Key Authentication
async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != config.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key

# API Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "langgraph-orchestrator"}

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()

@app.post("/agents/invoke", dependencies=[Depends(verify_api_key)])
async def invoke_agent(request: AgentRequest) -> AgentResponse:
    """Invoke a specific agent"""
    try:
        # Check cache
        cache_key = f"agent:{request.agent}:{hash(request.input)}"
        cached = await redis_client.get(cache_key)
        if cached:
            return AgentResponse(**json.loads(cached))

        # Run agent
        response = await orchestrator.run_agent(request)

        # Cache result
        await redis_client.setex(
            cache_key,
            86400,  # 24 hour TTL
            json.dumps(response.dict())
        )

        return response

    except Exception as e:
        logger.error(f"Agent invocation failed: {e}")
        request_counter.labels(agent=request.agent.value, status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/workflow/run", dependencies=[Depends(verify_api_key)])
async def run_workflow(goal: str, config: Optional[Dict[str, Any]] = None):
    """Run a complete multi-agent workflow"""
    try:
        result = await orchestrator.run_workflow(goal, config)
        return result

    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents")
async def list_agents():
    """List available agents"""
    return {
        "agents": [
            {
                "name": agent.value,
                "description": orchestrator.agents[agent].__class__.__doc__
            }
            for agent in AgentType
        ]
    }

# Main entry point
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        log_level=os.getenv("LOG_LEVEL", "info").lower()
    )