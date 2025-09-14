"""LangGraph-based orchestrator for complex multi-agent workflows."""

from typing import Dict, Any, Optional, List, TypedDict, Annotated, Sequence, Literal
from dataclasses import dataclass
import operator
import asyncio
import json
import time
from enum import Enum

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain.schema.runnable import RunnableConfig
from pydantic import BaseModel, Field

from ..models.router import ModelRouter, TaskProfile, TaskComplexity
from ..utils.logging import get_logger

logger = get_logger(__name__)


class AgentState(TypedDict):
    """State for the agent workflow."""

    messages: Annotated[Sequence[BaseMessage], operator.add]
    task: str
    context: Dict[str, Any]
    current_agent: str
    iteration: int
    max_iterations: int
    reflection_count: int
    should_continue: bool
    final_answer: Optional[str]
    metadata: Dict[str, Any]
    errors: List[str]
    quality_scores: List[float]


class WorkflowNode(str, Enum):
    """Nodes in the workflow graph."""

    ROUTER = "router"
    PLANNER = "planner"
    EXECUTOR = "executor"
    REFLECTOR = "reflector"
    TOOLS = "tools"
    EVALUATOR = "evaluator"
    SYNTHESIZER = "synthesizer"


@dataclass
class WorkflowConfig:
    """Configuration for workflow execution."""

    max_iterations: int = 10
    max_reflection_attempts: int = 3
    enable_parallel_execution: bool = True
    enable_caching: bool = True
    quality_threshold: float = 0.7
    timeout_seconds: int = 300


class LangGraphOrchestrator:
    """
    Advanced orchestrator using LangGraph for complex workflows.

    Features:
    - Multi-agent coordination
    - Conditional routing
    - State management
    - Parallel execution
    - Checkpointing and recovery
    """

    def __init__(
        self,
        model_router: ModelRouter,
        tools: List[BaseTool],
        config: Optional[WorkflowConfig] = None
    ):
        self.model_router = model_router
        self.tools = tools
        self.config = config or WorkflowConfig()
        self.checkpointer = MemorySaver()
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""

        # Create the graph
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node(WorkflowNode.ROUTER, self._route_task)
        workflow.add_node(WorkflowNode.PLANNER, self._plan_execution)
        workflow.add_node(WorkflowNode.EXECUTOR, self._execute_plan)
        workflow.add_node(WorkflowNode.REFLECTOR, self._reflect_on_execution)
        workflow.add_node(WorkflowNode.TOOLS, self._execute_tools)
        workflow.add_node(WorkflowNode.EVALUATOR, self._evaluate_quality)
        workflow.add_node(WorkflowNode.SYNTHESIZER, self._synthesize_answer)

        # Set entry point
        workflow.set_entry_point(WorkflowNode.ROUTER)

        # Add edges with conditions
        workflow.add_conditional_edges(
            WorkflowNode.ROUTER,
            self._route_condition,
            {
                "plan": WorkflowNode.PLANNER,
                "execute": WorkflowNode.EXECUTOR,
                "synthesize": WorkflowNode.SYNTHESIZER,
            }
        )

        workflow.add_conditional_edges(
            WorkflowNode.PLANNER,
            self._plan_condition,
            {
                "execute": WorkflowNode.EXECUTOR,
                "reflect": WorkflowNode.REFLECTOR,
            }
        )

        workflow.add_conditional_edges(
            WorkflowNode.EXECUTOR,
            self._execution_condition,
            {
                "tools": WorkflowNode.TOOLS,
                "evaluate": WorkflowNode.EVALUATOR,
                "reflect": WorkflowNode.REFLECTOR,
            }
        )

        workflow.add_conditional_edges(
            WorkflowNode.TOOLS,
            self._tool_condition,
            {
                "continue": WorkflowNode.EXECUTOR,
                "evaluate": WorkflowNode.EVALUATOR,
            }
        )

        workflow.add_conditional_edges(
            WorkflowNode.REFLECTOR,
            self._reflection_condition,
            {
                "replan": WorkflowNode.PLANNER,
                "retry": WorkflowNode.EXECUTOR,
                "synthesize": WorkflowNode.SYNTHESIZER,
            }
        )

        workflow.add_conditional_edges(
            WorkflowNode.EVALUATOR,
            self._evaluation_condition,
            {
                "improve": WorkflowNode.REFLECTOR,
                "synthesize": WorkflowNode.SYNTHESIZER,
                "end": END,
            }
        )

        workflow.add_edge(WorkflowNode.SYNTHESIZER, END)

        return workflow.compile(checkpointer=self.checkpointer)

    async def run(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        config: Optional[RunnableConfig] = None
    ) -> Dict[str, Any]:
        """Execute the workflow for a given task."""

        # Initialize state
        initial_state: AgentState = {
            "messages": [HumanMessage(content=task)],
            "task": task,
            "context": context or {},
            "current_agent": WorkflowNode.ROUTER,
            "iteration": 0,
            "max_iterations": self.config.max_iterations,
            "reflection_count": 0,
            "should_continue": True,
            "final_answer": None,
            "metadata": {},
            "errors": [],
            "quality_scores": [],
        }

        # Run the graph
        start_time = time.time()

        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                self.graph.ainvoke(initial_state, config=config),
                timeout=self.config.timeout_seconds
            )

            execution_time = time.time() - start_time

            return {
                "success": True,
                "answer": result.get("final_answer"),
                "execution_time": execution_time,
                "iterations": result.get("iteration"),
                "quality_scores": result.get("quality_scores"),
                "metadata": result.get("metadata"),
                "messages": [self._serialize_message(m) for m in result.get("messages", [])]
            }

        except asyncio.TimeoutError:
            logger.error(f"Workflow timeout after {self.config.timeout_seconds}s")
            return {
                "success": False,
                "error": "Workflow execution timeout",
                "execution_time": self.config.timeout_seconds
            }

        except Exception as e:
            logger.error(f"Workflow execution error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "execution_time": time.time() - start_time
            }

    async def _route_task(self, state: AgentState) -> AgentState:
        """Route task to appropriate starting point based on complexity."""

        task = state["task"]
        context = state["context"]

        # Analyze task complexity
        task_profile = TaskProfile(
            prompt_tokens=len(task) // 4,
            expected_output_tokens=500,
            requires_tools=self._requires_tools(task),
            requires_reasoning=self._requires_reasoning(task)
        )

        complexity = self.model_router.analyze_task_complexity(task_profile)

        # Select initial model
        model_id, routing_metadata = self.model_router.select_model(task_profile)

        # Update state
        state["metadata"]["complexity"] = complexity.value
        state["metadata"]["selected_model"] = model_id
        state["metadata"]["routing"] = routing_metadata
        state["current_agent"] = WorkflowNode.PLANNER if complexity in [
            TaskComplexity.COMPLEX, TaskComplexity.CRITICAL
        ] else WorkflowNode.EXECUTOR

        logger.info(f"Routed task to {state['current_agent']} with model {model_id}")

        return state

    async def _plan_execution(self, state: AgentState) -> AgentState:
        """Create execution plan for complex tasks."""

        task = state["task"]
        model_id = state["metadata"].get("selected_model")

        # Generate plan using selected model
        plan_prompt = f"""Create a detailed execution plan for this task:
{task}

Break it down into clear, actionable steps.
Consider what tools or resources might be needed.
Output a JSON array of steps."""

        # In production, this would call the actual model
        # For now, create a sample plan
        plan = [
            {"step": 1, "action": "analyze_requirements", "description": "Understand task requirements"},
            {"step": 2, "action": "gather_information", "description": "Collect necessary information"},
            {"step": 3, "action": "process_data", "description": "Process and analyze data"},
            {"step": 4, "action": "generate_response", "description": "Create final response"}
        ]

        state["metadata"]["execution_plan"] = plan
        state["messages"].append(AIMessage(
            content=f"Execution plan created: {json.dumps(plan, indent=2)}"
        ))

        return state

    async def _execute_plan(self, state: AgentState) -> AgentState:
        """Execute the planned steps."""

        state["iteration"] += 1

        if state["iteration"] > state["max_iterations"]:
            state["should_continue"] = False
            logger.warning("Max iterations reached")
            return state

        # Get current plan step
        plan = state["metadata"].get("execution_plan", [])
        current_step = min(state["iteration"] - 1, len(plan) - 1)

        if current_step < len(plan):
            step = plan[current_step]
            logger.info(f"Executing step {step['step']}: {step['description']}")

            # Execute step (placeholder for actual execution)
            state["messages"].append(AIMessage(
                content=f"Executing: {step['description']}"
            ))

            # Simulate step execution
            if step["action"] == "gather_information" and self.tools:
                state["metadata"]["needs_tools"] = True

        return state

    async def _reflect_on_execution(self, state: AgentState) -> AgentState:
        """Reflect on execution progress and adjust strategy."""

        state["reflection_count"] += 1

        # Analyze progress
        messages = state["messages"]
        errors = state["errors"]

        reflection_prompt = f"""Reflect on the execution so far:
- Progress made: {len(messages)} messages
- Errors encountered: {len(errors)}
- Current iteration: {state['iteration']}

Should we continue, adjust approach, or synthesize answer?"""

        # Simple reflection logic
        if state["reflection_count"] > self.config.max_reflection_attempts:
            state["should_continue"] = False
            state["metadata"]["reflection_decision"] = "max_reflections_reached"
        elif errors:
            state["metadata"]["reflection_decision"] = "retry_with_adjustments"
        else:
            state["metadata"]["reflection_decision"] = "continue_execution"

        state["messages"].append(SystemMessage(
            content=f"Reflection: {state['metadata']['reflection_decision']}"
        ))

        return state

    async def _execute_tools(self, state: AgentState) -> AgentState:
        """Execute tool calls."""

        # Get tool requirements from state
        tool_calls = state["metadata"].get("tool_calls", [])

        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("arguments", {})

            # Find and execute tool
            tool = next((t for t in self.tools if t.name == tool_name), None)
            if tool:
                try:
                    result = await tool.ainvoke(tool_args)
                    state["messages"].append(ToolMessage(
                        content=str(result),
                        tool_call_id=tool_call.get("id", "")
                    ))
                except Exception as e:
                    state["errors"].append(f"Tool error: {str(e)}")
                    logger.error(f"Tool execution error: {str(e)}")

        return state

    async def _evaluate_quality(self, state: AgentState) -> AgentState:
        """Evaluate quality of current execution."""

        # Calculate quality score based on various factors
        quality_score = 0.0

        # Factor 1: Completion ratio
        if state["metadata"].get("execution_plan"):
            plan = state["metadata"]["execution_plan"]
            completion_ratio = state["iteration"] / len(plan)
            quality_score += completion_ratio * 0.3

        # Factor 2: Error rate
        error_rate = len(state["errors"]) / max(state["iteration"], 1)
        quality_score += (1 - error_rate) * 0.3

        # Factor 3: Message coherence (simplified)
        if len(state["messages"]) > 2:
            quality_score += 0.4

        state["quality_scores"].append(quality_score)
        state["metadata"]["current_quality"] = quality_score

        logger.info(f"Quality evaluation: {quality_score:.2f}")

        return state

    async def _synthesize_answer(self, state: AgentState) -> AgentState:
        """Synthesize final answer from execution results."""

        messages = state["messages"]
        task = state["task"]

        # Generate synthesis prompt
        synthesis_prompt = f"""Based on the execution results, synthesize a final answer for:
{task}

Consider all gathered information and provide a comprehensive response."""

        # In production, this would use the model to synthesize
        # For now, create a sample answer
        final_answer = "Based on the comprehensive analysis and execution of the task, " \
                      "here is the synthesized final answer incorporating all gathered information..."

        state["final_answer"] = final_answer
        state["messages"].append(AIMessage(content=final_answer))

        return state

    # Routing conditions
    def _route_condition(self, state: AgentState) -> str:
        """Determine initial routing based on task complexity."""
        complexity = state["metadata"].get("complexity")
        if complexity in ["complex", "critical"]:
            return "plan"
        elif complexity in ["simple"]:
            return "synthesize"
        else:
            return "execute"

    def _plan_condition(self, state: AgentState) -> str:
        """Determine next step after planning."""
        if state["metadata"].get("execution_plan"):
            return "execute"
        else:
            return "reflect"

    def _execution_condition(self, state: AgentState) -> str:
        """Determine next step after execution."""
        if state["metadata"].get("needs_tools"):
            return "tools"
        elif state["iteration"] % 3 == 0:  # Periodic evaluation
            return "evaluate"
        elif state["errors"]:
            return "reflect"
        else:
            return "evaluate"

    def _tool_condition(self, state: AgentState) -> str:
        """Determine next step after tool execution."""
        if state["should_continue"]:
            return "continue"
        else:
            return "evaluate"

    def _reflection_condition(self, state: AgentState) -> str:
        """Determine next step after reflection."""
        decision = state["metadata"].get("reflection_decision")
        if decision == "max_reflections_reached":
            return "synthesize"
        elif decision == "retry_with_adjustments":
            return "retry"
        else:
            return "replan"

    def _evaluation_condition(self, state: AgentState) -> str:
        """Determine next step after evaluation."""
        quality = state["metadata"].get("current_quality", 0)
        if quality >= self.config.quality_threshold:
            return "synthesize"
        elif state["reflection_count"] < self.config.max_reflection_attempts:
            return "improve"
        else:
            return "end"

    # Helper methods
    def _requires_tools(self, task: str) -> bool:
        """Check if task requires tool usage."""
        tool_keywords = ["search", "calculate", "fetch", "query", "execute", "browse"]
        return any(keyword in task.lower() for keyword in tool_keywords)

    def _requires_reasoning(self, task: str) -> bool:
        """Check if task requires complex reasoning."""
        reasoning_keywords = ["explain", "analyze", "compare", "evaluate", "reason", "think"]
        return any(keyword in task.lower() for keyword in reasoning_keywords)

    def _serialize_message(self, message: BaseMessage) -> Dict[str, Any]:
        """Serialize message for output."""
        return {
            "type": message.__class__.__name__,
            "content": message.content,
            "additional_kwargs": message.additional_kwargs
        }