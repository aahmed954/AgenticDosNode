"""Comprehensive usage examples for the Agentic Orchestrator."""

import asyncio
import os
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain.tools import Tool, StructuredTool
from langchain.schema import Document
from pydantic import BaseModel, Field

# Import our framework
from src.orchestrator import (
    AgenticOrchestrator,
    OrchestrationRequest,
    ExecutionMode
)
from src.rag.advanced_rag import RAGConfig, RetrievalStrategy
from src.config import settings

# Load environment variables
load_dotenv()


# Example 1: Basic Usage with Auto Mode
async def example_basic_usage():
    """Basic usage with automatic mode selection."""

    print("\n=== Example 1: Basic Usage ===\n")

    # Initialize orchestrator
    orchestrator = AgenticOrchestrator()

    # Simple request
    request = OrchestrationRequest(
        task="Explain the concept of reinforcement learning in simple terms",
        mode=ExecutionMode.AUTO  # Let the system decide
    )

    response = await orchestrator.execute(request)

    print(f"Success: {response.success}")
    print(f"Model Used: {response.model_used}")
    print(f"Execution Time: {response.execution_time:.2f}s")
    print(f"Cost: ${response.total_cost:.4f}")
    print(f"Cached: {response.cached}")
    print(f"\nResult: {response.result}")


# Example 2: Tool Usage with ReAct Agent
async def example_tool_usage():
    """Example using tools with ReAct agent."""

    print("\n=== Example 2: Tool Usage with ReAct ===\n")

    # Define custom tools
    def calculate(expression: str) -> str:
        """Calculate mathematical expression."""
        try:
            result = eval(expression)
            return f"The result is: {result}"
        except:
            return "Invalid expression"

    def search_web(query: str) -> str:
        """Simulate web search."""
        return f"Search results for '{query}': [Simulated results about {query}]"

    tools = [
        Tool(
            name="calculator",
            description="Calculate mathematical expressions",
            func=calculate
        ),
        Tool(
            name="web_search",
            description="Search the web for information",
            func=search_web
        )
    ]

    # Initialize orchestrator with tools
    orchestrator = AgenticOrchestrator(tools=tools)

    # Request requiring tools
    request = OrchestrationRequest(
        task="What is 15% of 2500, and search for the current inflation rate?",
        mode=ExecutionMode.REACT,
        tools=tools
    )

    response = await orchestrator.execute(request)
    print(f"Result: {response.result}")
    print(f"Execution traces: {len(response.traces or [])} steps")


# Example 3: Complex Workflow with LangGraph
async def example_complex_workflow():
    """Example of complex multi-step workflow."""

    print("\n=== Example 3: Complex Workflow with LangGraph ===\n")

    orchestrator = AgenticOrchestrator()

    # Complex multi-step task
    request = OrchestrationRequest(
        task="""Analyze the following business scenario and provide recommendations:
        1. A SaaS company has 1000 customers
        2. Monthly churn rate is 5%
        3. Average revenue per user is $100
        4. Customer acquisition cost is $500

        Calculate the customer lifetime value, analyze the unit economics,
        and provide three strategic recommendations to improve profitability.""",
        mode=ExecutionMode.LANGGRAPH,
        context={
            "industry": "SaaS",
            "analysis_depth": "comprehensive"
        }
    )

    response = await orchestrator.execute(request)
    print(f"Complex analysis completed in {response.execution_time:.2f}s")
    print(f"Result: {response.result}")


# Example 4: RAG Integration
async def example_rag_integration():
    """Example with RAG system integration."""

    print("\n=== Example 4: RAG Integration ===\n")

    # Mock vector store for demonstration
    class MockVectorStore:
        async def add_documents(self, documents, embeddings):
            pass

        async def similarity_search(self, query_embedding, k):
            # Return mock documents
            return [
                (Document(page_content="LLMs are large language models trained on vast text data.", metadata={"source": "doc1"}), 0.95),
                (Document(page_content="Transformers architecture revolutionized NLP with attention mechanisms.", metadata={"source": "doc2"}), 0.89),
            ]

        async def hybrid_search(self, query, query_embedding, k):
            return await self.similarity_search(query_embedding, k)

    vector_store = MockVectorStore()
    orchestrator = AgenticOrchestrator(vector_store=vector_store)

    # Request with RAG
    request = OrchestrationRequest(
        task="Based on the documents, explain how LLMs work",
        rag_query="How do large language models work?",
        context={"use_citations": True}
    )

    response = await orchestrator.execute(request)
    print(f"RAG-enhanced response: {response.result}")


# Example 5: Streaming Results
async def example_streaming():
    """Example with streaming results."""

    print("\n=== Example 5: Streaming Results ===\n")

    orchestrator = AgenticOrchestrator()

    request = OrchestrationRequest(
        task="Write a short story about AI agents collaborating",
        mode=ExecutionMode.REACT,
        stream=True
    )

    print("Streaming results:")
    async for chunk in await orchestrator.execute(request):
        if chunk["type"] == "thought":
            print(f"\n💭 Thinking: {chunk['content']}")
        elif chunk["type"] == "action":
            print(f"⚡ Action: {chunk['content']}")
        elif chunk["type"] == "observation":
            print(f"👁️ Observation: {chunk['content']}")
        elif chunk["type"] == "answer":
            print(f"\n✅ Final Answer: {chunk['content']}")


# Example 6: Cost-Optimized Batch Processing
async def example_batch_processing():
    """Example of cost-optimized batch processing."""

    print("\n=== Example 6: Batch Processing ===\n")

    orchestrator = AgenticOrchestrator()

    # Multiple similar requests
    tasks = [
        "Summarize the benefits of cloud computing",
        "Summarize the benefits of edge computing",
        "Summarize the benefits of quantum computing",
    ]

    # Process in parallel with caching
    async def process_task(task):
        request = OrchestrationRequest(
            task=task,
            mode=ExecutionMode.DIRECT,
            use_cache=True,
            priority=1
        )
        return await orchestrator.execute(request)

    # Execute all tasks
    results = await asyncio.gather(*[process_task(task) for task in tasks])

    total_cost = sum(r.total_cost for r in results)
    total_time = sum(r.execution_time for r in results)

    print(f"Processed {len(tasks)} tasks")
    print(f"Total cost: ${total_cost:.4f}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average cost per task: ${total_cost/len(tasks):.4f}")


# Example 7: Multi-Model Routing
async def example_multi_model_routing():
    """Example showing intelligent model routing."""

    print("\n=== Example 7: Multi-Model Routing ===\n")

    orchestrator = AgenticOrchestrator()

    # Different complexity tasks
    tasks = [
        ("What is 2+2?", "simple"),
        ("Explain quantum entanglement to a 5-year-old", "moderate"),
        ("Design a distributed system for real-time ML inference at scale", "complex"),
    ]

    for task, expected_complexity in tasks:
        request = OrchestrationRequest(
            task=task,
            mode=ExecutionMode.AUTO
        )

        response = await orchestrator.execute(request)

        print(f"\nTask: {task[:50]}...")
        print(f"Expected Complexity: {expected_complexity}")
        print(f"Model Selected: {response.model_used}")
        print(f"Cost: ${response.total_cost:.4f}")
        print(f"Time: {response.execution_time:.2f}s")


# Example 8: Advanced RAG with Multiple Strategies
async def example_advanced_rag():
    """Example of advanced RAG with different retrieval strategies."""

    print("\n=== Example 8: Advanced RAG Strategies ===\n")

    class AdvancedVectorStore:
        async def add_documents(self, documents, embeddings):
            pass

        async def similarity_search(self, query_embedding, k):
            return [
                (Document(page_content=f"Document about {i}", metadata={"id": i}), 0.9 - i*0.1)
                for i in range(k)
            ]

        async def hybrid_search(self, query, query_embedding, k):
            # Combine vector and keyword search
            return await self.similarity_search(query_embedding, k)

    vector_store = AdvancedVectorStore()
    orchestrator = AgenticOrchestrator(vector_store=vector_store)

    strategies = [
        RetrievalStrategy.SIMILARITY,
        RetrievalStrategy.HYBRID,
        RetrievalStrategy.HYDE,
        RetrievalStrategy.FUSION
    ]

    query = "Explain the latest advances in generative AI"

    for strategy in strategies:
        # Configure RAG to use specific strategy
        request = OrchestrationRequest(
            task=f"Using {strategy.value} retrieval: {query}",
            rag_query=query,
            context={"retrieval_strategy": strategy.value}
        )

        response = await orchestrator.execute(request)
        print(f"\n{strategy.value} Strategy:")
        print(f"  Time: {response.execution_time:.2f}s")
        print(f"  Documents used: {len(response.metadata.get('rag', {}).get('documents', []))}")


# Example 9: Monitoring and Health Checks
async def example_monitoring():
    """Example of monitoring and health checks."""

    print("\n=== Example 9: Monitoring & Health ===\n")

    orchestrator = AgenticOrchestrator(enable_monitoring=True)

    # Execute some requests to generate metrics
    for i in range(5):
        request = OrchestrationRequest(
            task=f"Test task {i}",
            mode=ExecutionMode.DIRECT
        )
        await orchestrator.execute(request)

    # Get statistics
    stats = await orchestrator.get_statistics()
    print("Statistics:")
    print(f"  Total Requests: {stats['total_requests']}")
    print(f"  Total Cost: ${stats['total_cost']:.4f}")
    print(f"  Cache Hit Rate: {stats['cache_stats'].get('hit_rate', 0):.2%}")

    # Health check
    health = await orchestrator.health_check()
    print(f"\nHealth Status: {health['status']}")
    print(f"Components: {health['components']}")


# Example 10: Error Handling and Fallbacks
async def example_error_handling():
    """Example of error handling and fallback strategies."""

    print("\n=== Example 10: Error Handling ===\n")

    orchestrator = AgenticOrchestrator()

    # Request that might fail
    request = OrchestrationRequest(
        task="Process this with a non-existent model",
        mode=ExecutionMode.AUTO,
        context={"force_model": "non-existent-model"},
        budget_limit=0.001  # Very low budget to trigger fallback
    )

    response = await orchestrator.execute(request)

    if not response.success:
        print(f"Request failed: {response.metadata.get('error')}")
        print("Retrying with fallback...")

        # Retry with different parameters
        fallback_request = OrchestrationRequest(
            task=request.task,
            mode=ExecutionMode.DIRECT,
            budget_limit=1.0  # Higher budget
        )

        fallback_response = await orchestrator.execute(fallback_request)
        print(f"Fallback success: {fallback_response.success}")
        print(f"Fallback model: {fallback_response.model_used}")


# Main execution
async def main():
    """Run all examples."""

    examples = [
        example_basic_usage,
        example_tool_usage,
        example_complex_workflow,
        example_rag_integration,
        example_streaming,
        example_batch_processing,
        example_multi_model_routing,
        example_advanced_rag,
        example_monitoring,
        example_error_handling
    ]

    for example in examples:
        try:
            await example()
        except Exception as e:
            print(f"Error in {example.__name__}: {str(e)}")
        print("\n" + "="*50)

    print("\n✅ All examples completed!")


if __name__ == "__main__":
    # Run examples
    asyncio.run(main())