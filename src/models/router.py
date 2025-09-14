"""Intelligent multi-model routing system."""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio
import time
import hashlib
import json
from collections import deque
import numpy as np

from ..config import ModelProvider, TaskComplexity, settings
from ..monitoring.metrics import MetricsCollector
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TaskProfile:
    """Profile for task complexity analysis."""

    prompt_tokens: int
    expected_output_tokens: int
    requires_tools: bool = False
    requires_vision: bool = False
    requires_reasoning: bool = False
    requires_realtime: bool = False
    budget_constraint: Optional[float] = None
    latency_requirement: Optional[float] = None  # in seconds
    retry_count: int = 0
    previous_failures: List[str] = None

    def __post_init__(self):
        if self.previous_failures is None:
            self.previous_failures = []


@dataclass
class ModelMetrics:
    """Runtime metrics for model performance."""

    model_id: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency: float = 0.0
    total_cost: float = 0.0
    avg_quality_score: float = 1.0
    recent_latencies: deque = None
    recent_errors: deque = None
    last_updated: float = 0.0

    def __post_init__(self):
        if self.recent_latencies is None:
            self.recent_latencies = deque(maxlen=100)
        if self.recent_errors is None:
            self.recent_errors = deque(maxlen=10)
        self.last_updated = time.time()

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests

    @property
    def avg_latency(self) -> float:
        """Calculate average latency."""
        if not self.recent_latencies:
            return 0.0
        return np.mean(list(self.recent_latencies))

    @property
    def p95_latency(self) -> float:
        """Calculate P95 latency."""
        if not self.recent_latencies:
            return 0.0
        return np.percentile(list(self.recent_latencies), 95)

    def update(self, latency: float, success: bool, cost: float, quality_score: float = 1.0):
        """Update metrics with new request data."""
        self.total_requests += 1
        self.total_latency += latency
        self.total_cost += cost
        self.recent_latencies.append(latency)

        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
            self.recent_errors.append(time.time())

        # Update quality score with exponential moving average
        alpha = 0.1
        self.avg_quality_score = alpha * quality_score + (1 - alpha) * self.avg_quality_score
        self.last_updated = time.time()


class ModelRouter:
    """Intelligent model routing with cost optimization and fallback strategies."""

    def __init__(self, metrics_collector: Optional[MetricsCollector] = None):
        self.settings = settings
        self.metrics_collector = metrics_collector
        self.model_metrics: Dict[str, ModelMetrics] = {}
        self.routing_history: deque = deque(maxlen=1000)
        self.daily_cost_tracker: Dict[str, float] = {}
        self._initialize_metrics()

    def _initialize_metrics(self):
        """Initialize metrics for all configured models."""
        for model_id in self.settings.model.model_specs.keys():
            self.model_metrics[model_id] = ModelMetrics(model_id=model_id)

    def analyze_task_complexity(self, task: TaskProfile) -> TaskComplexity:
        """Analyze task and determine complexity level."""

        # Simple heuristics for complexity analysis
        complexity_score = 0

        # Token-based scoring
        if task.prompt_tokens < 500:
            complexity_score += 0
        elif task.prompt_tokens < 2000:
            complexity_score += 1
        elif task.prompt_tokens < 10000:
            complexity_score += 2
        else:
            complexity_score += 3

        # Feature requirements
        if task.requires_tools:
            complexity_score += 1
        if task.requires_vision:
            complexity_score += 2
        if task.requires_reasoning:
            complexity_score += 3

        # Output requirements
        if task.expected_output_tokens > 4000:
            complexity_score += 2

        # Retry escalation
        complexity_score += task.retry_count

        # Map score to complexity level
        if complexity_score <= 2:
            return TaskComplexity.SIMPLE
        elif complexity_score <= 5:
            return TaskComplexity.MODERATE
        elif complexity_score <= 8:
            return TaskComplexity.COMPLEX
        else:
            return TaskComplexity.CRITICAL

    def select_model(
        self,
        task: TaskProfile,
        preferred_models: Optional[List[str]] = None,
        excluded_models: Optional[List[str]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Select the optimal model for a given task.

        Returns:
            Tuple of (model_id, routing_metadata)
        """

        # Determine task complexity
        complexity = self.analyze_task_complexity(task)
        logger.info(f"Task complexity determined: {complexity}")

        # Get candidate models based on complexity
        candidate_models = self._get_candidate_models(
            complexity, task, preferred_models, excluded_models
        )

        if not candidate_models:
            # Fallback to default model
            logger.warning("No suitable candidates found, using default model")
            return self.settings.model.default_model, {
                "reason": "no_candidates",
                "complexity": complexity
            }

        # Score and rank models
        model_scores = self._score_models(candidate_models, task)

        # Select best model
        best_model = max(model_scores, key=model_scores.get)

        # Check budget constraints
        if not self._check_budget(best_model, task):
            # Find cheaper alternative
            for model in sorted(model_scores, key=model_scores.get, reverse=True):
                if self._check_budget(model, task):
                    best_model = model
                    break

        # Record routing decision
        routing_metadata = {
            "complexity": complexity.value,
            "candidates": list(candidate_models),
            "scores": model_scores,
            "selected": best_model,
            "reason": "optimal_score",
            "task_features": {
                "tokens": task.prompt_tokens,
                "requires_tools": task.requires_tools,
                "requires_vision": task.requires_vision,
                "requires_reasoning": task.requires_reasoning,
            }
        }

        self.routing_history.append({
            "timestamp": time.time(),
            "model": best_model,
            "metadata": routing_metadata
        })

        logger.info(f"Selected model: {best_model} (score: {model_scores[best_model]:.3f})")

        return best_model, routing_metadata

    def _get_candidate_models(
        self,
        complexity: TaskComplexity,
        task: TaskProfile,
        preferred_models: Optional[List[str]],
        excluded_models: Optional[List[str]]
    ) -> List[str]:
        """Get candidate models based on task requirements."""

        # Start with complexity-based candidates
        candidates = set(self.settings.routing_rules.get(complexity, []))

        # Apply preferences
        if preferred_models:
            candidates = candidates.intersection(set(preferred_models))

        # Apply exclusions
        if excluded_models:
            candidates = candidates - set(excluded_models)

        # Filter by task requirements
        filtered_candidates = []
        for model_id in candidates:
            spec = self.settings.model.model_specs.get(model_id, {})

            # Check capability requirements
            if task.requires_tools and not spec.get("supports_tools", False):
                continue
            if task.requires_vision and not spec.get("supports_vision", False):
                continue
            if task.requires_reasoning and not spec.get("supports_thinking", False):
                continue

            # Check context window
            if task.prompt_tokens > spec.get("context_window", 0):
                continue

            # Check if model has been failing recently
            metrics = self.model_metrics.get(model_id)
            if metrics and metrics.failed_requests > 5 and metrics.success_rate < 0.5:
                if model_id not in task.previous_failures:
                    logger.warning(f"Skipping {model_id} due to recent failures")
                    continue

            filtered_candidates.append(model_id)

        return filtered_candidates

    def _score_models(self, candidates: List[str], task: TaskProfile) -> Dict[str, float]:
        """Score candidate models based on multiple factors."""

        scores = {}

        for model_id in candidates:
            spec = self.settings.model.model_specs[model_id]
            metrics = self.model_metrics[model_id]

            # Base score
            score = 1.0

            # Cost factor (inverse - lower cost is better)
            estimated_cost = self._estimate_cost(model_id, task)
            if estimated_cost > 0:
                cost_score = 1.0 / (1.0 + estimated_cost)
                score *= cost_score

            # Performance factor
            if metrics.total_requests > 0:
                performance_score = (
                    metrics.success_rate * 0.4 +
                    (1.0 - min(metrics.avg_latency / 30.0, 1.0)) * 0.3 +
                    metrics.avg_quality_score * 0.3
                )
                score *= performance_score

            # Latency requirement
            if task.latency_requirement:
                if metrics.p95_latency > task.latency_requirement:
                    score *= 0.5  # Penalize high-latency models

            # Context utilization efficiency
            context_utilization = task.prompt_tokens / spec["context_window"]
            if context_utilization < 0.1:
                score *= 1.2  # Bonus for efficient context use
            elif context_utilization > 0.8:
                score *= 0.8  # Penalty for near-limit usage

            # Provider availability bonus
            if spec["provider"] == ModelProvider.VLLM:
                score *= 1.3  # Prefer local models when suitable

            # Retry bonus - prefer different model on retry
            if model_id in task.previous_failures:
                score *= 0.3

            scores[model_id] = score

        return scores

    def _estimate_cost(self, model_id: str, task: TaskProfile) -> float:
        """Estimate cost for running task on specific model."""

        spec = self.settings.model.model_specs[model_id]

        input_cost = (task.prompt_tokens / 1000) * spec["cost_per_1k_input"]
        output_cost = (task.expected_output_tokens / 1000) * spec["cost_per_1k_output"]

        return input_cost + output_cost

    def _check_budget(self, model_id: str, task: TaskProfile) -> bool:
        """Check if model selection fits within budget constraints."""

        estimated_cost = self._estimate_cost(model_id, task)

        # Check per-request budget
        if task.budget_constraint and estimated_cost > task.budget_constraint:
            return False

        # Check max request cost
        if estimated_cost > self.settings.performance.max_request_cost:
            return False

        # Check daily budget
        today = time.strftime("%Y-%m-%d")
        daily_cost = self.daily_cost_tracker.get(today, 0.0)
        if daily_cost + estimated_cost > self.settings.performance.daily_budget_limit:
            return False

        return True

    def update_metrics(
        self,
        model_id: str,
        latency: float,
        success: bool,
        cost: float,
        quality_score: float = 1.0
    ):
        """Update model metrics after request completion."""

        if model_id not in self.model_metrics:
            self.model_metrics[model_id] = ModelMetrics(model_id=model_id)

        self.model_metrics[model_id].update(latency, success, cost, quality_score)

        # Update daily cost tracker
        today = time.strftime("%Y-%m-%d")
        if today not in self.daily_cost_tracker:
            self.daily_cost_tracker = {today: 0.0}  # Reset for new day
        self.daily_cost_tracker[today] += cost

        # Send metrics to collector if available
        if self.metrics_collector:
            self.metrics_collector.record_model_request(
                model_id=model_id,
                latency=latency,
                success=success,
                cost=cost,
                quality_score=quality_score
            )

    def get_fallback_chain(self, primary_model: str, task: TaskProfile) -> List[str]:
        """Get ordered fallback chain for a model."""

        complexity = self.analyze_task_complexity(task)
        all_candidates = self.settings.routing_rules.get(complexity, [])

        # Remove primary model and previously failed models
        fallbacks = [
            m for m in all_candidates
            if m != primary_model and m not in task.previous_failures
        ]

        # Sort by performance metrics
        fallbacks.sort(
            key=lambda m: self.model_metrics.get(m, ModelMetrics(m)).success_rate,
            reverse=True
        )

        return fallbacks[:3]  # Return top 3 fallbacks

    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics."""

        stats = {
            "total_routes": len(self.routing_history),
            "model_usage": {},
            "complexity_distribution": {},
            "daily_cost": self.daily_cost_tracker.get(time.strftime("%Y-%m-%d"), 0.0),
            "model_metrics": {}
        }

        # Calculate usage distribution
        for entry in self.routing_history:
            model = entry["model"]
            complexity = entry["metadata"]["complexity"]

            stats["model_usage"][model] = stats["model_usage"].get(model, 0) + 1
            stats["complexity_distribution"][complexity] = \
                stats["complexity_distribution"].get(complexity, 0) + 1

        # Add model metrics
        for model_id, metrics in self.model_metrics.items():
            if metrics.total_requests > 0:
                stats["model_metrics"][model_id] = {
                    "requests": metrics.total_requests,
                    "success_rate": metrics.success_rate,
                    "avg_latency": metrics.avg_latency,
                    "p95_latency": metrics.p95_latency,
                    "total_cost": metrics.total_cost,
                    "quality_score": metrics.avg_quality_score
                }

        return stats