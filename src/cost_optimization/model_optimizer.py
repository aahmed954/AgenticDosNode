"""
Intelligent model selection optimization system.

This module provides sophisticated model routing based on:
- Cost/performance analysis
- Task complexity scoring
- Dynamic model selection
- Bulk processing optimization
- Performance-based routing decisions
"""

from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import json
import numpy as np
from collections import defaultdict, deque
import re
import hashlib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression

from ..config import settings, TaskComplexity
from ..utils.logging import get_logger
from .cost_tracker import CostTracker

logger = get_logger(__name__)


class TaskType(str, Enum):
    """Types of tasks for optimization."""
    CODE_GENERATION = "code_generation"
    TEXT_ANALYSIS = "text_analysis"
    QUESTION_ANSWERING = "question_answering"
    CREATIVE_WRITING = "creative_writing"
    DATA_EXTRACTION = "data_extraction"
    REASONING = "reasoning"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    CLASSIFICATION = "classification"
    RESEARCH = "research"


@dataclass
class ModelPerformanceMetrics:
    """Performance metrics for a model."""
    model_id: str
    avg_latency: float  # seconds
    success_rate: float  # 0-1
    quality_score: float  # 0-1
    cost_per_token: float
    cost_per_request: float
    context_efficiency: float  # how well it uses context
    instruction_following: float  # 0-1
    last_updated: datetime


@dataclass
class TaskComplexityFeatures:
    """Features used to determine task complexity."""
    input_length: int
    requires_reasoning: bool
    has_code_context: bool
    requires_creativity: bool
    domain_specific: bool
    multi_step: bool
    requires_accuracy: bool
    time_sensitive: bool
    complexity_score: float = 0.0


@dataclass
class ModelRecommendation:
    """Model recommendation with rationale."""
    model_id: str
    confidence: float  # 0-1
    estimated_cost: float
    estimated_latency: float
    estimated_quality: float
    rationale: str
    alternatives: List[str] = field(default_factory=list)


@dataclass
class BatchOptimization:
    """Batch processing optimization."""
    total_requests: int
    optimal_model: str
    batch_size: int
    estimated_total_cost: float
    estimated_total_time: float
    cost_savings: float
    time_savings: float


class ModelOptimizer:
    """
    Intelligent model selection and optimization system.

    Features:
    - Dynamic model routing based on task complexity
    - Cost/performance optimization matrices
    - Real-time performance learning
    - Bulk processing cost optimization
    - Quality-aware model selection
    - Budget-constrained optimization
    """

    def __init__(self, cost_tracker: Optional[CostTracker] = None):
        """Initialize model optimizer."""
        self.cost_tracker = cost_tracker

        # Performance history
        self.model_performance: Dict[str, ModelPerformanceMetrics] = {}
        self.task_history: deque = deque(maxlen=10000)

        # Task complexity analyzer
        self.complexity_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.complexity_model = None

        # Cost/performance matrices
        self.cost_performance_matrix = self._initialize_cost_performance_matrix()
        self.task_model_affinity = self._initialize_task_affinity_matrix()

        # Optimization state
        self.routing_decisions = deque(maxlen=1000)
        self.quality_feedback = defaultdict(list)

        # Load historical performance data
        self._load_performance_history()

        logger.info("Model optimizer initialized")

    def _initialize_cost_performance_matrix(self) -> Dict[str, Dict[str, float]]:
        """Initialize cost/performance matrix from config."""
        matrix = {}

        for model_id, specs in settings.model.model_specs.items():
            matrix[model_id] = {
                # Cost metrics (lower is better)
                "cost_per_1k_input": specs.get("cost_per_1k_input", 0.0),
                "cost_per_1k_output": specs.get("cost_per_1k_output", 0.0),
                "cost_efficiency": self._calculate_cost_efficiency(specs),

                # Performance metrics (higher is better)
                "context_window": specs.get("context_window", 0),
                "max_output": specs.get("max_output", 0),
                "supports_tools": 1.0 if specs.get("supports_tools") else 0.0,
                "supports_vision": 1.0 if specs.get("supports_vision") else 0.0,
                "supports_thinking": 1.0 if specs.get("supports_thinking") else 0.0,

                # Estimated quality metrics (based on model capabilities)
                "reasoning_ability": self._estimate_reasoning_ability(model_id),
                "code_quality": self._estimate_code_quality(model_id),
                "creativity": self._estimate_creativity(model_id),
                "accuracy": self._estimate_accuracy(model_id),
                "speed_score": self._estimate_speed_score(model_id),
            }

        return matrix

    def _calculate_cost_efficiency(self, specs: Dict[str, Any]) -> float:
        """Calculate cost efficiency score (higher is better)."""
        input_cost = specs.get("cost_per_1k_input", 0.001)
        output_cost = specs.get("cost_per_1k_output", 0.001)
        context_window = specs.get("context_window", 1000)

        # Efficiency = capability per dollar
        avg_cost = (input_cost + output_cost) / 2
        if avg_cost == 0:
            return 100.0  # Free local models

        return (context_window / 1000) / avg_cost

    def _estimate_reasoning_ability(self, model_id: str) -> float:
        """Estimate reasoning ability (0-1 score)."""
        if "opus" in model_id.lower():
            return 0.95
        elif "o1" in model_id.lower():
            return 0.98
        elif "sonnet-4" in model_id.lower():
            return 0.85
        elif "gpt-4o" in model_id.lower() and "mini" not in model_id:
            return 0.80
        elif "gpt-4o-mini" in model_id.lower():
            return 0.70
        elif "haiku" in model_id.lower():
            return 0.60
        elif "llama" in model_id.lower():
            return 0.65
        else:
            return 0.50

    def _estimate_code_quality(self, model_id: str) -> float:
        """Estimate code generation quality (0-1 score)."""
        if "opus" in model_id.lower():
            return 0.90
        elif "sonnet" in model_id.lower():
            return 0.85
        elif "gpt-4o" in model_id.lower():
            return 0.80
        elif "o1" in model_id.lower():
            return 0.75  # Less code-focused
        elif "gpt-4o-mini" in model_id.lower():
            return 0.65
        elif "llama" in model_id.lower():
            return 0.70
        else:
            return 0.50

    def _estimate_creativity(self, model_id: str) -> float:
        """Estimate creative writing ability (0-1 score)."""
        if "opus" in model_id.lower():
            return 0.95
        elif "gpt-4o" in model_id.lower() and "mini" not in model_id:
            return 0.85
        elif "sonnet" in model_id.lower():
            return 0.80
        elif "llama" in model_id.lower():
            return 0.75
        elif "gpt-4o-mini" in model_id.lower():
            return 0.70
        elif "haiku" in model_id.lower():
            return 0.65
        else:
            return 0.50

    def _estimate_accuracy(self, model_id: str) -> float:
        """Estimate factual accuracy (0-1 score)."""
        if "o1" in model_id.lower():
            return 0.95
        elif "opus" in model_id.lower():
            return 0.90
        elif "gpt-4o" in model_id.lower():
            return 0.85
        elif "sonnet" in model_id.lower():
            return 0.80
        elif "llama" in model_id.lower():
            return 0.75
        elif "haiku" in model_id.lower():
            return 0.70
        else:
            return 0.60

    def _estimate_speed_score(self, model_id: str) -> float:
        """Estimate relative speed (0-1 score, higher is faster)."""
        if "haiku" in model_id.lower():
            return 0.95
        elif "gpt-4o-mini" in model_id.lower():
            return 0.90
        elif "llama-3.1-8b" in model_id.lower():
            return 0.95  # Local is fast
        elif "sonnet" in model_id.lower():
            return 0.75
        elif "gpt-4o" in model_id.lower():
            return 0.70
        elif "llama-3.1-70b" in model_id.lower():
            return 0.65
        elif "opus" in model_id.lower():
            return 0.40
        elif "o1" in model_id.lower():
            return 0.30  # Slow but thoughtful
        else:
            return 0.60

    def _initialize_task_affinity_matrix(self) -> Dict[TaskType, Dict[str, float]]:
        """Initialize task-model affinity matrix."""
        affinity = {}

        models = list(settings.model.model_specs.keys())

        # Define affinities for each task type
        affinity[TaskType.CODE_GENERATION] = {
            "claude-opus-4-1": 0.95,
            "claude-sonnet-4": 0.90,
            "gpt-4o": 0.85,
            "gpt-4o-mini": 0.75,
            "llama-3.1-70b": 0.80,
            "llama-3.1-8b": 0.70,
            "claude-haiku-3": 0.65,
            "o1-preview": 0.70,
        }

        affinity[TaskType.REASONING] = {
            "o1-preview": 0.98,
            "claude-opus-4-1": 0.95,
            "claude-sonnet-4": 0.85,
            "gpt-4o": 0.80,
            "llama-3.1-70b": 0.75,
            "gpt-4o-mini": 0.70,
            "llama-3.1-8b": 0.65,
            "claude-haiku-3": 0.60,
        }

        affinity[TaskType.CREATIVE_WRITING] = {
            "claude-opus-4-1": 0.95,
            "gpt-4o": 0.90,
            "claude-sonnet-4": 0.85,
            "llama-3.1-70b": 0.80,
            "gpt-4o-mini": 0.75,
            "llama-3.1-8b": 0.70,
            "claude-haiku-3": 0.65,
            "o1-preview": 0.60,
        }

        affinity[TaskType.QUESTION_ANSWERING] = {
            "claude-sonnet-4": 0.90,
            "gpt-4o": 0.85,
            "claude-opus-4-1": 0.88,
            "gpt-4o-mini": 0.80,
            "llama-3.1-70b": 0.78,
            "llama-3.1-8b": 0.75,
            "claude-haiku-3": 0.70,
            "o1-preview": 0.85,
        }

        affinity[TaskType.SUMMARIZATION] = {
            "claude-haiku-3": 0.90,
            "gpt-4o-mini": 0.88,
            "claude-sonnet-4": 0.85,
            "llama-3.1-8b": 0.85,
            "gpt-4o": 0.80,
            "llama-3.1-70b": 0.78,
            "claude-opus-4-1": 0.75,
            "o1-preview": 0.70,
        }

        affinity[TaskType.CLASSIFICATION] = {
            "gpt-4o-mini": 0.95,
            "claude-haiku-3": 0.90,
            "llama-3.1-8b": 0.88,
            "claude-sonnet-4": 0.85,
            "gpt-4o": 0.80,
            "llama-3.1-70b": 0.78,
            "claude-opus-4-1": 0.70,
            "o1-preview": 0.65,
        }

        # Fill missing combinations with default values
        for task_type in TaskType:
            if task_type not in affinity:
                affinity[task_type] = {}

            for model in models:
                if model not in affinity[task_type]:
                    affinity[task_type][model] = 0.60  # Default moderate affinity

        return affinity

    def _load_performance_history(self):
        """Load historical performance data."""
        # Initialize with baseline metrics
        for model_id in settings.model.model_specs.keys():
            self.model_performance[model_id] = ModelPerformanceMetrics(
                model_id=model_id,
                avg_latency=self._get_baseline_latency(model_id),
                success_rate=0.95,
                quality_score=0.80,
                cost_per_token=self._get_model_cost_per_token(model_id),
                cost_per_request=0.01,
                context_efficiency=0.80,
                instruction_following=0.85,
                last_updated=datetime.utcnow()
            )

    def _get_baseline_latency(self, model_id: str) -> float:
        """Get baseline latency estimate for model."""
        if "haiku" in model_id.lower():
            return 1.0
        elif "gpt-4o-mini" in model_id.lower():
            return 1.5
        elif "llama-3.1-8b" in model_id.lower():
            return 0.8  # Local model
        elif "sonnet" in model_id.lower():
            return 2.5
        elif "gpt-4o" in model_id.lower():
            return 3.0
        elif "llama-3.1-70b" in model_id.lower():
            return 4.0
        elif "opus" in model_id.lower():
            return 8.0
        elif "o1" in model_id.lower():
            return 15.0
        else:
            return 3.0

    def _get_model_cost_per_token(self, model_id: str) -> float:
        """Get average cost per token for model."""
        if model_id not in settings.model.model_specs:
            return 0.001

        specs = settings.model.model_specs[model_id]
        input_cost = specs.get("cost_per_1k_input", 0.0)
        output_cost = specs.get("cost_per_1k_output", 0.0)

        # Average of input and output costs
        return (input_cost + output_cost) / 2000  # Per token

    def analyze_task_complexity(
        self,
        prompt: str,
        context: Optional[str] = None,
        task_type: Optional[TaskType] = None
    ) -> TaskComplexityFeatures:
        """Analyze task complexity from prompt and context."""

        full_text = prompt
        if context:
            full_text += " " + context

        features = TaskComplexityFeatures(
            input_length=len(full_text),
            requires_reasoning=self._detect_reasoning_required(full_text),
            has_code_context=self._detect_code_context(full_text),
            requires_creativity=self._detect_creativity_required(full_text),
            domain_specific=self._detect_domain_specific(full_text),
            multi_step=self._detect_multi_step(full_text),
            requires_accuracy=self._detect_accuracy_required(full_text),
            time_sensitive=False  # Would need additional context
        )

        # Calculate complexity score
        features.complexity_score = self._calculate_complexity_score(features)

        return features

    def _detect_reasoning_required(self, text: str) -> bool:
        """Detect if task requires reasoning."""
        reasoning_keywords = [
            "analyze", "reasoning", "logic", "because", "therefore", "conclude",
            "infer", "deduce", "explain why", "compare", "contrast", "evaluate",
            "assess", "judge", "determine", "prove", "solve", "calculate"
        ]

        text_lower = text.lower()
        return any(keyword in text_lower for keyword in reasoning_keywords)

    def _detect_code_context(self, text: str) -> bool:
        """Detect if task involves code."""
        code_indicators = [
            "```", "code", "function", "class", "import", "def ", "return",
            "variable", "algorithm", "programming", "script", "debug",
            "compile", "execute", "syntax", "API", "database", "SQL"
        ]

        return any(indicator in text for indicator in code_indicators)

    def _detect_creativity_required(self, text: str) -> bool:
        """Detect if task requires creativity."""
        creative_keywords = [
            "creative", "write", "story", "poem", "novel", "character",
            "imagine", "brainstorm", "design", "art", "original", "unique",
            "innovative", "generate ideas", "creative writing"
        ]

        text_lower = text.lower()
        return any(keyword in text_lower for keyword in creative_keywords)

    def _detect_domain_specific(self, text: str) -> bool:
        """Detect if task is domain-specific."""
        domain_keywords = [
            "medical", "legal", "financial", "scientific", "technical",
            "academic", "research", "clinical", "pharmaceutical", "engineering",
            "mathematical", "statistical", "biochemical", "legal case"
        ]

        text_lower = text.lower()
        return any(keyword in text_lower for keyword in domain_keywords)

    def _detect_multi_step(self, text: str) -> bool:
        """Detect if task has multiple steps."""
        multi_step_indicators = [
            "first", "then", "next", "finally", "step", "stage", "phase",
            "process", "procedure", "workflow", "sequence", "order",
            "1.", "2.", "3.", "•", "-", "afterwards", "subsequently"
        ]

        return any(indicator in text.lower() for indicator in multi_step_indicators)

    def _detect_accuracy_required(self, text: str) -> bool:
        """Detect if task requires high accuracy."""
        accuracy_keywords = [
            "accurate", "precise", "exact", "correct", "facts", "data",
            "research", "citation", "source", "verify", "validate",
            "important", "critical", "essential", "must be", "ensure"
        ]

        text_lower = text.lower()
        return any(keyword in text_lower for keyword in accuracy_keywords)

    def _calculate_complexity_score(self, features: TaskComplexityFeatures) -> float:
        """Calculate overall complexity score (0-1)."""
        score = 0.0

        # Length complexity
        if features.input_length > 5000:
            score += 0.2
        elif features.input_length > 1000:
            score += 0.1

        # Feature-based complexity
        if features.requires_reasoning:
            score += 0.25
        if features.has_code_context:
            score += 0.15
        if features.requires_creativity:
            score += 0.15
        if features.domain_specific:
            score += 0.1
        if features.multi_step:
            score += 0.1
        if features.requires_accuracy:
            score += 0.15
        if features.time_sensitive:
            score += 0.05

        return min(score, 1.0)

    async def recommend_model(
        self,
        prompt: str,
        context: Optional[str] = None,
        task_type: Optional[TaskType] = None,
        budget_limit: Optional[float] = None,
        quality_priority: float = 0.5,
        speed_priority: float = 0.3,
        cost_priority: float = 0.2,
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> ModelRecommendation:
        """Recommend optimal model for task."""

        # Analyze task complexity
        complexity = self.analyze_task_complexity(prompt, context, task_type)

        # Get candidate models
        candidates = self._get_candidate_models(complexity, budget_limit)

        # Score each candidate
        model_scores = {}
        for model_id in candidates:
            score = await self._score_model_for_task(
                model_id, complexity, task_type,
                quality_priority, speed_priority, cost_priority
            )
            model_scores[model_id] = score

        # Select best model
        best_model = max(model_scores.items(), key=lambda x: x[1])
        model_id, confidence = best_model

        # Calculate estimates
        estimated_cost = self._estimate_request_cost(model_id, len(prompt))
        estimated_latency = self.model_performance[model_id].avg_latency
        estimated_quality = self._estimate_quality_for_task(model_id, task_type)

        # Generate rationale
        rationale = self._generate_rationale(
            model_id, complexity, task_type, model_scores
        )

        # Get alternatives
        alternatives = sorted(
            [m for m, s in model_scores.items() if m != model_id],
            key=lambda m: model_scores[m],
            reverse=True
        )[:3]

        recommendation = ModelRecommendation(
            model_id=model_id,
            confidence=confidence,
            estimated_cost=estimated_cost,
            estimated_latency=estimated_latency,
            estimated_quality=estimated_quality,
            rationale=rationale,
            alternatives=alternatives
        )

        # Log routing decision for learning
        self.routing_decisions.append({
            "timestamp": datetime.utcnow(),
            "prompt_hash": hashlib.md5(prompt.encode()).hexdigest()[:10],
            "complexity_score": complexity.complexity_score,
            "recommended_model": model_id,
            "confidence": confidence,
            "task_type": task_type.value if task_type else "unknown"
        })

        return recommendation

    def _get_candidate_models(
        self,
        complexity: TaskComplexityFeatures,
        budget_limit: Optional[float]
    ) -> List[str]:
        """Get candidate models based on complexity and budget."""

        candidates = list(settings.model.model_specs.keys())

        # Filter by budget if specified
        if budget_limit:
            filtered_candidates = []
            for model_id in candidates:
                estimated_cost = self._estimate_request_cost(model_id, complexity.input_length)
                if estimated_cost <= budget_limit:
                    filtered_candidates.append(model_id)
            candidates = filtered_candidates

        # Filter by complexity requirements
        if complexity.complexity_score > 0.8:
            # High complexity - use premium models
            candidates = [m for m in candidates if "opus" in m or "o1" in m or ("gpt-4o" in m and "mini" not in m)]
        elif complexity.complexity_score < 0.3:
            # Low complexity - prefer efficient models
            candidates = [m for m in candidates if "haiku" in m or "mini" in m or "llama-3.1-8b" in m]

        # Ensure at least one candidate
        if not candidates:
            candidates = ["claude-sonnet-4"]  # Fallback

        return candidates

    async def _score_model_for_task(
        self,
        model_id: str,
        complexity: TaskComplexityFeatures,
        task_type: Optional[TaskType],
        quality_priority: float,
        speed_priority: float,
        cost_priority: float
    ) -> float:
        """Score model for specific task."""

        if model_id not in self.cost_performance_matrix:
            return 0.0

        model_metrics = self.cost_performance_matrix[model_id]
        performance_metrics = self.model_performance[model_id]

        # Quality score
        quality_score = 0.0
        if complexity.requires_reasoning:
            quality_score += model_metrics["reasoning_ability"] * 0.3
        if complexity.has_code_context:
            quality_score += model_metrics["code_quality"] * 0.3
        if complexity.requires_creativity:
            quality_score += model_metrics["creativity"] * 0.2
        if complexity.requires_accuracy:
            quality_score += model_metrics["accuracy"] * 0.3

        # Add task-specific affinity
        if task_type and task_type in self.task_model_affinity:
            quality_score += self.task_model_affinity[task_type].get(model_id, 0.6) * 0.4
        else:
            quality_score += performance_metrics.quality_score * 0.4

        quality_score = min(quality_score, 1.0)

        # Speed score (normalized inverse of latency)
        max_latency = 30.0  # seconds
        speed_score = 1.0 - (performance_metrics.avg_latency / max_latency)
        speed_score = max(0.0, min(speed_score, 1.0))

        # Cost score (normalized inverse of cost)
        estimated_cost = self._estimate_request_cost(model_id, complexity.input_length)
        max_cost = 1.0  # dollars
        cost_score = 1.0 - (estimated_cost / max_cost) if max_cost > 0 else 1.0
        cost_score = max(0.0, min(cost_score, 1.0))

        # Combined score
        total_score = (
            quality_score * quality_priority +
            speed_score * speed_priority +
            cost_score * cost_priority
        )

        # Apply success rate factor
        total_score *= performance_metrics.success_rate

        return total_score

    def _estimate_request_cost(self, model_id: str, input_length: int) -> float:
        """Estimate cost for a request."""
        if model_id not in settings.model.model_specs:
            return 0.01

        specs = settings.model.model_specs[model_id]

        # Estimate token count (rough approximation)
        estimated_input_tokens = input_length // 4
        estimated_output_tokens = min(estimated_input_tokens // 2, specs.get("max_output", 1000))

        input_cost = (estimated_input_tokens / 1000) * specs.get("cost_per_1k_input", 0.0)
        output_cost = (estimated_output_tokens / 1000) * specs.get("cost_per_1k_output", 0.0)

        return input_cost + output_cost

    def _estimate_quality_for_task(self, model_id: str, task_type: Optional[TaskType]) -> float:
        """Estimate quality score for specific task type."""
        if task_type and task_type in self.task_model_affinity:
            return self.task_model_affinity[task_type].get(model_id, 0.7)

        return self.model_performance[model_id].quality_score

    def _generate_rationale(
        self,
        selected_model: str,
        complexity: TaskComplexityFeatures,
        task_type: Optional[TaskType],
        all_scores: Dict[str, float]
    ) -> str:
        """Generate human-readable rationale for model selection."""

        reasons = []

        # Complexity-based reasons
        if complexity.complexity_score > 0.7:
            reasons.append(f"{selected_model} excels at complex reasoning tasks")
        elif complexity.complexity_score < 0.3:
            reasons.append(f"{selected_model} provides cost-efficient performance for simple tasks")

        # Feature-based reasons
        if complexity.has_code_context and "sonnet" in selected_model.lower():
            reasons.append("excellent code generation capabilities")

        if complexity.requires_reasoning and ("opus" in selected_model.lower() or "o1" in selected_model.lower()):
            reasons.append("superior reasoning and analysis abilities")

        if complexity.requires_creativity and "opus" in selected_model.lower():
            reasons.append("outstanding creative writing performance")

        # Cost considerations
        if selected_model in ["claude-haiku-3", "gpt-4o-mini", "llama-3.1-8b"]:
            reasons.append("optimal cost-efficiency for this task type")

        # Performance comparison
        selected_score = all_scores[selected_model]
        alternatives = [m for m, s in all_scores.items() if m != selected_model]

        if alternatives:
            best_alternative = max(alternatives, key=lambda m: all_scores[m])
            score_diff = selected_score - all_scores[best_alternative]

            if score_diff > 0.1:
                reasons.append(f"significantly outperforms {best_alternative}")

        if not reasons:
            reasons = ["balanced performance across all criteria"]

        return f"{selected_model} selected: " + ", ".join(reasons)

    async def optimize_batch_processing(
        self,
        requests: List[Dict[str, Any]],
        total_budget: Optional[float] = None,
        time_limit: Optional[float] = None,
        quality_threshold: float = 0.7
    ) -> BatchOptimization:
        """Optimize batch processing for multiple requests."""

        if not requests:
            return BatchOptimization(0, "", 0, 0.0, 0.0, 0.0, 0.0)

        # Analyze all requests
        request_analyses = []
        for req in requests:
            complexity = self.analyze_task_complexity(
                req.get("prompt", ""),
                req.get("context"),
                req.get("task_type")
            )
            request_analyses.append({
                "request": req,
                "complexity": complexity
            })

        # Find optimal model for batch
        model_costs = {}
        model_times = {}
        model_qualities = {}

        for model_id in settings.model.model_specs.keys():
            total_cost = 0.0
            total_time = 0.0
            avg_quality = 0.0

            for analysis in request_analyses:
                cost = self._estimate_request_cost(
                    model_id,
                    analysis["complexity"].input_length
                )
                latency = self.model_performance[model_id].avg_latency
                quality = self._estimate_quality_for_task(
                    model_id,
                    analysis["request"].get("task_type")
                )

                total_cost += cost
                total_time += latency
                avg_quality += quality

            avg_quality /= len(requests)

            # Only consider models that meet quality threshold
            if avg_quality >= quality_threshold:
                model_costs[model_id] = total_cost
                model_times[model_id] = total_time
                model_qualities[model_id] = avg_quality

        if not model_costs:
            # No models meet quality threshold, use best available
            best_model = max(
                settings.model.model_specs.keys(),
                key=lambda m: self.model_performance[m].quality_score
            )
        else:
            # Apply budget and time constraints
            valid_models = list(model_costs.keys())

            if total_budget:
                valid_models = [m for m in valid_models if model_costs[m] <= total_budget]

            if time_limit:
                valid_models = [m for m in valid_models if model_times[m] <= time_limit]

            if not valid_models:
                # Relax constraints and pick cheapest/fastest
                if total_budget:
                    best_model = min(model_costs.items(), key=lambda x: x[1])[0]
                elif time_limit:
                    best_model = min(model_times.items(), key=lambda x: x[1])[0]
                else:
                    best_model = "claude-sonnet-4"  # Balanced choice
            else:
                # Pick model with best cost/quality ratio
                best_model = min(
                    valid_models,
                    key=lambda m: model_costs[m] / model_qualities[m]
                )

        # Calculate optimization results
        optimal_cost = model_costs.get(best_model, 0.0)
        optimal_time = model_times.get(best_model, 0.0)

        # Calculate savings (compared to naive individual optimization)
        naive_cost = sum(
            min(model_costs.values()) for _ in requests
        ) if model_costs else 0.0

        naive_time = sum(
            min(model_times.values()) for _ in requests
        ) if model_times else 0.0

        cost_savings = max(0.0, naive_cost - optimal_cost)
        time_savings = max(0.0, naive_time - optimal_time)

        # Determine optimal batch size
        optimal_batch_size = self._calculate_optimal_batch_size(
            best_model, len(requests), total_budget
        )

        return BatchOptimization(
            total_requests=len(requests),
            optimal_model=best_model,
            batch_size=optimal_batch_size,
            estimated_total_cost=optimal_cost,
            estimated_total_time=optimal_time,
            cost_savings=cost_savings,
            time_savings=time_savings
        )

    def _calculate_optimal_batch_size(
        self,
        model_id: str,
        total_requests: int,
        budget: Optional[float]
    ) -> int:
        """Calculate optimal batch size for processing."""

        # Consider rate limits and performance
        if "anthropic" in settings.model.model_specs[model_id].get("provider", ""):
            max_concurrent = 5
        elif "openai" in settings.model.model_specs[model_id].get("provider", ""):
            max_concurrent = 10
        elif "local" in settings.model.model_specs[model_id].get("provider", ""):
            max_concurrent = 20
        else:
            max_concurrent = 5

        # Don't exceed total requests
        optimal_size = min(max_concurrent, total_requests)

        # Consider budget constraints
        if budget:
            avg_request_cost = self._estimate_request_cost(model_id, 1000)  # Average
            max_budget_size = int(budget / avg_request_cost) if avg_request_cost > 0 else optimal_size
            optimal_size = min(optimal_size, max_budget_size)

        return max(1, optimal_size)

    async def update_performance_metrics(
        self,
        model_id: str,
        latency: float,
        success: bool,
        quality_score: float,
        cost: float
    ):
        """Update model performance metrics based on actual usage."""

        if model_id not in self.model_performance:
            return

        metrics = self.model_performance[model_id]

        # Update metrics with exponential moving average
        alpha = 0.1  # Learning rate

        metrics.avg_latency = (1 - alpha) * metrics.avg_latency + alpha * latency
        metrics.success_rate = (1 - alpha) * metrics.success_rate + alpha * (1.0 if success else 0.0)
        metrics.quality_score = (1 - alpha) * metrics.quality_score + alpha * quality_score
        metrics.cost_per_request = (1 - alpha) * metrics.cost_per_request + alpha * cost
        metrics.last_updated = datetime.utcnow()

        logger.debug(
            f"Updated metrics for {model_id}",
            model_id=model_id,
            latency=metrics.avg_latency,
            success_rate=metrics.success_rate,
            quality_score=metrics.quality_score
        )

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization system statistics."""

        recent_decisions = list(self.routing_decisions)[-100:]  # Last 100 decisions

        model_usage = defaultdict(int)
        complexity_distribution = []

        for decision in recent_decisions:
            model_usage[decision["recommended_model"]] += 1
            complexity_distribution.append(decision["complexity_score"])

        avg_complexity = np.mean(complexity_distribution) if complexity_distribution else 0.0

        return {
            "total_routing_decisions": len(self.routing_decisions),
            "tracked_models": len(self.model_performance),
            "recent_model_usage": dict(model_usage),
            "average_task_complexity": avg_complexity,
            "complexity_distribution": {
                "simple": sum(1 for c in complexity_distribution if c < 0.3),
                "moderate": sum(1 for c in complexity_distribution if 0.3 <= c < 0.7),
                "complex": sum(1 for c in complexity_distribution if c >= 0.7),
            },
            "performance_metrics": {
                model_id: {
                    "avg_latency": metrics.avg_latency,
                    "success_rate": metrics.success_rate,
                    "quality_score": metrics.quality_score,
                }
                for model_id, metrics in self.model_performance.items()
            }
        }