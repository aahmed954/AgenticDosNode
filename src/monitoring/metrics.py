"""Production monitoring and metrics collection."""

from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import asyncio
from collections import deque, defaultdict
import json
import numpy as np
from datetime import datetime, timedelta

from prometheus_client import Counter, Histogram, Gauge, Summary, CollectorRegistry, generate_latest
from opentelemetry import trace, metrics
from opentelemetry.trace import Status, StatusCode
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
import structlog

from ..config import settings
from ..utils.logging import get_logger

logger = get_logger(__name__)


class MetricType(str, Enum):
    """Types of metrics to track."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricSnapshot:
    """Snapshot of metrics at a point in time."""

    timestamp: float
    metrics: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """Alert configuration and state."""

    name: str
    condition: Callable[[Dict[str, float]], bool]
    message: str
    severity: str  # info, warning, error, critical
    cooldown_seconds: int = 300
    last_triggered: Optional[float] = None
    triggered_count: int = 0


class MetricsCollector:
    """
    Comprehensive metrics collection for the orchestration framework.

    Features:
    - Prometheus metrics export
    - OpenTelemetry tracing
    - Custom business metrics
    - Alerting system
    - Performance profiling
    """

    def __init__(self, enable_prometheus: bool = True, enable_otel: bool = True):
        self.enable_prometheus = enable_prometheus
        self.enable_otel = enable_otel

        # Initialize Prometheus metrics
        if enable_prometheus:
            self.registry = CollectorRegistry()
            self._init_prometheus_metrics()

        # Initialize OpenTelemetry
        if enable_otel:
            self._init_opentelemetry()

        # Custom metrics storage
        self.custom_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.metric_aggregates: Dict[str, Dict[str, float]] = defaultdict(dict)

        # Alerting
        self.alerts: List[Alert] = []
        self._init_default_alerts()

        # Performance tracking
        self.performance_traces: deque = deque(maxlen=100)
        self.slow_queries: deque = deque(maxlen=50)

    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics."""

        # Request metrics
        self.request_counter = Counter(
            'agent_requests_total',
            'Total number of agent requests',
            ['model', 'status'],
            registry=self.registry
        )

        self.request_duration = Histogram(
            'agent_request_duration_seconds',
            'Agent request duration in seconds',
            ['model', 'operation'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
            registry=self.registry
        )

        # Model metrics
        self.model_cost = Counter(
            'model_cost_dollars',
            'Total cost in dollars by model',
            ['model'],
            registry=self.registry
        )

        self.model_tokens = Counter(
            'model_tokens_total',
            'Total tokens processed',
            ['model', 'type'],  # type: input/output
            registry=self.registry
        )

        # Cache metrics
        self.cache_hits = Counter(
            'cache_hits_total',
            'Total cache hits',
            ['cache_level'],
            registry=self.registry
        )

        self.cache_misses = Counter(
            'cache_misses_total',
            'Total cache misses',
            ['cache_level'],
            registry=self.registry
        )

        # RAG metrics
        self.retrieval_duration = Histogram(
            'rag_retrieval_duration_seconds',
            'RAG retrieval duration',
            ['strategy'],
            registry=self.registry
        )

        self.retrieval_relevance = Summary(
            'rag_retrieval_relevance_score',
            'RAG retrieval relevance scores',
            ['strategy'],
            registry=self.registry
        )

        # System metrics
        self.active_requests = Gauge(
            'active_requests',
            'Number of active requests',
            registry=self.registry
        )

        self.error_rate = Gauge(
            'error_rate',
            'Current error rate',
            registry=self.registry
        )

        self.daily_cost = Gauge(
            'daily_cost_dollars',
            'Daily cost in dollars',
            registry=self.registry
        )

    def _init_opentelemetry(self):
        """Initialize OpenTelemetry tracing and metrics."""

        # Set up resource
        resource = Resource.create({
            "service.name": "agentic-orchestrator",
            "service.version": "0.1.0",
        })

        # Initialize tracer
        trace.set_tracer_provider(TracerProvider(resource=resource))
        self.tracer = trace.get_tracer(__name__)

        # Initialize meter
        metrics.set_meter_provider(MeterProvider(resource=resource))
        self.meter = metrics.get_meter(__name__)

        # Create OTEL metrics
        self.otel_request_counter = self.meter.create_counter(
            "agent.requests",
            description="Agent request count"
        )

        self.otel_latency_histogram = self.meter.create_histogram(
            "agent.latency",
            description="Agent request latency",
            unit="ms"
        )

    def _init_default_alerts(self):
        """Initialize default alert rules."""

        # High error rate alert
        self.alerts.append(Alert(
            name="high_error_rate",
            condition=lambda m: m.get("error_rate", 0) > 0.1,
            message="Error rate exceeds 10%",
            severity="error",
            cooldown_seconds=300
        ))

        # High latency alert
        self.alerts.append(Alert(
            name="high_latency",
            condition=lambda m: m.get("p95_latency", 0) > 10.0,
            message="P95 latency exceeds 10 seconds",
            severity="warning",
            cooldown_seconds=600
        ))

        # Budget alert
        self.alerts.append(Alert(
            name="budget_exceeded",
            condition=lambda m: m.get("daily_cost", 0) > settings.performance.daily_budget_limit * 0.8,
            message="Daily budget 80% consumed",
            severity="warning",
            cooldown_seconds=3600
        ))

        # Cache performance alert
        self.alerts.append(Alert(
            name="low_cache_hit_rate",
            condition=lambda m: m.get("cache_hit_rate", 1.0) < 0.5,
            message="Cache hit rate below 50%",
            severity="info",
            cooldown_seconds=1800
        ))

    def record_request(
        self,
        model: str,
        operation: str,
        duration: float,
        success: bool,
        cost: float = 0.0,
        tokens: Optional[Dict[str, int]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record a model request with comprehensive metrics."""

        # Prometheus metrics
        if self.enable_prometheus:
            status = "success" if success else "failure"
            self.request_counter.labels(model=model, status=status).inc()
            self.request_duration.labels(model=model, operation=operation).observe(duration)
            self.model_cost.labels(model=model).inc(cost)

            if tokens:
                self.model_tokens.labels(model=model, type="input").inc(tokens.get("input", 0))
                self.model_tokens.labels(model=model, type="output").inc(tokens.get("output", 0))

        # OpenTelemetry metrics
        if self.enable_otel:
            self.otel_request_counter.add(1, {"model": model, "status": "success" if success else "failure"})
            self.otel_latency_histogram.record(duration * 1000, {"model": model, "operation": operation})

        # Custom metrics
        self.custom_metrics["request_durations"].append(duration)
        self.custom_metrics["request_costs"].append(cost)

        # Update aggregates
        self._update_aggregates({
            "total_requests": 1,
            "total_cost": cost,
            "total_duration": duration,
            "errors": 0 if success else 1
        })

        # Track slow queries
        if duration > 5.0:
            self.slow_queries.append({
                "model": model,
                "operation": operation,
                "duration": duration,
                "timestamp": time.time(),
                "metadata": metadata
            })

        # Log structured event
        logger.info(
            "request_completed",
            model=model,
            operation=operation,
            duration=duration,
            success=success,
            cost=cost,
            tokens=tokens
        )

    def record_cache_access(
        self,
        cache_level: str,
        hit: bool,
        response_time: float
    ):
        """Record cache access metrics."""

        if self.enable_prometheus:
            if hit:
                self.cache_hits.labels(cache_level=cache_level).inc()
            else:
                self.cache_misses.labels(cache_level=cache_level).inc()

        self.custom_metrics["cache_response_times"].append(response_time)

    def record_retrieval(
        self,
        strategy: str,
        duration: float,
        relevance_scores: List[float],
        document_count: int
    ):
        """Record RAG retrieval metrics."""

        if self.enable_prometheus:
            self.retrieval_duration.labels(strategy=strategy).observe(duration)

            for score in relevance_scores:
                self.retrieval_relevance.labels(strategy=strategy).observe(score)

        self.custom_metrics["retrieval_durations"].append(duration)
        self.custom_metrics["retrieval_relevance"].extend(relevance_scores)

        logger.info(
            "retrieval_completed",
            strategy=strategy,
            duration=duration,
            document_count=document_count,
            avg_relevance=np.mean(relevance_scores) if relevance_scores else 0
        )

    def record_model_request(
        self,
        model_id: str,
        latency: float,
        success: bool,
        cost: float,
        quality_score: float
    ):
        """Record model-specific request metrics."""

        self.record_request(
            model=model_id,
            operation="inference",
            duration=latency,
            success=success,
            cost=cost,
            metadata={"quality_score": quality_score}
        )

    @trace.instrument
    async def trace_operation(
        self,
        operation_name: str,
        attributes: Optional[Dict[str, Any]] = None
    ):
        """Context manager for tracing operations."""

        if not self.enable_otel:
            class NoOpContext:
                async def __aenter__(self): return self
                async def __aexit__(self, *args): pass
            return NoOpContext()

        span = self.tracer.start_span(operation_name)

        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, str(value))

        return span

    def _update_aggregates(self, updates: Dict[str, float]):
        """Update aggregate metrics."""

        for key, value in updates.items():
            if key not in self.metric_aggregates["current"]:
                self.metric_aggregates["current"][key] = 0
            self.metric_aggregates["current"][key] += value

        # Update time-based aggregates
        current_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
        hour_key = current_hour.isoformat()

        if hour_key not in self.metric_aggregates:
            self.metric_aggregates[hour_key] = {}

        for key, value in updates.items():
            if key not in self.metric_aggregates[hour_key]:
                self.metric_aggregates[hour_key][key] = 0
            self.metric_aggregates[hour_key][key] += value

    def check_alerts(self) -> List[Dict[str, Any]]:
        """Check alert conditions and return triggered alerts."""

        current_metrics = self.get_current_metrics()
        triggered_alerts = []

        for alert in self.alerts:
            # Check cooldown
            if alert.last_triggered:
                if time.time() - alert.last_triggered < alert.cooldown_seconds:
                    continue

            # Check condition
            if alert.condition(current_metrics):
                alert.triggered_count += 1
                alert.last_triggered = time.time()

                triggered_alerts.append({
                    "name": alert.name,
                    "message": alert.message,
                    "severity": alert.severity,
                    "timestamp": time.time(),
                    "triggered_count": alert.triggered_count,
                    "metrics": current_metrics
                })

                logger.warning(
                    f"Alert triggered: {alert.name}",
                    alert=alert.name,
                    message=alert.message,
                    severity=alert.severity
                )

        return triggered_alerts

    def get_current_metrics(self) -> Dict[str, float]:
        """Get current metric values."""

        metrics = {}

        # Calculate error rate
        total_requests = self.metric_aggregates["current"].get("total_requests", 0)
        errors = self.metric_aggregates["current"].get("errors", 0)
        metrics["error_rate"] = errors / max(total_requests, 1)

        # Calculate latencies
        if self.custom_metrics["request_durations"]:
            durations = list(self.custom_metrics["request_durations"])
            metrics["avg_latency"] = np.mean(durations)
            metrics["p50_latency"] = np.percentile(durations, 50)
            metrics["p95_latency"] = np.percentile(durations, 95)
            metrics["p99_latency"] = np.percentile(durations, 99)

        # Calculate cache hit rate
        cache_hits = sum(1 for _ in self.custom_metrics.get("cache_hits", []))
        cache_total = cache_hits + sum(1 for _ in self.custom_metrics.get("cache_misses", []))
        metrics["cache_hit_rate"] = cache_hits / max(cache_total, 1)

        # Daily cost
        metrics["daily_cost"] = self.metric_aggregates["current"].get("total_cost", 0)

        # Active requests (would need to track this separately)
        metrics["active_requests"] = 0  # Placeholder

        return metrics

    def get_prometheus_metrics(self) -> bytes:
        """Export metrics in Prometheus format."""
        if self.enable_prometheus:
            return generate_latest(self.registry)
        return b""

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""

        current_metrics = self.get_current_metrics()

        # Calculate trends
        trends = self._calculate_trends()

        # Get top slow queries
        slow_queries = list(self.slow_queries)[-10:]  # Last 10 slow queries

        # Calculate cost breakdown
        cost_breakdown = self._calculate_cost_breakdown()

        return {
            "timestamp": time.time(),
            "current_metrics": current_metrics,
            "trends": trends,
            "slow_queries": slow_queries,
            "cost_breakdown": cost_breakdown,
            "alerts": [
                {
                    "name": a.name,
                    "triggered_count": a.triggered_count,
                    "last_triggered": a.last_triggered
                }
                for a in self.alerts
            ],
            "aggregates": {
                "total_requests": self.metric_aggregates["current"].get("total_requests", 0),
                "total_cost": self.metric_aggregates["current"].get("total_cost", 0),
                "total_errors": self.metric_aggregates["current"].get("errors", 0)
            }
        }

    def _calculate_trends(self) -> Dict[str, str]:
        """Calculate metric trends."""

        trends = {}

        # Compare current hour with previous hour
        current_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
        previous_hour = current_hour - timedelta(hours=1)

        current_key = current_hour.isoformat()
        previous_key = previous_hour.isoformat()

        if current_key in self.metric_aggregates and previous_key in self.metric_aggregates:
            current = self.metric_aggregates[current_key]
            previous = self.metric_aggregates[previous_key]

            # Request trend
            current_requests = current.get("total_requests", 0)
            previous_requests = previous.get("total_requests", 0)
            if previous_requests > 0:
                request_change = (current_requests - previous_requests) / previous_requests * 100
                trends["requests"] = f"{request_change:+.1f}%"

            # Cost trend
            current_cost = current.get("total_cost", 0)
            previous_cost = previous.get("total_cost", 0)
            if previous_cost > 0:
                cost_change = (current_cost - previous_cost) / previous_cost * 100
                trends["cost"] = f"{cost_change:+.1f}%"

        return trends

    def _calculate_cost_breakdown(self) -> Dict[str, float]:
        """Calculate cost breakdown by model."""

        breakdown = defaultdict(float)

        # This would aggregate from actual model usage data
        # For now, returning placeholder
        return {
            "claude-opus-4-1": 0.0,
            "claude-sonnet-4": 0.0,
            "gpt-4o": 0.0,
            "gpt-4o-mini": 0.0,
            "local": 0.0
        }

    def reset_metrics(self):
        """Reset all metrics (useful for testing)."""

        self.custom_metrics.clear()
        self.metric_aggregates.clear()
        self.performance_traces.clear()
        self.slow_queries.clear()

        for alert in self.alerts:
            alert.triggered_count = 0
            alert.last_triggered = None

        logger.info("Metrics reset completed")