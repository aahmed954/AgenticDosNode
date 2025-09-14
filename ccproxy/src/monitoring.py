"""Comprehensive monitoring, logging, and metrics for the Claude Code proxy."""

import time
import json
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
import structlog
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Info,
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST
)
from fastapi import Response


@dataclass
class RequestMetrics:
    """Request metrics data."""
    timestamp: datetime
    endpoint: str
    method: str
    status_code: int
    response_time: float
    model: Optional[str]
    tokens_used: Optional[int]
    api_key_hash: Optional[str]
    error_type: Optional[str] = None


class MetricsCollector:
    """Prometheus metrics collector for the proxy."""

    def __init__(self):
        # Create custom registry
        self.registry = CollectorRegistry()

        # Request metrics
        self.request_total = Counter(
            'ccproxy_requests_total',
            'Total number of requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )

        self.request_duration = Histogram(
            'ccproxy_request_duration_seconds',
            'Request duration in seconds',
            ['method', 'endpoint', 'model'],
            registry=self.registry
        )

        self.tokens_processed = Counter(
            'ccproxy_tokens_total',
            'Total tokens processed',
            ['model', 'type'],  # type: input/output
            registry=self.registry
        )

        # Active connections and rate limiting
        self.active_requests = Gauge(
            'ccproxy_active_requests',
            'Number of active requests',
            registry=self.registry
        )

        self.rate_limit_hits = Counter(
            'ccproxy_rate_limit_hits_total',
            'Rate limit hits',
            ['reason'],
            registry=self.registry
        )

        # Claude API metrics
        self.claude_api_requests = Counter(
            'ccproxy_claude_api_requests_total',
            'Requests to Claude API',
            ['status'],
            registry=self.registry
        )

        self.claude_api_duration = Histogram(
            'ccproxy_claude_api_duration_seconds',
            'Claude API response time',
            ['model'],
            registry=self.registry
        )

        # Error tracking
        self.errors_total = Counter(
            'ccproxy_errors_total',
            'Total errors',
            ['error_type', 'endpoint'],
            registry=self.registry
        )

        # System info
        self.info = Info(
            'ccproxy_info',
            'Proxy information',
            registry=self.registry
        )

        # Set system info
        self.info.info({
            'version': '1.0.0',
            'python_version': '3.11',
            'service': 'ccproxy'
        })

    def record_request(self, metrics: RequestMetrics):
        """Record request metrics."""
        # Basic request metrics
        self.request_total.labels(
            method=metrics.method,
            endpoint=metrics.endpoint,
            status=str(metrics.status_code)
        ).inc()

        self.request_duration.labels(
            method=metrics.method,
            endpoint=metrics.endpoint,
            model=metrics.model or 'unknown'
        ).observe(metrics.response_time)

        # Token metrics
        if metrics.tokens_used:
            self.tokens_processed.labels(
                model=metrics.model or 'unknown',
                type='total'
            ).inc(metrics.tokens_used)

        # Error tracking
        if metrics.error_type:
            self.errors_total.labels(
                error_type=metrics.error_type,
                endpoint=metrics.endpoint
            ).inc()

    def record_claude_api_call(self, model: str, duration: float, success: bool):
        """Record Claude API call metrics."""
        status = 'success' if success else 'error'
        self.claude_api_requests.labels(status=status).inc()
        self.claude_api_duration.labels(model=model).observe(duration)

    def record_rate_limit_hit(self, reason: str):
        """Record rate limit hit."""
        self.rate_limit_hits.labels(reason=reason).inc()

    def set_active_requests(self, count: int):
        """Set active request count."""
        self.active_requests.set(count)

    def get_metrics(self) -> str:
        """Get Prometheus metrics in text format."""
        return generate_latest(self.registry).decode('utf-8')

    def get_content_type(self) -> str:
        """Get metrics content type."""
        return CONTENT_TYPE_LATEST


class StructuredLogger:
    """Structured logging for the proxy."""

    def __init__(self, log_level: str = "INFO", service_name: str = "ccproxy"):
        self.service_name = service_name

        # Configure structlog
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

        self.logger = structlog.get_logger(service_name)

    def log_request(
        self,
        request_id: str,
        method: str,
        endpoint: str,
        model: Optional[str] = None,
        api_key_hash: Optional[str] = None,
        **kwargs
    ):
        """Log incoming request."""
        self.logger.info(
            "Request received",
            request_id=request_id,
            method=method,
            endpoint=endpoint,
            model=model,
            api_key_hash=api_key_hash,
            **kwargs
        )

    def log_response(
        self,
        request_id: str,
        status_code: int,
        response_time: float,
        tokens_used: Optional[int] = None,
        **kwargs
    ):
        """Log response."""
        self.logger.info(
            "Request completed",
            request_id=request_id,
            status_code=status_code,
            response_time=response_time,
            tokens_used=tokens_used,
            **kwargs
        )

    def log_error(
        self,
        request_id: str,
        error_type: str,
        error_message: str,
        **kwargs
    ):
        """Log error."""
        self.logger.error(
            "Request error",
            request_id=request_id,
            error_type=error_type,
            error_message=error_message,
            **kwargs
        )

    def log_rate_limit(
        self,
        request_id: str,
        reason: str,
        api_key_hash: Optional[str] = None,
        **kwargs
    ):
        """Log rate limit hit."""
        self.logger.warning(
            "Rate limit exceeded",
            request_id=request_id,
            reason=reason,
            api_key_hash=api_key_hash,
            **kwargs
        )

    def log_security_event(
        self,
        event_type: str,
        description: str,
        severity: str = "medium",
        **kwargs
    ):
        """Log security-related events."""
        self.logger.warning(
            "Security event",
            event_type=event_type,
            description=description,
            severity=severity,
            **kwargs
        )


class HealthChecker:
    """Health check system for various components."""

    def __init__(self):
        self.checks: Dict[str, Dict] = {}
        self.start_time = time.time()

    async def register_check(
        self,
        name: str,
        check_func,
        timeout: float = 5.0,
        critical: bool = True
    ):
        """Register a health check."""
        self.checks[name] = {
            'func': check_func,
            'timeout': timeout,
            'critical': critical,
            'last_check': None,
            'last_result': None,
            'last_error': None
        }

    async def run_check(self, name: str) -> Dict[str, Any]:
        """Run a specific health check."""
        if name not in self.checks:
            return {'status': 'unknown', 'error': 'Check not found'}

        check = self.checks[name]

        try:
            # Run check with timeout
            result = await asyncio.wait_for(
                check['func'](),
                timeout=check['timeout']
            )

            check['last_check'] = time.time()
            check['last_result'] = result
            check['last_error'] = None

            return {
                'status': 'healthy' if result else 'unhealthy',
                'result': result,
                'last_check': check['last_check']
            }

        except asyncio.TimeoutError:
            error = f"Check timed out after {check['timeout']}s"
            check['last_error'] = error
            return {'status': 'timeout', 'error': error}

        except Exception as e:
            error = str(e)
            check['last_error'] = error
            return {'status': 'error', 'error': error}

    async def run_all_checks(self) -> Dict[str, Any]:
        """Run all registered health checks."""
        results = {}
        overall_status = 'healthy'

        # Run all checks concurrently
        check_tasks = [
            (name, self.run_check(name))
            for name in self.checks.keys()
        ]

        for name, task in check_tasks:
            result = await task
            results[name] = result

            # Determine overall status
            if result['status'] in ['unhealthy', 'error', 'timeout']:
                if self.checks[name]['critical']:
                    overall_status = 'unhealthy'
                elif overall_status == 'healthy':
                    overall_status = 'degraded'

        return {
            'status': overall_status,
            'uptime_seconds': int(time.time() - self.start_time),
            'checks': results,
            'timestamp': datetime.now().isoformat()
        }


class AlertManager:
    """Simple alerting system for critical issues."""

    def __init__(self):
        self.alert_history = deque(maxlen=100)
        self.alert_rules = {}
        self.cooldown_periods = {}  # Prevent alert spam

    def add_alert_rule(
        self,
        name: str,
        condition_func,
        message: str,
        severity: str = "warning",
        cooldown_seconds: int = 300
    ):
        """Add an alert rule."""
        self.alert_rules[name] = {
            'condition': condition_func,
            'message': message,
            'severity': severity,
            'cooldown': cooldown_seconds,
            'last_fired': None
        }

    async def check_alerts(self, metrics_data: Dict[str, Any]):
        """Check all alert rules and fire alerts if needed."""
        current_time = time.time()

        for name, rule in self.alert_rules.items():
            try:
                # Check if alert is in cooldown
                if (rule['last_fired'] and
                    current_time - rule['last_fired'] < rule['cooldown']):
                    continue

                # Evaluate condition
                if await rule['condition'](metrics_data):
                    await self._fire_alert(name, rule, current_time)

            except Exception as e:
                # Log alert evaluation error
                pass

    async def _fire_alert(self, name: str, rule: Dict, timestamp: float):
        """Fire an alert."""
        alert = {
            'name': name,
            'message': rule['message'],
            'severity': rule['severity'],
            'timestamp': timestamp,
            'datetime': datetime.fromtimestamp(timestamp).isoformat()
        }

        self.alert_history.append(alert)
        rule['last_fired'] = timestamp

        # Here you could integrate with external alerting systems:
        # - Send to Slack/Discord webhook
        # - Send email notification
        # - Push to PagerDuty
        # - Log to external monitoring system

    def get_active_alerts(self, max_age_seconds: int = 3600) -> List[Dict]:
        """Get recent active alerts."""
        cutoff_time = time.time() - max_age_seconds
        return [
            alert for alert in self.alert_history
            if alert['timestamp'] > cutoff_time
        ]


class ProxyMonitor:
    """Main monitoring orchestrator."""

    def __init__(self, config):
        self.config = config
        self.metrics = MetricsCollector()
        self.logger = StructuredLogger(
            log_level=config.log_level,
            service_name="ccproxy"
        )
        self.health_checker = HealthChecker()
        self.alert_manager = AlertManager()

        # Setup default health checks
        self._setup_health_checks()
        self._setup_default_alerts()

    def _setup_health_checks(self):
        """Setup default health checks."""
        # Add health checks here
        pass

    def _setup_default_alerts(self):
        """Setup default alert rules."""
        # High error rate alert
        self.alert_manager.add_alert_rule(
            name="high_error_rate",
            condition_func=lambda m: m.get('error_rate', 0) > 0.1,
            message="High error rate detected (>10%)",
            severity="warning",
            cooldown_seconds=300
        )

        # High response time alert
        self.alert_manager.add_alert_rule(
            name="high_response_time",
            condition_func=lambda m: m.get('avg_response_time', 0) > 10.0,
            message="High average response time detected (>10s)",
            severity="warning",
            cooldown_seconds=300
        )

        # Rate limit alerts
        self.alert_manager.add_alert_rule(
            name="frequent_rate_limits",
            condition_func=lambda m: m.get('rate_limit_hits', 0) > 50,
            message="Frequent rate limiting detected",
            severity="info",
            cooldown_seconds=600
        )

    async def record_request_metrics(self, metrics: RequestMetrics):
        """Record metrics for a request."""
        self.metrics.record_request(metrics)

        # Log structured data
        self.logger.log_response(
            request_id=f"req_{int(time.time() * 1000)}",
            status_code=metrics.status_code,
            response_time=metrics.response_time,
            tokens_used=metrics.tokens_used,
            model=metrics.model,
            endpoint=metrics.endpoint
        )

    async def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        return await self.health_checker.run_all_checks()

    async def get_metrics_response(self) -> Response:
        """Get Prometheus metrics response."""
        metrics_data = self.metrics.get_metrics()
        return Response(
            content=metrics_data,
            media_type=self.metrics.get_content_type()
        )

    async def get_proxy_stats(self) -> Dict[str, Any]:
        """Get detailed proxy statistics."""
        # This would aggregate various metrics
        return {
            'uptime_seconds': int(time.time() - self.health_checker.start_time),
            'active_alerts': self.alert_manager.get_active_alerts(),
            'health_status': await self.get_health_status(),
            'version': '1.0.0'
        }


# Global monitor instance
monitor = None

def get_monitor(config=None):
    """Get or create monitor instance."""
    global monitor
    if monitor is None and config is not None:
        monitor = ProxyMonitor(config)
    return monitor