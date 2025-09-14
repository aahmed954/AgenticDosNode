"""
Comprehensive cost tracking and analysis system for the agentic AI stack.

This module provides detailed cost tracking across all AI services including:
- Token usage and billing by model
- Infrastructure resource costs
- Service-level cost attribution
- Real-time cost monitoring and alerting
"""

from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import json
from collections import defaultdict, deque
import numpy as np
from decimal import Decimal, ROUND_HALF_UP

import redis
import sqlite3
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from ..config import settings
from ..utils.logging import get_logger

logger = get_logger(__name__)

Base = declarative_base()


class CostCategory(str, Enum):
    """Cost categories for attribution."""
    MODEL_INFERENCE = "model_inference"
    INFRASTRUCTURE = "infrastructure"
    STORAGE = "storage"
    NETWORK = "network"
    EMBEDDINGS = "embeddings"
    VECTOR_DB = "vector_db"
    CACHING = "caching"
    MONITORING = "monitoring"


class UsageType(str, Enum):
    """Types of usage tracked."""
    INPUT_TOKENS = "input_tokens"
    OUTPUT_TOKENS = "output_tokens"
    IMAGES_PROCESSED = "images_processed"
    AUDIO_MINUTES = "audio_minutes"
    REQUESTS = "requests"
    STORAGE_GB = "storage_gb"
    BANDWIDTH_GB = "bandwidth_gb"
    COMPUTE_HOURS = "compute_hours"


@dataclass
class CostRecord:
    """Individual cost record."""
    timestamp: datetime
    model_id: str
    category: CostCategory
    usage_type: UsageType
    quantity: float
    unit_cost: float
    total_cost: float
    project_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class CostRecordDB(Base):
    """SQLAlchemy model for cost records."""
    __tablename__ = 'cost_records'

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False)
    model_id = Column(String(50), nullable=False)
    category = Column(String(20), nullable=False)
    usage_type = Column(String(20), nullable=False)
    quantity = Column(Float, nullable=False)
    unit_cost = Column(Float, nullable=False)
    total_cost = Column(Float, nullable=False)
    project_id = Column(String(50), nullable=True)
    user_id = Column(String(50), nullable=True)
    session_id = Column(String(100), nullable=True)
    metadata = Column(JSON, nullable=True)


@dataclass
class CostSummary:
    """Cost summary for a time period."""
    period_start: datetime
    period_end: datetime
    total_cost: float
    cost_by_category: Dict[CostCategory, float]
    cost_by_model: Dict[str, float]
    cost_by_project: Dict[str, float]
    request_count: int
    token_usage: Dict[str, int]
    cost_trends: Dict[str, float]


@dataclass
class BudgetAlert:
    """Budget alert configuration."""
    name: str
    threshold: float
    period: str  # 'daily', 'weekly', 'monthly'
    category: Optional[CostCategory] = None
    project_id: Optional[str] = None
    enabled: bool = True
    alert_percentage: float = 0.8  # Alert at 80% of budget
    last_alerted: Optional[datetime] = None


class CostTracker:
    """
    Advanced cost tracking and analysis system.

    Features:
    - Real-time cost tracking across all services
    - Token-level billing accuracy
    - Project and user attribution
    - Budget monitoring and alerting
    - Cost optimization recommendations
    - Historical trend analysis
    - Export capabilities for accounting systems
    """

    def __init__(
        self,
        db_url: str = "sqlite:///cost_tracking.db",
        redis_url: Optional[str] = None
    ):
        """Initialize cost tracker."""
        self.db_url = db_url
        self.redis_client = redis.from_url(redis_url) if redis_url else None

        # Initialize database
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

        # In-memory caches for performance
        self.current_session_costs: Dict[str, List[CostRecord]] = defaultdict(list)
        self.daily_totals: Dict[str, float] = {}
        self.model_pricing: Dict[str, Dict[str, float]] = self._load_model_pricing()

        # Budget tracking
        self.budget_alerts: List[BudgetAlert] = []

        # Performance metrics
        self.cost_calculation_times: deque = deque(maxlen=1000)

        logger.info("Cost tracker initialized", db_url=db_url)

    def _load_model_pricing(self) -> Dict[str, Dict[str, float]]:
        """Load current model pricing from config."""
        pricing = {}

        for model_id, specs in settings.model.model_specs.items():
            pricing[model_id] = {
                "input_tokens_per_1k": specs.get("cost_per_1k_input", 0.0),
                "output_tokens_per_1k": specs.get("cost_per_1k_output", 0.0),
                "context_window": specs.get("context_window", 0),
                "max_output": specs.get("max_output", 0),
                "provider": specs.get("provider", "unknown")
            }

        # Add infrastructure costs (estimated per hour)
        infrastructure_costs = {
            "thanos_gpu_hour": 0.50,  # RTX 4090 equivalent
            "oracle1_cpu_hour": 0.10,  # VPS costs
            "vector_db_hour": 0.05,   # Qdrant hosting
            "redis_cache_hour": 0.02,  # Redis cache
            "storage_gb_month": 0.10,  # Storage costs
            "bandwidth_gb": 0.09,      # Data transfer
        }

        pricing["infrastructure"] = infrastructure_costs

        return pricing

    async def record_model_usage(
        self,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
        request_duration: float,
        success: bool = True,
        project_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> CostRecord:
        """Record model usage and calculate costs."""

        if model_id not in self.model_pricing:
            logger.warning(f"Unknown model pricing for {model_id}")
            return None

        pricing = self.model_pricing[model_id]

        # Calculate costs
        input_cost = (input_tokens / 1000) * pricing["input_tokens_per_1k"]
        output_cost = (output_tokens / 1000) * pricing["output_tokens_per_1k"]
        total_cost = input_cost + output_cost

        # Round to avoid floating point precision issues
        total_cost = float(Decimal(str(total_cost)).quantize(Decimal('0.000001'), rounding=ROUND_HALF_UP))

        # Create cost record
        record = CostRecord(
            timestamp=datetime.utcnow(),
            model_id=model_id,
            category=CostCategory.MODEL_INFERENCE,
            usage_type=UsageType.INPUT_TOKENS,
            quantity=input_tokens,
            unit_cost=pricing["input_tokens_per_1k"] / 1000,
            total_cost=input_cost,
            project_id=project_id,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata or {}
        )

        # Store record
        await self._store_cost_record(record)

        # Output tokens record
        if output_tokens > 0:
            output_record = CostRecord(
                timestamp=datetime.utcnow(),
                model_id=model_id,
                category=CostCategory.MODEL_INFERENCE,
                usage_type=UsageType.OUTPUT_TOKENS,
                quantity=output_tokens,
                unit_cost=pricing["output_tokens_per_1k"] / 1000,
                total_cost=output_cost,
                project_id=project_id,
                user_id=user_id,
                session_id=session_id,
                metadata=metadata or {}
            )
            await self._store_cost_record(output_record)

        # Update session tracking
        if session_id:
            self.current_session_costs[session_id].extend([record])
            if output_tokens > 0:
                self.current_session_costs[session_id].append(output_record)

        # Update daily totals
        today = datetime.utcnow().date().isoformat()
        self.daily_totals[today] = self.daily_totals.get(today, 0) + total_cost

        # Check budget alerts
        await self._check_budget_alerts()

        logger.debug(
            "Model usage recorded",
            model_id=model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_cost=total_cost,
            session_id=session_id
        )

        return record

    async def record_infrastructure_usage(
        self,
        resource_type: str,
        usage_hours: float,
        project_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> CostRecord:
        """Record infrastructure usage costs."""

        if "infrastructure" not in self.model_pricing:
            logger.warning("Infrastructure pricing not configured")
            return None

        pricing = self.model_pricing["infrastructure"]
        cost_key = f"{resource_type}_hour"

        if cost_key not in pricing:
            logger.warning(f"Unknown infrastructure cost for {resource_type}")
            return None

        unit_cost = pricing[cost_key]
        total_cost = usage_hours * unit_cost

        record = CostRecord(
            timestamp=datetime.utcnow(),
            model_id="infrastructure",
            category=CostCategory.INFRASTRUCTURE,
            usage_type=UsageType.COMPUTE_HOURS,
            quantity=usage_hours,
            unit_cost=unit_cost,
            total_cost=total_cost,
            project_id=project_id,
            metadata=metadata or {}
        )

        await self._store_cost_record(record)

        # Update daily totals
        today = datetime.utcnow().date().isoformat()
        self.daily_totals[today] = self.daily_totals.get(today, 0) + total_cost

        return record

    async def _store_cost_record(self, record: CostRecord):
        """Store cost record in database."""

        try:
            session = self.Session()

            db_record = CostRecordDB(
                timestamp=record.timestamp,
                model_id=record.model_id,
                category=record.category.value,
                usage_type=record.usage_type.value,
                quantity=record.quantity,
                unit_cost=record.unit_cost,
                total_cost=record.total_cost,
                project_id=record.project_id,
                user_id=record.user_id,
                session_id=record.session_id,
                metadata=record.metadata
            )

            session.add(db_record)
            session.commit()

            # Cache in Redis for fast access
            if self.redis_client:
                key = f"cost:record:{db_record.id}"
                self.redis_client.setex(
                    key,
                    3600,  # 1 hour TTL
                    json.dumps(record.to_dict())
                )

        except Exception as e:
            logger.error(f"Failed to store cost record: {e}")
            session.rollback()
        finally:
            session.close()

    async def get_cost_summary(
        self,
        start_date: datetime,
        end_date: datetime,
        project_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> CostSummary:
        """Get cost summary for a time period."""

        session = self.Session()

        try:
            query = session.query(CostRecordDB).filter(
                CostRecordDB.timestamp >= start_date,
                CostRecordDB.timestamp <= end_date
            )

            if project_id:
                query = query.filter(CostRecordDB.project_id == project_id)

            if user_id:
                query = query.filter(CostRecordDB.user_id == user_id)

            records = query.all()

            # Calculate summary statistics
            total_cost = sum(r.total_cost for r in records)

            cost_by_category = defaultdict(float)
            cost_by_model = defaultdict(float)
            cost_by_project = defaultdict(float)
            token_usage = defaultdict(int)

            for record in records:
                cost_by_category[CostCategory(record.category)] += record.total_cost
                cost_by_model[record.model_id] += record.total_cost

                if record.project_id:
                    cost_by_project[record.project_id] += record.total_cost

                if record.usage_type in [UsageType.INPUT_TOKENS.value, UsageType.OUTPUT_TOKENS.value]:
                    token_usage[record.usage_type] += int(record.quantity)

            # Calculate trends (compare with previous period)
            previous_start = start_date - (end_date - start_date)
            previous_summary = await self.get_cost_summary(
                previous_start, start_date, project_id, user_id
            )

            cost_trends = {}
            if previous_summary.total_cost > 0:
                cost_trends["total_change"] = (total_cost - previous_summary.total_cost) / previous_summary.total_cost

            return CostSummary(
                period_start=start_date,
                period_end=end_date,
                total_cost=total_cost,
                cost_by_category=dict(cost_by_category),
                cost_by_model=dict(cost_by_model),
                cost_by_project=dict(cost_by_project),
                request_count=len(records),
                token_usage=dict(token_usage),
                cost_trends=cost_trends
            )

        finally:
            session.close()

    async def get_session_cost(self, session_id: str) -> float:
        """Get total cost for a session."""
        if session_id in self.current_session_costs:
            return sum(r.total_cost for r in self.current_session_costs[session_id])

        # Fallback to database query
        session = self.Session()
        try:
            records = session.query(CostRecordDB).filter(
                CostRecordDB.session_id == session_id
            ).all()
            return sum(r.total_cost for r in records)
        finally:
            session.close()

    async def get_daily_cost(self, date: Optional[datetime] = None) -> float:
        """Get total cost for a specific day."""
        if date is None:
            date = datetime.utcnow()

        date_str = date.date().isoformat()

        if date_str in self.daily_totals:
            return self.daily_totals[date_str]

        # Calculate from database
        start_date = datetime.combine(date.date(), datetime.min.time())
        end_date = start_date + timedelta(days=1)

        summary = await self.get_cost_summary(start_date, end_date)
        self.daily_totals[date_str] = summary.total_cost

        return summary.total_cost

    def add_budget_alert(self, alert: BudgetAlert):
        """Add a budget alert."""
        self.budget_alerts.append(alert)
        logger.info(f"Budget alert added: {alert.name}")

    async def _check_budget_alerts(self):
        """Check if any budget alerts should be triggered."""

        for alert in self.budget_alerts:
            if not alert.enabled:
                continue

            # Skip if recently alerted
            if (alert.last_alerted and
                datetime.utcnow() - alert.last_alerted < timedelta(hours=1)):
                continue

            current_spend = await self._get_current_period_spend(alert)
            threshold_amount = alert.threshold * alert.alert_percentage

            if current_spend >= threshold_amount:
                await self._trigger_budget_alert(alert, current_spend)

    async def _get_current_period_spend(self, alert: BudgetAlert) -> float:
        """Get current spending for alert period."""
        now = datetime.utcnow()

        if alert.period == "daily":
            start_date = datetime.combine(now.date(), datetime.min.time())
        elif alert.period == "weekly":
            start_date = now - timedelta(days=now.weekday())
            start_date = datetime.combine(start_date.date(), datetime.min.time())
        elif alert.period == "monthly":
            start_date = datetime.combine(now.replace(day=1).date(), datetime.min.time())
        else:
            return 0.0

        summary = await self.get_cost_summary(
            start_date=start_date,
            end_date=now,
            project_id=alert.project_id
        )

        if alert.category:
            return summary.cost_by_category.get(alert.category, 0.0)
        else:
            return summary.total_cost

    async def _trigger_budget_alert(self, alert: BudgetAlert, current_spend: float):
        """Trigger a budget alert."""
        alert.last_alerted = datetime.utcnow()

        percentage_used = (current_spend / alert.threshold) * 100

        logger.warning(
            f"Budget alert triggered: {alert.name}",
            alert_name=alert.name,
            current_spend=current_spend,
            threshold=alert.threshold,
            percentage_used=percentage_used,
            period=alert.period
        )

        # Here you could integrate with notification services
        # (Slack, email, webhook, etc.)

    def get_cost_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Generate cost optimization recommendations."""
        recommendations = []

        # Analyze recent usage patterns
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=7)

        try:
            session = self.Session()
            records = session.query(CostRecordDB).filter(
                CostRecordDB.timestamp >= start_date,
                CostRecordDB.timestamp <= end_date
            ).all()

            # Model usage analysis
            model_costs = defaultdict(float)
            model_requests = defaultdict(int)

            for record in records:
                if record.category == CostCategory.MODEL_INFERENCE.value:
                    model_costs[record.model_id] += record.total_cost
                    model_requests[record.model_id] += 1

            # Recommend switching expensive models to cheaper alternatives
            for model_id, cost in model_costs.items():
                if cost > 10.0 and model_requests[model_id] > 100:  # High usage, high cost
                    if "opus" in model_id.lower():
                        recommendations.append({
                            "type": "model_optimization",
                            "priority": "high",
                            "title": f"Consider switching from {model_id} to Claude Sonnet",
                            "description": f"You spent ${cost:.2f} on {model_id} this week. "
                                         f"Claude Sonnet could reduce costs by ~80% for many tasks.",
                            "estimated_savings": cost * 0.8,
                            "action": f"Review tasks using {model_id} and switch suitable ones to claude-sonnet-4"
                        })

                    elif "gpt-4o" in model_id and "mini" not in model_id:
                        recommendations.append({
                            "type": "model_optimization",
                            "priority": "medium",
                            "title": f"Consider GPT-4o Mini for simpler tasks",
                            "description": f"You spent ${cost:.2f} on {model_id} this week. "
                                         f"GPT-4o Mini could reduce costs by ~94% for simpler tasks.",
                            "estimated_savings": cost * 0.5,  # Conservative estimate
                            "action": f"Implement task complexity routing to use gpt-4o-mini for simple tasks"
                        })

            # Cache optimization recommendations
            total_cost = sum(model_costs.values())
            if total_cost > 50.0:  # Significant spend
                recommendations.append({
                    "type": "caching_optimization",
                    "priority": "medium",
                    "title": "Implement aggressive caching",
                    "description": f"With ${total_cost:.2f} weekly model spend, "
                                 f"semantic caching could reduce costs by 20-40%.",
                    "estimated_savings": total_cost * 0.3,
                    "action": "Enable semantic caching with lower similarity thresholds"
                })

            # Local model recommendations
            local_suitable_cost = sum(
                cost for model_id, cost in model_costs.items()
                if model_requests[model_id] > 50 and cost < 5.0  # High frequency, low complexity
            )

            if local_suitable_cost > 5.0:
                recommendations.append({
                    "type": "infrastructure_optimization",
                    "priority": "low",
                    "title": "Consider local model deployment",
                    "description": f"${local_suitable_cost:.2f} of your spend could potentially "
                                 f"be served by local models for near-zero marginal cost.",
                    "estimated_savings": local_suitable_cost * 0.95,
                    "action": "Deploy llama-3.1-8b for high-frequency simple tasks"
                })

        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")

        finally:
            session.close()

        return sorted(recommendations, key=lambda x: x.get("estimated_savings", 0), reverse=True)

    async def export_cost_data(
        self,
        start_date: datetime,
        end_date: datetime,
        format: str = "csv",
        project_id: Optional[str] = None
    ) -> str:
        """Export cost data for accounting/billing systems."""

        session = self.Session()

        try:
            query = session.query(CostRecordDB).filter(
                CostRecordDB.timestamp >= start_date,
                CostRecordDB.timestamp <= end_date
            )

            if project_id:
                query = query.filter(CostRecordDB.project_id == project_id)

            records = query.all()

            if format == "csv":
                import csv
                import io

                output = io.StringIO()
                writer = csv.writer(output)

                # Write header
                writer.writerow([
                    'timestamp', 'model_id', 'category', 'usage_type',
                    'quantity', 'unit_cost', 'total_cost', 'project_id',
                    'user_id', 'session_id'
                ])

                # Write data
                for record in records:
                    writer.writerow([
                        record.timestamp.isoformat(),
                        record.model_id,
                        record.category,
                        record.usage_type,
                        record.quantity,
                        record.unit_cost,
                        record.total_cost,
                        record.project_id,
                        record.user_id,
                        record.session_id
                    ])

                return output.getvalue()

            elif format == "json":
                data = []
                for record in records:
                    data.append({
                        'timestamp': record.timestamp.isoformat(),
                        'model_id': record.model_id,
                        'category': record.category,
                        'usage_type': record.usage_type,
                        'quantity': record.quantity,
                        'unit_cost': record.unit_cost,
                        'total_cost': record.total_cost,
                        'project_id': record.project_id,
                        'user_id': record.user_id,
                        'session_id': record.session_id,
                        'metadata': record.metadata
                    })

                return json.dumps(data, indent=2)

        finally:
            session.close()

    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the cost tracker."""

        avg_calc_time = 0
        if self.cost_calculation_times:
            avg_calc_time = sum(self.cost_calculation_times) / len(self.cost_calculation_times)

        return {
            "tracked_sessions": len(self.current_session_costs),
            "daily_totals_cached": len(self.daily_totals),
            "budget_alerts": len(self.budget_alerts),
            "avg_calculation_time_ms": avg_calc_time * 1000,
            "active_models": len(self.model_pricing) - 1,  # Exclude infrastructure
        }

    async def cleanup_old_data(self, days_to_keep: int = 90):
        """Clean up old cost records to manage database size."""

        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)

        session = self.Session()
        try:
            deleted_count = session.query(CostRecordDB).filter(
                CostRecordDB.timestamp < cutoff_date
            ).delete()

            session.commit()

            logger.info(f"Cleaned up {deleted_count} old cost records")

        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
            session.rollback()
        finally:
            session.close()