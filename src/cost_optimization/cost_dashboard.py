"""
Real-time cost monitoring and alerting dashboard.

This module provides:
- Real-time cost tracking dashboards
- Budget alerts and thresholds
- Cost attribution by project/user
- Predictive cost analysis
- Interactive cost visualization
- Alert management system
"""

from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import json
from collections import defaultdict, deque
import pandas as pd
import numpy as np
from decimal import Decimal

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc

from ..config import settings
from ..utils.logging import get_logger
from .cost_tracker import CostTracker, CostSummary, CostCategory
from .model_optimizer import ModelOptimizer

logger = get_logger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class DashboardTheme(str, Enum):
    """Dashboard themes."""
    DARK = "dark"
    LIGHT = "light"
    AUTO = "auto"


@dataclass
class AlertRule:
    """Cost alert rule configuration."""
    name: str
    condition: str  # e.g., "daily_cost > 100"
    threshold: float
    severity: AlertSeverity
    enabled: bool = True
    cooldown_minutes: int = 60
    notification_channels: List[str] = field(default_factory=list)
    last_triggered: Optional[datetime] = None


@dataclass
class DashboardMetrics:
    """Real-time dashboard metrics."""
    current_hour_cost: float
    current_day_cost: float
    current_month_cost: float
    projected_month_cost: float
    top_cost_models: List[Tuple[str, float]]
    top_cost_projects: List[Tuple[str, float]]
    recent_alerts: List[Dict[str, Any]]
    cost_trend_7d: List[float]
    efficiency_score: float


class CostDashboard:
    """
    Interactive cost monitoring and alerting dashboard.

    Features:
    - Real-time cost visualization
    - Budget monitoring and alerts
    - Project and user cost attribution
    - Model usage analytics
    - Predictive cost forecasting
    - Alert management interface
    - Export capabilities
    """

    def __init__(
        self,
        cost_tracker: CostTracker,
        model_optimizer: Optional[ModelOptimizer] = None,
        theme: DashboardTheme = DashboardTheme.DARK
    ):
        """Initialize cost dashboard."""
        self.cost_tracker = cost_tracker
        self.model_optimizer = model_optimizer
        self.theme = theme

        # Alert system
        self.alert_rules: List[AlertRule] = []
        self.active_alerts: deque = deque(maxlen=1000)

        # Dashboard state
        self.app = None
        self.update_interval = 30  # seconds
        self.cache_duration = 300  # seconds

        # Cached data
        self.cached_metrics: Optional[DashboardMetrics] = None
        self.cache_timestamp: Optional[datetime] = None

        # Initialize default alert rules
        self._setup_default_alerts()

        logger.info("Cost dashboard initialized")

    def _setup_default_alerts(self):
        """Set up default alert rules."""

        self.alert_rules = [
            AlertRule(
                name="Daily Budget Exceeded",
                condition="daily_cost > daily_budget",
                threshold=settings.performance.daily_budget_limit,
                severity=AlertSeverity.WARNING,
                notification_channels=["dashboard", "log"]
            ),
            AlertRule(
                name="Hourly Spend Spike",
                condition="hourly_cost > hourly_average * 5",
                threshold=0.0,  # Calculated dynamically
                severity=AlertSeverity.ERROR,
                notification_channels=["dashboard", "log"]
            ),
            AlertRule(
                name="Model Cost Anomaly",
                condition="model_cost_change > 200%",
                threshold=2.0,  # 200% increase
                severity=AlertSeverity.WARNING,
                notification_channels=["dashboard"]
            ),
            AlertRule(
                name="High Error Rate Cost Impact",
                condition="error_rate > 0.1 AND hourly_cost > 10",
                threshold=0.1,
                severity=AlertSeverity.CRITICAL,
                notification_channels=["dashboard", "log"]
            ),
        ]

    async def get_dashboard_metrics(self, force_refresh: bool = False) -> DashboardMetrics:
        """Get current dashboard metrics with caching."""

        now = datetime.utcnow()

        # Check cache
        if (not force_refresh and
            self.cached_metrics and
            self.cache_timestamp and
            (now - self.cache_timestamp).total_seconds() < self.cache_duration):
            return self.cached_metrics

        # Calculate current costs
        current_hour_cost = await self._get_current_hour_cost()
        current_day_cost = await self.cost_tracker.get_daily_cost()

        # Calculate monthly costs
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        month_summary = await self.cost_tracker.get_cost_summary(month_start, now)
        current_month_cost = month_summary.total_cost

        # Project monthly cost
        days_in_month = (month_start + timedelta(days=32)).replace(day=1) - timedelta(days=1)
        days_elapsed = (now - month_start).days + 1
        daily_average = current_month_cost / days_elapsed if days_elapsed > 0 else 0
        projected_month_cost = daily_average * days_in_month.day

        # Get top cost contributors
        top_cost_models = sorted(
            month_summary.cost_by_model.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        top_cost_projects = sorted(
            month_summary.cost_by_project.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        # Get cost trend for last 7 days
        cost_trend_7d = await self._get_7day_cost_trend()

        # Calculate efficiency score
        efficiency_score = await self._calculate_efficiency_score()

        # Get recent alerts
        recent_alerts = list(self.active_alerts)[-10:]

        metrics = DashboardMetrics(
            current_hour_cost=current_hour_cost,
            current_day_cost=current_day_cost,
            current_month_cost=current_month_cost,
            projected_month_cost=projected_month_cost,
            top_cost_models=top_cost_models,
            top_cost_projects=top_cost_projects,
            recent_alerts=recent_alerts,
            cost_trend_7d=cost_trend_7d,
            efficiency_score=efficiency_score
        )

        # Update cache
        self.cached_metrics = metrics
        self.cache_timestamp = now

        return metrics

    async def _get_current_hour_cost(self) -> float:
        """Get cost for current hour."""
        now = datetime.utcnow()
        hour_start = now.replace(minute=0, second=0, microsecond=0)

        summary = await self.cost_tracker.get_cost_summary(hour_start, now)
        return summary.total_cost

    async def _get_7day_cost_trend(self) -> List[float]:
        """Get daily costs for last 7 days."""
        daily_costs = []
        now = datetime.utcnow()

        for i in range(7):
            date = now - timedelta(days=i)
            daily_cost = await self.cost_tracker.get_daily_cost(date)
            daily_costs.append(daily_cost)

        return list(reversed(daily_costs))  # Oldest to newest

    async def _calculate_efficiency_score(self) -> float:
        """Calculate overall cost efficiency score (0-1)."""

        # Get recent performance data
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(hours=24)

        summary = await self.cost_tracker.get_cost_summary(start_date, end_date)

        if summary.total_cost == 0 or summary.request_count == 0:
            return 0.0

        # Calculate metrics
        cost_per_request = summary.total_cost / summary.request_count
        tokens_per_dollar = sum(summary.token_usage.values()) / max(summary.total_cost, 0.001)

        # Normalize scores (based on typical good performance)
        cost_efficiency = min(1.0, 0.01 / max(cost_per_request, 0.001))  # $0.01 per request is good
        token_efficiency = min(1.0, tokens_per_dollar / 10000)  # 10k tokens per dollar is good

        # Model efficiency (prefer using cheaper models when appropriate)
        model_efficiency = 0.8  # Default, would calculate based on model choice optimality

        if self.model_optimizer:
            # Get model efficiency from optimizer
            stats = self.model_optimizer.get_optimization_stats()
            complexity_dist = stats.get("complexity_distribution", {})
            total_tasks = sum(complexity_dist.values())

            if total_tasks > 0:
                # Good efficiency = using cheap models for simple tasks
                simple_tasks = complexity_dist.get("simple", 0)
                model_efficiency = min(1.0, (simple_tasks / total_tasks) * 2)

        # Combined efficiency score
        efficiency_score = (cost_efficiency * 0.4 + token_efficiency * 0.3 + model_efficiency * 0.3)

        return efficiency_score

    def create_dash_app(self) -> dash.Dash:
        """Create and configure Dash application."""

        # Choose theme
        if self.theme == DashboardTheme.DARK:
            external_stylesheets = [dbc.themes.CYBORG]
        elif self.theme == DashboardTheme.LIGHT:
            external_stylesheets = [dbc.themes.BOOTSTRAP]
        else:
            external_stylesheets = [dbc.themes.BOOTSTRAP]

        app = dash.Dash(
            __name__,
            external_stylesheets=external_stylesheets,
            suppress_callback_exceptions=True
        )

        app.title = "AI Cost Monitoring Dashboard"

        # Define layout
        app.layout = self._create_layout()

        # Register callbacks
        self._register_callbacks(app)

        self.app = app
        return app

    def _create_layout(self):
        """Create dashboard layout."""

        return dbc.Container([
            # Header
            dbc.Row([
                dbc.Col([
                    html.H1("AI Cost Monitoring Dashboard", className="text-center mb-4"),
                    html.Hr()
                ])
            ]),

            # Key metrics cards
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Current Hour", className="card-title"),
                            html.H2(id="current-hour-cost", className="text-info"),
                            html.P("Cost this hour", className="card-text")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Today", className="card-title"),
                            html.H2(id="current-day-cost", className="text-success"),
                            html.P("Total today", className="card-text")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("This Month", className="card-title"),
                            html.H2(id="current-month-cost", className="text-warning"),
                            html.P("Month to date", className="card-text")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Projected", className="card-title"),
                            html.H2(id="projected-month-cost", className="text-danger"),
                            html.P("Month projection", className="card-text")
                        ])
                    ])
                ], width=3),
            ], className="mb-4"),

            # Charts row
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("7-Day Cost Trend", className="card-title"),
                            dcc.Graph(id="cost-trend-chart")
                        ])
                    ])
                ], width=8),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Efficiency Score", className="card-title"),
                            dcc.Graph(id="efficiency-gauge")
                        ])
                    ])
                ], width=4),
            ], className="mb-4"),

            # Model and project breakdown
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Top Cost Models", className="card-title"),
                            dcc.Graph(id="model-cost-chart")
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Top Cost Projects", className="card-title"),
                            dcc.Graph(id="project-cost-chart")
                        ])
                    ])
                ], width=6),
            ], className="mb-4"),

            # Alerts section
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Recent Alerts", className="card-title"),
                            html.Div(id="alerts-list")
                        ])
                    ])
                ], width=8),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Budget Status", className="card-title"),
                            dcc.Graph(id="budget-status-chart")
                        ])
                    ])
                ], width=4),
            ], className="mb-4"),

            # Auto-refresh interval
            dcc.Interval(
                id='interval-component',
                interval=self.update_interval * 1000,  # in milliseconds
                n_intervals=0
            ),

            # Store for sharing data between callbacks
            dcc.Store(id='dashboard-data-store'),

        ], fluid=True)

    def _register_callbacks(self, app: dash.Dash):
        """Register dashboard callbacks."""

        @app.callback(
            [
                Output('current-hour-cost', 'children'),
                Output('current-day-cost', 'children'),
                Output('current-month-cost', 'children'),
                Output('projected-month-cost', 'children'),
                Output('dashboard-data-store', 'data'),
            ],
            Input('interval-component', 'n_intervals')
        )
        def update_metrics(n):
            """Update key metrics."""
            try:
                # This would be async in a real implementation
                # For now, we'll simulate the data
                metrics = {
                    'current_hour_cost': 2.45,
                    'current_day_cost': 15.32,
                    'current_month_cost': 342.18,
                    'projected_month_cost': 1025.54,
                    'top_cost_models': [
                        ('claude-opus-4-1', 125.45),
                        ('gpt-4o', 89.32),
                        ('claude-sonnet-4', 67.89),
                    ],
                    'top_cost_projects': [
                        ('research-analysis', 156.78),
                        ('customer-support', 98.45),
                        ('content-generation', 76.23),
                    ],
                    'cost_trend_7d': [12.4, 15.6, 18.9, 14.2, 16.8, 13.5, 15.3],
                    'efficiency_score': 0.78
                }

                return (
                    f"${metrics['current_hour_cost']:.2f}",
                    f"${metrics['current_day_cost']:.2f}",
                    f"${metrics['current_month_cost']:.2f}",
                    f"${metrics['projected_month_cost']:.2f}",
                    metrics
                )
            except Exception as e:
                logger.error(f"Error updating metrics: {e}")
                return "$0.00", "$0.00", "$0.00", "$0.00", {}

        @app.callback(
            Output('cost-trend-chart', 'figure'),
            Input('dashboard-data-store', 'data')
        )
        def update_cost_trend(data):
            """Update cost trend chart."""
            if not data:
                return go.Figure()

            trend_data = data.get('cost_trend_7d', [])
            if not trend_data:
                return go.Figure()

            dates = [(datetime.now() - timedelta(days=6-i)).strftime('%m/%d') for i in range(7)]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=trend_data,
                mode='lines+markers',
                name='Daily Cost',
                line=dict(color='#17a2b8', width=3),
                marker=dict(size=8)
            ))

            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Cost ($)",
                template="plotly_dark" if self.theme == DashboardTheme.DARK else "plotly_white",
                height=300
            )

            return fig

        @app.callback(
            Output('efficiency-gauge', 'figure'),
            Input('dashboard-data-store', 'data')
        )
        def update_efficiency_gauge(data):
            """Update efficiency gauge."""
            if not data:
                return go.Figure()

            efficiency = data.get('efficiency_score', 0.0)

            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=efficiency * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Efficiency %"},
                delta={'reference': 80},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "lightgreen" if efficiency > 0.8 else "orange" if efficiency > 0.6 else "red"},
                    'steps': [
                        {'range': [0, 60], 'color': "lightgray"},
                        {'range': [60, 80], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))

            fig.update_layout(
                height=300,
                template="plotly_dark" if self.theme == DashboardTheme.DARK else "plotly_white"
            )

            return fig

        @app.callback(
            Output('model-cost-chart', 'figure'),
            Input('dashboard-data-store', 'data')
        )
        def update_model_costs(data):
            """Update model cost breakdown."""
            if not data:
                return go.Figure()

            model_data = data.get('top_cost_models', [])
            if not model_data:
                return go.Figure()

            models, costs = zip(*model_data)

            fig = go.Figure(data=[
                go.Bar(x=models, y=costs, marker_color='#ffc107')
            ])

            fig.update_layout(
                xaxis_title="Model",
                yaxis_title="Cost ($)",
                template="plotly_dark" if self.theme == DashboardTheme.DARK else "plotly_white",
                height=300
            )

            return fig

        @app.callback(
            Output('project-cost-chart', 'figure'),
            Input('dashboard-data-store', 'data')
        )
        def update_project_costs(data):
            """Update project cost breakdown."""
            if not data:
                return go.Figure()

            project_data = data.get('top_cost_projects', [])
            if not project_data:
                return go.Figure()

            projects, costs = zip(*project_data)

            fig = go.Figure(data=[
                go.Pie(labels=projects, values=costs, hole=.3)
            ])

            fig.update_layout(
                template="plotly_dark" if self.theme == DashboardTheme.DARK else "plotly_white",
                height=300
            )

            return fig

        @app.callback(
            Output('alerts-list', 'children'),
            Input('interval-component', 'n_intervals')
        )
        def update_alerts(n):
            """Update alerts list."""
            try:
                # Simulate recent alerts
                alerts = [
                    {
                        'timestamp': datetime.now() - timedelta(minutes=15),
                        'severity': 'warning',
                        'message': 'Daily budget at 85% capacity'
                    },
                    {
                        'timestamp': datetime.now() - timedelta(hours=2),
                        'severity': 'info',
                        'message': 'Model switching reduced costs by 15%'
                    },
                    {
                        'timestamp': datetime.now() - timedelta(hours=6),
                        'severity': 'error',
                        'message': 'Unusual spike in GPT-4o usage'
                    },
                ]

                alert_items = []
                for alert in alerts:
                    color = {
                        'info': 'primary',
                        'warning': 'warning',
                        'error': 'danger',
                        'critical': 'dark'
                    }.get(alert['severity'], 'secondary')

                    alert_items.append(
                        dbc.Alert([
                            html.Strong(f"[{alert['severity'].upper()}] "),
                            alert['message'],
                            html.Small(
                                f" - {alert['timestamp'].strftime('%H:%M')}",
                                className="text-muted"
                            )
                        ], color=color, dismissable=True, className="mb-2")
                    )

                return alert_items

            except Exception as e:
                logger.error(f"Error updating alerts: {e}")
                return [html.P("Error loading alerts")]

        @app.callback(
            Output('budget-status-chart', 'figure'),
            Input('dashboard-data-store', 'data')
        )
        def update_budget_status(data):
            """Update budget status chart."""
            if not data:
                return go.Figure()

            current_month = data.get('current_month_cost', 0)
            projected_month = data.get('projected_month_cost', 0)
            budget_limit = settings.performance.daily_budget_limit * 30  # Monthly budget

            fig = go.Figure()

            # Current spend
            fig.add_trace(go.Bar(
                x=['Current', 'Projected', 'Budget'],
                y=[current_month, projected_month, budget_limit],
                marker_color=['#28a745', '#ffc107', '#dc3545'],
                name='Monthly Cost'
            ))

            fig.update_layout(
                yaxis_title="Cost ($)",
                template="plotly_dark" if self.theme == DashboardTheme.DARK else "plotly_white",
                height=300,
                showlegend=False
            )

            return fig

    async def check_alert_conditions(self):
        """Check all alert conditions and trigger alerts."""

        current_metrics = await self.get_dashboard_metrics(force_refresh=True)

        for rule in self.alert_rules:
            if not rule.enabled:
                continue

            # Check cooldown
            if rule.last_triggered:
                cooldown_delta = timedelta(minutes=rule.cooldown_minutes)
                if datetime.utcnow() - rule.last_triggered < cooldown_delta:
                    continue

            # Evaluate condition (simplified evaluation)
            triggered = await self._evaluate_alert_condition(rule, current_metrics)

            if triggered:
                await self._trigger_alert(rule, current_metrics)

    async def _evaluate_alert_condition(
        self,
        rule: AlertRule,
        metrics: DashboardMetrics
    ) -> bool:
        """Evaluate if alert condition is met."""

        # Simplified condition evaluation
        # In a real implementation, you'd use a proper expression parser

        if rule.name == "Daily Budget Exceeded":
            return metrics.current_day_cost > rule.threshold

        elif rule.name == "Hourly Spend Spike":
            # Calculate hourly average from 7-day trend
            if len(metrics.cost_trend_7d) > 0:
                daily_avg = sum(metrics.cost_trend_7d) / len(metrics.cost_trend_7d)
                hourly_avg = daily_avg / 24
                return metrics.current_hour_cost > (hourly_avg * 5)

        elif rule.name == "High Error Rate Cost Impact":
            # Would need error rate data
            return False  # Placeholder

        return False

    async def _trigger_alert(self, rule: AlertRule, metrics: DashboardMetrics):
        """Trigger an alert."""

        rule.last_triggered = datetime.utcnow()

        alert = {
            'id': f"{rule.name}_{int(rule.last_triggered.timestamp())}",
            'timestamp': rule.last_triggered,
            'rule_name': rule.name,
            'severity': rule.severity.value,
            'message': self._generate_alert_message(rule, metrics),
            'acknowledged': False
        }

        self.active_alerts.append(alert)

        # Send notifications
        for channel in rule.notification_channels:
            await self._send_alert_notification(alert, channel)

        logger.warning(
            f"Alert triggered: {rule.name}",
            alert_id=alert['id'],
            severity=rule.severity.value,
            message=alert['message']
        )

    def _generate_alert_message(self, rule: AlertRule, metrics: DashboardMetrics) -> str:
        """Generate alert message."""

        if rule.name == "Daily Budget Exceeded":
            return (f"Daily cost of ${metrics.current_day_cost:.2f} exceeds "
                   f"budget limit of ${rule.threshold:.2f}")

        elif rule.name == "Hourly Spend Spike":
            return (f"Current hour cost of ${metrics.current_hour_cost:.2f} "
                   f"is significantly higher than normal")

        return f"Alert condition met: {rule.condition}"

    async def _send_alert_notification(self, alert: Dict[str, Any], channel: str):
        """Send alert notification to specified channel."""

        if channel == "dashboard":
            # Already added to active_alerts
            pass

        elif channel == "log":
            logger.warning(
                f"COST ALERT: {alert['message']}",
                alert_id=alert['id'],
                severity=alert['severity']
            )

        elif channel == "webhook":
            # Would implement webhook notification
            pass

        elif channel == "email":
            # Would implement email notification
            pass

    def run_dashboard(
        self,
        host: str = "0.0.0.0",
        port: int = 8050,
        debug: bool = False
    ):
        """Run the dashboard server."""

        if not self.app:
            self.create_dash_app()

        logger.info(f"Starting cost dashboard on http://{host}:{port}")

        try:
            self.app.run_server(
                host=host,
                port=port,
                debug=debug,
                use_reloader=False  # Avoid issues in production
            )
        except Exception as e:
            logger.error(f"Failed to start dashboard: {e}")
            raise

    async def export_dashboard_data(
        self,
        format: str = "json",
        date_range: Tuple[datetime, datetime] = None
    ) -> str:
        """Export dashboard data for external use."""

        if date_range is None:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=30)
        else:
            start_date, end_date = date_range

        # Get comprehensive data
        summary = await self.cost_tracker.get_cost_summary(start_date, end_date)
        metrics = await self.get_dashboard_metrics(force_refresh=True)

        export_data = {
            'export_timestamp': datetime.utcnow().isoformat(),
            'date_range': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'summary': {
                'total_cost': summary.total_cost,
                'request_count': summary.request_count,
                'cost_by_category': summary.cost_by_category,
                'cost_by_model': summary.cost_by_model,
                'cost_by_project': summary.cost_by_project,
                'token_usage': summary.token_usage
            },
            'current_metrics': {
                'current_hour_cost': metrics.current_hour_cost,
                'current_day_cost': metrics.current_day_cost,
                'current_month_cost': metrics.current_month_cost,
                'projected_month_cost': metrics.projected_month_cost,
                'efficiency_score': metrics.efficiency_score
            },
            'alerts': [
                {
                    'timestamp': alert['timestamp'].isoformat() if isinstance(alert['timestamp'], datetime) else alert['timestamp'],
                    'severity': alert['severity'],
                    'message': alert['message'],
                    'rule_name': alert['rule_name']
                }
                for alert in list(self.active_alerts)[-50:]  # Last 50 alerts
            ]
        }

        if format == "json":
            return json.dumps(export_data, indent=2, default=str)
        elif format == "csv":
            # Convert to DataFrame and export as CSV
            import pandas as pd
            df = pd.json_normalize(export_data)
            return df.to_csv(index=False)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def get_dashboard_status(self) -> Dict[str, Any]:
        """Get dashboard system status."""

        return {
            'status': 'running' if self.app else 'not_started',
            'theme': self.theme.value,
            'update_interval_seconds': self.update_interval,
            'cache_duration_seconds': self.cache_duration,
            'active_alert_rules': len([r for r in self.alert_rules if r.enabled]),
            'total_alerts': len(self.active_alerts),
            'cache_status': {
                'cached': self.cached_metrics is not None,
                'last_update': self.cache_timestamp.isoformat() if self.cache_timestamp else None,
                'expires_in_seconds': (
                    self.cache_duration - (datetime.utcnow() - self.cache_timestamp).total_seconds()
                    if self.cache_timestamp else 0
                )
            }
        }