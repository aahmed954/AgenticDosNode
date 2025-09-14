"""
Comprehensive cost modeling and scenario calculators.

This module provides:
- Cost calculators for different usage scenarios
- Usage pattern analysis and forecasting
- Proxy cost modeling and optimization
- ROI analysis for different AI configurations
- Budget planning and scenario modeling
"""

from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import math
import numpy as np
from decimal import Decimal, ROUND_HALF_UP

from ..config import settings
from ..utils.logging import get_logger

logger = get_logger(__name__)


class UsageProfile(str, Enum):
    """Standard usage profiles."""
    HOBBYIST = "hobbyist"          # Light personal use
    DEVELOPER = "developer"        # Professional development
    STARTUP = "startup"           # Small team/startup
    SMB = "smb"                   # Small-medium business
    ENTERPRISE = "enterprise"     # Large organization
    RESEARCH = "research"         # Academic/research institution


class ProxyStrategy(str, Enum):
    """Different proxy deployment strategies."""
    DIRECT_API = "direct_api"              # Direct API calls
    SINGLE_PROXY = "single_proxy"          # One proxy instance
    LOAD_BALANCED = "load_balanced"        # Multiple proxy instances
    REGIONAL_PROXY = "regional_proxy"      # Regional proxy deployment
    HYBRID = "hybrid"                      # Mix of proxy and direct


@dataclass
class UsagePattern:
    """Usage pattern specification."""
    requests_per_hour: float
    avg_input_tokens: int
    avg_output_tokens: int
    peak_multiplier: float = 2.0
    hours_per_day: int = 8
    days_per_month: int = 22
    seasonal_variation: float = 1.0
    growth_rate_monthly: float = 0.0  # As decimal (0.1 = 10% growth)


@dataclass
class ModelMix:
    """Model usage distribution."""
    model_percentages: Dict[str, float]  # Model ID -> percentage (0-1)

    def __post_init__(self):
        """Validate percentages sum to 1."""
        total = sum(self.model_percentages.values())
        if abs(total - 1.0) > 0.01:
            logger.warning(f"Model percentages sum to {total}, not 1.0")


@dataclass
class InfrastructureCosts:
    """Infrastructure cost breakdown."""
    gpu_node_hourly: float = 0.50
    cpu_node_hourly: float = 0.10
    storage_gb_monthly: float = 0.10
    network_gb: float = 0.09
    monitoring_monthly: float = 25.0
    backup_monthly: float = 15.0
    security_monthly: float = 30.0


@dataclass
class CostBreakdown:
    """Detailed cost breakdown."""
    model_inference: float
    infrastructure: float
    storage: float
    network: float
    monitoring: float
    total: float
    cost_per_request: float
    cost_per_token: float


@dataclass
class ScenarioResult:
    """Cost scenario calculation result."""
    scenario_name: str
    monthly_cost: float
    annual_cost: float
    cost_breakdown: CostBreakdown
    usage_stats: Dict[str, Any]
    optimization_opportunities: List[str]
    confidence_level: float  # 0-1, how accurate the estimate is


class CostCalculator:
    """
    Comprehensive cost calculator for AI infrastructure.

    Features:
    - Multi-scenario cost modeling
    - Usage pattern analysis
    - Proxy strategy optimization
    - Infrastructure cost modeling
    - Growth and scaling projections
    - ROI analysis
    """

    def __init__(self):
        """Initialize cost calculator."""

        # Load current model pricing
        self.model_pricing = self._load_model_pricing()
        self.infrastructure_costs = InfrastructureCosts()

        # Predefined usage profiles
        self.usage_profiles = self._initialize_usage_profiles()

        # Predefined model mixes
        self.model_mixes = self._initialize_model_mixes()

        logger.info("Cost calculator initialized")

    def _load_model_pricing(self) -> Dict[str, Dict[str, float]]:
        """Load current model pricing."""
        pricing = {}

        for model_id, specs in settings.model.model_specs.items():
            pricing[model_id] = {
                "input_cost_per_1k": specs.get("cost_per_1k_input", 0.0),
                "output_cost_per_1k": specs.get("cost_per_1k_output", 0.0),
                "provider": specs.get("provider", "unknown")
            }

        return pricing

    def _initialize_usage_profiles(self) -> Dict[UsageProfile, UsagePattern]:
        """Initialize standard usage profiles."""

        return {
            UsageProfile.HOBBYIST: UsagePattern(
                requests_per_hour=2.0,
                avg_input_tokens=500,
                avg_output_tokens=200,
                peak_multiplier=1.5,
                hours_per_day=4,
                days_per_month=15,
                seasonal_variation=0.8,
                growth_rate_monthly=0.05
            ),

            UsageProfile.DEVELOPER: UsagePattern(
                requests_per_hour=10.0,
                avg_input_tokens=1000,
                avg_output_tokens=500,
                peak_multiplier=2.0,
                hours_per_day=8,
                days_per_month=22,
                seasonal_variation=1.0,
                growth_rate_monthly=0.10
            ),

            UsageProfile.STARTUP: UsagePattern(
                requests_per_hour=25.0,
                avg_input_tokens=800,
                avg_output_tokens=400,
                peak_multiplier=3.0,
                hours_per_day=12,
                days_per_month=25,
                seasonal_variation=1.2,
                growth_rate_monthly=0.20
            ),

            UsageProfile.SMB: UsagePattern(
                requests_per_hour=50.0,
                avg_input_tokens=1200,
                avg_output_tokens=600,
                peak_multiplier=2.5,
                hours_per_day=10,
                days_per_month=22,
                seasonal_variation=1.1,
                growth_rate_monthly=0.15
            ),

            UsageProfile.ENTERPRISE: UsagePattern(
                requests_per_hour=200.0,
                avg_input_tokens=1500,
                avg_output_tokens=750,
                peak_multiplier=4.0,
                hours_per_day=24,
                days_per_month=30,
                seasonal_variation=1.3,
                growth_rate_monthly=0.08
            ),

            UsageProfile.RESEARCH: UsagePattern(
                requests_per_hour=15.0,
                avg_input_tokens=2000,
                avg_output_tokens=1000,
                peak_multiplier=2.0,
                hours_per_day=10,
                days_per_month=20,
                seasonal_variation=0.7,  # Lower during holidays
                growth_rate_monthly=0.02
            )
        }

    def _initialize_model_mixes(self) -> Dict[str, ModelMix]:
        """Initialize common model usage distributions."""

        return {
            "cost_optimized": ModelMix({
                "claude-haiku-3": 0.4,
                "gpt-4o-mini": 0.3,
                "llama-3.1-8b": 0.2,
                "claude-sonnet-4": 0.1
            }),

            "balanced": ModelMix({
                "claude-sonnet-4": 0.4,
                "gpt-4o": 0.25,
                "gpt-4o-mini": 0.2,
                "claude-haiku-3": 0.15
            }),

            "performance_focused": ModelMix({
                "claude-opus-4-1": 0.3,
                "o1-preview": 0.2,
                "claude-sonnet-4": 0.25,
                "gpt-4o": 0.25
            }),

            "research_heavy": ModelMix({
                "o1-preview": 0.4,
                "claude-opus-4-1": 0.35,
                "claude-sonnet-4": 0.25
            }),

            "code_focused": ModelMix({
                "claude-sonnet-4": 0.4,
                "claude-opus-4-1": 0.3,
                "gpt-4o": 0.3
            })
        }

    def calculate_scenario_cost(
        self,
        scenario_name: str,
        usage_pattern: UsagePattern,
        model_mix: ModelMix,
        proxy_strategy: ProxyStrategy = ProxyStrategy.DIRECT_API,
        months: int = 1
    ) -> ScenarioResult:
        """Calculate costs for a specific scenario."""

        # Calculate base monthly usage
        monthly_requests = (
            usage_pattern.requests_per_hour *
            usage_pattern.hours_per_day *
            usage_pattern.days_per_month
        )

        # Apply seasonal variation
        monthly_requests *= usage_pattern.seasonal_variation

        # Calculate total tokens
        total_input_tokens = monthly_requests * usage_pattern.avg_input_tokens
        total_output_tokens = monthly_requests * usage_pattern.avg_output_tokens

        # Calculate model costs
        model_cost = 0.0
        for model_id, percentage in model_mix.model_percentages.items():
            if model_id in self.model_pricing:
                pricing = self.model_pricing[model_id]

                model_input_tokens = total_input_tokens * percentage
                model_output_tokens = total_output_tokens * percentage

                input_cost = (model_input_tokens / 1000) * pricing["input_cost_per_1k"]
                output_cost = (model_output_tokens / 1000) * pricing["output_cost_per_1k"]

                model_cost += input_cost + output_cost

        # Apply proxy strategy costs
        proxy_overhead = self._calculate_proxy_overhead(proxy_strategy, monthly_requests)
        model_cost += proxy_overhead

        # Calculate infrastructure costs
        infra_cost = self._calculate_infrastructure_cost(usage_pattern, months)

        # Calculate storage costs (based on usage)
        storage_gb = max(50, monthly_requests * 0.001)  # Minimum 50GB
        storage_cost = storage_gb * self.infrastructure_costs.storage_gb_monthly

        # Calculate network costs
        network_cost = self._calculate_network_cost(total_input_tokens, total_output_tokens)

        # Calculate monitoring costs
        monitoring_cost = self.infrastructure_costs.monitoring_monthly

        # Apply growth over multiple months
        if months > 1:
            costs_by_month = []
            current_model_cost = model_cost

            for month in range(months):
                growth_factor = (1 + usage_pattern.growth_rate_monthly) ** month
                month_cost = current_model_cost * growth_factor
                costs_by_month.append(month_cost)

            model_cost = sum(costs_by_month)
            infra_cost *= months
            storage_cost *= months
            monitoring_cost *= months

        # Create cost breakdown
        total_cost = model_cost + infra_cost + storage_cost + network_cost + monitoring_cost

        breakdown = CostBreakdown(
            model_inference=model_cost,
            infrastructure=infra_cost,
            storage=storage_cost,
            network=network_cost,
            monitoring=monitoring_cost,
            total=total_cost,
            cost_per_request=total_cost / (monthly_requests * months) if monthly_requests > 0 else 0,
            cost_per_token=total_cost / ((total_input_tokens + total_output_tokens) * months) if (total_input_tokens + total_output_tokens) > 0 else 0
        )

        # Calculate usage stats
        usage_stats = {
            "monthly_requests": monthly_requests,
            "monthly_input_tokens": total_input_tokens,
            "monthly_output_tokens": total_output_tokens,
            "peak_requests_per_hour": usage_pattern.requests_per_hour * usage_pattern.peak_multiplier,
            "models_used": list(model_mix.model_percentages.keys()),
            "proxy_strategy": proxy_strategy.value
        }

        # Generate optimization opportunities
        optimization_opportunities = self._identify_optimizations(breakdown, usage_pattern, model_mix)

        # Calculate confidence level
        confidence_level = self._calculate_confidence(usage_pattern, model_mix)

        return ScenarioResult(
            scenario_name=scenario_name,
            monthly_cost=total_cost / months if months > 1 else total_cost,
            annual_cost=total_cost * (12 / months),
            cost_breakdown=breakdown,
            usage_stats=usage_stats,
            optimization_opportunities=optimization_opportunities,
            confidence_level=confidence_level
        )

    def _calculate_proxy_overhead(self, strategy: ProxyStrategy, monthly_requests: float) -> float:
        """Calculate proxy infrastructure overhead costs."""

        if strategy == ProxyStrategy.DIRECT_API:
            return 0.0

        # Estimate proxy infrastructure costs based on strategy
        base_proxy_cost = 50.0  # Monthly base cost

        overhead_multipliers = {
            ProxyStrategy.SINGLE_PROXY: 1.0,
            ProxyStrategy.LOAD_BALANCED: 2.5,
            ProxyStrategy.REGIONAL_PROXY: 3.0,
            ProxyStrategy.HYBRID: 1.5
        }

        multiplier = overhead_multipliers.get(strategy, 1.0)

        # Add request-based costs for high-volume scenarios
        if monthly_requests > 100000:
            volume_cost = (monthly_requests - 100000) * 0.00001  # $0.00001 per request over 100k
        else:
            volume_cost = 0.0

        return base_proxy_cost * multiplier + volume_cost

    def _calculate_infrastructure_cost(self, usage_pattern: UsagePattern, months: int) -> float:
        """Calculate infrastructure costs."""

        # Estimate required compute resources
        peak_rps = usage_pattern.requests_per_hour * usage_pattern.peak_multiplier / 3600

        # GPU node utilization (for model inference)
        gpu_utilization_hours = usage_pattern.hours_per_day * usage_pattern.days_per_month
        if peak_rps > 5:  # High load requires more GPU time
            gpu_utilization_hours *= 1.5

        gpu_cost = gpu_utilization_hours * self.infrastructure_costs.gpu_node_hourly

        # CPU node utilization (for supporting services)
        cpu_utilization_hours = 24 * usage_pattern.days_per_month  # Always running
        cpu_cost = cpu_utilization_hours * self.infrastructure_costs.cpu_node_hourly

        return (gpu_cost + cpu_cost) * months

    def _calculate_network_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate network transfer costs."""

        # Estimate data transfer (rough approximation)
        # Average token ~= 4 bytes, plus protocol overhead
        total_bytes = (input_tokens + output_tokens) * 4 * 1.3  # 30% overhead
        total_gb = total_bytes / (1024 ** 3)

        return total_gb * self.infrastructure_costs.network_gb

    def _identify_optimizations(
        self,
        breakdown: CostBreakdown,
        usage_pattern: UsagePattern,
        model_mix: ModelMix
    ) -> List[str]:
        """Identify cost optimization opportunities."""

        opportunities = []

        # Model optimization opportunities
        if breakdown.model_inference / breakdown.total > 0.7:  # >70% of costs
            opportunities.append(
                "High model costs - consider using cheaper models for simple tasks"
            )

        expensive_models = ["claude-opus-4-1", "o1-preview"]
        expensive_usage = sum(
            model_mix.model_percentages.get(model, 0)
            for model in expensive_models
        )

        if expensive_usage > 0.5:
            opportunities.append(
                "Heavy usage of premium models - evaluate task complexity routing"
            )

        # Infrastructure optimization opportunities
        if breakdown.infrastructure / breakdown.total > 0.3:  # >30% of costs
            opportunities.append(
                "High infrastructure costs - consider auto-scaling or resource optimization"
            )

        # Caching opportunities
        if usage_pattern.requests_per_hour > 10:
            opportunities.append(
                "High request volume - implement caching to reduce API calls"
            )

        # Batch processing opportunities
        if usage_pattern.peak_multiplier > 3.0:
            opportunities.append(
                "High peak load variation - implement request batching"
            )

        # Local deployment opportunities
        simple_models = ["gpt-4o-mini", "claude-haiku-3"]
        simple_usage = sum(
            model_mix.model_percentages.get(model, 0)
            for model in simple_models
        )

        if simple_usage > 0.4 and usage_pattern.requests_per_hour > 20:
            opportunities.append(
                "High volume of simple tasks - consider local model deployment"
            )

        return opportunities

    def _calculate_confidence(self, usage_pattern: UsagePattern, model_mix: ModelMix) -> float:
        """Calculate confidence level in cost estimate."""

        confidence = 1.0

        # Reduce confidence for high growth rates (uncertain scaling)
        if usage_pattern.growth_rate_monthly > 0.15:
            confidence *= 0.8

        # Reduce confidence for high seasonal variation
        if abs(usage_pattern.seasonal_variation - 1.0) > 0.3:
            confidence *= 0.9

        # Reduce confidence if using many different models (complex mix)
        if len(model_mix.model_percentages) > 4:
            confidence *= 0.9

        # Reduce confidence for very high or very low usage patterns
        if usage_pattern.requests_per_hour > 100 or usage_pattern.requests_per_hour < 1:
            confidence *= 0.8

        return max(0.5, confidence)  # Minimum 50% confidence

    def compare_scenarios(self, scenarios: List[ScenarioResult]) -> Dict[str, Any]:
        """Compare multiple cost scenarios."""

        if not scenarios:
            return {}

        # Find best and worst scenarios
        best_cost = min(scenarios, key=lambda s: s.monthly_cost)
        worst_cost = max(scenarios, key=lambda s: s.monthly_cost)

        # Calculate average costs
        avg_monthly_cost = sum(s.monthly_cost for s in scenarios) / len(scenarios)

        # Cost range
        cost_range = worst_cost.monthly_cost - best_cost.monthly_cost

        # Model usage analysis
        all_models = set()
        for scenario in scenarios:
            all_models.update(scenario.usage_stats.get("models_used", []))

        # Optimization opportunity frequency
        all_opportunities = []
        for scenario in scenarios:
            all_opportunities.extend(scenario.optimization_opportunities)

        opportunity_frequency = {}
        for opp in all_opportunities:
            opportunity_frequency[opp] = opportunity_frequency.get(opp, 0) + 1

        most_common_opportunities = sorted(
            opportunity_frequency.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        return {
            "scenario_count": len(scenarios),
            "cost_range": {
                "best": {
                    "scenario": best_cost.scenario_name,
                    "monthly_cost": best_cost.monthly_cost
                },
                "worst": {
                    "scenario": worst_cost.scenario_name,
                    "monthly_cost": worst_cost.monthly_cost
                },
                "range": cost_range,
                "savings_potential": cost_range / worst_cost.monthly_cost if worst_cost.monthly_cost > 0 else 0
            },
            "average_monthly_cost": avg_monthly_cost,
            "models_analyzed": list(all_models),
            "common_optimizations": most_common_opportunities,
            "confidence_range": {
                "min": min(s.confidence_level for s in scenarios),
                "max": max(s.confidence_level for s in scenarios),
                "avg": sum(s.confidence_level for s in scenarios) / len(scenarios)
            }
        }

    def calculate_roi_analysis(
        self,
        current_scenario: ScenarioResult,
        optimized_scenario: ScenarioResult,
        implementation_cost: float = 0.0,
        time_to_implement_months: int = 1
    ) -> Dict[str, Any]:
        """Calculate ROI for optimization implementation."""

        monthly_savings = current_scenario.monthly_cost - optimized_scenario.monthly_cost
        annual_savings = monthly_savings * 12

        if monthly_savings <= 0:
            return {
                "roi_valid": False,
                "reason": "No cost savings identified"
            }

        # Calculate payback period
        if monthly_savings > 0:
            payback_months = implementation_cost / monthly_savings
        else:
            payback_months = float('inf')

        # Calculate 3-year NPV (assuming 5% discount rate)
        discount_rate = 0.05
        npv_3_year = 0

        for month in range(36):
            if month < time_to_implement_months:
                # Implementation period - costs only
                monthly_value = -implementation_cost / time_to_implement_months
            else:
                # Savings period
                monthly_value = monthly_savings

            discount_factor = (1 + discount_rate / 12) ** -month
            npv_3_year += monthly_value * discount_factor

        # Calculate ROI percentage
        roi_percentage = (annual_savings / max(implementation_cost, 1)) * 100

        return {
            "roi_valid": True,
            "monthly_savings": monthly_savings,
            "annual_savings": annual_savings,
            "implementation_cost": implementation_cost,
            "payback_months": payback_months,
            "roi_percentage": roi_percentage,
            "npv_3_year": npv_3_year,
            "break_even_months": payback_months,
            "recommendation": self._get_roi_recommendation(
                roi_percentage, payback_months, npv_3_year
            )
        }

    def _get_roi_recommendation(
        self,
        roi_percentage: float,
        payback_months: float,
        npv_3_year: float
    ) -> str:
        """Get ROI-based recommendation."""

        if npv_3_year < 0:
            return "Not recommended - negative NPV"
        elif payback_months < 3:
            return "Highly recommended - quick payback"
        elif payback_months < 12:
            return "Recommended - good ROI"
        elif payback_months < 24:
            return "Consider - moderate ROI"
        else:
            return "Not recommended - long payback period"

    def generate_budget_forecast(
        self,
        usage_pattern: UsagePattern,
        model_mix: ModelMix,
        months_ahead: int = 12,
        confidence_interval: float = 0.95
    ) -> Dict[str, Any]:
        """Generate budget forecast with confidence intervals."""

        monthly_forecasts = []

        for month in range(months_ahead):
            # Apply growth
            growth_factor = (1 + usage_pattern.growth_rate_monthly) ** month

            # Apply seasonal variation (simplified sine wave)
            seasonal_factor = 1 + 0.2 * math.sin(2 * math.pi * month / 12) * (usage_pattern.seasonal_variation - 1)

            # Calculate base scenario
            adjusted_pattern = UsagePattern(
                requests_per_hour=usage_pattern.requests_per_hour * growth_factor * seasonal_factor,
                avg_input_tokens=usage_pattern.avg_input_tokens,
                avg_output_tokens=usage_pattern.avg_output_tokens,
                peak_multiplier=usage_pattern.peak_multiplier,
                hours_per_day=usage_pattern.hours_per_day,
                days_per_month=usage_pattern.days_per_month,
                seasonal_variation=1.0,  # Already applied
                growth_rate_monthly=0.0  # Already applied
            )

            base_result = self.calculate_scenario_cost(
                f"month_{month+1}",
                adjusted_pattern,
                model_mix
            )

            # Add uncertainty for confidence intervals
            uncertainty_factor = 1 + (month * 0.02)  # Increasing uncertainty over time

            lower_bound = base_result.monthly_cost * (1 - uncertainty_factor * (1 - confidence_interval/2))
            upper_bound = base_result.monthly_cost * (1 + uncertainty_factor * (1 - confidence_interval/2))

            monthly_forecasts.append({
                "month": month + 1,
                "forecast": base_result.monthly_cost,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "requests": adjusted_pattern.requests_per_hour * adjusted_pattern.hours_per_day * adjusted_pattern.days_per_month
            })

        total_forecast = sum(f["forecast"] for f in monthly_forecasts)
        total_lower = sum(f["lower_bound"] for f in monthly_forecasts)
        total_upper = sum(f["upper_bound"] for f in monthly_forecasts)

        return {
            "months_forecasted": months_ahead,
            "confidence_interval": confidence_interval,
            "monthly_forecasts": monthly_forecasts,
            "total_forecast": {
                "amount": total_forecast,
                "lower_bound": total_lower,
                "upper_bound": total_upper
            },
            "average_monthly": total_forecast / months_ahead,
            "growth_assumptions": {
                "monthly_growth_rate": usage_pattern.growth_rate_monthly,
                "seasonal_variation": usage_pattern.seasonal_variation
            }
        }

    def get_predefined_scenarios(self) -> Dict[str, ScenarioResult]:
        """Get results for all predefined usage profiles."""

        scenarios = {}

        for profile_name, pattern in self.usage_profiles.items():
            # Use balanced model mix as default
            model_mix = self.model_mixes["balanced"]

            result = self.calculate_scenario_cost(
                profile_name.value,
                pattern,
                model_mix
            )

            scenarios[profile_name.value] = result

        return scenarios

    def optimize_model_mix(
        self,
        usage_pattern: UsagePattern,
        budget_limit: Optional[float] = None,
        quality_threshold: float = 0.7
    ) -> Tuple[ModelMix, ScenarioResult]:
        """Find optimal model mix for given constraints."""

        # Try different model mixes
        best_mix = None
        best_result = None
        best_score = -1

        for mix_name, model_mix in self.model_mixes.items():
            result = self.calculate_scenario_cost(
                f"optimized_{mix_name}",
                usage_pattern,
                model_mix
            )

            # Check budget constraint
            if budget_limit and result.monthly_cost > budget_limit:
                continue

            # Calculate score (lower cost is better, but consider quality)
            # Simplified quality scoring based on model mix
            quality_score = self._estimate_mix_quality(model_mix)

            if quality_score < quality_threshold:
                continue

            # Combined score: cost efficiency + quality
            cost_score = 1000 / max(result.monthly_cost, 1)  # Inverse cost
            combined_score = cost_score * 0.7 + quality_score * 0.3

            if combined_score > best_score:
                best_score = combined_score
                best_mix = model_mix
                best_result = result

        if best_mix is None:
            # Fallback to cost-optimized mix
            best_mix = self.model_mixes["cost_optimized"]
            best_result = self.calculate_scenario_cost(
                "fallback_optimized",
                usage_pattern,
                best_mix
            )

        return best_mix, best_result

    def _estimate_mix_quality(self, model_mix: ModelMix) -> float:
        """Estimate overall quality of a model mix."""

        # Simplified quality scoring based on model capabilities
        model_quality_scores = {
            "claude-opus-4-1": 0.95,
            "o1-preview": 0.90,
            "claude-sonnet-4": 0.85,
            "gpt-4o": 0.80,
            "llama-3.1-70b": 0.75,
            "gpt-4o-mini": 0.65,
            "claude-haiku-3": 0.60,
            "llama-3.1-8b": 0.55
        }

        weighted_quality = 0.0
        for model_id, percentage in model_mix.model_percentages.items():
            quality = model_quality_scores.get(model_id, 0.5)
            weighted_quality += quality * percentage

        return weighted_quality