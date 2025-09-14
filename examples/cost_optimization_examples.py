#!/usr/bin/env python3
"""
Cost Optimization Examples and Usage Patterns

This file demonstrates practical usage of the cost optimization system
with real-world examples and common scenarios.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Import our cost optimization modules
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cost_optimization.cost_tracker import CostTracker, BudgetAlert, CostCategory, AlertSeverity
from cost_optimization.model_optimizer import ModelOptimizer, TaskType
from cost_optimization.infrastructure_optimizer import InfrastructureOptimizer, WorkloadType
from cost_optimization.cost_calculator import CostCalculator, UsageProfile, ModelMix, UsagePattern
from cost_optimization.cost_dashboard import CostDashboard


class CostOptimizationExamples:
    """
    Comprehensive examples of cost optimization usage patterns.
    """

    def __init__(self):
        """Initialize optimization components."""
        self.cost_tracker = CostTracker()
        self.model_optimizer = ModelOptimizer(self.cost_tracker)
        self.infrastructure_optimizer = InfrastructureOptimizer(self.cost_tracker)
        self.cost_calculator = CostCalculator()
        self.dashboard = CostDashboard(self.cost_tracker, self.model_optimizer)

    async def example_basic_cost_tracking(self):
        """
        Example 1: Basic cost tracking for model usage.

        This shows how to track costs for different types of AI requests.
        """
        print("🔍 Example 1: Basic Cost Tracking")
        print("=" * 50)

        # Track various model usages
        usage_examples = [
            {
                "model_id": "claude-sonnet-4",
                "input_tokens": 1200,
                "output_tokens": 800,
                "project_id": "content-generation",
                "user_id": "writer@company.com",
                "task": "Blog post generation"
            },
            {
                "model_id": "gpt-4o-mini",
                "input_tokens": 500,
                "output_tokens": 200,
                "project_id": "customer-support",
                "user_id": "support@company.com",
                "task": "Email classification"
            },
            {
                "model_id": "claude-opus-4-1",
                "input_tokens": 2000,
                "output_tokens": 1500,
                "project_id": "research-analysis",
                "user_id": "researcher@company.com",
                "task": "Complex data analysis"
            }
        ]

        for example in usage_examples:
            record = await self.cost_tracker.record_model_usage(
                model_id=example["model_id"],
                input_tokens=example["input_tokens"],
                output_tokens=example["output_tokens"],
                request_duration=2.5,
                project_id=example["project_id"],
                user_id=example["user_id"],
                metadata={"task_type": example["task"]}
            )

            if record:
                print(f"✅ Tracked: {example['task']}")
                print(f"   Model: {example['model_id']}")
                print(f"   Cost: ${record.total_cost:.4f}")
                print(f"   Tokens: {example['input_tokens']} + {example['output_tokens']}")
                print()

        # Get cost summary
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(hours=1)
        summary = await self.cost_tracker.get_cost_summary(start_date, end_date)

        print(f"📊 Summary (last hour):")
        print(f"Total Cost: ${summary.total_cost:.4f}")
        print(f"Total Requests: {summary.request_count}")
        print(f"Cost per Request: ${summary.total_cost / max(summary.request_count, 1):.4f}")
        print()

    async def example_intelligent_model_routing(self):
        """
        Example 2: Intelligent model selection based on task complexity.

        Shows how the system automatically chooses optimal models.
        """
        print("🤖 Example 2: Intelligent Model Routing")
        print("=" * 50)

        # Different types of tasks with varying complexity
        tasks = [
            {
                "prompt": "What's the capital of France?",
                "task_type": TaskType.QUESTION_ANSWERING,
                "expected_complexity": "simple"
            },
            {
                "prompt": "Write a comprehensive analysis of the impact of artificial intelligence on healthcare systems, including ethical considerations and implementation challenges.",
                "task_type": TaskType.RESEARCH,
                "expected_complexity": "complex"
            },
            {
                "prompt": "def fibonacci(n):\n    # Complete this function to calculate fibonacci numbers\n    pass",
                "task_type": TaskType.CODE_GENERATION,
                "expected_complexity": "moderate"
            },
            {
                "prompt": "Summarize this meeting transcript: [10 minutes of discussion about quarterly planning...]",
                "task_type": TaskType.SUMMARIZATION,
                "expected_complexity": "simple"
            }
        ]

        total_cost_optimized = 0.0
        total_cost_naive = 0.0
        naive_model_cost = 0.075  # Assume always using expensive model

        for i, task in enumerate(tasks, 1):
            print(f"Task {i}: {task['expected_complexity'].title()} {task['task_type'].value}")
            print(f"Prompt: {task['prompt'][:80]}...")

            # Get optimized recommendation
            recommendation = await self.model_optimizer.recommend_model(
                prompt=task["prompt"],
                task_type=task["task_type"],
                quality_priority=0.5,
                speed_priority=0.3,
                cost_priority=0.2
            )

            print(f"🎯 Recommended Model: {recommendation.model_id}")
            print(f"💰 Estimated Cost: ${recommendation.estimated_cost:.4f}")
            print(f"⚡ Estimated Latency: {recommendation.estimated_latency:.1f}s")
            print(f"📝 Rationale: {recommendation.rationale}")
            print(f"🔄 Alternatives: {', '.join(recommendation.alternatives[:2])}")

            total_cost_optimized += recommendation.estimated_cost
            total_cost_naive += naive_model_cost

            print()

        savings = total_cost_naive - total_cost_optimized
        savings_percent = (savings / total_cost_naive) * 100

        print(f"💡 Optimization Results:")
        print(f"Optimized Total Cost: ${total_cost_optimized:.4f}")
        print(f"Naive Total Cost: ${total_cost_naive:.4f}")
        print(f"Total Savings: ${savings:.4f} ({savings_percent:.1f}%)")
        print()

    async def example_infrastructure_optimization(self):
        """
        Example 3: Infrastructure resource optimization.

        Demonstrates workload distribution and scaling recommendations.
        """
        print("🏗️ Example 3: Infrastructure Optimization")
        print("=" * 50)

        # Start infrastructure monitoring
        await self.infrastructure_optimizer.start_monitoring()

        # Simulate different workload scenarios
        workloads = [
            {
                "name": "llm-inference-service",
                "type": WorkloadType.LLM_INFERENCE,
                "current_rps": 5.0,
                "target_latency": 2000
            },
            {
                "name": "vector-search-service",
                "type": WorkloadType.VECTOR_SEARCH,
                "current_rps": 50.0,
                "target_latency": 500
            },
            {
                "name": "embedding-service",
                "type": WorkloadType.EMBEDDING_GENERATION,
                "current_rps": 20.0,
                "target_latency": 1000
            },
            {
                "name": "api-gateway",
                "type": WorkloadType.API_GATEWAY,
                "current_rps": 100.0,
                "target_latency": 200
            }
        ]

        print("🎯 Workload Analysis and Recommendations:")
        print()

        for workload in workloads:
            # Get optimal node assignment
            optimal_node = self.infrastructure_optimizer.get_optimal_node_for_workload(
                workload["type"]
            )

            # Get scaling recommendations
            scaling = self.infrastructure_optimizer.calculate_scaling_requirements(
                workload["type"],
                workload["current_rps"],
                workload["target_latency"]
            )

            print(f"🔧 {workload['name']}:")
            print(f"   Optimal Node: {optimal_node}")
            print(f"   Recommended Replicas: {scaling['replicas']}")
            print(f"   Reasoning: {scaling['reason']}")
            print(f"   Resource Requirements:")
            print(f"     CPU: {scaling['resource_requirements']['cpu']:.1f} cores")
            print(f"     Memory: {scaling['resource_requirements']['memory_gb']:.1f} GB")
            if scaling['resource_requirements']['gpu_memory_gb'] > 0:
                print(f"     GPU Memory: {scaling['resource_requirements']['gpu_memory_gb']:.1f} GB")
            print()

        # Get optimization recommendations
        recommendations = self.infrastructure_optimizer.get_optimization_recommendations()

        if recommendations:
            print("💡 Infrastructure Optimization Recommendations:")
            for rec in recommendations[:3]:  # Show top 3
                print(f"🚀 {rec.title}")
                print(f"   Priority: {rec.priority.title()}")
                print(f"   Estimated Savings: ${rec.estimated_savings:.2f}/month")
                print(f"   Implementation Effort: {rec.implementation_effort.title()}")
                print(f"   Description: {rec.description}")
                print()

        await self.infrastructure_optimizer.stop_monitoring()

    async def example_budget_management(self):
        """
        Example 4: Budget management and alerting.

        Shows how to set up budget controls and monitoring.
        """
        print("💰 Example 4: Budget Management and Alerting")
        print("=" * 50)

        # Set up various budget alerts
        budget_alerts = [
            {
                "name": "Daily Development Budget",
                "threshold": 25.0,
                "period": "daily",
                "project_id": "development",
                "alert_percentage": 0.8
            },
            {
                "name": "Monthly Production Budget",
                "threshold": 500.0,
                "period": "monthly",
                "project_id": "production",
                "alert_percentage": 0.9
            },
            {
                "name": "Research Project Budget",
                "threshold": 100.0,
                "period": "weekly",
                "project_id": "research-analysis",
                "alert_percentage": 0.7
            }
        ]

        print("🔔 Setting up budget alerts:")

        for alert_config in budget_alerts:
            alert = BudgetAlert(
                name=alert_config["name"],
                threshold=alert_config["threshold"],
                period=alert_config["period"],
                project_id=alert_config.get("project_id"),
                alert_percentage=alert_config["alert_percentage"]
            )

            self.cost_tracker.add_budget_alert(alert)
            print(f"✅ {alert.name}: ${alert.threshold} ({alert.period})")

        # Simulate some spending to trigger alerts
        print(f"\n💳 Simulating usage to test alerts...")

        # High-cost usage that might trigger alerts
        await self.cost_tracker.record_model_usage(
            model_id="claude-opus-4-1",
            input_tokens=5000,
            output_tokens=3000,
            request_duration=15.0,
            project_id="research-analysis",
            metadata={"simulation": True}
        )

        # Check for triggered alerts
        await self.cost_tracker._check_budget_alerts()

        print("🔍 Alert system configured and tested!")
        print()

    async def example_cost_forecasting(self):
        """
        Example 5: Cost forecasting and scenario planning.

        Demonstrates predictive cost analysis and scenario comparison.
        """
        print("📈 Example 5: Cost Forecasting and Scenario Planning")
        print("=" * 50)

        # Compare different usage scenarios
        scenarios_to_analyze = [
            ("hobbyist", "Hobbyist Usage"),
            ("developer", "Professional Developer"),
            ("startup", "Growing Startup"),
            ("smb", "Small-Medium Business")
        ]

        print("📊 Scenario Comparison:")
        scenario_results = []

        for profile_key, profile_name in scenarios_to_analyze:
            try:
                usage_profile = UsageProfile(profile_key.upper())
                usage_pattern = self.cost_calculator.usage_profiles[usage_profile]
                model_mix = self.cost_calculator.model_mixes["balanced"]

                result = self.cost_calculator.calculate_scenario_cost(
                    scenario_name=profile_name,
                    usage_pattern=usage_pattern,
                    model_mix=model_mix
                )

                scenario_results.append(result)

                print(f"\n🎯 {profile_name}:")
                print(f"   Monthly Cost: ${result.monthly_cost:.2f}")
                print(f"   Annual Cost: ${result.annual_cost:.2f}")
                print(f"   Requests/Month: {result.usage_stats['monthly_requests']:,.0f}")
                print(f"   Cost/Request: ${result.cost_breakdown.cost_per_request:.4f}")
                print(f"   Confidence: {result.confidence_level:.1%}")

                if result.optimization_opportunities:
                    print(f"   Top Optimization: {result.optimization_opportunities[0]}")

            except ValueError:
                print(f"❌ Invalid profile: {profile_key}")

        # Generate 6-month forecast for developer profile
        print(f"\n🔮 6-Month Cost Forecast (Developer Profile):")

        usage_pattern = self.cost_calculator.usage_profiles[UsageProfile.DEVELOPER]
        model_mix = self.cost_calculator.model_mixes["balanced"]

        forecast = self.cost_calculator.generate_budget_forecast(
            usage_pattern=usage_pattern,
            model_mix=model_mix,
            months_ahead=6,
            confidence_interval=0.90
        )

        print(f"Total 6-Month Forecast: ${forecast['total_forecast']['amount']:.2f}")
        print(f"Confidence Range: ${forecast['total_forecast']['lower_bound']:.2f} - ${forecast['total_forecast']['upper_bound']:.2f}")
        print(f"Average Monthly: ${forecast['average_monthly']:.2f}")

        print(f"\nMonthly Breakdown:")
        for month_data in forecast["monthly_forecasts"]:
            print(f"  Month {month_data['month']:2d}: ${month_data['forecast']:7.2f} "
                  f"({month_data['requests']:,} requests)")

        # Compare scenarios
        comparison = self.cost_calculator.compare_scenarios(scenario_results)

        print(f"\n📈 Scenario Analysis Results:")
        print(f"Best Cost Scenario: {comparison['cost_range']['best']['scenario']} "
              f"(${comparison['cost_range']['best']['monthly_cost']:.2f})")
        print(f"Worst Cost Scenario: {comparison['cost_range']['worst']['scenario']} "
              f"(${comparison['cost_range']['worst']['monthly_cost']:.2f})")
        print(f"Potential Savings: ${comparison['cost_range']['range']:.2f} "
              f"({comparison['cost_range']['savings_potential']:.1%})")

        print()

    async def example_cache_optimization(self):
        """
        Example 6: Cache optimization strategies.

        Shows how to optimize caching for cost reduction.
        """
        print("💾 Example 6: Cache Optimization Strategies")
        print("=" * 50)

        # Simulate cache scenarios
        cache_scenarios = [
            {
                "scenario": "Low Cache Hit Rate",
                "hit_rate": 0.45,
                "monthly_requests": 10000,
                "avg_cost_per_request": 0.005
            },
            {
                "scenario": "Moderate Cache Performance",
                "hit_rate": 0.70,
                "monthly_requests": 10000,
                "avg_cost_per_request": 0.005
            },
            {
                "scenario": "High Cache Performance",
                "hit_rate": 0.90,
                "monthly_requests": 10000,
                "avg_cost_per_request": 0.005
            }
        ]

        print("📊 Cache Impact Analysis:")

        for scenario in cache_scenarios:
            hit_rate = scenario["hit_rate"]
            requests = scenario["monthly_requests"]
            cost_per_request = scenario["avg_cost_per_request"]

            # Calculate costs with and without cache
            cache_hits = requests * hit_rate
            cache_misses = requests * (1 - hit_rate)

            # Assume cache hits cost nothing, misses cost full amount
            total_cost = cache_misses * cost_per_request
            cost_without_cache = requests * cost_per_request

            savings = cost_without_cache - total_cost
            savings_percent = (savings / cost_without_cache) * 100

            print(f"\n🎯 {scenario['scenario']}:")
            print(f"   Hit Rate: {hit_rate:.1%}")
            print(f"   Cache Hits: {cache_hits:,.0f}")
            print(f"   Cache Misses: {cache_misses:,.0f}")
            print(f"   Monthly Cost: ${total_cost:.2f}")
            print(f"   Cost Savings: ${savings:.2f} ({savings_percent:.1f}%)")

        # Cache optimization recommendations
        print(f"\n💡 Cache Optimization Recommendations:")

        recommendations = [
            {
                "optimization": "Increase Cache Size",
                "current": "512MB",
                "recommended": "1GB",
                "expected_improvement": "Hit rate: 65% → 80%",
                "monthly_savings": 25.50
            },
            {
                "optimization": "Implement Semantic Caching",
                "current": "Exact match only",
                "recommended": "85% similarity threshold",
                "expected_improvement": "Hit rate: 70% → 85%",
                "monthly_savings": 35.75
            },
            {
                "optimization": "Cache Warming Strategy",
                "current": "Cold cache startup",
                "recommended": "Preload frequent queries",
                "expected_improvement": "Startup performance: 5s → 0.5s",
                "monthly_savings": 15.25
            }
        ]

        for rec in recommendations:
            print(f"\n🚀 {rec['optimization']}:")
            print(f"   Current: {rec['current']}")
            print(f"   Recommended: {rec['recommended']}")
            print(f"   Expected Improvement: {rec['expected_improvement']}")
            print(f"   Monthly Savings: ${rec['monthly_savings']:.2f}")

        print()

    async def example_roi_analysis(self):
        """
        Example 7: ROI analysis for optimization investments.

        Demonstrates cost-benefit analysis of optimization initiatives.
        """
        print("💼 Example 7: ROI Analysis for Optimization Investments")
        print("=" * 50)

        # Define optimization investments
        optimization_investments = [
            {
                "name": "Advanced Caching System",
                "upfront_cost": 5000,
                "monthly_cost": 200,
                "expected_monthly_savings": 800,
                "implementation_months": 1
            },
            {
                "name": "Local Model Deployment",
                "upfront_cost": 12000,
                "monthly_cost": 400,
                "expected_monthly_savings": 1500,
                "implementation_months": 2
            },
            {
                "name": "Intelligent Model Routing",
                "upfront_cost": 3000,
                "monthly_cost": 100,
                "expected_monthly_savings": 600,
                "implementation_months": 1
            }
        ]

        print("💰 Investment Analysis:")

        for investment in optimization_investments:
            # Calculate ROI metrics
            net_monthly_savings = investment["expected_monthly_savings"] - investment["monthly_cost"]
            annual_savings = net_monthly_savings * 12

            # Payback period
            if net_monthly_savings > 0:
                payback_months = investment["upfront_cost"] / net_monthly_savings
            else:
                payback_months = float('inf')

            # 3-year NPV (5% annual discount rate)
            npv = -investment["upfront_cost"]  # Initial investment
            for month in range(36):
                if month >= investment["implementation_months"]:
                    monthly_value = net_monthly_savings
                    discount_factor = (1.05) ** (-month/12)
                    npv += monthly_value * discount_factor

            # ROI percentage
            roi_percent = (annual_savings / investment["upfront_cost"]) * 100 if investment["upfront_cost"] > 0 else 0

            print(f"\n🎯 {investment['name']}:")
            print(f"   Upfront Investment: ${investment['upfront_cost']:,}")
            print(f"   Monthly Operating Cost: ${investment['monthly_cost']}")
            print(f"   Expected Monthly Savings: ${investment['expected_monthly_savings']}")
            print(f"   Net Monthly Benefit: ${net_monthly_savings}")
            print(f"   Payback Period: {payback_months:.1f} months")
            print(f"   Annual ROI: {roi_percent:.1f}%")
            print(f"   3-Year NPV: ${npv:,.0f}")

            # Recommendation
            if payback_months < 6:
                recommendation = "🟢 Highly Recommended - Quick payback"
            elif payback_months < 12:
                recommendation = "🟡 Recommended - Good ROI"
            elif payback_months < 24:
                recommendation = "🟠 Consider - Moderate ROI"
            else:
                recommendation = "🔴 Not Recommended - Long payback"

            print(f"   Recommendation: {recommendation}")

        print(f"\n📊 Portfolio Analysis:")
        total_investment = sum(inv["upfront_cost"] for inv in optimization_investments)
        total_monthly_savings = sum(inv["expected_monthly_savings"] - inv["monthly_cost"] for inv in optimization_investments)
        portfolio_payback = total_investment / total_monthly_savings if total_monthly_savings > 0 else float('inf')

        print(f"Total Portfolio Investment: ${total_investment:,}")
        print(f"Total Monthly Net Savings: ${total_monthly_savings:,}")
        print(f"Portfolio Payback Period: {portfolio_payback:.1f} months")
        print(f"Annual ROI (Portfolio): {(total_monthly_savings * 12 / total_investment) * 100:.1f}%")

        print()


async def main():
    """Run all cost optimization examples."""

    print("🚀 Cost Optimization System - Comprehensive Examples")
    print("=" * 60)
    print()

    examples = CostOptimizationExamples()

    # Run all examples
    example_functions = [
        examples.example_basic_cost_tracking,
        examples.example_intelligent_model_routing,
        examples.example_infrastructure_optimization,
        examples.example_budget_management,
        examples.example_cost_forecasting,
        examples.example_cache_optimization,
        examples.example_roi_analysis
    ]

    for i, example_func in enumerate(example_functions, 1):
        try:
            await example_func()
            if i < len(example_functions):
                print("🔄 " + "=" * 60)
                print()
        except Exception as e:
            print(f"❌ Error in example {i}: {e}")
            print()

    print("✅ All examples completed!")
    print()
    print("🎯 Key Takeaways:")
    print("• Intelligent model routing can save 30-60% on AI costs")
    print("• Proper caching strategies reduce API calls by 70-90%")
    print("• Infrastructure optimization improves utilization by 50%+")
    print("• Budget monitoring prevents cost overruns")
    print("• ROI on optimization investments typically 300-500%")
    print()
    print("📚 Next steps:")
    print("• Review the cost optimization guide: docs/cost_optimization_guide.md")
    print("• Use the optimization scripts: scripts/cost_optimization_suite.py")
    print("• Set up monitoring dashboard for your environment")
    print("• Implement optimizations based on your usage patterns")


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())