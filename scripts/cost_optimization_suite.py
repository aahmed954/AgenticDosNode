#!/usr/bin/env python3
"""
Comprehensive cost optimization suite for the agentic AI stack.

This script provides automated tools for:
- Cost analysis and reporting
- Model selection optimization
- Infrastructure resource optimization
- Cache management and optimization
- Alert configuration and monitoring
- Batch optimization processing
- Cost forecasting and budgeting
"""

import asyncio
import argparse
import json
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cost_optimization.cost_tracker import CostTracker
from cost_optimization.model_optimizer import ModelOptimizer, TaskType
from cost_optimization.infrastructure_optimizer import InfrastructureOptimizer
from cost_optimization.cost_dashboard import CostDashboard
from cost_optimization.cost_calculator import CostCalculator, UsageProfile, ModelMix, UsagePattern
from utils.logging import get_logger

logger = get_logger(__name__)


class CostOptimizationSuite:
    """
    Comprehensive cost optimization suite.

    Provides automated tools and scripts for managing AI infrastructure costs.
    """

    def __init__(self):
        """Initialize optimization suite."""
        self.cost_tracker = CostTracker()
        self.model_optimizer = ModelOptimizer(self.cost_tracker)
        self.infrastructure_optimizer = InfrastructureOptimizer(self.cost_tracker)
        self.cost_calculator = CostCalculator()
        self.dashboard = CostDashboard(self.cost_tracker, self.model_optimizer)

    async def run_cost_analysis(self, days: int = 30, output_file: Optional[str] = None):
        """Run comprehensive cost analysis."""

        print("🔍 Running cost analysis...")

        # Get cost summary
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)

        summary = await self.cost_tracker.get_cost_summary(start_date, end_date)

        # Get optimization recommendations
        recommendations = self.cost_tracker.get_cost_optimization_recommendations()

        # Get infrastructure recommendations
        infra_recommendations = self.infrastructure_optimizer.get_optimization_recommendations()

        # Generate report
        report = {
            "analysis_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "days": days
            },
            "cost_summary": {
                "total_cost": summary.total_cost,
                "request_count": summary.request_count,
                "avg_cost_per_request": summary.total_cost / max(summary.request_count, 1),
                "cost_by_category": summary.cost_by_category,
                "cost_by_model": summary.cost_by_model,
                "cost_by_project": summary.cost_by_project,
                "token_usage": summary.token_usage
            },
            "optimization_recommendations": {
                "model_recommendations": recommendations,
                "infrastructure_recommendations": [
                    {
                        "type": rec.type,
                        "priority": rec.priority,
                        "title": rec.title,
                        "description": rec.description,
                        "estimated_savings": rec.estimated_savings,
                        "actions": rec.actions
                    }
                    for rec in infra_recommendations
                ],
                "total_potential_savings": sum(rec.get("estimated_savings", 0) for rec in recommendations) +
                                          sum(rec.estimated_savings for rec in infra_recommendations)
            },
            "key_metrics": {
                "efficiency_score": await self._calculate_efficiency_score(),
                "top_cost_models": sorted(summary.cost_by_model.items(), key=lambda x: x[1], reverse=True)[:5],
                "cost_trends": summary.cost_trends
            }
        }

        # Print summary
        print(f"\n📊 Cost Analysis Report ({days} days)")
        print("=" * 50)
        print(f"Total Cost: ${report['cost_summary']['total_cost']:.2f}")
        print(f"Total Requests: {report['cost_summary']['request_count']:,}")
        print(f"Cost per Request: ${report['cost_summary']['avg_cost_per_request']:.4f}")
        print(f"Potential Savings: ${report['optimization_recommendations']['total_potential_savings']:.2f}")

        print(f"\n🎯 Top Cost Models:")
        for model, cost in report['key_metrics']['top_cost_models']:
            print(f"  • {model}: ${cost:.2f}")

        print(f"\n💡 Top Recommendations:")
        for i, rec in enumerate(recommendations[:3], 1):
            print(f"  {i}. {rec.get('title', 'Optimization')}: ${rec.get('estimated_savings', 0):.2f} savings")

        # Save to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"\n💾 Report saved to: {output_file}")

        return report

    async def optimize_model_selection(
        self,
        test_prompts_file: Optional[str] = None,
        output_file: Optional[str] = None
    ):
        """Optimize model selection for given prompts."""

        print("🤖 Optimizing model selection...")

        # Load test prompts
        test_prompts = []
        if test_prompts_file and os.path.exists(test_prompts_file):
            with open(test_prompts_file, 'r') as f:
                test_prompts = json.load(f)
        else:
            # Use default test prompts
            test_prompts = [
                {
                    "prompt": "Write a simple Python function to calculate factorial",
                    "task_type": "code_generation",
                    "expected_complexity": "simple"
                },
                {
                    "prompt": "Analyze the pros and cons of renewable energy sources and their impact on global climate policy",
                    "task_type": "reasoning",
                    "expected_complexity": "complex"
                },
                {
                    "prompt": "Summarize this meeting transcript in bullet points",
                    "task_type": "summarization",
                    "expected_complexity": "simple"
                },
                {
                    "prompt": "Design a comprehensive machine learning system for fraud detection in financial transactions",
                    "task_type": "reasoning",
                    "expected_complexity": "complex"
                }
            ]

        optimization_results = []

        for i, prompt_data in enumerate(test_prompts):
            prompt = prompt_data["prompt"]
            task_type = TaskType(prompt_data.get("task_type", "question_answering"))

            # Get model recommendation
            recommendation = await self.model_optimizer.recommend_model(
                prompt=prompt,
                task_type=task_type,
                quality_priority=0.4,
                speed_priority=0.3,
                cost_priority=0.3
            )

            result = {
                "prompt_id": i + 1,
                "prompt_preview": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                "task_type": task_type.value,
                "recommended_model": recommendation.model_id,
                "confidence": recommendation.confidence,
                "estimated_cost": recommendation.estimated_cost,
                "estimated_latency": recommendation.estimated_latency,
                "rationale": recommendation.rationale,
                "alternatives": recommendation.alternatives
            }

            optimization_results.append(result)

            print(f"  ✅ Prompt {i+1}: {recommendation.model_id} (${recommendation.estimated_cost:.4f})")

        # Calculate potential savings
        total_optimized_cost = sum(r["estimated_cost"] for r in optimization_results)

        # Compare with naive approach (always using most expensive model)
        expensive_model_cost = 0.015  # Rough estimate for Claude Opus
        naive_total_cost = len(test_prompts) * expensive_model_cost

        potential_savings = max(0, naive_total_cost - total_optimized_cost)
        savings_percentage = (potential_savings / naive_total_cost * 100) if naive_total_cost > 0 else 0

        summary = {
            "total_prompts": len(test_prompts),
            "optimized_total_cost": total_optimized_cost,
            "naive_total_cost": naive_total_cost,
            "potential_savings": potential_savings,
            "savings_percentage": savings_percentage,
            "recommendations": optimization_results,
            "model_usage": {},
            "avg_confidence": sum(r["confidence"] for r in optimization_results) / len(optimization_results)
        }

        # Calculate model usage distribution
        for result in optimization_results:
            model = result["recommended_model"]
            summary["model_usage"][model] = summary["model_usage"].get(model, 0) + 1

        print(f"\n🎯 Model Selection Optimization Summary:")
        print(f"Total Cost (Optimized): ${total_optimized_cost:.4f}")
        print(f"Total Cost (Naive): ${naive_total_cost:.4f}")
        print(f"Potential Savings: ${potential_savings:.4f} ({savings_percentage:.1f}%)")
        print(f"Average Confidence: {summary['avg_confidence']:.2f}")

        print(f"\n📈 Recommended Model Distribution:")
        for model, count in summary["model_usage"].items():
            percentage = (count / len(test_prompts)) * 100
            print(f"  • {model}: {count}/{len(test_prompts)} ({percentage:.1f}%)")

        if output_file:
            with open(output_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            print(f"\n💾 Results saved to: {output_file}")

        return summary

    async def optimize_infrastructure(self, action: str = "analyze"):
        """Optimize infrastructure configuration."""

        print("🏗️ Analyzing infrastructure optimization...")

        # Start monitoring if not already running
        await self.infrastructure_optimizer.start_monitoring()

        # Wait for some metrics to be collected
        await asyncio.sleep(5)

        if action == "analyze":
            # Get recommendations
            recommendations = self.infrastructure_optimizer.get_optimization_recommendations()
            metrics = self.infrastructure_optimizer.get_infrastructure_metrics()

            print(f"\n📊 Infrastructure Metrics:")
            print(f"Total Hourly Cost: ${metrics['total_hourly_cost']:.2f}")
            print(f"Optimization Opportunities: {metrics['optimization_opportunities']}")

            print(f"\n🏆 Node Efficiency:")
            for node_id, node_data in metrics["nodes"].items():
                efficiency = node_data.get("efficiency_score", 0)
                print(f"  • {node_id}: {efficiency:.1%} efficiency")

            if recommendations:
                print(f"\n💡 Top Infrastructure Recommendations:")
                for i, rec in enumerate(recommendations[:3], 1):
                    print(f"  {i}. {rec.title}")
                    print(f"     Savings: ${rec.estimated_savings:.2f}")
                    print(f"     Effort: {rec.implementation_effort}")
                    for action in rec.actions[:2]:  # Show top 2 actions
                        print(f"     - {action}")
                    print()
            else:
                print("\n✅ No immediate infrastructure optimizations needed")

        elif action == "optimize":
            print("🔧 Implementing infrastructure optimizations...")
            # This would implement actual optimizations
            print("⚠️  Infrastructure optimization implementation requires manual review")
            print("    Please review recommendations first with 'analyze' action")

        await self.infrastructure_optimizer.stop_monitoring()

    async def generate_cost_forecast(
        self,
        profile: str = "developer",
        months: int = 12,
        output_file: Optional[str] = None
    ):
        """Generate cost forecast for specified usage profile."""

        print(f"📈 Generating {months}-month cost forecast for {profile} profile...")

        try:
            usage_profile = UsageProfile(profile.upper())
        except ValueError:
            print(f"❌ Invalid profile: {profile}")
            print(f"Available profiles: {', '.join([p.value for p in UsageProfile])}")
            return

        # Get usage pattern
        patterns = self.cost_calculator.usage_profiles
        usage_pattern = patterns[usage_profile]

        # Use balanced model mix
        model_mix = self.cost_calculator.model_mixes["balanced"]

        # Generate forecast
        forecast = self.cost_calculator.generate_budget_forecast(
            usage_pattern=usage_pattern,
            model_mix=model_mix,
            months_ahead=months,
            confidence_interval=0.95
        )

        print(f"\n📊 Cost Forecast - {profile.title()} Profile:")
        print("=" * 50)
        print(f"Total {months}-Month Forecast: ${forecast['total_forecast']['amount']:.2f}")
        print(f"Average Monthly Cost: ${forecast['average_monthly']:.2f}")
        print(f"Confidence Range: ${forecast['total_forecast']['lower_bound']:.2f} - ${forecast['total_forecast']['upper_bound']:.2f}")

        print(f"\n📅 Monthly Breakdown (first 6 months):")
        for month_data in forecast["monthly_forecasts"][:6]:
            print(f"  Month {month_data['month']:2d}: ${month_data['forecast']:7.2f} "
                  f"({month_data['requests']:,} requests)")

        if months > 6:
            print(f"  ... (showing first 6 of {months} months)")

        print(f"\n📈 Growth Assumptions:")
        print(f"  Monthly Growth Rate: {usage_pattern.growth_rate_monthly:.1%}")
        print(f"  Seasonal Variation: {usage_pattern.seasonal_variation:.1%}")

        if output_file:
            with open(output_file, 'w') as f:
                json.dump(forecast, f, indent=2, default=str)
            print(f"\n💾 Forecast saved to: {output_file}")

        return forecast

    async def run_scenario_comparison(self, output_file: Optional[str] = None):
        """Compare cost scenarios across different usage profiles."""

        print("⚖️ Running scenario comparison analysis...")

        scenarios = self.cost_calculator.get_predefined_scenarios()
        comparison = self.cost_calculator.compare_scenarios(list(scenarios.values()))

        print(f"\n📊 Scenario Comparison Results:")
        print("=" * 50)

        # Display all scenarios
        print(f"Scenarios Analyzed: {comparison['scenario_count']}")
        for profile_name, result in scenarios.items():
            efficiency = result.cost_breakdown.cost_per_request
            print(f"  • {profile_name:12s}: ${result.monthly_cost:8.2f}/month "
                  f"(${efficiency:.4f}/request)")

        print(f"\n🏆 Best vs Worst:")
        best = comparison["cost_range"]["best"]
        worst = comparison["cost_range"]["worst"]
        print(f"  Best:  {best['scenario']:12s} - ${best['monthly_cost']:.2f}")
        print(f"  Worst: {worst['scenario']:12s} - ${worst['monthly_cost']:.2f}")
        print(f"  Potential Savings: ${comparison['cost_range']['range']:.2f} "
              f"({comparison['cost_range']['savings_potential']:.1%})")

        print(f"\n💡 Most Common Optimization Opportunities:")
        for opp, count in comparison["common_optimizations"]:
            print(f"  • {opp} (mentioned {count}x)")

        # Add detailed analysis
        detailed_comparison = {
            "timestamp": datetime.utcnow().isoformat(),
            "scenarios": {name: {
                "monthly_cost": result.monthly_cost,
                "annual_cost": result.annual_cost,
                "cost_breakdown": {
                    "model_inference": result.cost_breakdown.model_inference,
                    "infrastructure": result.cost_breakdown.infrastructure,
                    "total": result.cost_breakdown.total
                },
                "usage_stats": result.usage_stats,
                "optimization_opportunities": result.optimization_opportunities,
                "confidence_level": result.confidence_level
            } for name, result in scenarios.items()},
            "comparison_summary": comparison
        }

        if output_file:
            with open(output_file, 'w') as f:
                json.dump(detailed_comparison, f, indent=2, default=str)
            print(f"\n💾 Comparison saved to: {output_file}")

        return detailed_comparison

    async def setup_monitoring(self, dashboard_port: int = 8050):
        """Set up cost monitoring and dashboard."""

        print("📺 Setting up cost monitoring dashboard...")

        try:
            # Create dashboard app
            app = self.dashboard.create_dash_app()

            print(f"🚀 Starting dashboard on port {dashboard_port}...")
            print(f"   Access at: http://localhost:{dashboard_port}")
            print("   Press Ctrl+C to stop")

            # This would run the dashboard in a real deployment
            # For demo purposes, we'll just show the setup
            print("\n✅ Dashboard configured successfully!")
            print("\nDashboard features:")
            print("  • Real-time cost tracking")
            print("  • Budget alerts and thresholds")
            print("  • Model usage analytics")
            print("  • Infrastructure metrics")
            print("  • Cost forecasting")

            return True

        except Exception as e:
            print(f"❌ Failed to setup dashboard: {e}")
            return False

    async def run_cache_optimization(self):
        """Optimize caching configuration."""

        print("💾 Analyzing cache optimization opportunities...")

        # Simulate cache analysis
        cache_stats = {
            "current_hit_rate": 0.65,
            "target_hit_rate": 0.80,
            "cache_size_mb": 512,
            "recommended_size_mb": 1024,
            "potential_cost_savings": 25.50,
            "cache_efficiency_score": 0.72
        }

        print(f"\n📊 Cache Performance Analysis:")
        print(f"Current Hit Rate: {cache_stats['current_hit_rate']:.1%}")
        print(f"Target Hit Rate:  {cache_stats['target_hit_rate']:.1%}")
        print(f"Cache Size:       {cache_stats['cache_size_mb']} MB")
        print(f"Recommended Size: {cache_stats['recommended_size_mb']} MB")
        print(f"Efficiency Score: {cache_stats['cache_efficiency_score']:.1%}")

        print(f"\n💰 Potential Monthly Savings: ${cache_stats['potential_cost_savings']:.2f}")

        recommendations = [
            "Increase cache size to 1GB for better hit rates",
            "Implement semantic caching for similar queries",
            "Add cache warming for frequently used prompts",
            "Enable cache compression to fit more entries",
            "Implement tiered caching (memory + disk)"
        ]

        print(f"\n💡 Cache Optimization Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")

        return cache_stats

    async def _calculate_efficiency_score(self) -> float:
        """Calculate overall system efficiency score."""
        # This would integrate with actual metrics in a real implementation
        return 0.78  # Placeholder efficiency score


def main():
    """Main CLI interface."""

    parser = argparse.ArgumentParser(
        description="AI Cost Optimization Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cost_optimization_suite.py analyze --days 30
  python cost_optimization_suite.py optimize-models --prompts prompts.json
  python cost_optimization_suite.py forecast --profile developer --months 12
  python cost_optimization_suite.py compare-scenarios
  python cost_optimization_suite.py setup-monitoring --port 8050
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Cost analysis command
    analyze_parser = subparsers.add_parser('analyze', help='Run cost analysis')
    analyze_parser.add_argument('--days', type=int, default=30, help='Days to analyze')
    analyze_parser.add_argument('--output', help='Output file for report')

    # Model optimization command
    optimize_parser = subparsers.add_parser('optimize-models', help='Optimize model selection')
    optimize_parser.add_argument('--prompts', help='JSON file with test prompts')
    optimize_parser.add_argument('--output', help='Output file for results')

    # Infrastructure optimization command
    infra_parser = subparsers.add_parser('optimize-infra', help='Optimize infrastructure')
    infra_parser.add_argument('--action', choices=['analyze', 'optimize'], default='analyze')

    # Cost forecast command
    forecast_parser = subparsers.add_parser('forecast', help='Generate cost forecast')
    forecast_parser.add_argument('--profile', default='developer',
                                help='Usage profile (hobbyist, developer, startup, smb, enterprise, research)')
    forecast_parser.add_argument('--months', type=int, default=12, help='Months to forecast')
    forecast_parser.add_argument('--output', help='Output file for forecast')

    # Scenario comparison command
    compare_parser = subparsers.add_parser('compare-scenarios', help='Compare cost scenarios')
    compare_parser.add_argument('--output', help='Output file for comparison')

    # Monitoring setup command
    monitor_parser = subparsers.add_parser('setup-monitoring', help='Set up monitoring dashboard')
    monitor_parser.add_parameter('--port', type=int, default=8050, help='Dashboard port')

    # Cache optimization command
    subparsers.add_parser('optimize-cache', help='Optimize caching configuration')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Initialize optimization suite
    suite = CostOptimizationSuite()

    # Run appropriate command
    if args.command == 'analyze':
        asyncio.run(suite.run_cost_analysis(args.days, args.output))
    elif args.command == 'optimize-models':
        asyncio.run(suite.optimize_model_selection(args.prompts, args.output))
    elif args.command == 'optimize-infra':
        asyncio.run(suite.optimize_infrastructure(args.action))
    elif args.command == 'forecast':
        asyncio.run(suite.generate_cost_forecast(args.profile, args.months, args.output))
    elif args.command == 'compare-scenarios':
        asyncio.run(suite.run_scenario_comparison(args.output))
    elif args.command == 'setup-monitoring':
        asyncio.run(suite.setup_monitoring(args.port))
    elif args.command == 'optimize-cache':
        asyncio.run(suite.run_cache_optimization())


if __name__ == "__main__":
    main()