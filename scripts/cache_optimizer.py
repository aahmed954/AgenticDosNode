#!/usr/bin/env python3
"""
Intelligent cache optimization script.

This script provides:
- Cache performance analysis
- Cache size optimization
- Cache key optimization
- Semantic cache configuration
- Cache warming strategies
"""

import asyncio
import argparse
import json
import redis
import hashlib
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class CacheStats:
    """Cache performance statistics."""
    total_requests: int
    cache_hits: int
    cache_misses: int
    hit_rate: float
    avg_response_time_ms: float
    cache_size_mb: float
    eviction_count: int
    memory_usage_percent: float


@dataclass
class CacheOptimization:
    """Cache optimization recommendation."""
    optimization_type: str
    current_value: Any
    recommended_value: Any
    expected_improvement: float
    implementation_effort: str
    description: str


class CacheOptimizer:
    """
    Intelligent cache optimization system.

    Features:
    - Cache performance analysis
    - Optimal cache size calculation
    - Cache key optimization
    - Semantic caching configuration
    - Cache warming strategies
    """

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        """Initialize cache optimizer."""
        self.redis_client = redis.from_url(redis_url)
        self.cache_stats_history = []

        # Semantic cache configuration
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.semantic_threshold = 0.85

    async def analyze_cache_performance(self, hours_back: int = 24) -> CacheStats:
        """Analyze cache performance over specified time period."""

        print(f"📊 Analyzing cache performance for last {hours_back} hours...")

        # Get Redis info
        info = self.redis_client.info()

        # Calculate statistics from Redis info
        total_commands = info.get('total_commands_processed', 0)
        keyspace_hits = info.get('keyspace_hits', 0)
        keyspace_misses = info.get('keyspace_misses', 0)

        total_requests = keyspace_hits + keyspace_misses
        hit_rate = keyspace_hits / max(total_requests, 1)

        # Memory usage
        used_memory = info.get('used_memory', 0)
        max_memory = info.get('maxmemory', used_memory)
        memory_usage_percent = (used_memory / max(max_memory, 1)) * 100

        # Eviction count
        evicted_keys = info.get('evicted_keys', 0)

        stats = CacheStats(
            total_requests=total_requests,
            cache_hits=keyspace_hits,
            cache_misses=keyspace_misses,
            hit_rate=hit_rate,
            avg_response_time_ms=self._estimate_avg_response_time(),
            cache_size_mb=used_memory / (1024 * 1024),
            eviction_count=evicted_keys,
            memory_usage_percent=memory_usage_percent
        )

        self.cache_stats_history.append({
            'timestamp': datetime.now(),
            'stats': stats
        })

        return stats

    def _estimate_avg_response_time(self) -> float:
        """Estimate average cache response time."""
        # Benchmark cache response time
        start_time = time.time()
        test_key = f"benchmark_{int(start_time)}"

        # Set a test value
        self.redis_client.set(test_key, "benchmark_value", ex=60)

        # Time multiple get operations
        times = []
        for _ in range(10):
            start = time.time()
            self.redis_client.get(test_key)
            end = time.time()
            times.append((end - start) * 1000)  # Convert to milliseconds

        # Cleanup
        self.redis_client.delete(test_key)

        return np.mean(times)

    def analyze_cache_keys(self, sample_size: int = 1000) -> Dict[str, Any]:
        """Analyze cache key patterns and efficiency."""

        print(f"🔍 Analyzing cache key patterns (sampling {sample_size} keys)...")

        # Get sample of keys
        all_keys = self.redis_client.keys("*")
        if len(all_keys) > sample_size:
            sample_keys = np.random.choice(all_keys, sample_size, replace=False)
        else:
            sample_keys = all_keys

        key_analysis = {
            'total_keys': len(all_keys),
            'sampled_keys': len(sample_keys),
            'key_patterns': defaultdict(int),
            'key_lengths': [],
            'ttl_analysis': defaultdict(int),
            'memory_per_key': [],
            'key_types': defaultdict(int)
        }

        for key in sample_keys:
            key_str = key.decode('utf-8') if isinstance(key, bytes) else key

            # Pattern analysis (prefix-based)
            if ':' in key_str:
                pattern = key_str.split(':')[0]
                key_analysis['key_patterns'][pattern] += 1

            # Length analysis
            key_analysis['key_lengths'].append(len(key_str))

            # TTL analysis
            ttl = self.redis_client.ttl(key)
            if ttl > 0:
                if ttl < 3600:  # < 1 hour
                    key_analysis['ttl_analysis']['short'] += 1
                elif ttl < 86400:  # < 1 day
                    key_analysis['ttl_analysis']['medium'] += 1
                else:  # >= 1 day
                    key_analysis['ttl_analysis']['long'] += 1
            elif ttl == -1:  # No expiration
                key_analysis['ttl_analysis']['permanent'] += 1

            # Memory usage per key (approximate)
            try:
                memory_usage = self.redis_client.memory_usage(key)
                if memory_usage:
                    key_analysis['memory_per_key'].append(memory_usage)
            except:
                pass  # MEMORY USAGE command might not be available

            # Key type analysis
            key_type = self.redis_client.type(key).decode('utf-8')
            key_analysis['key_types'][key_type] += 1

        # Calculate statistics
        key_analysis['avg_key_length'] = np.mean(key_analysis['key_lengths'])
        key_analysis['avg_memory_per_key'] = np.mean(key_analysis['memory_per_key']) if key_analysis['memory_per_key'] else 0

        return key_analysis

    def optimize_cache_size(self, current_stats: CacheStats, target_hit_rate: float = 0.85) -> CacheOptimization:
        """Calculate optimal cache size for target hit rate."""

        current_hit_rate = current_stats.hit_rate
        current_size_mb = current_stats.cache_size_mb

        if current_hit_rate >= target_hit_rate:
            # Already meeting target, potentially can reduce size
            recommended_size = current_size_mb * 0.9
            expected_improvement = (current_size_mb - recommended_size) / current_size_mb

            return CacheOptimization(
                optimization_type="cache_size_reduction",
                current_value=f"{current_size_mb:.1f} MB",
                recommended_value=f"{recommended_size:.1f} MB",
                expected_improvement=expected_improvement,
                implementation_effort="low",
                description=f"Reduce cache size by 10% while maintaining {current_hit_rate:.1%} hit rate"
            )
        else:
            # Need to increase size to improve hit rate
            # Rough estimation: doubling cache size typically improves hit rate by 15-25%
            hit_rate_deficit = target_hit_rate - current_hit_rate
            size_multiplier = 1 + (hit_rate_deficit / 0.2)  # Rough heuristic

            recommended_size = current_size_mb * size_multiplier
            expected_improvement = hit_rate_deficit

            return CacheOptimization(
                optimization_type="cache_size_increase",
                current_value=f"{current_size_mb:.1f} MB",
                recommended_value=f"{recommended_size:.1f} MB",
                expected_improvement=expected_improvement,
                implementation_effort="medium",
                description=f"Increase cache size to achieve {target_hit_rate:.1%} hit rate"
            )

    def optimize_ttl_strategy(self, key_analysis: Dict[str, Any]) -> List[CacheOptimization]:
        """Optimize TTL (Time To Live) strategy."""

        optimizations = []
        ttl_dist = key_analysis['ttl_analysis']

        # Check for too many permanent keys
        total_keys = sum(ttl_dist.values())
        permanent_ratio = ttl_dist.get('permanent', 0) / max(total_keys, 1)

        if permanent_ratio > 0.3:  # More than 30% permanent keys
            optimizations.append(CacheOptimization(
                optimization_type="ttl_permanent_keys",
                current_value=f"{permanent_ratio:.1%} permanent keys",
                recommended_value="< 20% permanent keys",
                expected_improvement=0.15,  # 15% memory reduction
                implementation_effort="medium",
                description="Set appropriate TTL for permanent keys to prevent memory bloat"
            ))

        # Check for too many short-lived keys
        short_ratio = ttl_dist.get('short', 0) / max(total_keys, 1)
        if short_ratio > 0.5:  # More than 50% short-lived keys
            optimizations.append(CacheOptimization(
                optimization_type="ttl_short_keys",
                current_value=f"{short_ratio:.1%} short-lived keys",
                recommended_value="30-40% short-lived keys",
                expected_improvement=0.10,  # 10% efficiency improvement
                implementation_effort="low",
                description="Increase TTL for frequently accessed short-lived keys"
            ))

        return optimizations

    def setup_semantic_caching(self, threshold: float = 0.85) -> Dict[str, Any]:
        """Set up semantic caching for similar queries."""

        print(f"🧠 Setting up semantic caching with threshold {threshold}")

        # Create semantic cache configuration
        config = {
            "enabled": True,
            "similarity_threshold": threshold,
            "max_cache_entries": 10000,
            "vector_dimensions": 1000,
            "recompute_interval_hours": 24
        }

        # Store configuration in Redis
        self.redis_client.set(
            "semantic_cache_config",
            json.dumps(config),
            ex=86400 * 7  # 7 days
        )

        # Initialize semantic cache index
        self.redis_client.delete("semantic_cache:*")  # Clear existing

        return config

    def find_similar_cached_queries(self, query: str, threshold: float = None) -> List[Dict[str, Any]]:
        """Find semantically similar cached queries."""

        if threshold is None:
            threshold = self.semantic_threshold

        # Get all cached query keys
        query_keys = self.redis_client.keys("query_cache:*")

        if not query_keys:
            return []

        # Extract queries and their responses
        cached_queries = []
        for key in query_keys[:100]:  # Limit to 100 for performance
            try:
                cached_data = self.redis_client.get(key)
                if cached_data:
                    data = json.loads(cached_data)
                    cached_queries.append({
                        'key': key.decode('utf-8') if isinstance(key, bytes) else key,
                        'query': data.get('query', ''),
                        'response': data.get('response', ''),
                        'timestamp': data.get('timestamp', '')
                    })
            except:
                continue

        if not cached_queries:
            return []

        # Compute similarity
        all_queries = [query] + [cq['query'] for cq in cached_queries]

        try:
            # Vectorize queries
            tfidf_matrix = self.vectorizer.fit_transform(all_queries)

            # Compute similarity scores
            query_vector = tfidf_matrix[0:1]  # First row is our query
            similarities = cosine_similarity(query_vector, tfidf_matrix[1:]).flatten()

            # Find similar queries above threshold
            similar_queries = []
            for i, similarity in enumerate(similarities):
                if similarity >= threshold:
                    similar_queries.append({
                        **cached_queries[i],
                        'similarity': similarity
                    })

            # Sort by similarity
            similar_queries.sort(key=lambda x: x['similarity'], reverse=True)

            return similar_queries

        except Exception as e:
            print(f"Error computing similarities: {e}")
            return []

    def warm_cache(self, warming_strategy: str = "frequency_based") -> Dict[str, Any]:
        """Implement cache warming strategy."""

        print(f"🔥 Warming cache using {warming_strategy} strategy...")

        warming_stats = {
            "strategy": warming_strategy,
            "keys_warmed": 0,
            "total_time_seconds": 0,
            "success_rate": 0
        }

        start_time = time.time()

        if warming_strategy == "frequency_based":
            # Warm most frequently accessed keys
            # This would typically involve analyzing access logs
            # For demo, we'll simulate this

            frequent_patterns = [
                "user_profile:*",
                "session:*",
                "model_response:*"
            ]

            warmed_keys = 0
            for pattern in frequent_patterns:
                keys = self.redis_client.keys(pattern)
                for key in keys[:50]:  # Warm top 50 of each pattern
                    try:
                        # Touch the key to move it to hot cache
                        self.redis_client.touch(key)
                        warmed_keys += 1
                    except:
                        continue

            warming_stats["keys_warmed"] = warmed_keys

        elif warming_strategy == "predictive":
            # Warm keys likely to be accessed soon
            # This would involve ML prediction of future access patterns
            warming_stats["keys_warmed"] = 0  # Would be implemented with ML

        warming_stats["total_time_seconds"] = time.time() - start_time
        warming_stats["success_rate"] = 1.0 if warming_stats["keys_warmed"] > 0 else 0.0

        return warming_stats

    def generate_optimization_report(self, output_file: str = None) -> Dict[str, Any]:
        """Generate comprehensive cache optimization report."""

        print("📋 Generating cache optimization report...")

        # Analyze current performance
        current_stats = asyncio.run(self.analyze_cache_performance())
        key_analysis = self.analyze_cache_keys()

        # Generate optimizations
        size_optimization = self.optimize_cache_size(current_stats)
        ttl_optimizations = self.optimize_ttl_strategy(key_analysis)

        # Calculate potential impact
        total_improvement = size_optimization.expected_improvement
        for opt in ttl_optimizations:
            total_improvement += opt.expected_improvement

        report = {
            "timestamp": datetime.now().isoformat(),
            "current_performance": {
                "hit_rate": current_stats.hit_rate,
                "cache_size_mb": current_stats.cache_size_mb,
                "memory_usage_percent": current_stats.memory_usage_percent,
                "avg_response_time_ms": current_stats.avg_response_time_ms,
                "total_requests": current_stats.total_requests
            },
            "key_analysis": {
                "total_keys": key_analysis["total_keys"],
                "avg_key_length": key_analysis["avg_key_length"],
                "key_patterns": dict(key_analysis["key_patterns"]),
                "ttl_distribution": dict(key_analysis["ttl_analysis"])
            },
            "optimizations": {
                "cache_size": {
                    "type": size_optimization.optimization_type,
                    "current": size_optimization.current_value,
                    "recommended": size_optimization.recommended_value,
                    "expected_improvement": size_optimization.expected_improvement,
                    "description": size_optimization.description
                },
                "ttl_optimizations": [
                    {
                        "type": opt.optimization_type,
                        "current": opt.current_value,
                        "recommended": opt.recommended_value,
                        "expected_improvement": opt.expected_improvement,
                        "description": opt.description
                    }
                    for opt in ttl_optimizations
                ]
            },
            "potential_impact": {
                "total_improvement_percent": total_improvement * 100,
                "estimated_cost_savings_monthly": self._estimate_cost_savings(total_improvement),
                "implementation_priority": self._get_implementation_priority(total_improvement)
            },
            "recommendations": self._generate_recommendations(current_stats, key_analysis, total_improvement)
        }

        # Print summary
        print(f"\n📊 Cache Optimization Report Summary:")
        print(f"Current Hit Rate: {current_stats.hit_rate:.1%}")
        print(f"Cache Size: {current_stats.cache_size_mb:.1f} MB")
        print(f"Total Keys: {key_analysis['total_keys']:,}")
        print(f"Potential Improvement: {total_improvement:.1%}")
        print(f"Estimated Monthly Savings: ${self._estimate_cost_savings(total_improvement):.2f}")

        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"\n💾 Report saved to: {output_file}")

        return report

    def _estimate_cost_savings(self, improvement_percent: float) -> float:
        """Estimate monthly cost savings from cache improvements."""
        # Rough estimation based on reduced API calls and infrastructure usage
        base_monthly_cost = 100.0  # Assume $100 baseline
        return base_monthly_cost * improvement_percent

    def _get_implementation_priority(self, improvement_percent: float) -> str:
        """Get implementation priority based on improvement potential."""
        if improvement_percent > 0.3:
            return "high"
        elif improvement_percent > 0.15:
            return "medium"
        else:
            return "low"

    def _generate_recommendations(self, stats: CacheStats, key_analysis: Dict, improvement: float) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        if stats.hit_rate < 0.7:
            recommendations.append("Increase cache size to improve hit rate")

        if stats.memory_usage_percent > 90:
            recommendations.append("Implement cache eviction policies to prevent memory issues")

        if key_analysis["ttl_analysis"].get("permanent", 0) > key_analysis["total_keys"] * 0.3:
            recommendations.append("Set appropriate TTL for permanent keys")

        if stats.eviction_count > stats.total_requests * 0.1:
            recommendations.append("Optimize key priorities to reduce unnecessary evictions")

        if improvement > 0.2:
            recommendations.append("Consider implementing semantic caching for similar queries")
            recommendations.append("Set up cache warming for frequently accessed data")

        return recommendations


def main():
    """Main CLI interface."""

    parser = argparse.ArgumentParser(description="Cache Optimization Tool")

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze cache performance')
    analyze_parser.add_argument('--hours', type=int, default=24, help='Hours to analyze')
    analyze_parser.add_argument('--output', help='Output file for report')

    # Optimize command
    optimize_parser = subparsers.add_parser('optimize', help='Generate optimization recommendations')
    optimize_parser.add_argument('--target-hit-rate', type=float, default=0.85, help='Target hit rate')
    optimize_parser.add_argument('--output', help='Output file for report')

    # Warm cache command
    warm_parser = subparsers.add_parser('warm', help='Warm cache with frequently accessed data')
    warm_parser.add_argument('--strategy', choices=['frequency_based', 'predictive'], default='frequency_based')

    # Semantic cache command
    semantic_parser = subparsers.add_parser('setup-semantic', help='Set up semantic caching')
    semantic_parser.add_argument('--threshold', type=float, default=0.85, help='Similarity threshold')

    # Similar queries command
    similar_parser = subparsers.add_parser('find-similar', help='Find similar cached queries')
    similar_parser.add_argument('query', help='Query to find similar matches for')
    similar_parser.add_argument('--threshold', type=float, default=0.85, help='Similarity threshold')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    optimizer = CacheOptimizer()

    try:
        if args.command == 'analyze':
            stats = asyncio.run(optimizer.analyze_cache_performance(args.hours))
            print(f"\n📊 Cache Performance (last {args.hours} hours):")
            print(f"Hit Rate: {stats.hit_rate:.1%}")
            print(f"Total Requests: {stats.total_requests:,}")
            print(f"Cache Size: {stats.cache_size_mb:.1f} MB")
            print(f"Memory Usage: {stats.memory_usage_percent:.1f}%")
            print(f"Avg Response Time: {stats.avg_response_time_ms:.2f} ms")

        elif args.command == 'optimize':
            optimizer.generate_optimization_report(args.output)

        elif args.command == 'warm':
            result = optimizer.warm_cache(args.strategy)
            print(f"✅ Cache warming completed:")
            print(f"Strategy: {result['strategy']}")
            print(f"Keys warmed: {result['keys_warmed']}")
            print(f"Time taken: {result['total_time_seconds']:.2f} seconds")

        elif args.command == 'setup-semantic':
            config = optimizer.setup_semantic_caching(args.threshold)
            print(f"🧠 Semantic caching configured:")
            print(f"Threshold: {config['similarity_threshold']}")
            print(f"Max entries: {config['max_cache_entries']}")

        elif args.command == 'find-similar':
            similar = optimizer.find_similar_cached_queries(args.query, args.threshold)
            if similar:
                print(f"🔍 Found {len(similar)} similar queries:")
                for i, sim in enumerate(similar[:5], 1):
                    print(f"{i}. Similarity: {sim['similarity']:.2f}")
                    print(f"   Query: {sim['query'][:100]}...")
            else:
                print("No similar queries found")

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()