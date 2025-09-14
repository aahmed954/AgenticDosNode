"""Advanced caching system for cost optimization."""

from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import time
import asyncio
import pickle
from collections import OrderedDict, deque
import numpy as np

import redis
from redis.asyncio import Redis as AsyncRedis
import aiofiles
from cachetools import TTLCache, LRUCache

from ..config import settings
from ..utils.logging import get_logger

logger = get_logger(__name__)


class CacheLevel(str, Enum):
    """Cache levels in the hierarchy."""
    MEMORY = "memory"
    REDIS = "redis"
    DISK = "disk"
    SEMANTIC = "semantic"


@dataclass
class CacheEntry:
    """Entry in the cache system."""

    key: str
    value: Any
    timestamp: float
    ttl: float
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    cost_saved: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheStats:
    """Statistics for cache performance."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_cost_saved: float = 0.0
    avg_response_time: float = 0.0
    memory_usage_bytes: int = 0
    semantic_matches: int = 0


class PromptCache:
    """
    Multi-level prompt caching system for cost optimization.

    Features:
    - Hierarchical caching (memory -> Redis -> disk)
    - Semantic similarity matching
    - TTL and LRU eviction policies
    - Cost tracking and optimization
    - Batch request deduplication
    """

    def __init__(
        self,
        redis_client: Optional[AsyncRedis] = None,
        memory_size: int = 1000,
        ttl_seconds: int = 3600,
        semantic_threshold: float = 0.95
    ):
        self.redis_client = redis_client
        self.memory_cache = TTLCache(maxsize=memory_size, ttl=ttl_seconds)
        self.semantic_cache = OrderedDict()  # For semantic similarity matching
        self.ttl_seconds = ttl_seconds
        self.semantic_threshold = semantic_threshold
        self.stats = CacheStats()
        self.pending_requests: Dict[str, asyncio.Future] = {}  # For request deduplication

    async def get(
        self,
        key: str,
        level: CacheLevel = CacheLevel.MEMORY,
        check_semantic: bool = True
    ) -> Optional[Any]:
        """Get value from cache."""

        start_time = time.time()

        # Check memory cache first
        if level == CacheLevel.MEMORY or level == CacheLevel.SEMANTIC:
            result = self._get_from_memory(key)
            if result is not None:
                self.stats.hits += 1
                self.stats.avg_response_time = self._update_avg_time(
                    time.time() - start_time
                )
                return result

        # Check semantic cache if enabled
        if check_semantic and level == CacheLevel.SEMANTIC:
            result = await self._get_semantic_match(key)
            if result is not None:
                self.stats.hits += 1
                self.stats.semantic_matches += 1
                return result

        # Check Redis if available
        if self.redis_client and level in [CacheLevel.REDIS, CacheLevel.DISK]:
            result = await self._get_from_redis(key)
            if result is not None:
                # Promote to memory cache
                self.memory_cache[key] = result
                self.stats.hits += 1
                return result

        # Check disk cache
        if level == CacheLevel.DISK:
            result = await self._get_from_disk(key)
            if result is not None:
                # Promote to higher levels
                await self._promote_to_cache(key, result)
                self.stats.hits += 1
                return result

        self.stats.misses += 1
        return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        cost: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Set value in cache with optional TTL and cost tracking."""

        ttl = ttl or self.ttl_seconds

        # Create cache entry
        entry = CacheEntry(
            key=key,
            value=value,
            timestamp=time.time(),
            ttl=ttl,
            cost_saved=cost,
            metadata=metadata or {}
        )

        # Store in all cache levels
        await self._set_all_levels(key, entry)

        # Update stats
        self.stats.total_cost_saved += cost

        # Notify any waiting requests
        if key in self.pending_requests:
            self.pending_requests[key].set_result(value)
            del self.pending_requests[key]

    async def get_or_compute(
        self,
        key: str,
        compute_fn,
        ttl: Optional[int] = None,
        cost: float = 0.0
    ) -> Any:
        """Get from cache or compute if missing, with request deduplication."""

        # Check cache first
        cached = await self.get(key)
        if cached is not None:
            return cached

        # Check if computation is already in progress
        if key in self.pending_requests:
            # Wait for ongoing computation
            return await self.pending_requests[key]

        # Start new computation
        future = asyncio.Future()
        self.pending_requests[key] = future

        try:
            # Compute value
            result = await compute_fn()

            # Cache result
            await self.set(key, result, ttl, cost)

            return result

        except Exception as e:
            # Remove from pending on error
            if key in self.pending_requests:
                del self.pending_requests[key]
            raise e

    async def batch_get(
        self,
        keys: List[str],
        check_semantic: bool = False
    ) -> Dict[str, Optional[Any]]:
        """Get multiple values from cache in batch."""

        results = {}

        # Batch memory lookups
        for key in keys:
            results[key] = self._get_from_memory(key)

        # Batch Redis lookups for missing keys
        missing_keys = [k for k, v in results.items() if v is None]

        if missing_keys and self.redis_client:
            redis_results = await self._batch_get_from_redis(missing_keys)
            results.update(redis_results)

        # Check semantic matches for still missing keys
        if check_semantic:
            still_missing = [k for k, v in results.items() if v is None]
            for key in still_missing:
                semantic_match = await self._get_semantic_match(key)
                if semantic_match:
                    results[key] = semantic_match

        return results

    async def invalidate(self, pattern: Optional[str] = None):
        """Invalidate cache entries matching pattern."""

        if pattern is None:
            # Clear all caches
            self.memory_cache.clear()
            self.semantic_cache.clear()
            if self.redis_client:
                await self.redis_client.flushdb()
        else:
            # Pattern-based invalidation
            keys_to_remove = []

            # Memory cache
            for key in self.memory_cache:
                if pattern in key:
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                del self.memory_cache[key]
                self.stats.evictions += 1

            # Redis cache
            if self.redis_client:
                cursor = 0
                while True:
                    cursor, keys = await self.redis_client.scan(
                        cursor, match=f"*{pattern}*"
                    )
                    if keys:
                        await self.redis_client.delete(*keys)
                    if cursor == 0:
                        break

    def _get_from_memory(self, key: str) -> Optional[Any]:
        """Get value from memory cache."""
        if key in self.memory_cache:
            entry = self.memory_cache[key]
            if isinstance(entry, CacheEntry):
                entry.access_count += 1
                entry.last_access = time.time()
                return entry.value
            return entry
        return None

    async def _get_from_redis(self, key: str) -> Optional[Any]:
        """Get value from Redis cache."""
        if not self.redis_client:
            return None

        try:
            data = await self.redis_client.get(f"prompt_cache:{key}")
            if data:
                return pickle.loads(data)
        except Exception as e:
            logger.error(f"Redis get error: {str(e)}")

        return None

    async def _get_from_disk(self, key: str) -> Optional[Any]:
        """Get value from disk cache."""
        cache_file = f"/tmp/cache/{key[:2]}/{key}.pkl"

        try:
            async with aiofiles.open(cache_file, 'rb') as f:
                data = await f.read()
                return pickle.loads(data)
        except FileNotFoundError:
            return None
        except Exception as e:
            logger.error(f"Disk cache read error: {str(e)}")
            return None

    async def _get_semantic_match(self, key: str) -> Optional[Any]:
        """Find semantically similar cached entry."""

        # Simple similarity check based on key similarity
        # In production, use embeddings for better semantic matching

        for cached_key, entry in self.semantic_cache.items():
            similarity = self._calculate_similarity(key, cached_key)
            if similarity >= self.semantic_threshold:
                logger.info(f"Semantic cache hit: {similarity:.3f}")
                return entry.value

        return None

    async def _set_all_levels(self, key: str, entry: CacheEntry):
        """Set value in all cache levels."""

        # Memory cache
        self.memory_cache[key] = entry

        # Semantic cache
        self.semantic_cache[key] = entry
        if len(self.semantic_cache) > 1000:
            # Remove oldest entry
            self.semantic_cache.popitem(last=False)

        # Redis cache
        if self.redis_client:
            try:
                await self.redis_client.setex(
                    f"prompt_cache:{key}",
                    entry.ttl,
                    pickle.dumps(entry.value)
                )
            except Exception as e:
                logger.error(f"Redis set error: {str(e)}")

        # Disk cache (async write)
        asyncio.create_task(self._write_to_disk(key, entry.value))

    async def _write_to_disk(self, key: str, value: Any):
        """Write value to disk cache asynchronously."""
        import os

        cache_dir = f"/tmp/cache/{key[:2]}"
        cache_file = f"{cache_dir}/{key}.pkl"

        try:
            os.makedirs(cache_dir, exist_ok=True)
            async with aiofiles.open(cache_file, 'wb') as f:
                await f.write(pickle.dumps(value))
        except Exception as e:
            logger.error(f"Disk cache write error: {str(e)}")

    async def _promote_to_cache(self, key: str, value: Any):
        """Promote value to higher cache levels."""
        # Add to memory cache
        self.memory_cache[key] = value

        # Add to Redis if available
        if self.redis_client:
            try:
                await self.redis_client.setex(
                    f"prompt_cache:{key}",
                    self.ttl_seconds,
                    pickle.dumps(value)
                )
            except Exception as e:
                logger.error(f"Redis promotion error: {str(e)}")

    async def _batch_get_from_redis(self, keys: List[str]) -> Dict[str, Optional[Any]]:
        """Batch get from Redis."""
        if not self.redis_client:
            return {k: None for k in keys}

        results = {}
        redis_keys = [f"prompt_cache:{k}" for k in keys]

        try:
            values = await self.redis_client.mget(redis_keys)
            for key, value in zip(keys, values):
                if value:
                    results[key] = pickle.loads(value)
                else:
                    results[key] = None
        except Exception as e:
            logger.error(f"Redis batch get error: {str(e)}")
            return {k: None for k in keys}

        return results

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        # Simple Levenshtein-based similarity
        # In production, use embeddings for semantic similarity

        if text1 == text2:
            return 1.0

        # Jaccard similarity of words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union)

    def _update_avg_time(self, new_time: float) -> float:
        """Update average response time."""
        alpha = 0.1  # Exponential moving average factor
        if self.stats.avg_response_time == 0:
            return new_time
        return alpha * new_time + (1 - alpha) * self.stats.avg_response_time

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        hit_rate = self.stats.hits / max(self.stats.hits + self.stats.misses, 1)

        return {
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "hit_rate": hit_rate,
            "evictions": self.stats.evictions,
            "total_cost_saved": self.stats.total_cost_saved,
            "avg_response_time_ms": self.stats.avg_response_time * 1000,
            "semantic_matches": self.stats.semantic_matches,
            "memory_entries": len(self.memory_cache),
            "semantic_entries": len(self.semantic_cache)
        }


class BatchOptimizer:
    """
    Optimize costs through intelligent batching of requests.

    Features:
    - Dynamic batch size adjustment
    - Priority-based batching
    - Timeout management
    - Cost-aware grouping
    """

    def __init__(
        self,
        max_batch_size: int = 10,
        batch_timeout_ms: int = 100,
        enable_priority: bool = True
    ):
        self.max_batch_size = max_batch_size
        self.batch_timeout_ms = batch_timeout_ms
        self.enable_priority = enable_priority
        self.pending_requests: deque = deque()
        self.batch_lock = asyncio.Lock()
        self.processing = False

    async def add_request(
        self,
        request: Dict[str, Any],
        priority: int = 0,
        callback: Optional[callable] = None
    ) -> asyncio.Future:
        """Add request to batch queue."""

        future = asyncio.Future()

        batch_request = {
            "request": request,
            "priority": priority,
            "future": future,
            "callback": callback,
            "timestamp": time.time()
        }

        async with self.batch_lock:
            self.pending_requests.append(batch_request)

            # Start batch processing if not already running
            if not self.processing:
                asyncio.create_task(self._process_batches())

        return future

    async def _process_batches(self):
        """Process pending requests in batches."""

        self.processing = True

        while self.pending_requests:
            batch = await self._create_batch()

            if batch:
                await self._execute_batch(batch)

            # Small delay to allow more requests to accumulate
            await asyncio.sleep(self.batch_timeout_ms / 1000)

        self.processing = False

    async def _create_batch(self) -> List[Dict[str, Any]]:
        """Create optimal batch from pending requests."""

        batch = []

        async with self.batch_lock:
            # Sort by priority if enabled
            if self.enable_priority:
                sorted_requests = sorted(
                    self.pending_requests,
                    key=lambda x: (-x["priority"], x["timestamp"])
                )
            else:
                sorted_requests = list(self.pending_requests)

            # Select requests for batch
            while sorted_requests and len(batch) < self.max_batch_size:
                request = sorted_requests.pop(0)
                batch.append(request)
                self.pending_requests.remove(request)

        return batch

    async def _execute_batch(self, batch: List[Dict[str, Any]]):
        """Execute a batch of requests."""

        try:
            # Combine requests for batch processing
            combined_request = self._combine_requests([r["request"] for r in batch])

            # Execute batch request (placeholder for actual execution)
            result = await self._execute_combined_request(combined_request)

            # Distribute results
            for i, request in enumerate(batch):
                request["future"].set_result(result.get(i))

                # Execute callback if provided
                if request["callback"]:
                    await request["callback"](result.get(i))

        except Exception as e:
            # Set exception for all requests in batch
            for request in batch:
                request["future"].set_exception(e)

    def _combine_requests(self, requests: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine multiple requests into a single batch request."""
        return {
            "batch": True,
            "requests": requests,
            "count": len(requests)
        }

    async def _execute_combined_request(self, combined: Dict[str, Any]) -> Dict[int, Any]:
        """Execute combined batch request."""
        # Placeholder for actual batch execution
        # This would call the model API with batched inputs
        return {i: f"Result for request {i}" for i in range(combined["count"])}