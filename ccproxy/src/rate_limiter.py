"""Rate limiting and request throttling for the Claude Code proxy."""

import asyncio
import time
from typing import Dict, Optional
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class RateLimitRule:
    """Rate limit rule configuration."""
    requests_per_minute: int
    tokens_per_minute: int
    concurrent_requests: int


class TokenBucket:
    """Token bucket implementation for rate limiting."""

    def __init__(self, capacity: int, refill_rate: int):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate  # tokens per second
        self.last_refill = time.time()
        self._lock = asyncio.Lock()

    async def consume(self, tokens: int) -> bool:
        """Try to consume tokens from the bucket."""
        async with self._lock:
            await self._refill()
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    async def _refill(self):
        """Refill the token bucket based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        tokens_to_add = int(elapsed * self.refill_rate)

        if tokens_to_add > 0:
            self.tokens = min(self.capacity, self.tokens + tokens_to_add)
            self.last_refill = now


class SlidingWindowCounter:
    """Sliding window counter for request rate limiting."""

    def __init__(self, window_size_minutes: int = 1):
        self.window_size = timedelta(minutes=window_size_minutes)
        self.requests = deque()
        self._lock = asyncio.Lock()

    async def add_request(self) -> bool:
        """Add a request and return if within limits."""
        async with self._lock:
            now = datetime.now()
            # Remove old requests outside the window
            while self.requests and now - self.requests[0] > self.window_size:
                self.requests.popleft()

            self.requests.append(now)
            return True

    async def get_count(self) -> int:
        """Get current request count in the window."""
        async with self._lock:
            now = datetime.now()
            # Clean old requests
            while self.requests and now - self.requests[0] > self.window_size:
                self.requests.popleft()
            return len(self.requests)

    async def can_add_request(self, limit: int) -> bool:
        """Check if we can add another request within limits."""
        count = await self.get_count()
        return count < limit


class ConcurrencyLimiter:
    """Limits concurrent requests."""

    def __init__(self, max_concurrent: int):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.active_count = 0
        self._lock = asyncio.Lock()

    async def acquire(self):
        """Acquire a concurrency slot."""
        await self.semaphore.acquire()
        async with self._lock:
            self.active_count += 1

    async def release(self):
        """Release a concurrency slot."""
        async with self._lock:
            self.active_count = max(0, self.active_count - 1)
        self.semaphore.release()

    async def get_active_count(self) -> int:
        """Get current active request count."""
        async with self._lock:
            return self.active_count


class RateLimiter:
    """Comprehensive rate limiter with multiple strategies."""

    def __init__(
        self,
        requests_per_minute: int = 60,
        tokens_per_minute: int = 100000,
        concurrent_requests: int = 10
    ):
        # Request rate limiting (sliding window)
        self.request_counter = SlidingWindowCounter()
        self.requests_per_minute = requests_per_minute

        # Token rate limiting (token bucket)
        tokens_per_second = tokens_per_minute / 60
        self.token_bucket = TokenBucket(tokens_per_minute, tokens_per_second)

        # Concurrency limiting
        self.concurrency_limiter = ConcurrencyLimiter(concurrent_requests)

        # Per-key rate limiting
        self.per_key_limiters: Dict[str, 'RateLimiter'] = {}
        self._key_lock = asyncio.Lock()

        # Statistics
        self.total_requests = 0
        self.rate_limited_requests = 0
        self.start_time = time.time()

    async def check_rate_limit(
        self,
        api_key: Optional[str] = None,
        estimated_tokens: int = 0
    ) -> tuple[bool, str]:
        """
        Check if request is within rate limits.
        Returns (allowed, reason_if_denied).
        """
        # Global request rate check
        if not await self.request_counter.can_add_request(self.requests_per_minute):
            self.rate_limited_requests += 1
            return False, "Request rate limit exceeded"

        # Token rate check
        if estimated_tokens > 0:
            if not await self.token_bucket.consume(estimated_tokens):
                self.rate_limited_requests += 1
                return False, "Token rate limit exceeded"

        # Per-key rate limiting
        if api_key:
            key_limiter = await self._get_key_limiter(api_key)
            key_allowed, key_reason = await key_limiter.check_rate_limit()
            if not key_allowed:
                self.rate_limited_requests += 1
                return False, f"Per-key {key_reason}"

        # All checks passed
        await self.request_counter.add_request()
        self.total_requests += 1
        return True, ""

    async def acquire_concurrency_slot(self):
        """Acquire a concurrency slot."""
        await self.concurrency_limiter.acquire()

    async def release_concurrency_slot(self):
        """Release a concurrency slot."""
        await self.concurrency_limiter.release()

    async def _get_key_limiter(self, api_key: str) -> 'RateLimiter':
        """Get or create a per-key rate limiter."""
        async with self._key_lock:
            if api_key not in self.per_key_limiters:
                # Create more restrictive per-key limits
                self.per_key_limiters[api_key] = RateLimiter(
                    requests_per_minute=self.requests_per_minute // 2,
                    tokens_per_minute=50000,  # Lower per-key token limit
                    concurrent_requests=5     # Lower per-key concurrency
                )
            return self.per_key_limiters[api_key]

    async def get_stats(self) -> Dict[str, any]:
        """Get rate limiting statistics."""
        uptime = time.time() - self.start_time
        active_requests = await self.concurrency_limiter.get_active_count()
        current_request_count = await self.request_counter.get_count()

        return {
            "total_requests": self.total_requests,
            "rate_limited_requests": self.rate_limited_requests,
            "success_rate": (self.total_requests - self.rate_limited_requests) / max(self.total_requests, 1),
            "active_requests": active_requests,
            "current_request_rate": current_request_count,
            "uptime_seconds": uptime,
            "requests_per_second": self.total_requests / max(uptime, 1)
        }

    async def reset_stats(self):
        """Reset statistics."""
        self.total_requests = 0
        self.rate_limited_requests = 0
        self.start_time = time.time()


class AdaptiveRateLimiter(RateLimiter):
    """Rate limiter that adapts based on error rates and response times."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.error_count = 0
        self.response_times = deque(maxlen=100)  # Keep last 100 response times
        self.last_adaptation = time.time()
        self.adaptation_interval = 60  # Adapt every 60 seconds

    async def record_response(self, success: bool, response_time: float):
        """Record response outcome and time."""
        if not success:
            self.error_count += 1

        self.response_times.append(response_time)

        # Adapt rate limits if needed
        await self._maybe_adapt()

    async def _maybe_adapt(self):
        """Adapt rate limits based on error rates and performance."""
        now = time.time()
        if now - self.last_adaptation < self.adaptation_interval:
            return

        self.last_adaptation = now

        # Calculate error rate
        error_rate = self.error_count / max(self.total_requests, 1)

        # Calculate average response time
        if self.response_times:
            avg_response_time = sum(self.response_times) / len(self.response_times)
        else:
            avg_response_time = 0

        # Adapt based on error rate
        if error_rate > 0.1:  # High error rate
            # Reduce rate limits by 20%
            self.requests_per_minute = int(self.requests_per_minute * 0.8)
            self.token_bucket.refill_rate = int(self.token_bucket.refill_rate * 0.8)
        elif error_rate < 0.02 and avg_response_time < 2.0:  # Low error rate and fast responses
            # Increase rate limits by 10%
            self.requests_per_minute = int(self.requests_per_minute * 1.1)
            self.token_bucket.refill_rate = int(self.token_bucket.refill_rate * 1.1)

        # Reset error count for next period
        self.error_count = 0