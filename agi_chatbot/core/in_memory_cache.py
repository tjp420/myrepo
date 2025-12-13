"""Simple in-memory cache shim for dev/test.

Exports `get_response_cache` to satisfy imports in api_server.
"""

from typing import Any, Dict, Optional


class SimpleCache:
    def __init__(self) -> None:
        self._store: Dict[str, Any] = {}

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        return self._store.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._store[key] = value

    def stats(self) -> Dict[str, int]:
        return {"size": len(self._store)}


_GLOBAL_CACHE = SimpleCache()


def get_response_cache() -> SimpleCache:
    return _GLOBAL_CACHE


__all__ = ["SimpleCache", "get_response_cache"]
"""
In-Memory Response Cache for AGI Chatbot
Provides instant responses for repeated queries with LRU eviction.
Enhanced with multi-level caching and performance optimizations.
"""

import asyncio
import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class MultiLevelCache:
    """
    Multi-level caching system with L1 (ultra-fast), L2 (semantic), and L3 (persistent) layers.
    """

    def __init__(self):
        # L1 Cache: Ultra-fast in-memory (small, fast)
        self.l1_cache: Dict[str, Dict[str, Any]] = {}
        self.l1_max_size = 1000
        self.l1_ttl = 300  # 5 minutes

        # L2 Cache: Semantic cache (medium, intelligent)
        self.l2_cache = InMemoryResponseCache(
            maxsize=5000, default_ttl=1800
        )  # 30 minutes

        # L3 Cache: Persistent cache (large, durable)
        self.l3_cache = InMemoryResponseCache(maxsize=50000, default_ttl=3600)  # 1 hour

        # Performance tracking
        self.performance_stats = {
            "l1_hits": 0,
            "l1_misses": 0,
            "l2_hits": 0,
            "l2_misses": 0,
            "l3_hits": 0,
            "l3_misses": 0,
            "total_requests": 0,
        }

        # Thread safety
        self._lock = threading.Lock()

    def _generate_cache_key(self, query: str, context: Optional[Dict] = None) -> str:
        """Generate optimized cache key."""
        if context:
            # Sort context for consistent hashing
            context_str = json.dumps(sorted(context.items()), sort_keys=True)
            content = f"{query.lower().strip()}_{context_str}"
        else:
            content = query.lower().strip()

        # Use faster hashing for L1 cache
        return hashlib.md5(content.encode()).hexdigest()

    def _generate_semantic_key(self, query: str) -> str:
        """Generate semantic key for L2 cache based on query intent."""
        # Simple semantic hashing - can be enhanced with NLP
        words = query.lower().split()
        key_words = [w for w in words if len(w) > 3][
            :5
        ]  # Take first 5 significant words
        return "_".join(sorted(key_words)) if key_words else query[:50]

    async def get(
        self, query: str, context: Optional[Dict] = None
    ) -> Optional[Dict[str, Any]]:
        """Get from multi-level cache with fallback strategy."""
        self.performance_stats["total_requests"] += 1
        cache_key = self._generate_cache_key(query, context)

        # L1 Cache Check (ultra-fast)
        with self._lock:
            if cache_key in self.l1_cache:
                entry = self.l1_cache[cache_key]
                if time.time() - entry["timestamp"] < self.l1_ttl:
                    self.performance_stats["l1_hits"] += 1
                    entry["hits"] = entry.get("hits", 0) + 1
                    logger.debug(f"🚀 L1 CACHE HIT! ({entry['hits']} hits)")
                    return entry["data"]

        # L2 Cache Check (semantic)
        semantic_key = self._generate_semantic_key(query)
        l2_result = await self.l2_cache.get(semantic_key)
        if l2_result:
            self.performance_stats["l2_hits"] += 1
            # Promote to L1 cache
            with self._lock:
                if len(self.l1_cache) >= self.l1_max_size:
                    # Remove oldest entry
                    oldest_key = min(
                        self.l1_cache.keys(),
                        key=lambda k: self.l1_cache[k]["timestamp"],
                    )
                    del self.l1_cache[oldest_key]

                self.l1_cache[cache_key] = {
                    "data": l2_result,
                    "timestamp": time.time(),
                    "hits": 1,
                }
            logger.debug("🔄 L2 CACHE HIT - Promoted to L1")
            return l2_result

        # L3 Cache Check (persistent)
        l3_result = await self.l3_cache.get(query, context)
        if l3_result:
            self.performance_stats["l3_hits"] += 1
            # Promote to L2 cache
            await self.l2_cache.set(semantic_key, l3_result, ttl_seconds=1800)
            logger.debug("💾 L3 CACHE HIT - Promoted to L2")
            return l3_result

        # Cache miss
        self.performance_stats["l1_misses"] += 1
        self.performance_stats["l2_misses"] += 1
        self.performance_stats["l3_misses"] += 1
        return None

    async def set(
        self,
        query: str,
        response: Dict[str, Any],
        context: Optional[Dict] = None,
        priority: str = "normal",
    ) -> None:
        """Set in multi-level cache with intelligent distribution."""
        cache_key = self._generate_cache_key(query, context)
        semantic_key = self._generate_semantic_key(query)

        # Always set in L3 cache
        await self.l3_cache.set(query, response, context)

        # Set in L2 cache for semantic similarity
        await self.l2_cache.set(semantic_key, response, ttl_seconds=1800)

        # Set in L1 cache for ultra-fast access (if high priority or small response)
        if priority == "high" or len(str(response)) < 1000:
            with self._lock:
                if len(self.l1_cache) >= self.l1_max_size:
                    # Remove oldest entry
                    oldest_key = min(
                        self.l1_cache.keys(),
                        key=lambda k: self.l1_cache[k]["timestamp"],
                    )
                    del self.l1_cache[oldest_key]

                self.l1_cache[cache_key] = {
                    "data": response,
                    "timestamp": time.time(),
                    "hits": 0,
                }

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        total_requests = self.performance_stats["total_requests"]
        if total_requests == 0:
            return self.performance_stats

        l1_hit_rate = self.performance_stats["l1_hits"] / total_requests
        l2_hit_rate = self.performance_stats["l2_hits"] / total_requests
        l3_hit_rate = self.performance_stats["l3_hits"] / total_requests
        overall_hit_rate = (
            self.performance_stats["l1_hits"]
            + self.performance_stats["l2_hits"]
            + self.performance_stats["l3_hits"]
        ) / total_requests

        return {
            **self.performance_stats,
            "l1_hit_rate": l1_hit_rate,
            "l2_hit_rate": l2_hit_rate,
            "l3_hit_rate": l3_hit_rate,
            "overall_hit_rate": overall_hit_rate,
            "l1_size": len(self.l1_cache),
            "l2_size": len(self.l2_cache.cache),
            "l3_size": len(self.l3_cache.cache),
        }

    async def cleanup(self) -> None:
        """Clean up expired entries across all cache levels."""
        # L1 cleanup (manual since it's a dict)
        current_time = time.time()
        with self._lock:
            expired_keys = [
                key
                for key, entry in self.l1_cache.items()
                if current_time - entry["timestamp"] > self.l1_ttl
            ]
            for key in expired_keys:
                del self.l1_cache[key]

        # L2 and L3 cleanup
        await self.l2_cache.cleanup_expired()
        await self.l3_cache.cleanup_expired()

        logger.debug(f"Cache cleanup: removed {len(expired_keys)} L1 entries")


class InMemoryResponseCache:
    """
    High-performance in-memory cache for response caching.
    Features LRU eviction, TTL support, and async operations.
    """

    def __init__(self, maxsize: int = 10000, default_ttl: int = 3600):
        self.maxsize = maxsize
        self.default_ttl = default_ttl
        self.cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._lock = asyncio.Lock()
        self.stats = {"hits": 0, "misses": 0, "evictions": 0, "sets": 0}

    def _generate_cache_key(self, query: str, context: Optional[Dict] = None) -> str:
        """Generate a unique cache key from query and context."""
        content = (
            f"{query.lower().strip()}_{str(sorted(context.items()) if context else '')}"
        )
        return hashlib.md5(content.encode()).hexdigest()

    async def get(
        self, query: str, context: Optional[Dict] = None
    ) -> Optional[Dict[str, Any]]:
        """Get cached response if available and not expired."""
        key = self._generate_cache_key(query, context)

        async with self._lock:
            if key not in self.cache:
                self.stats["misses"] += 1
                return None

            entry = self.cache[key]
            current_time = time.time()

            # Check if expired
            if current_time > entry["expires_at"]:
                del self.cache[key]
                self.stats["misses"] += 1
                return None

            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.stats["hits"] += 1

            return entry["data"]

    async def set(
        self,
        query: str,
        response: Dict[str, Any],
        context: Optional[Dict] = None,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """Cache a response with optional TTL."""
        key = self._generate_cache_key(query, context)
        expires_at = time.time() + (ttl_seconds or self.default_ttl)

        async with self._lock:
            # Evict if at capacity
            if len(self.cache) >= self.maxsize:
                evicted_key, evicted_entry = self.cache.popitem(last=False)
                self.stats["evictions"] += 1
                logger.debug(f"Evicted cache entry: {evicted_key[:8]}...")

            self.cache[key] = {
                "data": response,
                "expires_at": expires_at,
                "created_at": time.time(),
                "query": query[:100],  # Store truncated query for debugging
            }
            self.cache.move_to_end(key)
            self.stats["sets"] += 1

    async def invalidate(self, query: str, context: Optional[Dict] = None) -> bool:
        """Remove a specific entry from cache."""
        key = self._generate_cache_key(query, context)

        async with self._lock:
            if key in self.cache:
                del self.cache[key]
                return True
        return False

    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            self.cache.clear()
            logger.info("Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = (self.stats["hits"] / total_requests) if total_requests > 0 else 0

        return {
            **self.stats,
            "cache_size": len(self.cache),
            "max_size": self.maxsize,
            "hit_rate": hit_rate,
            "utilization": len(self.cache) / self.maxsize,
        }

    async def cleanup_expired(self) -> int:
        """Remove expired entries. Returns number of entries removed."""
        current_time = time.time()
        removed_count = 0

        async with self._lock:
            keys_to_remove = [
                key
                for key, entry in self.cache.items()
                if current_time > entry["expires_at"]
            ]

            for key in keys_to_remove:
                del self.cache[key]
                removed_count += 1

        if removed_count > 0:
            logger.debug(f"Cleaned up {removed_count} expired cache entries")

        return removed_count


# Global cache instance - now using multi-level caching
_response_cache = MultiLevelCache()


def get_response_cache() -> MultiLevelCache:
    """Get the global multi-level cache instance."""
    return _response_cache
