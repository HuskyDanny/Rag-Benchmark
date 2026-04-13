"""LLM client wrapper with prompt-level response caching.

Wraps OpenAIGenericClient to cache LLM responses by message content hash.
Same prompt → cached response → saves the LLM API call entirely.

Use case: when pipeline crashes and resumes, or when the same episodes are
processed across phases (pipeline vs pipeline_presplit), the extract_nodes
and extract_edges calls return cached results instead of hitting the API.
"""

from __future__ import annotations

import hashlib
import json
import typing
from collections import OrderedDict
from pathlib import Path

from graphiti_core.llm_client.config import LLMConfig, ModelSize
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.prompts.models import Message
from pydantic import BaseModel

CACHE_DIR = Path("results/llm_cache")
DEFAULT_MAX_MEMORY_ENTRIES = 2000  # evict oldest when exceeded


class CachingLLMClient(OpenAIGenericClient):
    """OpenAIGenericClient with persistent prompt-level response caching.

    Cache key = sha256(messages_content + response_model_schema).
    Cache is stored as individual JSON files in results/llm_cache/ (per-file).
    Memory cache is bounded by max_memory_entries (LRU eviction).

    Per-file storage means corruption of one entry doesn't affect others.
    Disk cache is unbounded (files persist until manually cleared).
    Memory cache is bounded (oldest entries evicted when limit reached).
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        client: typing.Any = None,
        max_tokens: int = 16384,
        max_memory_entries: int = DEFAULT_MAX_MEMORY_ENTRIES,
    ):
        super().__init__(
            config=config, cache=False, client=client, max_tokens=max_tokens
        )
        self._memory_cache: OrderedDict[str, dict] = OrderedDict()
        self._max_memory_entries = max_memory_entries
        self._cache_dir = CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._hits = 0
        self._misses = 0

    def _cache_key(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None,
        model_size: ModelSize,
    ) -> str:
        """Deterministic cache key from message content + schema."""
        parts = []
        for m in messages:
            parts.append(f"{m.role}:{m.content}")
        if response_model is not None:
            parts.append(f"schema:{response_model.__name__}")
        parts.append(f"size:{model_size.value}")
        content = "\n".join(parts)
        return hashlib.sha256(content.encode()).hexdigest()[:20]

    def _load_from_disk(self, key: str) -> dict | None:
        """Try loading a cached response from disk."""
        path = self._cache_dir / f"{key}.json"
        if path.exists():
            try:
                return json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                return None
        return None

    def _save_to_disk(self, key: str, response: dict) -> None:
        """Persist a response to disk cache."""
        path = self._cache_dir / f"{key}.json"
        tmp = path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(response, indent=2, default=str))
            tmp.rename(path)
        except OSError as e:
            print(f"  WARNING: LLM cache write failed for {key}: {e}")

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = 16384,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, typing.Any]:
        key = self._cache_key(messages, response_model, model_size)

        # Check memory cache (LRU: move to end on access)
        if key in self._memory_cache:
            self._memory_cache.move_to_end(key)
            self._hits += 1
            return self._memory_cache[key]

        # Check disk cache
        cached = self._load_from_disk(key)
        if cached is not None:
            self._put_memory(key, cached)
            self._hits += 1
            return cached

        # Cache miss — call the actual LLM
        self._misses += 1
        response = await super()._generate_response(
            messages, response_model, max_tokens, model_size
        )

        # Cache the response (memory + disk)
        self._put_memory(key, response)
        self._save_to_disk(key, response)

        return response

    def _put_memory(self, key: str, value: dict) -> None:
        """Add to memory cache with LRU eviction."""
        self._memory_cache[key] = value
        self._memory_cache.move_to_end(key)
        while len(self._memory_cache) > self._max_memory_entries:
            self._memory_cache.popitem(last=False)  # evict oldest

    def cache_stats(self) -> dict[str, int]:
        """Return cache hit/miss statistics."""
        disk_files = len(list(self._cache_dir.glob("*.json")))
        return {
            "memory_entries": len(self._memory_cache),
            "disk_entries": disk_files,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / max(self._hits + self._misses, 1), 3),
        }
