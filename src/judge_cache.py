"""Global LLM judge result cache — avoids re-judging identical fact pairs across experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.judge import facts_match

CACHE_DIR = Path("results/judge_cache")

_memory_cache: dict[str, bool] = {}
_disk_loaded = False


def _cache_key(fact_a: str, fact_b: str) -> str:
    """Deterministic key for a fact pair (order-independent)."""
    pair = sorted([fact_a.strip().lower(), fact_b.strip().lower()])
    return hashlib.sha256(json.dumps(pair).encode()).hexdigest()[:16]


def _load_disk_cache() -> None:
    """Load cached judgments from disk into memory (once)."""
    global _memory_cache, _disk_loaded
    if _disk_loaded:
        return
    _disk_loaded = True
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / "judgments.json"
    if cache_file.exists():
        try:
            _memory_cache = json.loads(cache_file.read_text())
        except (json.JSONDecodeError, OSError):
            _memory_cache = {}


def _persist() -> None:
    """Atomic write of memory cache to disk."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / "judgments.json"
    tmp = cache_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(_memory_cache, indent=2))
    tmp.rename(cache_file)


async def cached_facts_match(returned_fact: str, expected_fact: str) -> bool:
    """Like judge.facts_match but with persistent caching.

    Cache is keyed by sorted(fact_a, fact_b) hash — order-independent.
    Survives across runs and experiments.
    """
    _load_disk_cache()
    key = _cache_key(returned_fact, expected_fact)
    if key in _memory_cache:
        return _memory_cache[key]

    result = await facts_match(returned_fact, expected_fact)
    _memory_cache[key] = result
    _persist()
    return result


def cache_stats() -> dict[str, int]:
    """Return cache statistics."""
    _load_disk_cache()
    return {
        "total_entries": len(_memory_cache),
        "true_matches": sum(1 for v in _memory_cache.values() if v),
        "false_matches": sum(1 for v in _memory_cache.values() if not v),
    }
