"""Post-ingestion contradiction resolver.

After all episodes are ingested, detects entities with multiple CURRENT edges
that contradict each other and sets invalid_at on the older edge.

This fixes the core Graphiti limitation where LLM-extracted edges with different
relationship names (e.g., WORKS_AT vs LEFT_COMPANY) aren't detected as
contradictions by resolve_extracted_edges.
"""

from __future__ import annotations

import os
from collections import defaultdict
from datetime import datetime
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

_client: AsyncOpenAI | None = None
_MODEL = os.getenv("JUDGE_MODEL", "Qwen/Qwen2.5-7B-Instruct")


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )
    return _client


async def resolve_contradictions(driver: Any, group_id: str) -> int:
    """Detect and resolve contradicting CURRENT edges within a group.

    For each entity with 2+ CURRENT edges, uses LLM to check if any pair
    describes the same attribute with conflicting values. If yes, sets
    invalid_at on the older edge.

    Returns the number of edges invalidated.
    """
    # Step 1: Get all CURRENT edges grouped by source entity
    records, _, _ = await driver.execute_query(
        """
        MATCH (s:Entity)-[r:RELATES_TO]->(t:Entity)
        WHERE r.group_id = $group_id AND r.invalid_at IS NULL
        RETURN s.name AS source, t.name AS target,
               r.name AS relation, r.fact AS fact,
               r.uuid AS uuid, r.valid_at AS valid_at,
               r.created_at AS created_at
        ORDER BY s.name, r.valid_at
        """,
        group_id=group_id,
    )

    edges_by_entity: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        d = dict(r)
        edges_by_entity[d["source"]].append(d)

    invalidated = 0

    # Step 2: For each entity with 2+ current edges, check pairs
    for entity, edges in edges_by_entity.items():
        if len(edges) < 2:
            continue

        # Check all pairs (O(n²) but n is small per entity, typically 2-5)
        for i in range(len(edges)):
            for j in range(i + 1, len(edges)):
                older = edges[i]
                newer = edges[j]

                # Skip if already invalidated in this pass
                if older.get("_invalidated") or newer.get("_invalidated"):
                    continue

                contradicts = await _facts_contradict(older["fact"], newer["fact"])
                if contradicts:
                    # Invalidate the older one
                    to_invalidate = older
                    valid_edge = newer

                    # Determine which is older by valid_at, fallback to created_at
                    older_time = _get_time(older)
                    newer_time = _get_time(newer)
                    if newer_time and older_time and newer_time < older_time:
                        to_invalidate = newer
                        valid_edge = older

                    invalidation_time = _get_time(valid_edge) or datetime.now()

                    await driver.execute_query(
                        """
                        MATCH ()-[r:RELATES_TO]->()
                        WHERE r.uuid = $uuid
                        SET r.invalid_at = $invalid_at
                        """,
                        uuid=to_invalidate["uuid"],
                        invalid_at=invalidation_time,
                    )
                    to_invalidate["_invalidated"] = True
                    invalidated += 1
                    print(
                        f"    Resolved: '{to_invalidate['fact']}' "
                        f"invalidated by '{valid_edge['fact']}'"
                    )

    return invalidated


def _get_time(edge: dict) -> datetime | None:
    """Extract the best available timestamp from an edge."""
    for key in ("valid_at", "created_at"):
        val = edge.get(key)
        if val is not None:
            if isinstance(val, datetime):
                return val
    return None


async def _facts_contradict(fact_a: str, fact_b: str) -> bool:
    """Use LLM to check if two facts about the same entity contradict each other.

    Returns True if the facts describe the same attribute with different values,
    meaning the newer one supersedes the older one.
    """
    client = _get_client()
    response = await client.chat.completions.create(
        model=_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a fact contradiction detector. Two facts about the same "
                    "entity are given. Determine if they CONTRADICT each other — meaning "
                    "they describe the same attribute (e.g., workplace, location, role, "
                    "quantity) but with different values, and only one can be true at a "
                    "given time.\n\n"
                    "Examples of contradictions:\n"
                    '- "Alice works at Google" vs "Alice works at Meta" (same attribute: workplace)\n'
                    '- "Company has 500 employees" vs "Company has 750 employees" (same attribute: count)\n'
                    '- "Bob lives in NYC" vs "Bob moved to SF" (same attribute: residence)\n\n'
                    "NOT contradictions:\n"
                    '- "Alice works at Google" vs "Alice lives in NYC" (different attributes)\n'
                    '- "Alice manages Team A" vs "Alice reports to Bob" (different relationships)\n\n'
                    "Respond with only YES or NO."
                ),
            },
            {
                "role": "user",
                "content": f"Fact A: {fact_a}\nFact B: {fact_b}\n\nDo these contradict?",
            },
        ],
    )
    answer = response.choices[0].message.content or ""
    return answer.strip().upper().startswith("YES")
