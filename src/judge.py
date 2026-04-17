"""LLM-based semantic fact matching judge.

Supports two match modes (selectable via FACT_MATCH_MODE env var):
- "llm" (default): Qwen2.5-7B semantic equivalence judge
- "contains": token-subset check — all normalized tokens of expected must
  appear in returned. Free, deterministic, suitable when expected_facts
  are short entity names and returned are long edge-fact sentences.
"""

from __future__ import annotations

import os
import re

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

_client: AsyncOpenAI | None = None


JUDGE_MODEL = os.getenv("JUDGE_MODEL", "Qwen/Qwen2.5-7B-Instruct")


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )
    return _client


_NONWORD_RE = re.compile(r"[^\w\s,]")  # keep commas, strip other punct
_WS_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """Lowercase, strip non-word punctuation (keep commas), collapse whitespace."""
    norm = _NONWORD_RE.sub(" ", text.lower())
    norm = _WS_RE.sub(" ", norm).strip()
    return norm


def contains_match(returned_fact: str, expected_fact: str) -> bool:
    """True if every comma-separated phrase of expected is a substring of returned.

    Use when expected is a short entity name ("University of Arizona, Tucson")
    and returned is a long edge-fact sentence. No LLM call.

    Phrase-based (preserves word order) to avoid false positives where scattered
    tokens coincide across unrelated parts of the returned text. Commas split
    TimeQA-style compound names ("University of Arizona , Tucson") so each
    piece is checked independently — rejects partial matches like a different
    UC campus for "University of California, Santa Barbara".
    """
    ret_norm = _normalize(returned_fact).replace(",", " ")
    ret_norm = _WS_RE.sub(" ", ret_norm).strip()
    exp_norm = _normalize(expected_fact)
    if not exp_norm:
        return False
    for piece in exp_norm.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if piece not in ret_norm:
            return False
    return True


def _current_mode() -> str:
    """Read FACT_MATCH_MODE on every call so tests/runners can override it."""
    return os.getenv("FACT_MATCH_MODE", "llm").lower()


async def facts_match(returned_fact: str, expected_fact: str) -> bool:
    """Check if two facts are semantically equivalent.

    Mode "contains": pure token-subset check, no LLM.
    Mode "llm" (default): exact match fast-path, then Qwen judge.
    """
    if _current_mode() == "contains":
        return contains_match(returned_fact, expected_fact)
    if returned_fact.strip() == expected_fact.strip():
        return True

    client = _get_client()
    try:
        response = await client.chat.completions.create(
            model=JUDGE_MODEL,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a fact-matching judge. Determine if two facts express "
                        "the same information. Respond with only YES or NO."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Fact A: {returned_fact}\n"
                        f"Fact B: {expected_fact}\n\n"
                        "Do these two facts express the same information?"
                    ),
                },
            ],
        )
        answer = response.choices[0].message.content or ""
        return answer.strip().upper().startswith("YES")
    except Exception as e:
        print(f"  WARNING: Judge API error: {e}. Falling back to exact match.")
        return returned_fact.strip().lower() == expected_fact.strip().lower()


async def find_matches(
    returned_facts: list[str],
    expected_facts: list[str],
) -> list[tuple[str, str]]:
    """Find pairs of (returned, expected) facts that semantically match."""
    matches: list[tuple[str, str]] = []
    matched_expected: set[int] = set()

    for returned in returned_facts:
        for i, expected in enumerate(expected_facts):
            if i in matched_expected:
                continue
            if await facts_match(returned, expected):
                matches.append((returned, expected))
                matched_expected.add(i)
                break

    return matches


async def any_match(
    returned_facts: list[str],
    unwanted_facts: list[str],
) -> list[str]:
    """Return any unwanted facts that appear in the returned results."""
    found: list[str] = []
    for unwanted in unwanted_facts:
        for returned in returned_facts:
            if await facts_match(returned, unwanted):
                found.append(unwanted)
                break
    return found
