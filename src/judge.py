"""LLM-based semantic fact matching judge."""

from __future__ import annotations

import os

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


async def facts_match(returned_fact: str, expected_fact: str) -> bool:
    """Check if two facts are semantically equivalent.

    Short-circuits on exact string match. Falls back to GPT-4o-mini.
    """
    if returned_fact.strip() == expected_fact.strip():
        return True

    client = _get_client()
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
