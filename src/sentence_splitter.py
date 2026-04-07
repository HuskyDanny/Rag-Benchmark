"""Decompose compound/noisy text into atomic facts via LLM."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

_client: AsyncOpenAI | None = None

SPLITTER_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen3.5-397B-A17B")


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )
    return _client


def _looks_clean(text: str) -> bool:
    """Heuristic: does the text look like properly written English?"""
    if not text.isascii():
        return False
    if not text[0].isupper():
        return False
    noise_indicators = [
        "→",
        "@",
        " & ",
        "lol",
        "ngl",
        " rn",
        " 2 ",
        "prev.",
        "sr.",
        "mgr",
        " ft ",
    ]
    text_lower = text.lower()
    return not any(ind in text_lower for ind in noise_indicators)


def is_likely_compound(text: str) -> bool:
    """Heuristic check: does the text contain multiple facts?"""
    indicators = [" and ", " then ", " but ", ", then", " & ", "; "]
    text_lower = text.lower()
    return any(ind in text_lower for ind in indicators)


async def split_into_atomic_facts(text: str) -> list[str]:
    """Split text into atomic factual statements.

    Simple sentences pass through unchanged. Compound/noisy text
    is decomposed via LLM into clean, independent facts.
    """
    if not is_likely_compound(text) and _looks_clean(text) and len(text.split()) < 15:
        return [text]

    client = _get_client()
    response = await client.chat.completions.create(
        model=SPLITTER_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You decompose text into simple, independent factual statements. "
                    "Rules:\n"
                    "1. Each output fact must be one complete sentence with one subject-verb-object.\n"
                    "2. Fix typos and expand abbreviations into proper English.\n"
                    "3. If the input is already a simple fact, return it as-is.\n"
                    "4. Return ONLY the facts, one per line. No numbering, no bullets."
                ),
            },
            {"role": "user", "content": text},
        ],
    )
    result = response.choices[0].message.content or text
    facts = [line.strip() for line in result.strip().split("\n") if line.strip()]
    return facts if facts else [text]
