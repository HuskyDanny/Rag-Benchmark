"""LLM-based semantic fact matching judge.

Two match modes (via FACT_MATCH_MODE env var):
- "llm" (default): exact-match fast-path → auto-route entity-style expected
  facts to `contains_match` → Qwen judge with mention/support prompt for the
  rest. Paraphrases and supersets count (e.g., "works at" supports "employed
  by"); strict equivalence is not required.
- "contains": phrase-substring check only, no LLM. Free, deterministic,
  suitable when expected_facts are short entity names.

Auto-routing: `_looks_like_entity` treats short (≤6 tokens) text with no
common verbs as an entity name, sending it through `contains_match` even in
"llm" mode. Avoids LLM false-negatives on entity-vs-sentence comparisons.
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

    Case-insensitive — both sides are lowercased via `_normalize` before matching.

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


# Common verbs / auxiliaries that mark a string as a sentence (not an entity name).
# Used by `_looks_like_entity` to auto-route short entity-style expected facts
# through `contains_match` (free) instead of the LLM judge.
_SENTENCE_VERBS = frozenset(
    {
        "is",
        "was",
        "are",
        "were",
        "be",
        "been",
        "being",
        "am",
        "has",
        "have",
        "had",
        "having",
        "do",
        "does",
        "did",
        "doing",
        "done",
        "work",
        "works",
        "worked",
        "working",
        "live",
        "lives",
        "lived",
        "living",
        "move",
        "moves",
        "moved",
        "moving",
        "join",
        "joins",
        "joined",
        "joining",
        "leave",
        "leaves",
        "left",
        "leaving",
        "become",
        "becomes",
        "became",
        "becoming",
        "found",
        "founds",
        "founded",
        "founding",
        "own",
        "owns",
        "owned",
        "owning",
        "lead",
        "leads",
        "led",
        "leading",
        "employ",
        "employs",
        "employed",
        "employing",
        "coach",
        "coaches",
        "coached",
        "coaching",
        "complete",
        "completes",
        "completed",
        "completing",
        "study",
        "studies",
        "studied",
        "studying",
        "attend",
        "attends",
        "attended",
        "attending",
        "graduate",
        "graduates",
        "graduated",
        "graduating",
        "serve",
        "serves",
        "served",
        "serving",
        "make",
        "makes",
        "made",
        "making",
        "go",
        "goes",
        "went",
        "going",
        "take",
        "takes",
        "took",
        "taking",
        "give",
        "gives",
        "gave",
        "giving",
        "get",
        "gets",
        "got",
        "getting",
        "say",
        "says",
        "said",
        "saying",
    }
)


def _looks_like_entity(text: str) -> bool:
    """Heuristic: short text with no common verbs → likely an entity name.

    Entity-style expected facts ("University of Arizona, Tucson", "Grantham
    Town") route to `contains_match` for free strict matching. Sentence-style
    expected facts ("Amy works at Facebook") still go through the LLM judge
    for paraphrase recognition.
    """
    tokens = text.lower().split()
    if not tokens or len(tokens) > 6:
        return False
    return not (set(tokens) & _SENTENCE_VERBS)


async def facts_match(returned_fact: str, expected_fact: str) -> bool:
    """Check if two facts are semantically equivalent.

    Mode "contains": phrase-substring check, no LLM (see `contains_match`).
    Mode "llm" (default):
      1. Exact match fast-path
      2. Auto-route: if expected looks like a short entity name, use
         `contains_match` (free, strict). Avoids LLM false-negatives on
         entity-vs-sentence comparisons.
      3. Otherwise, Qwen judge with a mention/support prompt.
    """
    if _current_mode() == "contains":
        return contains_match(returned_fact, expected_fact)
    if returned_fact.strip() == expected_fact.strip():
        return True
    if _looks_like_entity(expected_fact):
        return contains_match(returned_fact, expected_fact)

    client = _get_client()
    try:
        response = await client.chat.completions.create(
            model=JUDGE_MODEL,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a fact-matching judge. Determine if Fact A "
                        "mentions, paraphrases, or supports Fact B. A supports B "
                        "when the information in B can be inferred from A — "
                        "synonyms, paraphrases, and supersets all count. "
                        "Respond with only YES or NO."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Fact A: {returned_fact}\n"
                        f"Fact B: {expected_fact}\n\n"
                        "Does Fact A mention, paraphrase, or support Fact B?"
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
