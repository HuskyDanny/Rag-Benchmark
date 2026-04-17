import os
import pytest
from dotenv import load_dotenv

load_dotenv()

needs_openai = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)


@pytest.mark.asyncio
async def test_exact_match():
    from src.judge import facts_match

    assert await facts_match("Amy works at Facebook", "Amy works at Facebook") is True


@needs_openai
@pytest.mark.asyncio
async def test_semantic_match():
    from src.judge import facts_match

    assert (
        await facts_match("Amy is employed by Facebook", "Amy works at Facebook")
        is True
    )


@needs_openai
@pytest.mark.asyncio
async def test_no_match():
    from src.judge import facts_match

    assert await facts_match("Amy works at Google", "Amy works at Facebook") is False


@needs_openai
@pytest.mark.asyncio
async def test_batch_match():
    from src.judge import find_matches

    returned = ["Amy works at Facebook", "Bob lives in NYC", "Carol is a CEO"]
    expected = ["Amy is employed by Facebook"]
    matches = await find_matches(returned, expected)
    assert len(matches) == 1
    assert matches[0][0] == "Amy works at Facebook"
    assert matches[0][1] == "Amy is employed by Facebook"


# ── contains mode (comma-split phrase substring, no LLM) ──


def test_contains_match_comma_split_each_piece():
    from src.judge import contains_match

    # TimeQA-style: expected "X , Y", returned has both X and Y in context
    assert contains_match(
        "Hossenfelder completed a research fellowship at the University of Arizona in Tucson after moving to North America.",
        "University of Arizona , Tucson",
    )


def test_contains_match_exact_entity():
    from src.judge import contains_match

    assert contains_match(
        "Hossenfelder completed a research fellowship at the University of California, Santa Barbara.",
        "University of California , Santa Barbara",
    )


def test_contains_match_rejects_wrong_campus():
    """Comma-split avoids false positive: 'University of California' alone isn't enough."""
    from src.judge import contains_match

    # UC Berkeley in returned, but expected is Santa Barbara
    assert not contains_match(
        "Hossenfelder worked at the University of California, Berkeley.",
        "University of California , Santa Barbara",
    )


def test_contains_match_rejects_scattered_tokens():
    """Phrase substring (not bag-of-tokens) avoids false positives on scattered words."""
    from src.judge import contains_match

    # "Arizona" and "Tucson" and "University" all appear but not as phrase
    assert not contains_match(
        "She lived in Tucson and studied Arizona history at a different university.",
        "University of Arizona , Tucson",
    )


def test_contains_match_simple_phrase():
    from src.judge import contains_match

    assert contains_match(
        "Gary Mills coached Grantham Town from 1996 to 1998.",
        "Grantham Town",
    )


def test_contains_match_missing_phrase_fails():
    from src.judge import contains_match

    assert not contains_match(
        "Hossenfelder worked at the University of Arizona.",
        "Stanford University",
    )


def test_contains_match_empty_expected():
    from src.judge import contains_match

    assert not contains_match("anything", "")


def test_contains_match_case_insensitive():
    """Both returned and expected are lowercased before matching."""
    from src.judge import contains_match

    # Mixed case returned, lowercase expected
    assert contains_match(
        "Hossenfelder worked at the UNIVERSITY OF ARIZONA in Tucson.",
        "university of arizona , tucson",
    )
    # Lowercase returned, uppercase expected
    assert contains_match(
        "ucsb is short for university of california, santa barbara.",
        "UNIVERSITY OF CALIFORNIA , SANTA BARBARA",
    )
    # Title case on both
    assert contains_match(
        "The Grantham Town Club Was Coached By Gary Mills.",
        "Grantham Town",
    )


@pytest.mark.asyncio
async def test_facts_match_contains_mode_routes_correctly(monkeypatch):
    monkeypatch.setenv("FACT_MATCH_MODE", "contains")
    from src.judge import facts_match

    # Token-subset match, no LLM call
    assert await facts_match(
        "The University of Arizona in Tucson hosted the event.",
        "University of Arizona , Tucson",
    )
    assert not await facts_match(
        "University of California hosted the event.",
        "University of Arizona , Tucson",
    )


@pytest.mark.asyncio
async def test_facts_match_default_mode_is_llm(monkeypatch):
    """Ensure default behavior unchanged when FACT_MATCH_MODE unset."""
    monkeypatch.delenv("FACT_MATCH_MODE", raising=False)
    from src.judge import facts_match

    # Exact match fast-path (doesn't require LLM)
    assert await facts_match("Amy works at Facebook", "Amy works at Facebook") is True


# ── _looks_like_entity heuristic (auto-routing signal) ──


def test_looks_like_entity_short_no_verb():
    from src.judge import _looks_like_entity

    assert _looks_like_entity("University of Arizona , Tucson")
    assert _looks_like_entity("Grantham Town")
    assert _looks_like_entity("University of California , Santa Barbara")


def test_looks_like_entity_rejects_sentence_with_verb():
    from src.judge import _looks_like_entity

    assert not _looks_like_entity("Amy works at Facebook")
    assert not _looks_like_entity("Amy is employed by Facebook")
    assert not _looks_like_entity("Bob lives in NYC")


def test_looks_like_entity_rejects_long_text():
    """Text over 6 tokens routes to LLM regardless of verb content."""
    from src.judge import _looks_like_entity

    assert not _looks_like_entity("the Grantham Town Football Club of England in 1996")


def test_looks_like_entity_handles_empty():
    from src.judge import _looks_like_entity

    assert not _looks_like_entity("")
    assert not _looks_like_entity("   ")


# ── facts_match auto-routing (llm mode, entity-style expected) ──


@pytest.mark.asyncio
async def test_facts_match_auto_routes_entity_to_contains(monkeypatch):
    """Short entity-style expected should skip LLM and use contains_match.

    Works even without OPENAI_API_KEY because no LLM call is made.
    """
    monkeypatch.delenv("FACT_MATCH_MODE", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from src.judge import facts_match

    # Entity-style expected → contains_match (no LLM) → match succeeds
    assert await facts_match(
        "Hossenfelder completed a fellowship at the University of Arizona in Tucson.",
        "University of Arizona , Tucson",
    )
    # Same routing rejects wrong campus
    assert not await facts_match(
        "Hossenfelder worked at the University of California, Berkeley.",
        "University of California , Santa Barbara",
    )
