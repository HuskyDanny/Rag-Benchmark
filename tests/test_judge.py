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


# ── contains mode (token-subset, no LLM) ──


def test_contains_match_punctuation_invariant():
    from src.judge import contains_match

    # TimeQA-style: expected has "X , Y", returned has "X in Y"
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


def test_contains_match_missing_token_fails():
    from src.judge import contains_match

    # Expected has "Stanford" but returned doesn't mention it
    assert not contains_match(
        "Hossenfelder worked at the University of Arizona.",
        "Stanford University",
    )


def test_contains_match_empty_expected():
    from src.judge import contains_match

    assert not contains_match("anything", "")


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
