import os

import pytest
from dotenv import load_dotenv

load_dotenv()

needs_llm = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)


def test_simple_sentence_not_compound():
    from src.sentence_splitter import is_likely_compound

    assert is_likely_compound("Amy works at Google.") is False
    assert is_likely_compound("Paris is the capital of France.") is False


def test_compound_detection():
    from src.sentence_splitter import is_likely_compound

    assert is_likely_compound("Amy left Google and joined Facebook.") is True
    assert is_likely_compound("Bob sold his house, then moved to SF.") is True


@needs_llm
@pytest.mark.asyncio
async def test_split_compound_sentence():
    from src.sentence_splitter import split_into_atomic_facts

    facts = await split_into_atomic_facts(
        "Amy left Google and joined Facebook as a tech lead."
    )
    assert len(facts) >= 2
    combined = " ".join(facts).lower()
    assert "google" in combined
    assert "facebook" in combined


@needs_llm
@pytest.mark.asyncio
async def test_split_simple_passthrough():
    from src.sentence_splitter import split_into_atomic_facts

    facts = await split_into_atomic_facts("Amy works at Google.")
    assert len(facts) == 1
    assert "google" in facts[0].lower()


@needs_llm
@pytest.mark.asyncio
async def test_split_noisy_text():
    from src.sentence_splitter import split_into_atomic_facts

    facts = await split_into_atomic_facts("amy workz at gogle as enginere")
    assert len(facts) >= 1
    combined = " ".join(facts).lower()
    assert "google" in combined or "gogle" in combined
