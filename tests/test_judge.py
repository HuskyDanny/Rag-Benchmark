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
