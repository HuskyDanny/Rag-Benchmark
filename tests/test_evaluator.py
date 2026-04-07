import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_precision_at_k_perfect():
    from src.evaluator import compute_precision_at_k

    with patch("src.evaluator.find_matches", new_callable=AsyncMock) as mock:
        mock.return_value = [("A", "A"), ("B", "B")]
        result = await compute_precision_at_k(
            returned=["A", "B", "C", "D", "E"], expected=["A", "B"], k=5
        )
        assert result == 2 / 5


@pytest.mark.asyncio
async def test_recall_at_k_perfect():
    from src.evaluator import compute_recall_at_k

    with patch("src.evaluator.find_matches", new_callable=AsyncMock) as mock:
        mock.return_value = [("A", "A"), ("B", "B")]
        result = await compute_recall_at_k(
            returned=["A", "B", "C"], expected=["A", "B"], k=5
        )
        assert result == 1.0


@pytest.mark.asyncio
async def test_recall_at_k_partial():
    from src.evaluator import compute_recall_at_k

    with patch("src.evaluator.find_matches", new_callable=AsyncMock) as mock:
        mock.return_value = [("A", "A")]
        result = await compute_recall_at_k(
            returned=["A", "C"], expected=["A", "B"], k=5
        )
        assert result == 0.5


@pytest.mark.asyncio
async def test_mrr_first_result():
    from src.evaluator import compute_mrr

    with patch("src.evaluator.facts_match", new_callable=AsyncMock) as mock:
        mock.side_effect = lambda r, e: r == "A"
        result = await compute_mrr(returned=["A", "B", "C"], expected=["A"])
        assert result == 1.0


@pytest.mark.asyncio
async def test_mrr_second_result():
    from src.evaluator import compute_mrr

    async def mock_match(r, e):
        return r == "B" and e == "A"

    with patch("src.evaluator.facts_match", side_effect=mock_match):
        result = await compute_mrr(returned=["X", "B", "C"], expected=["A"])
        assert result == 0.5


@pytest.mark.asyncio
async def test_temporal_accuracy_pass():
    from src.evaluator import compute_temporal_accuracy

    with patch("src.evaluator.any_match", new_callable=AsyncMock) as mock:
        mock.return_value = []
        result = await compute_temporal_accuracy(
            returned=["Amy works at Facebook"], expected_not=["Amy works at Google"]
        )
        assert result is True


@pytest.mark.asyncio
async def test_temporal_accuracy_fail():
    from src.evaluator import compute_temporal_accuracy

    with patch("src.evaluator.any_match", new_callable=AsyncMock) as mock:
        mock.return_value = ["Amy works at Google"]
        result = await compute_temporal_accuracy(
            returned=["Amy works at Facebook", "Amy works at Google"],
            expected_not=["Amy works at Google"],
        )
        assert result is False
