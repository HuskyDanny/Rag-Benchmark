import pytest
from src.models import QueryResult, CategoryReport


def test_aggregate_results():
    from src.reporter import aggregate_results

    results = [
        QueryResult(
            test_case_id="evolving_001",
            query="q1",
            strategy="hybrid",
            returned_facts=[],
            expected_facts=[],
            expected_not=[],
            precision_at_5=0.8,
            recall_at_5=1.0,
            mrr=1.0,
            temporal_accuracy=True,
        ),
        QueryResult(
            test_case_id="evolving_002",
            query="q2",
            strategy="hybrid",
            returned_facts=[],
            expected_facts=[],
            expected_not=[],
            precision_at_5=0.6,
            recall_at_5=0.5,
            mrr=0.5,
            temporal_accuracy=False,
        ),
    ]
    reports = aggregate_results(results, phase="controlled", category="evolving_fact")
    assert len(reports) == 1
    r = reports[0]
    assert r.phase == "controlled"
    assert r.category == "evolving_fact"
    assert r.strategy == "hybrid"
    assert r.avg_precision_at_5 == pytest.approx(0.7)
    assert r.avg_recall_at_5 == pytest.approx(0.75)
    assert r.avg_mrr == pytest.approx(0.75)
    assert r.temporal_accuracy_pct == pytest.approx(50.0)
    assert r.num_queries == 2


def test_format_report_table():
    from src.reporter import format_report_table

    reports = [
        CategoryReport(
            phase="controlled",
            category="evolving_fact",
            strategy="hybrid",
            avg_precision_at_5=0.8,
            avg_recall_at_5=0.9,
            avg_mrr=0.85,
            temporal_accuracy_pct=90.0,
            num_queries=15,
        ),
    ]
    table = format_report_table(reports)
    assert "evolving_fact" in table
    assert "hybrid" in table
    assert "0.80" in table
