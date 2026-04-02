"""Reporter module: aggregate benchmark results and format reports."""

from __future__ import annotations

from collections import defaultdict

from tabulate import tabulate

from src.models import CategoryReport, QueryResult


def aggregate_results(
    results: list[QueryResult],
    phase: str,
    category: str,
) -> list[CategoryReport]:
    """Group results by strategy and compute averaged metrics.

    Returns one CategoryReport per unique strategy found in results.
    """
    by_strategy: dict[str, list[QueryResult]] = defaultdict(list)
    for r in results:
        by_strategy[r.strategy].append(r)

    reports: list[CategoryReport] = []
    for strategy, group in sorted(by_strategy.items()):
        n = len(group)
        reports.append(
            CategoryReport(
                phase=phase,
                category=category,
                strategy=strategy,
                avg_precision_at_5=sum(r.precision_at_5 for r in group) / n,
                avg_recall_at_5=sum(r.recall_at_5 for r in group) / n,
                avg_mrr=sum(r.mrr for r in group) / n,
                temporal_accuracy_pct=sum(1 for r in group if r.temporal_accuracy)
                / n
                * 100,
                num_queries=n,
            )
        )
    return reports


def format_report_table(reports: list[CategoryReport]) -> str:
    """Format a list of CategoryReports as a tabulated string."""
    headers = [
        "Phase",
        "Category",
        "Strategy",
        "P@5",
        "R@5",
        "MRR",
        "Temporal%",
        "Queries",
    ]
    rows = [
        [
            r.phase,
            r.category,
            r.strategy,
            f"{r.avg_precision_at_5:.2f}",
            f"{r.avg_recall_at_5:.2f}",
            f"{r.avg_mrr:.2f}",
            f"{r.temporal_accuracy_pct:.1f}",
            r.num_queries,
        ]
        for r in reports
    ]
    return tabulate(rows, headers=headers, tablefmt="grid", disable_numparse=True)


def print_full_report(all_reports: list[CategoryReport]) -> None:
    """Print reports grouped by phase with a summary."""
    phases: dict[str, list[CategoryReport]] = defaultdict(list)
    for r in all_reports:
        phases[r.phase].append(r)

    for phase, reports in sorted(phases.items()):
        print(f"\n{'='*60}")
        print(f"  Phase: {phase}")
        print(f"{'='*60}")
        print(format_report_table(reports))

    # Summary across all phases
    if all_reports:
        total = len(all_reports)
        print(f"\n{'='*60}")
        print(f"  Summary ({total} category-strategy combinations)")
        print(f"{'='*60}")
        avg_p = sum(r.avg_precision_at_5 for r in all_reports) / total
        avg_r = sum(r.avg_recall_at_5 for r in all_reports) / total
        avg_mrr = sum(r.avg_mrr for r in all_reports) / total
        avg_temp = sum(r.temporal_accuracy_pct for r in all_reports) / total
        print(f"  Avg P@5:      {avg_p:.2f}")
        print(f"  Avg R@5:      {avg_r:.2f}")
        print(f"  Avg MRR:      {avg_mrr:.2f}")
        print(f"  Avg Temporal:  {avg_temp:.1f}%")
