"""TimeQA benchmark runner — external validation of Graphiti on real Wikipedia data.

Loads TimeQA human_annotated_test.json, converts docs to TestCase via the
adapter, ingests via add_episode, and runs the existing retrieval experiment
(hybrid/bm25_only/cosine_only) with LLM-judged P@5, R@5, MRR, TempAcc.

Usage:
    python scripts/run_timeqa_benchmark.py --max-docs 10 --port 7688
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))
os.chdir(_project_root)

# TimeQA expected_facts are short entity names; Graphiti returns long
# edge-fact sentences. Use token-subset containment instead of LLM judge.
os.environ.setdefault("FACT_MATCH_MODE", "contains")

from dotenv import load_dotenv  # noqa: E402
from graphiti_core.nodes import EpisodeType  # noqa: E402

from src.benchmark_runner import create_graphiti, wipe_graph  # noqa: E402
from src.checkpoint import Checkpoint  # noqa: E402
from src.experiments import RunConfig, get_experiment  # noqa: E402
from src.timeqa_adapter import load_timeqa_testcases  # noqa: E402

load_dotenv()

TIMEQA_GROUP_ID = "timeqa_pipeline"


async def insert_timeqa_testcases(graphiti, test_cases, checkpoint: Checkpoint) -> None:
    """Insert TimeQA episodes via add_episode under a dedicated group_id."""
    print(f"\n--- [timeqa] INSERT stage ({len(test_cases)} cases) ---")
    for tc in test_cases:
        if checkpoint.is_done(tc.id):
            print(f"  {tc.id} — skipped (checkpoint)")
            continue
        print(f"  {tc.id} — inserting ({len(tc.episodes)} episodes)...")
        for episode in sorted(tc.episodes, key=lambda e: e.order):
            await graphiti.add_episode(
                name=f"{tc.id}_ep{episode.order}",
                episode_body=episode.text,
                source=EpisodeType.text,
                source_description="timeqa wikipedia passage",
                reference_time=episode.reference_time,
                group_id=TIMEQA_GROUP_ID,
            )
        checkpoint.mark_done(tc.id)
    checkpoint.mark_stage_complete()
    print("  INSERT complete for timeqa")


async def run_timeqa_benchmark(
    data_path: str,
    max_docs: int | None,
    port: int,
    clean: bool,
    stage: str | None,
) -> None:
    print(f"Loading TimeQA from {data_path} (max_docs={max_docs})")
    test_cases = load_timeqa_testcases(data_path, max_docs=max_docs)
    total_qs = sum(len(tc.queries) for tc in test_cases)
    print(f"Loaded {len(test_cases)} test cases, {total_qs} queries")

    graphiti = await create_graphiti(neo4j_port=port)

    try:
        if clean:
            print("Cleaning checkpoint + graph data...")
            Checkpoint("timeqa", "insert").clear()
            await wipe_graph(graphiti, TIMEQA_GROUP_ID)

        # ── Insert ──
        if stage is None or stage == "insert":
            ckpt = Checkpoint("timeqa", "insert")
            if ckpt.load()["status"] != "completed":
                await insert_timeqa_testcases(graphiti, test_cases, ckpt)
            else:
                print("  [timeqa] INSERT already complete (checkpoint)")

        # ── Retrieval experiment ──
        if stage is None or stage == "evaluate":
            exp = get_experiment("retrieval")
            rc = RunConfig(
                run_id=f"timeqa_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
                experiment_type="retrieval",
                phase="timeqa",
                group_id=TIMEQA_GROUP_ID,
                params={},
            )
            eval_ckpt = Checkpoint("timeqa", "evaluate")
            print("\n--- [timeqa] RETRIEVAL evaluation ---")
            results = await exp.measure(graphiti, test_cases, rc, eval_ckpt)
            print(f"  {len(results)} query evaluations")

            print("\n--- [timeqa] REPORT ---")
            reports = exp.report(results, test_cases, rc)
            exp.print_report(reports)

            # Save results
            out_dir = Path("results/runs") / rc.run_id
            out_dir.mkdir(parents=True, exist_ok=True)
            import json

            (out_dir / "results.json").write_text(
                json.dumps(
                    [r.model_dump(mode="json") for r in results], indent=2, default=str
                )
            )
            (out_dir / "report.json").write_text(
                json.dumps(
                    [r.model_dump(mode="json") for r in reports], indent=2, default=str
                )
            )
            print(f"\n  Saved to {out_dir}")
    finally:
        await graphiti.close()


def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="TimeQA benchmark runner")
    p.add_argument(
        "--data",
        default="data/timeqa_human_annotated_test.json",
        help="Path to TimeQA JSON file",
    )
    p.add_argument(
        "--max-docs",
        type=int,
        default=10,
        help="Max TimeQA documents to load (default: 10 for smoke test)",
    )
    p.add_argument(
        "--port",
        type=int,
        default=7690,
        help="Neo4j bolt port (default: 7690, shared Neo4j instance)",
    )
    p.add_argument(
        "--clean",
        action="store_true",
        help="Wipe checkpoint + graph before running",
    )
    p.add_argument(
        "--stage",
        choices=("insert", "evaluate"),
        default=None,
        help="Run only this stage",
    )
    return p


def main():
    args = build_cli().parse_args()
    asyncio.run(
        run_timeqa_benchmark(
            data_path=args.data,
            max_docs=args.max_docs,
            port=args.port,
            clean=args.clean,
            stage=args.stage,
        )
    )


if __name__ == "__main__":
    main()
