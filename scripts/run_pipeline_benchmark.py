"""Run full pipeline benchmark with checkpoint resume support."""

import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on sys.path (allows running from any directory)
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))
os.chdir(_project_root)

from dotenv import load_dotenv

load_dotenv()

from src.benchmark_runner import (
    create_graphiti,
    load_test_cases,
    run_insert,
    GROUP_IDS,
    _save_run_results,
    _save_run_report,
    _save_run_metadata,
)
from src.experiments import get_experiment, RunConfig
from src.checkpoint import Checkpoint


async def main():
    test_cases = load_test_cases()
    phase = "pipeline"
    port = 7688
    group_id = GROUP_IDS[phase]

    print(f"=== PIPELINE BENCHMARK ({len(test_cases)} cases) ===")

    g = await create_graphiti(neo4j_port=port)
    try:
        # Step 1: Insert (RESUME — do NOT clear checkpoint)
        await run_insert(g, phase, test_cases)

        # Step 2: Ingestion experiment
        print("\n=== INGESTION EXPERIMENT ===")
        ing_exp = get_experiment("ingestion")
        rc_ing = RunConfig("ingestion", phase, group_id, run_id="ingestion_pipeline")
        ckpt_ing = Checkpoint(phase, "ingestion", run_id="ingestion_pipeline")
        ckpt_ing.clear()
        started = datetime.now(timezone.utc)
        ing_results = await ing_exp.measure(g, test_cases, rc_ing, ckpt_ing)
        _save_run_results(rc_ing, ing_results)
        ing_reports = ing_exp.report(ing_results, test_cases, rc_ing)
        ing_exp.print_report(ing_reports)
        _save_run_report(rc_ing, ing_reports)
        _save_run_metadata(rc_ing, started)

        # Step 3: Retrieval experiment
        print("\n=== RETRIEVAL EXPERIMENT ===")
        ret_exp = get_experiment("retrieval")
        rc_ret = RunConfig("retrieval", phase, group_id, run_id="retrieval_pipeline")
        ckpt_ret = Checkpoint(phase, "retrieval", run_id="retrieval_pipeline")
        ckpt_ret.clear()
        started = datetime.now(timezone.utc)
        ret_results = await ret_exp.measure(g, test_cases, rc_ret, ckpt_ret)
        _save_run_results(rc_ret, ret_results)
        ret_reports = ret_exp.report(ret_results, test_cases, rc_ret)
        ret_exp.print_report(ret_reports)
        _save_run_report(rc_ret, ret_reports)
        _save_run_metadata(rc_ret, started)

    finally:
        await g.close()

    print("\n=== PIPELINE BENCHMARK COMPLETE ===")


asyncio.run(main())
