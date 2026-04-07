"""Benchmark orchestrator with pluggable experiments, checkpoint/resume, and CLI."""

import argparse
import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from graphiti_core import Graphiti
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.utils.maintenance.graph_data_operations import clear_data

from src.checkpoint import Checkpoint
from src.experiments import RunConfig, get_experiment, list_experiments
from src.models import RunMetadata, TestCase
from src import controlled_inserter, pipeline_inserter, presplit_inserter

load_dotenv()

# ── Phase configuration ──

GROUP_IDS = {
    "controlled": controlled_inserter.GROUP_ID,
    "pipeline": pipeline_inserter.GROUP_ID,
    "pipeline_presplit": presplit_inserter.GROUP_ID,
}

PHASE_PORTS = {
    "controlled": 7687,
    "pipeline": 7688,
    "pipeline_presplit": 7689,
}

INSERTERS = {
    "controlled": controlled_inserter,
    "pipeline": pipeline_inserter,
    "pipeline_presplit": presplit_inserter,
}

# ── Shared infrastructure ──


def load_test_cases(path: str = "data/test_cases.json") -> list[TestCase]:
    with open(path) as f:
        return [TestCase.model_validate(d) for d in json.load(f)]


async def create_graphiti(neo4j_port: int = 7687) -> Graphiti:
    """Create and initialize a Graphiti instance with configurable Neo4j port."""
    uri = f"bolt://localhost:{neo4j_port}"
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model = os.getenv("LLM_MODEL", "Qwen/Qwen3.5-397B-A17B")
    small_model = os.getenv("LLM_SMALL_MODEL", "Qwen/Qwen2.5-7B-Instruct")

    llm_client = OpenAIGenericClient(
        config=LLMConfig(
            api_key=api_key, base_url=base_url, model=model, small_model=small_model
        )
    )
    embedder = OpenAIEmbedder(
        config=OpenAIEmbedderConfig(
            embedding_model=os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B"),
            api_key=api_key,
            base_url=base_url,
        )
    )

    graphiti = Graphiti(uri, user, password, llm_client=llm_client, embedder=embedder)
    await graphiti.build_indices_and_constraints()
    return graphiti


async def wipe_graph(graphiti: Graphiti, group_id: str | None = None) -> None:
    if group_id:
        await clear_data(graphiti.clients.driver, group_ids=[group_id])
    else:
        await clear_data(graphiti.clients.driver)
    await graphiti.build_indices_and_constraints()


# ── Shared insertion (once per phase) ──


async def run_insert(
    graphiti: Graphiti, phase: str, test_cases: list[TestCase]
) -> None:
    """Insert test data with checkpoint/resume support."""
    ckpt = Checkpoint(phase, "insert")
    inserter = INSERTERS[phase]

    print(f"\n--- [{phase}] INSERT stage ---")
    for tc in test_cases:
        if ckpt.is_done(tc.id):
            print(f"  {tc.id} — skipped (checkpoint)")
            continue
        print(f"  {tc.id} — inserting...")
        await inserter.insert_test_case(graphiti, tc)
        ckpt.mark_done(tc.id)

    ckpt.mark_stage_complete()
    print(f"  INSERT complete for {phase}")


# ── Run result persistence ──


def _run_dir(run_id: str) -> Path:
    """Get or create the directory for a run's results."""
    d = Path("results/runs") / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_run_results(run_config: RunConfig, results: list) -> None:
    """Persist raw results for a run."""
    d = _run_dir(run_config.run_id)
    data = []
    for r in results:
        if hasattr(r, "model_dump"):
            data.append(r.model_dump(mode="json"))
        else:
            data.append(r)
    (d / "results.json").write_text(json.dumps(data, indent=2, default=str))


def _load_run_results(run_config: RunConfig) -> list:
    """Load results from disk for a run."""
    d = _run_dir(run_config.run_id)
    results_path = d / "results.json"
    if not results_path.exists():
        # Fallback to legacy cache path for backward compat
        legacy = Path("results/checkpoints") / f"{run_config.phase}_results.json"
        if legacy.exists():
            return json.loads(legacy.read_text())
        return []
    experiment = get_experiment(run_config.experiment_type)
    raw = json.loads(results_path.read_text())
    if hasattr(experiment.result_model, "model_validate"):
        return [experiment.result_model.model_validate(d) for d in raw]
    return raw


def _save_run_report(run_config: RunConfig, reports: list) -> None:
    """Persist aggregated reports for a run."""
    d = _run_dir(run_config.run_id)
    data = []
    for r in reports:
        if hasattr(r, "model_dump"):
            data.append(r.model_dump(mode="json"))
        else:
            data.append(r)
    (d / "report.json").write_text(json.dumps(data, indent=2, default=str))


def _save_run_metadata(run_config: RunConfig, started_at: datetime) -> None:
    """Persist run metadata."""
    d = _run_dir(run_config.run_id)
    meta = RunMetadata(
        run_id=run_config.run_id,
        experiment_type=run_config.experiment_type,
        phase=run_config.phase,
        params=run_config.params,
        started_at=started_at,
        completed_at=datetime.now(timezone.utc),
    )
    (d / "metadata.json").write_text(meta.model_dump_json(indent=2))


# ── Legacy backward-compat: results cache for --stage report ──


def _save_legacy_results_cache(phase: str, results: list) -> None:
    """Save to legacy path so standalone --stage report still works."""
    cache = Path("results/checkpoints") / f"{phase}_results.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    data = [
        r.model_dump(mode="json") if hasattr(r, "model_dump") else r for r in results
    ]
    cache.write_text(json.dumps(data, indent=2))


# ── Orchestrator ──

LEGACY_STAGES = ("insert", "evaluate", "report")


async def run_benchmark(
    phase: str,
    port: int | None = None,
    stage: str | None = None,
    clean: bool = False,
    experiments: list[str] | None = None,
    params: dict | None = None,
    run_id: str | None = None,
) -> None:
    """Run benchmark: shared insertion + pluggable experiments."""
    if port is None:
        port = PHASE_PORTS.get(phase, 7687)

    # Default to retrieval for backward compat
    experiment_names = experiments or ["retrieval"]
    user_params = params or {}

    test_cases = load_test_cases()
    group_id = GROUP_IDS[phase]
    print(f"Loaded {len(test_cases)} test cases | phase={phase} port={port}")

    graphiti = await create_graphiti(neo4j_port=port)

    try:
        if clean:
            print("Cleaning checkpoint + graph data...")
            Checkpoint(phase, "insert").clear()
            await wipe_graph(graphiti, group_id)

        # ── Shared insertion (once per phase) ──
        should_insert = stage is None or stage == "insert"
        if should_insert:
            insert_ckpt = Checkpoint(phase, "insert")
            if insert_ckpt.load()["status"] != "completed":
                await run_insert(graphiti, phase, test_cases)
            else:
                print(f"  [{phase}] INSERT already complete (checkpoint)")

        # ── Run each experiment ──
        should_measure = stage is None or stage == "evaluate"
        should_report = stage is None or stage == "report"

        for exp_name in experiment_names:
            experiment = get_experiment(exp_name)
            effective_params = {**experiment.default_params(), **user_params}
            experiment.validate_params(effective_params)

            started_at = datetime.now(timezone.utc)
            rc = RunConfig(
                experiment_type=exp_name,
                phase=phase,
                group_id=group_id,
                params=effective_params,
                run_id=run_id or "",
            )

            print(f"\n=== Experiment: {exp_name} | Run: {rc.run_id} ===")

            results = []

            if should_measure:
                # Use run_id for experiment checkpoint, None for legacy compat
                ckpt_run_id = rc.run_id if run_id else None
                ckpt = Checkpoint(phase, exp_name, run_id=ckpt_run_id)
                if clean:
                    ckpt.clear()

                print(f"\n--- [{phase}] {exp_name.upper()} MEASURE ---")
                results = await experiment.measure(graphiti, test_cases, rc, ckpt)

                # Persist results
                _save_run_results(rc, results)
                # Legacy compat: also save to old path for retrieval experiment
                if exp_name == "retrieval":
                    _save_legacy_results_cache(phase, results)

                print(f"  {exp_name} measure complete ({len(results)} results)")

            if should_report:
                if not results:
                    results = _load_run_results(rc)
                    if results:
                        print(f"  Loaded {len(results)} results from cache")

                if not results:
                    print(f"  No results for {exp_name} — skipping report")
                    continue

                print(f"\n--- [{phase}] {exp_name.upper()} REPORT ---")
                reports = experiment.report(results, test_cases, rc)
                experiment.print_report(reports)
                _save_run_report(rc, reports)
                _save_run_metadata(rc, started_at)

                # Legacy compat: also save timestamped report for retrieval
                if exp_name == "retrieval":
                    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                    legacy_path = f"results/{phase}_{ts}.json"
                    data = [r.model_dump(mode="json") for r in reports]
                    Path(legacy_path).write_text(json.dumps(data, indent=2))
                    print(f"  Report saved to {legacy_path}")

    finally:
        await graphiti.close()


def _parse_params(param_list: list[str] | None) -> dict:
    """Parse --param key=value pairs into a dict."""
    if not param_list:
        return {}
    params = {}
    for item in param_list:
        if "=" not in item:
            raise ValueError(f"Invalid param format: '{item}'. Expected key=value.")
        key, val = item.split("=", 1)
        # Auto-convert numeric values
        try:
            params[key] = float(val) if "." in val else int(val)
        except ValueError:
            params[key] = val
    return params


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Graphiti benchmark runner with pluggable experiments"
    )
    parser.add_argument(
        "phase",
        nargs="?",
        choices=list(GROUP_IDS.keys()),
        help="Benchmark phase to run",
    )
    parser.add_argument(
        "--stage",
        choices=list(LEGACY_STAGES),
        default=None,
        help="Run only this stage (default: all stages)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Neo4j bolt port (default: phase-specific from PHASE_PORTS)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Wipe checkpoint + graph data before running",
    )
    parser.add_argument(
        "--experiment",
        "-e",
        action="append",
        default=None,
        help="Experiment type(s) to run (default: retrieval). Repeatable.",
    )
    parser.add_argument(
        "--param",
        "-p",
        action="append",
        default=None,
        help="Parameter overrides as key=value (e.g., --param mmr_lambda=0.3)",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Explicit run ID (auto-generated if omitted)",
    )
    parser.add_argument(
        "--list-experiments",
        action="store_true",
        help="List available experiment types and exit",
    )
    return parser


def main():
    args = build_cli().parse_args()

    if args.list_experiments:
        print("Available experiments:")
        for name in list_experiments():
            exp = get_experiment(name)
            print(f"  {name}: params={exp.default_params()}")
        return

    if not args.phase:
        build_cli().print_help()
        return

    asyncio.run(
        run_benchmark(
            phase=args.phase,
            port=args.port,
            stage=args.stage,
            clean=args.clean,
            experiments=args.experiment,
            params=_parse_params(args.param),
            run_id=args.run_id,
        )
    )


if __name__ == "__main__":
    main()
