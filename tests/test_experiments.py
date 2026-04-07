"""Tests for the experiment framework: base class, registry, and RunConfig."""

from __future__ import annotations

import pytest

from src.experiments import (
    ExperimentBase,
    RunConfig,
    get_experiment,
    list_experiments,
    register_experiment,
)


# ── RunConfig ──


def test_run_config_auto_generates_run_id():
    rc = RunConfig(experiment_type="test", phase="controlled", group_id="g1")
    assert rc.run_id.startswith("test_controlled_")
    assert len(rc.run_id) > len("test_controlled_")


def test_run_config_preserves_explicit_run_id():
    rc = RunConfig(
        experiment_type="test",
        phase="pipeline",
        group_id="g1",
        run_id="my_run",
    )
    assert rc.run_id == "my_run"


def test_run_config_default_params():
    rc = RunConfig(experiment_type="t", phase="p", group_id="g")
    assert rc.params == {}


# ── Registry ──


def test_list_experiments_includes_builtins():
    names = list_experiments()
    assert "retrieval" in names
    assert "ingestion" in names
    assert "search_tuning" in names


def test_get_experiment_returns_instance():
    exp = get_experiment("retrieval")
    assert isinstance(exp, ExperimentBase)
    assert exp.name == "retrieval"


def test_get_experiment_unknown_raises():
    with pytest.raises(ValueError, match="Unknown experiment"):
        get_experiment("nonexistent")


def test_retrieval_default_params():
    exp = get_experiment("retrieval")
    params = exp.default_params()
    assert "strategies" in params
    assert "hybrid" in params["strategies"]


def test_ingestion_default_params():
    exp = get_experiment("ingestion")
    assert exp.default_params() == {}


def test_search_tuning_default_params():
    exp = get_experiment("search_tuning")
    params = exp.default_params()
    assert "mmr_lambda" in params
    assert "sim_min_score" in params
    assert "bfs_max_depth" in params


# ── Validate params ──


def test_search_tuning_validate_params_rejects_bad_mmr():
    exp = get_experiment("search_tuning")
    with pytest.raises(ValueError, match="mmr_lambda"):
        exp.validate_params({"mmr_lambda": 5.0})


def test_search_tuning_validate_params_accepts_valid():
    exp = get_experiment("search_tuning")
    exp.validate_params({"mmr_lambda": 0.3, "sim_min_score": 0.8})


# ── Result/report model ──


def test_retrieval_result_model():
    from src.models import QueryResult, CategoryReport

    exp = get_experiment("retrieval")
    assert exp.result_model is QueryResult
    assert exp.report_model is CategoryReport


def test_ingestion_result_model():
    from src.models import IngestionResult, IngestionReport

    exp = get_experiment("ingestion")
    assert exp.result_model is IngestionResult
    assert exp.report_model is IngestionReport
