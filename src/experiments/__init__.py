"""Experiment base class and registry for pluggable benchmark measurements."""

from __future__ import annotations

import abc
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class RunConfig:
    """Identifies a single experiment run."""

    experiment_type: str  # "retrieval", "ingestion", "search_tuning"
    phase: str  # "controlled", "pipeline", "pipeline_presplit"
    group_id: str  # Neo4j group_id for this phase
    params: dict[str, Any] = field(default_factory=dict)
    run_id: str = ""  # auto-generated if empty

    def __post_init__(self):
        if not self.run_id:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            self.run_id = f"{self.experiment_type}_{self.phase}_{ts}"


class ExperimentBase(abc.ABC):
    """Base class all experiments implement."""

    name: str = ""  # e.g., "retrieval", "ingestion", "search_tuning"
    result_model: type = object  # concrete result type for deserialization
    report_model: type = object  # concrete report type for deserialization

    @abc.abstractmethod
    def default_params(self) -> dict[str, Any]:
        """Return default parameters for this experiment type."""
        ...

    def validate_params(self, params: dict[str, Any]) -> None:
        """Validate params before measure(). Raise ValueError if invalid.

        Default: no-op. Override in subclass to add validation.
        """

    @abc.abstractmethod
    async def measure(
        self,
        graphiti: Any,
        test_cases: Any,
        run_config: RunConfig,
        checkpoint: Any,
    ) -> list:
        """Run measurement logic. Returns a list of result models."""
        ...

    @abc.abstractmethod
    def report(self, results: Any, test_cases: Any, run_config: RunConfig) -> list:
        """Aggregate results and produce reports."""
        ...

    def print_report(self, reports: Any) -> None:
        """Print human-readable report. Default: JSON dump."""
        from pydantic import BaseModel

        if reports and isinstance(reports[0], BaseModel):
            print(json.dumps([r.model_dump(mode="json") for r in reports], indent=2))
        else:
            print(json.dumps(reports, indent=2, default=str))


# ── Registry ──

_REGISTRY: dict[str, type[ExperimentBase]] = {}


def register_experiment(cls: type[ExperimentBase]) -> type[ExperimentBase]:
    """Class decorator to register an experiment type."""
    _REGISTRY[cls.name] = cls
    return cls


def get_experiment(name: str) -> ExperimentBase:
    """Instantiate a registered experiment by name."""
    if name not in _REGISTRY:
        available = list(_REGISTRY.keys())
        raise ValueError(f"Unknown experiment '{name}'. Available: {available}")
    return _REGISTRY[name]()


def list_experiments() -> list[str]:
    """Return names of all registered experiments."""
    return sorted(_REGISTRY.keys())


# Auto-register built-in experiments on import
from src.experiments import retrieval  # noqa: F401, E402
from src.experiments import ingestion  # noqa: F401, E402
from src.experiments import search_tuning  # noqa: F401, E402
