"""Checkpoint persistence for resumable benchmark runs."""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_DIR = "results/checkpoints"


class Checkpoint:
    """Track completion state for a benchmark stage."""

    def __init__(
        self,
        phase: str,
        stage: str,
        run_id: str | None = None,
        base_dir: str = DEFAULT_DIR,
    ):
        self._dir = Path(base_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        # run_id adds isolation: phase_stage_runid.json
        # Without run_id: legacy format phase_stage.json (backward compat)
        name = f"{phase}_{stage}" if run_id is None else f"{phase}_{stage}_{run_id}"
        self.path = self._dir / f"{name}.json"
        self._state: dict | None = None

    def load(self) -> dict:
        if self._state is not None:
            return self._state
        if self.path.exists():
            self._state = json.loads(self.path.read_text())
        else:
            self._state = {"completed": [], "status": "pending"}
        return self._state

    def _save(self) -> None:
        state = self.load()
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2))
        tmp.rename(self.path)

    def mark_done(self, key: str) -> None:
        state = self.load()
        if key not in state["completed"]:
            state["completed"].append(key)
        state["status"] = "in_progress"
        self._save()

    def is_done(self, key: str) -> bool:
        return key in self.load()["completed"]

    def mark_stage_complete(self) -> None:
        self.load()["status"] = "completed"
        self._save()

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()
        self._state = {"completed": [], "status": "pending"}
