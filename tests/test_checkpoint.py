import json
import pytest
from pathlib import Path


@pytest.fixture
def tmp_checkpoint_dir(tmp_path):
    return tmp_path / "checkpoints"


def test_load_empty_returns_default(tmp_checkpoint_dir):
    from src.checkpoint import Checkpoint

    cp = Checkpoint("controlled", "insert", base_dir=str(tmp_checkpoint_dir))
    state = cp.load()
    assert state == {"completed": [], "status": "pending"}


def test_save_and_load_roundtrip(tmp_checkpoint_dir):
    from src.checkpoint import Checkpoint

    cp = Checkpoint("controlled", "insert", base_dir=str(tmp_checkpoint_dir))
    cp.mark_done("static_001")
    cp.mark_done("static_002")
    state = cp.load()
    assert "static_001" in state["completed"]
    assert "static_002" in state["completed"]
    assert state["status"] == "in_progress"


def test_is_done(tmp_checkpoint_dir):
    from src.checkpoint import Checkpoint

    cp = Checkpoint("controlled", "insert", base_dir=str(tmp_checkpoint_dir))
    cp.mark_done("static_001")
    assert cp.is_done("static_001") is True
    assert cp.is_done("static_002") is False


def test_clear(tmp_checkpoint_dir):
    from src.checkpoint import Checkpoint

    cp = Checkpoint("controlled", "insert", base_dir=str(tmp_checkpoint_dir))
    cp.mark_done("static_001")
    cp.clear()
    state = cp.load()
    assert state["completed"] == []


def test_mark_complete(tmp_checkpoint_dir):
    from src.checkpoint import Checkpoint

    cp = Checkpoint("controlled", "insert", base_dir=str(tmp_checkpoint_dir))
    cp.mark_done("static_001")
    cp.mark_stage_complete()
    state = cp.load()
    assert state["status"] == "completed"


def test_atomic_write_survives(tmp_checkpoint_dir):
    from src.checkpoint import Checkpoint

    cp = Checkpoint("controlled", "insert", base_dir=str(tmp_checkpoint_dir))
    cp.mark_done("case_1")
    assert cp.path.exists()
    data = json.loads(cp.path.read_text())
    assert "case_1" in data["completed"]
