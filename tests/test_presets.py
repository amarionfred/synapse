"""Immutable lineage checks without model inference or checkpoint assets."""
from copy import deepcopy
from pathlib import Path
import sqlite3
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "explorer"))
from presets import PresetStore, normalize_pushes


IDENTITY = {"checkpoint": {"model_key": "fixture", "revision": "one"},
            "runtime": {"backend": "fixture", "implementation": "research-v1"},
            "calibration": {"reference_hash": "fixed", "norms": {"0": 1.0}}}


def test_content_identity_is_stable_and_returned_objects_are_detached(tmp_path):
    identity = deepcopy(IDENTITY)
    store = PresetStore(tmp_path, identity)
    first = store.save([(1, 2, .5, "add"), (0, 1, .4, "mul")], label="v1")
    second = store.save([(0, 1, .4, "mul"), (1, 2, .5, "add")], label="v1")
    assert first == second
    identity["checkpoint"]["model_key"] = "changed"
    first["config"]["pushes"].clear()
    assert store.replay(second["id"])["pushes"] == [[0, 1, .4, "mul"], [1, 2, .5, "add"]]
    assert store.get(second["id"])["identity"] == IDENTITY
    store.close()
    reopened = PresetStore(tmp_path, IDENTITY)
    assert reopened.get(second["id"]) == second
    assert reopened.path == tmp_path / "research.sqlite3"


def test_ancestry_evaluation_compare_compose_and_restore(tmp_path):
    store = PresetStore(tmp_path, IDENTITY)
    base = store.save([], label="v0")
    a = store.save([(0, 1, .2, "add")], parents=[base["id"]],
                   evaluation={"job_id": "job-a", "split": "validation", "correct": 3, "total": 4})
    b = store.save([(1, 2, -.3, "add")], parents=[base["id"]])
    combined = store.compose([a["id"], b["id"]])
    ancestry = store.ancestry(combined["id"])
    assert {row["id"] for row in ancestry} == {base["id"], a["id"], b["id"], combined["id"]}
    assert ancestry[0]["id"] == base["id"] and ancestry[-1]["id"] == combined["id"]
    assert store.compare(a["id"], combined["id"])["compatible"] is True
    assert store.compare(a["id"], combined["id"])["same_config"] is False
    assert store.restore(a["id"]) == {"pushes": [[0, 1, .2, "add"]]}
    assert store.restore() == {"pushes": []}
    assert store.get(a["id"])["evaluation"]["job_id"] == "job-a"


@pytest.mark.parametrize("component", ["checkpoint", "runtime", "calibration"])
def test_replay_rejects_each_incompatible_identity_component(tmp_path, component):
    store = PresetStore(tmp_path, IDENTITY)
    preset = store.save([(0, 1, .2, "add")])
    incompatible = deepcopy(IDENTITY)
    incompatible[component]["changed"] = True
    with pytest.raises(ValueError, match="incompatible"):
        store.replay(preset["id"], identity=incompatible)
    other = PresetStore(tmp_path, incompatible)
    with pytest.raises(ValueError, match="incompatible"):
        other.save([], parents=[preset["id"]])
    assert other.compare(preset["id"], preset["id"])["same_config"]


@pytest.mark.parametrize("second", [[(0, 1, .3, "add")], [(0, None, 0, "mul")], [(0, 1, .2, "mul")]])
def test_overlapping_compositions_fail_explicitly(tmp_path, second):
    store = PresetStore(tmp_path, IDENTITY)
    a = store.save([(0, 1, .2, "add")])
    b = store.save(second)
    with pytest.raises(ValueError, match="Overlapping"):
        store.compose([a["id"], b["id"]])
    assert len(store.list()) == 2


def test_immutable_sql_table_blocks_update_and_delete(tmp_path):
    store = PresetStore(tmp_path, IDENTITY)
    record = store.save([])
    with pytest.raises(sqlite3.IntegrityError, match="immutable preset"):
        store._db.execute("UPDATE research_presets SET payload='{}' WHERE id=?", (record["id"],))
    store._db.rollback()
    with pytest.raises(sqlite3.IntegrityError, match="immutable preset"):
        store._db.execute("DELETE FROM research_presets WHERE id=?", (record["id"],))
    store._db.rollback()
    assert store.get(record["id"])["config"] == {"pushes": []}


@pytest.mark.parametrize("pushes", [[(0, 1, float("nan"), "add")], [(True, 1, 1., "add")],
                                   [(0, None, 1., "add")], [(0, -1, 1., "mul")],
                                   [(0, 1, .2, "add"), (0, 1, .3, "add")]])
def test_invalid_interventions_are_not_recorded(pushes, tmp_path):
    store = PresetStore(tmp_path, IDENTITY)
    with pytest.raises(ValueError):
        store.save(pushes)
    assert store.list() == []


def test_unknown_parent_and_nonfinite_provenance_fail(tmp_path):
    store = PresetStore(tmp_path, IDENTITY)
    with pytest.raises(KeyError):
        store.save([], parents=["0" * 64])
    with pytest.raises(ValueError):
        store.save([], evaluation={"score": float("inf")})
    with pytest.raises(ValueError):
        store.save([], parents=[{}])
    assert store.list() == []
