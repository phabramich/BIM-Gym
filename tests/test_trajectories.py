from __future__ import annotations

from pathlib import Path

import pytest

from bim_gym.data import JsonlRecordKind, collect_trajectories, iter_jsonl, write_run
from bim_gym.generators import TaskGeneratorConfig, generate
from bim_gym.generators.enums import TaskFamilyKind
from bim_gym.tasks import DOOR_ON_SOUTH_WALL, COLUMN_CLEAR_OF_WALL
from bim_gym.telemetry import VERIFIER_VERSION


def test_collect_trajectories_mixed_tasks(tmp_path: Path) -> None:
    config = TaskGeneratorConfig(seed=11, n_tasks=3)
    generated = generate(config)
    tasks = [DOOR_ON_SOUTH_WALL, COLUMN_CLEAR_OF_WALL, *generated]
    out_path = tmp_path / "run.jsonl"

    trajectories = collect_trajectories(tasks, path=out_path)

    assert len(trajectories) == len(tasks)
    assert out_path.exists()
    assert sum(1 for _ in out_path.open(encoding="utf-8")) == len(tasks)

    records = list(iter_jsonl(out_path))
    assert len(records) == len(tasks)
    for record in records:
        assert record["verifier_version"] == VERIFIER_VERSION

    telemetry_records = list(iter_jsonl(out_path, record_kind=JsonlRecordKind.Telemetry))
    assert len(telemetry_records) == len(tasks)
    assert all(t.verifier_version == VERIFIER_VERSION for t in telemetry_records)


def test_write_run_from_collected_trajectories(tmp_path: Path) -> None:
    tasks = generate(TaskGeneratorConfig(seed=5, n_tasks=2))
    trajectories = collect_trajectories(tasks)
    out_path = tmp_path / "batch.jsonl"

    write_run(trajectories, out_path)

    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        assert f'"verifier_version": "{VERIFIER_VERSION}"' in line


def test_collect_trajectories_static_task_has_zero_actions() -> None:
    trajectories = collect_trajectories([DOOR_ON_SOUTH_WALL])
    assert len(trajectories) == 1
    telemetry = trajectories[0]
    assert telemetry.success is True
    assert telemetry.n_actions == 0
    assert telemetry.turns == []


def test_collect_trajectories_reference_actions_record_turns() -> None:
    trajectories = collect_trajectories([COLUMN_CLEAR_OF_WALL], use_reference_actions=False)
    assert trajectories[0].n_actions == 0

    config = TaskGeneratorConfig(
        seed=99,
        n_tasks=1,
        families=(TaskFamilyKind.Clearance,),
    )
    generated = generate(config)
    assert generated[0].reference_actions

    trajectories = collect_trajectories(generated)
    assert trajectories[0].n_actions > 0
    assert len(trajectories[0].turns) == trajectories[0].n_actions
