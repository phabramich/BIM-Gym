from __future__ import annotations

import json
from pathlib import Path

from bim_gym.dsl.actions import assert_predicates, move, query
from bim_gym.dsl.enums import QueryOp
from bim_gym.generators import TaskGeneratorConfig, generate
from bim_gym.sft.builder import build_sft_example
from bim_gym.sft.export import export_jsonl
from bim_gym.sft.models import SftDatasetConfig, SftExample, SftMessage, SftMeta
from bim_gym.sft.enums import SftMessageRole
from bim_gym.sft.validate import (
    ValidationCheckKind,
    _check_assert_goals,
    _check_executable,
    _check_leak,
    _check_schema,
    _check_example_deep,
    validate_dataset,
    validate_jsonl,
)

def test_validate_dataset_passes_without_hard_fails() -> None:
    config = SftDatasetConfig(seed=0, n_tasks=16, error_recovery_rate=0.0)
    report = validate_dataset(config)
    assert report.n == 16
    assert report.ok
    assert report.hard_fail_count == 0
    assert report.metrics["family_balance"]
    assert "opening_shift" in report.metrics["family_balance"]
    assert report.metrics["action_kind_histogram"]


def test_validate_dataset_no_clearance_leaks() -> None:
    config = SftDatasetConfig(seed=0, n_tasks=16, error_recovery_rate=0.0)
    report = validate_dataset(config)
    leak_ids = [failure.id for failure in report.failures if failure.check == ValidationCheckKind.Leak]
    assert leak_ids == []


def test_validate_jsonl_round_trip(tmp_path: Path) -> None:
    config = SftDatasetConfig(seed=3, n_tasks=12, error_recovery_rate=0.0)
    from bim_gym.sft.builder import build_dataset

    examples = build_dataset(config)
    path = tmp_path / "sft.jsonl"
    export_jsonl(examples, path)
    report = validate_jsonl(str(path), config)
    assert report.n == 12
    assert report.ok
    assert report.hard_fail_count == 0


def test_self_contained_jsonl_does_not_depend_on_regeneration_seed(tmp_path: Path) -> None:
    config = SftDatasetConfig(seed=23, n_tasks=25)
    from bim_gym.sft.builder import build_dataset

    path = tmp_path / "self_contained.jsonl"
    export_jsonl(build_dataset(config), path)
    report = validate_jsonl(
        str(path),
        SftDatasetConfig(seed=999, n_tasks=1, strict_self_contained=True),
    )
    assert report.ok
    assert report.metrics["self_contained_rate"] == 1.0
    assert report.metrics["prompt_collision_count"] == 0


def test_check_schema_rejects_bad_message_count() -> None:
    example = SftExample(
        id="sft_test",
        messages=[
            SftMessage(role=SftMessageRole.System, content="sys"),
            SftMessage(role=SftMessageRole.User, content="user"),
        ],
        meta=SftMeta(task_id="test"),
    )
    failures = _check_schema(example)
    assert failures
    assert failures[0].check == ValidationCheckKind.Schema


def test_check_assert_goals_mismatch() -> None:
    tasks = generate(TaskGeneratorConfig(seed=0, n_tasks=5))
    task = tasks[0]
    wrong_assert = assert_predicates([])
    error = _check_assert_goals([wrong_assert], task.goals)
    assert error is not None


def test_check_executable_fails_on_bad_move() -> None:
    tasks = generate(TaskGeneratorConfig(seed=0, n_tasks=20))
    task = next(t for t in tasks if t.id.startswith("gen_spacing") and not t.static)
    actions = [
        query(QueryOp.Summary),
        move("column-b", (0.0, 0.0, 0.0)),
        assert_predicates(list(task.goals)),
    ]
    error = _check_executable(task, actions)
    assert error is not None


def test_check_leak_passes_on_fixed_clearance() -> None:
    tasks = generate(TaskGeneratorConfig(seed=0, n_tasks=20))
    task = next(t for t in tasks if t.id.startswith("gen_clearance"))
    assert _check_leak(task) is None


def test_check_leak_detects_already_solved_move_task() -> None:
    tasks = generate(TaskGeneratorConfig(seed=0, n_tasks=20))
    # Static spacing already satisfies goals; mark as move-task → leak
    static_spacing = next(t for t in tasks if t.id.startswith("gen_spacing") and t.static)
    fake = static_spacing.model_copy(
        update={
            "static": False,
            "reference_actions": [move("column-b", (1.0, 0.0, 0.0))],
        }
    )
    assert _check_leak(fake) is not None


def test_check_leak_passes_for_spacing_move() -> None:
    tasks = generate(TaskGeneratorConfig(seed=0, n_tasks=20))
    task = next(t for t in tasks if t.id.startswith("gen_spacing") and not t.static)
    assert _check_leak(task) is None


def test_check_leak_passes_for_opening_shift() -> None:
    tasks = generate(TaskGeneratorConfig(seed=0, n_tasks=20))
    task = next(t for t in tasks if t.id.startswith("gen_opening_shift"))
    assert _check_leak(task) is None


def test_validate_dataset_detects_assert_goals_failure() -> None:
    tasks = generate(TaskGeneratorConfig(seed=5, n_tasks=4))
    task = tasks[0]
    example = build_sft_example(task, [assert_predicates([])])
    failures = _check_schema(example)
    assert not failures

    deep_failures, _ = _check_example_deep(
        example,
        task,
        SftDatasetConfig(seed=5, n_tasks=4),
    )
    assert any(f.check == ValidationCheckKind.AssertGoals for f in deep_failures)


def test_validate_jsonl_schema_only_row_fails_lookup(tmp_path: Path) -> None:
    config = SftDatasetConfig(seed=0, n_tasks=5)
    bad_row = {
        "id": "sft_unknown_0000",
        "messages": [
            {"role": "system", "content": "x"},
            {"role": "user", "content": "y"},
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "format": "bim_gym_actions_v1",
                        "actions": [{"kind": "noop"}],
                    }
                ),
            },
        ],
        "meta": {"task_id": "unknown_0000", "family": None, "verifier_version": "0.1.0"},
    }
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(bad_row) + "\n", encoding="utf-8")
    report = validate_jsonl(str(path), config)
    assert not report.ok
    assert any(f.check == ValidationCheckKind.TaskLookup for f in report.failures)
