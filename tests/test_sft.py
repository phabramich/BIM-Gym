from __future__ import annotations

import json
from pathlib import Path

import pytest

from bim_gym.dsl.actions import Action
from bim_gym.generators.enums import TaskFamilyKind
from bim_gym.sft import (
    AssistantContentFormat,
    SftDatasetConfig,
    SftMessageRole,
    build_dataset,
    build_sft_example,
)
from bim_gym.sft.export import export_jsonl, load_jsonl
from bim_gym.sft.builder import _resolve_base_actions
from bim_gym.tasks.runner import run_task
from bim_gym.generators import TaskGeneratorConfig, generate


def _parse_assistant_content(content: str) -> dict:
    return json.loads(content)


def _assert_example_shape(example) -> None:
    roles = [message.role for message in example.messages]
    assert roles == [
        SftMessageRole.System,
        SftMessageRole.User,
        SftMessageRole.Assistant,
    ]
    assert example.meta.task_id
    assert example.meta.verifier_version

    assistant_payload = _parse_assistant_content(example.messages[2].content)
    assert assistant_payload["format"] == AssistantContentFormat.BimGymActionsV1.value
    actions_raw = assistant_payload["actions"]
    assert isinstance(actions_raw, list)
    assert len(actions_raw) >= 1
    for item in actions_raw:
        Action.model_validate(item)


def test_build_dataset_at_least_ten_examples() -> None:
    config = SftDatasetConfig(seed=0, n_tasks=12, error_recovery_rate=0.0)
    examples = build_dataset(config)
    assert len(examples) >= 10
    for example in examples:
        _assert_example_shape(example)


def test_query_prefix_present_for_move_tasks() -> None:
    config = SftDatasetConfig(seed=1, n_tasks=20, include_query_prefix=True)
    examples = build_dataset(config)
    move_examples = [
        ex
        for ex in examples
        if ex.meta.family in (TaskFamilyKind.Clearance, TaskFamilyKind.Spacing)
        and not ex.meta.has_error_recovery
    ]
    assert move_examples
    for example in move_examples[:3]:
        payload = _parse_assistant_content(example.messages[2].content)
        kinds = [action["kind"] for action in payload["actions"]]
        assert kinds[0] == "query"


def test_error_recovery_injects_zero_move() -> None:
    config = SftDatasetConfig(seed=2, n_tasks=30, error_recovery_rate=1.0)
    examples = build_dataset(config)
    recovered = [ex for ex in examples if ex.meta.has_error_recovery]
    assert recovered
    payload = _parse_assistant_content(recovered[0].messages[2].content)
    actions = payload["actions"]
    zero_moves = [
        a
        for i, a in enumerate(actions)
        if a.get("kind") == "move"
        and a.get("offset") == [0.0, 0.0, 0.0]
        and i + 1 < len(actions)
        and actions[i + 1].get("kind") == "move"
    ]
    assert zero_moves


def test_export_and_reload_jsonl(tmp_path: Path) -> None:
    examples = build_dataset(SftDatasetConfig(seed=3, n_tasks=10))
    out = tmp_path / "sft_bootstrap.jsonl"
    export_jsonl(examples, out)
    rows = load_jsonl(out)
    assert len(rows) == len(examples)
    assert rows[0]["id"] == examples[0].id
    assert len(rows[0]["messages"]) == 3


def test_build_sft_example_from_run_task() -> None:
    tasks = generate(TaskGeneratorConfig(seed=4, n_tasks=5))
    task = next(t for t in tasks if t.reference_actions)
    actions = _resolve_base_actions(task)
    result = run_task(task, actions)
    assert result.success
    example = build_sft_example(task, actions)
    _assert_example_shape(example)
    assert task.instruction in example.messages[1].content
