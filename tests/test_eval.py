from __future__ import annotations

import json

import pytest

from bim_gym.dsl.actions import PredicateSpec, assert_predicates
from bim_gym.dsl.enums import PredicateKind
from bim_gym.eval import (
    EvalScore,
    HeldoutEvalConfig,
    evaluate_reference_baseline,
    parse_assistant_actions,
    run_heldout,
    score_generation,
)
from bim_gym.generators import TaskGeneratorConfig, generate
from bim_gym.sft.enums import AssistantContentFormat


def _valid_assistant_text() -> str:
    action = assert_predicates(
        [PredicateSpec(kind=PredicateKind.HostedBy, a="opening-wall-s", b="wall-s")]
    )
    payload = {
        "format": AssistantContentFormat.BimGymActionsV1.value,
        "actions": [action.model_dump(mode="json")],
    }
    return json.dumps(payload)


def test_parse_assistant_actions_valid() -> None:
    text = _valid_assistant_text()
    actions = parse_assistant_actions(text)
    assert actions is not None
    assert len(actions) == 1
    assert actions[0].kind.value == "assert"


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        json.dumps({"format": "wrong", "actions": []}),
        json.dumps({"format": AssistantContentFormat.BimGymActionsV1.value}),
        json.dumps(
            {
                "format": AssistantContentFormat.BimGymActionsV1.value,
                "actions": "not-a-list",
            }
        ),
    ],
)
def test_parse_assistant_actions_invalid(text: str) -> None:
    assert parse_assistant_actions(text) is None


def test_score_generation_unknown_primitive() -> None:
    task = generate(TaskGeneratorConfig(seed=1, n_tasks=1))[0]
    text = json.dumps(
        {
            "format": AssistantContentFormat.BimGymActionsV1.value,
            "actions": [{"kind": "teleport", "entity_id": "x"}],
        }
    )
    score = score_generation(task, text)
    assert score == EvalScore(
        protocol_valid=False,
        unknown_primitive=True,
        success=False,
        error="invalid assistant protocol",
    )


def test_score_generation_valid_protocol() -> None:
    task = generate(TaskGeneratorConfig(seed=2, n_tasks=1))[0]
    action = assert_predicates(list(task.goals))
    text = json.dumps(
        {
            "format": AssistantContentFormat.BimGymActionsV1.value,
            "actions": [action.model_dump(mode="json")],
        }
    )
    score = score_generation(task, text)
    assert score.protocol_valid is True
    assert score.unknown_primitive is False
    assert score.success is True


def test_heldout_config_rejects_same_seed() -> None:
    with pytest.raises(ValueError, match="held-out seed must differ"):
        HeldoutEvalConfig(seed=0, train_seed=0)


def test_evaluate_reference_baseline_near_perfect() -> None:
    config = HeldoutEvalConfig(seed=10_000, train_seed=0, n_tasks=5)
    report = evaluate_reference_baseline(config)
    assert report.metrics.n == 5
    assert report.metrics.protocol_valid_rate == 1.0
    assert report.metrics.success_rate == 1.0
    assert report.metrics.unknown_primitive_rate == 0.0
    assert report.metrics.exact_task_overlap_rate == 0.0
    assert 0.0 <= report.metrics.structural_overlap_rate <= 1.0


def test_run_heldout_precomputed() -> None:
    config = HeldoutEvalConfig(seed=10_001, train_seed=0, n_tasks=3)
    tasks = generate(config.to_generator_config())
    precomputed = [(task, "not json") for task in tasks]
    report = run_heldout(config, precomputed=precomputed)
    assert report.metrics.n == 3
    assert report.metrics.protocol_valid_rate == 0.0
    assert report.metrics.success_rate == 0.0


def test_run_heldout_requires_exactly_one_source() -> None:
    config = HeldoutEvalConfig(seed=10_002, train_seed=0, n_tasks=1)
    with pytest.raises(ValueError, match="exactly one"):
        run_heldout(config)
    with pytest.raises(ValueError, match="exactly one"):
        run_heldout(config, generate_fn=lambda _m: "{}", precomputed=[])
