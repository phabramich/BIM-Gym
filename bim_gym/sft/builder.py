from __future__ import annotations

import json
from typing import Any

from bim_gym.contract.enums import EntityType
from bim_gym.dsl.actions import Action, assert_predicates
from bim_gym.dsl.enums import ActionKind, PredicateKind, QueryOp
from bim_gym.dsl import query
from bim_gym.generators import TaskGeneratorConfig, generate
from bim_gym.generators.enums import TaskFamilyKind
from bim_gym.runtime.episode import Episode
from bim_gym.sft.enums import AssistantContentFormat, SftMessageRole
from bim_gym.sft.models import SftDatasetConfig, SftExample, SftMessage, SftMeta
from bim_gym.sft.context import (
    GENERATOR_VERSION,
    canonical_json,
    difficulty_for,
    structural_fingerprint,
    task_context,
    task_fingerprint,
    task_to_jsonable,
)
from bim_gym.tasks.runner import build_scene, run_task
from bim_gym.tasks.spec import TaskSpec

_SYSTEM_PROMPT = """You are a BIM spatial reasoning agent. Respond with a JSON object containing an "actions" array.

Available action kinds: query, note, recall, move, place, assert, noop.
Query ops: list, get, children, openings_on, measure_distance, bbox, summary.
Mutations: move (offset or host_offset), place (position).
The user message includes a canonical scene snapshot and formal goals. Predicate
field expected=false means the underlying relation must be false. Finish with
assert using the task goal predicates when verifying or after edits."""

_ASSISTANT_FORMAT = AssistantContentFormat.BimGymActionsV1


def _infer_family(task: TaskSpec) -> TaskFamilyKind | None:
    if task.id.startswith("gen_hosting_qa"):
        return TaskFamilyKind.HostingQA
    if task.id.startswith("gen_clearance"):
        return TaskFamilyKind.Clearance
    if task.id.startswith("gen_spacing"):
        return TaskFamilyKind.Spacing
    if task.id.startswith("gen_opening_shift"):
        return TaskFamilyKind.OpeningShift
    if task.id.startswith("gen_alignment_qa"):
        return TaskFamilyKind.AlignmentQA
    return None


def _scene_summary_text(task: TaskSpec) -> str:
    return "Task context (canonical JSON):\n" + canonical_json(task_context(task))


def _user_content(task: TaskSpec) -> str:
    return f"{task.instruction}\n\n{_scene_summary_text(task)}"


def _actions_to_assistant_content(actions: list[Action]) -> str:
    payload = {
        "format": _ASSISTANT_FORMAT.value,
        "actions": [action.model_dump(mode="json") for action in actions],
    }
    return json.dumps(payload, ensure_ascii=False)


def _resolve_base_actions(task: TaskSpec) -> list[Action]:
    actions = list(task.reference_actions)
    if not actions or actions[-1].kind != ActionKind.Assert:
        actions.append(assert_predicates(list(task.goals)))
    return actions


def _query_prefix_for_task(task: TaskSpec) -> list[Action]:
    family = _infer_family(task)
    if family is TaskFamilyKind.HostingQA:
        prefix = [query(QueryOp.Summary)]
        for goal in task.goals:
            if goal.kind == PredicateKind.HostedBy:
                prefix.append(query(QueryOp.Get, entity_id=goal.a))
                prefix.append(query(QueryOp.Get, entity_id=goal.b))
                prefix.append(query(QueryOp.OpeningsOn, entity_id=goal.b))
        return prefix

    if family is TaskFamilyKind.Clearance:
        target_ids = sorted({goal.a for goal in task.goals})
        return [
            query(QueryOp.Summary),
            query(QueryOp.List, entity_type=EntityType.Column.value, parent="storey-1"),
            *(query(QueryOp.Get, entity_id=entity_id) for entity_id in target_ids),
        ]

    if family is TaskFamilyKind.Spacing:
        prefix = [
            query(QueryOp.Summary),
            query(QueryOp.List, entity_type=EntityType.Column.value, parent="storey-1"),
        ]
        for goal in task.goals:
            if goal.kind == PredicateKind.DistanceInRange and goal.a and goal.b:
                prefix.append(query(QueryOp.MeasureDistance, a=goal.a, b=goal.b))
        return prefix

    if family is TaskFamilyKind.OpeningShift:
        prefix = [query(QueryOp.Summary)]
        for goal in task.goals:
            if goal.kind == PredicateKind.HostedBy and goal.b:
                prefix.append(query(QueryOp.OpeningsOn, entity_id=goal.b))
                prefix.append(query(QueryOp.Get, entity_id=goal.a))
        return prefix

    return [query(QueryOp.Summary)]


def _actions_from_episode_or_list(
    episode_or_actions: Episode | list[Action],
) -> list[Action]:
    if isinstance(episode_or_actions, Episode):
        return [Action.model_validate(record.action) for record in episode_or_actions.state.history]
    return list(episode_or_actions)


def build_sft_example(
    task: TaskSpec,
    episode_or_actions: Episode | list[Action],
    *,
    has_error_recovery: bool = False,
    scene_seed: int | None = None,
) -> SftExample:
    actions = _actions_from_episode_or_list(episode_or_actions)
    messages = [
        SftMessage(role=SftMessageRole.System, content=_SYSTEM_PROMPT),
        SftMessage(role=SftMessageRole.User, content=_user_content(task)),
        SftMessage(
            role=SftMessageRole.Assistant,
            content=_actions_to_assistant_content(actions),
        ),
    ]
    return SftExample(
        id=f"sft_{task.id}",
        messages=messages,
        meta=SftMeta(
            task_id=task.id,
            family=_infer_family(task),
            has_error_recovery=has_error_recovery,
            generator_version=GENERATOR_VERSION,
            scene_seed=scene_seed,
            task_spec=task_to_jsonable(task),
            task_fingerprint=task_fingerprint(task),
            structural_fingerprint=structural_fingerprint(task),
            difficulty=difficulty_for(task),
        ),
    )


def _prepare_actions(
    task: TaskSpec,
    config: SftDatasetConfig,
) -> tuple[list[Action], bool]:
    actions = _resolve_base_actions(task)
    if config.include_query_prefix:
        prefix = _query_prefix_for_task(task)
        actions = prefix + actions

    return actions, False


def _verify_actions(task: TaskSpec, actions: list[Action]) -> None:
    if task.static and not any(a.kind == ActionKind.Move for a in actions):
        result = run_task(task, actions)
        if not result.success:
            raise ValueError(f"static task {task.id} failed verification: {result.error}")
        return

    result = run_task(task, actions)
    if result.error is not None:
        raise ValueError(f"task {task.id} reference path error: {result.error}")
    if not result.success:
        raise ValueError(f"task {task.id} reference path did not succeed")


def build_dataset(config: SftDatasetConfig) -> list[SftExample]:
    gen_config = TaskGeneratorConfig(seed=config.seed, n_tasks=config.n_tasks)
    tasks = generate(gen_config)
    examples: list[SftExample] = []

    for task in tasks:
        actions, has_recovery = _prepare_actions(task, config)
        _verify_actions(task, actions)
        examples.append(
            build_sft_example(
                task,
                actions,
                has_error_recovery=has_recovery,
                scene_seed=config.seed,
            )
        )

    return examples
