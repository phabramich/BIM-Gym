from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from bim_gym.dsl.actions import Action
from bim_gym.dsl.enums import ActionKind
from bim_gym.generators import TaskGeneratorConfig, generate
from bim_gym.generators.enums import TaskFamilyKind
from bim_gym.sft.builder import _infer_family, build_dataset
from bim_gym.sft.builder import _user_content
from bim_gym.sft.context import structural_fingerprint, task_fingerprint, task_from_jsonable
from bim_gym.sft.enums import AssistantContentFormat, SftMessageRole
from bim_gym.sft.export import load_jsonl
from bim_gym.sft.models import SftDatasetConfig, SftExample
from bim_gym.tasks.runner import build_scene, evaluate_task, run_task
from bim_gym.tasks.spec import TaskSpec

_DUPLICATE_WARN_RATE = 0.05
_EXPECTED_FAMILIES = frozenset(TaskFamilyKind)


class ValidationCheckKind(str, Enum):
    Schema = "schema"
    Executable = "executable"
    MissingEntity = "missing_entity"
    AssertGoals = "assert_goals"
    Leak = "leak"
    EmptyActions = "empty_actions"
    EmptyInstruction = "empty_instruction"
    IdConsistency = "id_consistency"
    TaskLookup = "task_lookup"
    Context = "context"
    Fingerprint = "fingerprint"
    PromptCollision = "prompt_collision"
    ReferenceMinimality = "reference_minimality"
    ProtectedState = "protected_state"


class ValidationFailure(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    check: ValidationCheckKind
    detail: str


class ValidationWarning(BaseModel):
    model_config = ConfigDict(frozen=True)

    check: str
    detail: str


class ValidationReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    ok: bool
    n: int
    hard_fail_count: int
    warnings: list[ValidationWarning] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    failures: list[ValidationFailure] = Field(default_factory=list)


def _failure(
    example_id: str,
    check: ValidationCheckKind,
    detail: str,
) -> ValidationFailure:
    return ValidationFailure(id=example_id, check=check, detail=detail)


def _entity_refs_in_action(action: Action) -> set[str]:
    refs: set[str] = set()
    if action.entity_id:
        refs.add(action.entity_id)
    if action.a:
        refs.add(action.a)
    if action.b:
        refs.add(action.b)
    if action.parent:
        refs.add(action.parent)
    for item in action.items:
        refs.add(item.a)
        if item.b:
            refs.add(item.b)
    return refs


def _parse_actions_from_example(example: SftExample) -> tuple[list[Action] | None, str | None]:
    content = example.messages[2].content
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        return None, f"assistant JSON parse error: {exc}"

    if not isinstance(payload, dict):
        return None, "assistant content is not a JSON object"
    if payload.get("format") != AssistantContentFormat.BimGymActionsV1.value:
        return None, f"unexpected format: {payload.get('format')!r}"

    raw_actions = payload.get("actions")
    if not isinstance(raw_actions, list):
        return None, "actions is not a list"
    if not raw_actions:
        return None, "actions list is empty"

    actions: list[Action] = []
    for index, item in enumerate(raw_actions):
        if not isinstance(item, dict):
            return None, f"action[{index}] is not an object"
        try:
            actions.append(Action.model_validate(item))
        except ValidationError as exc:
            return None, f"action[{index}] validation error: {exc}"
    return actions, None


def _check_schema(example: SftExample) -> list[ValidationFailure]:
    failures: list[ValidationFailure] = []
    example_id = example.id

    if len(example.messages) != 3:
        failures.append(
            _failure(
                example_id,
                ValidationCheckKind.Schema,
                f"expected 3 messages, got {len(example.messages)}",
            )
        )
        return failures

    expected_roles = [
        SftMessageRole.System,
        SftMessageRole.User,
        SftMessageRole.Assistant,
    ]
    roles = [message.role for message in example.messages]
    if roles != expected_roles:
        failures.append(
            _failure(
                example_id,
                ValidationCheckKind.Schema,
                f"unexpected roles: {[role.value for role in roles]}",
            )
        )

    if example.id != f"sft_{example.meta.task_id}":
        failures.append(
            _failure(
                example_id,
                ValidationCheckKind.IdConsistency,
                f"id {example.id!r} != sft_{example.meta.task_id!r}",
            )
        )

    _, parse_error = _parse_actions_from_example(example)
    if parse_error is not None:
        failures.append(
            _failure(example_id, ValidationCheckKind.Schema, parse_error)
        )

    return failures


def _check_executable(task: TaskSpec, actions: list[Action]) -> str | None:
    has_move = any(action.kind == ActionKind.Move for action in actions)
    if task.static and not has_move:
        result = run_task(task, actions)
        if not result.success:
            return result.error or "static task run_task failed"
        return None

    result = run_task(task, actions)
    if result.error is not None:
        return result.error
    if not result.success:
        return "run_task did not succeed"
    return None


def _check_assert_goals(actions: list[Action], goals: list) -> str | None:
    assert_actions = [action for action in actions if action.kind == ActionKind.Assert]
    if not assert_actions:
        return "no assert action"
    last_assert = assert_actions[-1]
    if list(last_assert.items) != list(goals):
        return "final assert items do not match task.goals"
    return None


def _check_missing_entities(task: TaskSpec, actions: list[Action]) -> str | None:
    scene = build_scene(task)
    known = set(scene.model.entities.keys())
    missing: list[str] = []
    for action in actions:
        for entity_id in _entity_refs_in_action(action):
            if entity_id not in known:
                missing.append(entity_id)
    if missing:
        return f"unknown entity ids: {sorted(set(missing))}"
    return None


def _check_leak(task: TaskSpec) -> str | None:
    if task.static:
        return None
    family = _infer_family(task)
    if family not in (
        TaskFamilyKind.Clearance,
        TaskFamilyKind.Spacing,
        TaskFamilyKind.OpeningShift,
    ):
        return None
    result = evaluate_task(task)
    if result.success:
        return "initial scene already satisfies goals before any moves"
    return None


def _check_reference_minimality(task: TaskSpec, actions: list[Action]) -> str | None:
    mutation_indices = [
        index for index, action in enumerate(actions)
        if action.kind in (ActionKind.Move, ActionKind.Place)
    ]
    for index in mutation_indices:
        candidate = actions[:index] + actions[index + 1:]
        if run_task(task, candidate).success:
            return f"mutation action[{index}] is unnecessary"
    return None


def _check_mutation_targets(task: TaskSpec, actions: list[Action]) -> str | None:
    allowed = {goal.a for goal in task.goals}
    allowed.update(goal.b for goal in task.goals if goal.b is not None)
    unexpected = sorted({
        action.entity_id for action in actions
        if action.kind in (ActionKind.Move, ActionKind.Place)
        and action.entity_id is not None
        and action.entity_id not in allowed
    })
    if unexpected:
        return f"mutations target entities outside task goals: {unexpected}"
    return None


def _check_example_deep(
    example: SftExample,
    task: TaskSpec,
    config: SftDatasetConfig,
) -> tuple[list[ValidationFailure], list[ValidationWarning]]:
    failures: list[ValidationFailure] = []
    warnings: list[ValidationWarning] = []
    example_id = example.id

    if not task.instruction.strip():
        failures.append(
            _failure(
                example_id,
                ValidationCheckKind.EmptyInstruction,
                "task instruction is empty",
            )
        )

    if example.meta.task_spec is not None and example.messages[1].content != _user_content(task):
        failures.append(_failure(
            example_id,
            ValidationCheckKind.Context,
            "user content does not match embedded task context",
        ))
    if example.meta.task_fingerprint is not None and example.meta.task_fingerprint != task_fingerprint(task):
        failures.append(_failure(
            example_id,
            ValidationCheckKind.Fingerprint,
            "task fingerprint does not match task semantics",
        ))
    if (example.meta.structural_fingerprint is not None
            and example.meta.structural_fingerprint != structural_fingerprint(task)):
        failures.append(_failure(
            example_id,
            ValidationCheckKind.Fingerprint,
            "structural fingerprint does not match task structure",
        ))

    actions, parse_error = _parse_actions_from_example(example)
    if parse_error is not None or actions is None:
        return failures, warnings

    exec_error = _check_executable(task, actions)
    if exec_error is not None:
        failures.append(
            _failure(example_id, ValidationCheckKind.Executable, exec_error)
        )

    goals_error = _check_assert_goals(actions, task.goals)
    if goals_error is not None:
        failures.append(
            _failure(example_id, ValidationCheckKind.AssertGoals, goals_error)
        )

    entity_error = _check_missing_entities(task, actions)
    if entity_error is not None:
        failures.append(
            _failure(example_id, ValidationCheckKind.MissingEntity, entity_error)
        )

    leak_error = _check_leak(task)
    if leak_error is not None:
        failures.append(_failure(example_id, ValidationCheckKind.Leak, leak_error))

    minimality_error = _check_reference_minimality(task, actions)
    if minimality_error is not None:
        failures.append(_failure(
            example_id,
            ValidationCheckKind.ReferenceMinimality,
            minimality_error,
        ))

    target_error = _check_mutation_targets(task, actions)
    if target_error is not None:
        failures.append(_failure(
            example_id,
            ValidationCheckKind.ProtectedState,
            target_error,
        ))

    if actions[-1].kind != ActionKind.Assert:
        warnings.append(
            ValidationWarning(
                check="no_assert_at_end",
                detail=f"{example_id} ends with {actions[-1].kind.value}",
            )
        )

    if config.include_query_prefix and not example.meta.has_error_recovery:
        family = _infer_family(task)
        if family in (
            TaskFamilyKind.Clearance,
            TaskFamilyKind.Spacing,
            TaskFamilyKind.OpeningShift,
        ):
            if not actions or actions[0].kind != ActionKind.Query:
                warnings.append(
                    ValidationWarning(
                        check="query_prefix_missing",
                        detail=f"{example_id} first action is not query",
                    )
                )

    return failures, warnings


def _action_kind_histogram(examples: list[SftExample]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for example in examples:
        actions, _ = _parse_actions_from_example(example)
        if actions is None:
            continue
        for action in actions:
            counts[action.kind.value] += 1
    return dict(sorted(counts.items()))


def _family_histogram(examples: list[SftExample]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for example in examples:
        family = example.meta.family
        key = family.value if family is not None else "unknown"
        counts[key] += 1
    return dict(sorted(counts.items()))


def _duplicate_rates(examples: list[SftExample]) -> tuple[float, float]:
    if not examples:
        return 0.0, 0.0
    task_ids = [example.meta.task_id for example in examples]
    task_dup = len(task_ids) - len(set(task_ids))
    payloads = [example.messages[2].content for example in examples]
    payload_dup = len(payloads) - len(set(payloads))
    n = len(examples)
    return task_dup / n, payload_dup / n


def _validate_examples(
    examples: list[SftExample],
    tasks_by_id: dict[str, TaskSpec],
    config: SftDatasetConfig,
    *,
    require_task_lookup: bool,
) -> ValidationReport:
    failures: list[ValidationFailure] = []
    warnings: list[ValidationWarning] = []

    prompt_semantics: dict[str, set[str]] = {}
    for example in examples:
        failures.extend(_check_schema(example))

        task: TaskSpec | None = None
        if example.meta.task_spec is not None:
            try:
                task = task_from_jsonable(example.meta.task_spec)
            except ValidationError as exc:
                failures.append(_failure(
                    example.id,
                    ValidationCheckKind.Schema,
                    f"embedded task validation: {exc}",
                ))
        if task is None:
            task = tasks_by_id.get(example.meta.task_id)
        if task is None:
            if require_task_lookup:
                failures.append(
                    _failure(
                        example.id,
                        ValidationCheckKind.TaskLookup,
                        f"no task for task_id {example.meta.task_id!r}",
                    )
                )
            continue

        prompt = example.messages[1].content
        prompt_semantics.setdefault(prompt, set()).add(task_fingerprint(task))

        deep_failures, deep_warnings = _check_example_deep(example, task, config)
        failures.extend(deep_failures)
        warnings.extend(deep_warnings)

    for prompt, fingerprints in prompt_semantics.items():
        if len(fingerprints) > 1:
            failures.append(_failure(
                "dataset",
                ValidationCheckKind.PromptCollision,
                f"one visible prompt maps to {len(fingerprints)} task semantics",
            ))

    task_dup_rate, payload_dup_rate = _duplicate_rates(examples)
    legacy_count = sum(example.meta.task_spec is None for example in examples)
    if legacy_count:
        warnings.append(ValidationWarning(
            check="not_self_contained",
            detail=f"{legacy_count} rows require generator lookup",
        ))
        if config.strict_self_contained:
            failures.append(_failure(
                "dataset",
                ValidationCheckKind.Schema,
                f"strict mode rejects {legacy_count} non-self-contained rows",
            ))
    if task_dup_rate > 0:
        warnings.append(
            ValidationWarning(
                check="duplicate_task_ids",
                detail=f"duplicate task_id rate {task_dup_rate:.2%}",
            )
        )
    if payload_dup_rate > _DUPLICATE_WARN_RATE:
        warnings.append(
            ValidationWarning(
                check="duplicate_assistant_payloads",
                detail=f"duplicate assistant payload rate {payload_dup_rate:.2%}",
            )
        )

    family_hist = _family_histogram(examples)
    missing_families = sorted(
        family.value for family in _EXPECTED_FAMILIES if family_hist.get(family.value, 0) == 0
    )
    if missing_families and examples:
        warnings.append(
            ValidationWarning(
                check="family_imbalance",
                detail=f"no examples for families: {missing_families}",
            )
        )

    query_warn_count = sum(1 for warning in warnings if warning.check == "query_prefix_missing")
    if config.include_query_prefix and examples:
        query_warn_rate = query_warn_count / len(examples)
        metrics_query = {"query_prefix_missing_rate": query_warn_rate}
    else:
        metrics_query = {}

    hard_fail_count = len(failures)
    structural_values = [
        example.meta.structural_fingerprint
        for example in examples
        if example.meta.structural_fingerprint is not None
    ]
    structural_dup_rate = (
        (len(structural_values) - len(set(structural_values))) / len(structural_values)
        if structural_values else 0.0
    )
    goal_hist = Counter()
    expected_hist = Counter()
    expected_by_family: dict[str, Counter[str]] = {}
    entity_hist = Counter()
    for example in examples:
        if example.meta.task_spec is None:
            continue
        task = task_from_jsonable(example.meta.task_spec)
        goal_hist[len(task.goals)] += 1
        expected_hist.update(str(goal.expected).lower() for goal in task.goals)
        family_key = example.meta.family.value if example.meta.family else "unknown"
        family_outcomes = expected_by_family.setdefault(family_key, Counter())
        family_outcomes.update(str(goal.expected).lower() for goal in task.goals)
        entity_hist[len(build_scene(task).model.entities)] += 1
    metrics: dict[str, Any] = {
        "family_balance": family_hist,
        "action_kind_histogram": _action_kind_histogram(examples),
        "duplicate_task_id_rate": task_dup_rate,
        "duplicate_assistant_payload_rate": payload_dup_rate,
        "structural_duplicate_rate": structural_dup_rate,
        "unique_task_fingerprints": len({e.meta.task_fingerprint for e in examples if e.meta.task_fingerprint}),
        "unique_structural_fingerprints": len(set(structural_values)),
        "self_contained_rate": sum(e.meta.task_spec is not None for e in examples) / len(examples) if examples else 0.0,
        "prompt_collision_count": sum(len(v) > 1 for v in prompt_semantics.values()),
        "goal_count_histogram": dict(sorted(goal_hist.items())),
        "expected_outcome_histogram": dict(sorted(expected_hist.items())),
        "expected_outcomes_by_family": {
            family: dict(sorted(counts.items()))
            for family, counts in sorted(expected_by_family.items())
        },
        "entity_count_histogram": dict(sorted(entity_hist.items())),
        **metrics_query,
    }

    return ValidationReport(
        ok=hard_fail_count == 0,
        n=len(examples),
        hard_fail_count=hard_fail_count,
        warnings=warnings,
        metrics=metrics,
        failures=failures,
    )


def _load_examples(rows: list[dict[str, Any]]) -> tuple[list[SftExample], list[ValidationFailure]]:
    examples: list[SftExample] = []
    failures: list[ValidationFailure] = []
    for index, row in enumerate(rows):
        row_id = str(row.get("id", f"row_{index}"))
        try:
            examples.append(SftExample.model_validate(row))
        except ValidationError as exc:
            failures.append(
                _failure(row_id, ValidationCheckKind.Schema, f"record validation: {exc}")
            )
    return examples, failures


def _tasks_for_config(config: SftDatasetConfig) -> dict[str, TaskSpec]:
    gen_config = TaskGeneratorConfig(seed=config.seed, n_tasks=config.n_tasks)
    return {task.id: task for task in generate(gen_config)}


def validate_dataset(config: SftDatasetConfig) -> ValidationReport:
    examples = build_dataset(config)
    tasks_by_id = _tasks_for_config(config)
    return _validate_examples(
        examples,
        tasks_by_id,
        config,
        require_task_lookup=True,
    )


def validate_jsonl(path: str, config: SftDatasetConfig) -> ValidationReport:
    rows = load_jsonl(path)
    examples, load_failures = _load_examples(rows)
    tasks_by_id = _tasks_for_config(config)
    report = _validate_examples(
        examples,
        tasks_by_id,
        config,
        require_task_lookup=True,
    )
    if not load_failures:
        return report
    merged_failures = list(load_failures) + list(report.failures)
    return report.model_copy(
        update={
            "ok": False,
            "hard_fail_count": len(merged_failures),
            "failures": merged_failures,
        }
    )


def _group_failures(failures: list[ValidationFailure]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for failure in failures:
        grouped.setdefault(failure.check.value, []).append(failure.id)
    return grouped


def _print_report(report: ValidationReport) -> None:
    status = "PASS" if report.ok else "FAIL"
    print(f"Validation {status}: n={report.n}, hard_fails={report.hard_fail_count}")
    if report.metrics:
        print("Metrics:")
        for key, value in report.metrics.items():
            print(f"  {key}: {value}")
    if report.warnings:
        print(f"Warnings ({len(report.warnings)}):")
        for warning in report.warnings[:20]:
            print(f"  [{warning.check}] {warning.detail}")
        if len(report.warnings) > 20:
            print(f"  ... and {len(report.warnings) - 20} more")
    if report.failures:
        print("Hard failures by check:")
        for check, ids in sorted(_group_failures(report.failures).items()):
            print(f"  {check} ({len(ids)}): {', '.join(ids)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate SFT bootstrap dataset quality")
    parser.add_argument("--jsonl", type=str, default=None, help="Path to exported JSONL")
    parser.add_argument("--n", type=int, default=20, help="Number of tasks (must match export)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (must match export)")
    parser.add_argument(
        "--no-query-prefix",
        action="store_true",
        help="Disable query-prefix expectation (must match export)",
    )
    parser.add_argument(
        "--strict-self-contained",
        action="store_true",
        help="Reject legacy rows without embedded task specs",
    )
    parser.add_argument(
        "--error-recovery-rate",
        type=float,
        default=0.0,
        help="Error recovery rate used during export",
    )
    args = parser.parse_args(argv)

    config = SftDatasetConfig(
        seed=args.seed,
        n_tasks=args.n,
        include_query_prefix=not args.no_query_prefix,
        error_recovery_rate=args.error_recovery_rate,
        strict_self_contained=args.strict_self_contained,
    )

    if args.jsonl:
        report = validate_jsonl(args.jsonl, config)
    else:
        report = validate_dataset(config)

    _print_report(report)
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
