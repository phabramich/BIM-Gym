from __future__ import annotations

from collections.abc import Callable
from typing import Any

from bim_gym.eval.models import (
    EvalScore,
    HeldoutEvalConfig,
    HeldoutReport,
    TaskEvalResult,
    aggregate_scores,
)
from bim_gym.eval.score import score_generation
from bim_gym.generators import generate
from bim_gym.sft.builder import _actions_to_assistant_content, _resolve_base_actions
from bim_gym.sft.enums import SftMessageRole
from bim_gym.tasks.spec import TaskSpec
from bim_gym.generators import TaskGeneratorConfig
from bim_gym.sft.context import structural_fingerprint, task_fingerprint

GenerateFn = Callable[[list[dict[str, str]]], str]
PrecomputedPair = tuple[TaskSpec, str]


def task_eval_messages(task: TaskSpec) -> list[dict[str, str]]:
    from bim_gym.sft.builder import _SYSTEM_PROMPT, _user_content

    return [
        {"role": SftMessageRole.System.value, "content": _SYSTEM_PROMPT},
        {"role": SftMessageRole.User.value, "content": _user_content(task)},
    ]


def _reference_assistant_text(task: TaskSpec) -> str:
    actions = _resolve_base_actions(task)
    return _actions_to_assistant_content(actions)


def _score_tasks(pairs: list[PrecomputedPair]) -> HeldoutReport:
    results = [
        TaskEvalResult(task_id=task.id, score=score_generation(task, assistant_text))
        for task, assistant_text in pairs
    ]
    scores: list[EvalScore] = [result.score for result in results]
    return HeldoutReport(metrics=aggregate_scores(scores), results=results)


def _add_overlap_metrics(
    report: HeldoutReport,
    tasks: list[TaskSpec],
    config: HeldoutEvalConfig,
) -> HeldoutReport:
    train_tasks = generate(TaskGeneratorConfig(
        seed=config.train_seed,
        n_tasks=config.n_tasks,
        families=config.families,
    ))
    train_exact = {task_fingerprint(task) for task in train_tasks}
    train_structural = {structural_fingerprint(task) for task in train_tasks}
    n = len(tasks)
    metrics = report.metrics.model_copy(update={
        "exact_task_overlap_rate": (
            sum(task_fingerprint(task) in train_exact for task in tasks) / n if n else 0.0
        ),
        "structural_overlap_rate": (
            sum(structural_fingerprint(task) in train_structural for task in tasks) / n
            if n else 0.0
        ),
    })
    return report.model_copy(update={"metrics": metrics})


def run_heldout(
    config: HeldoutEvalConfig,
    *,
    generate_fn: GenerateFn | None = None,
    precomputed: list[PrecomputedPair] | None = None,
) -> HeldoutReport:
    if (generate_fn is None) == (precomputed is None):
        raise ValueError("provide exactly one of generate_fn or precomputed")

    if precomputed is not None:
        return _add_overlap_metrics(
            _score_tasks(precomputed),
            [task for task, _ in precomputed],
            config,
        )

    assert generate_fn is not None
    tasks = generate(config.to_generator_config())
    pairs = [(task, generate_fn(task_eval_messages(task))) for task in tasks]
    return _add_overlap_metrics(_score_tasks(pairs), tasks, config)


def evaluate_reference_baseline(config: HeldoutEvalConfig) -> HeldoutReport:
    tasks = generate(config.to_generator_config())
    pairs = [(task, _reference_assistant_text(task)) for task in tasks]
    return _add_overlap_metrics(_score_tasks(pairs), tasks, config)
