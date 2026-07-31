from __future__ import annotations

from bim_gym.generators import TaskGeneratorConfig, generate
from bim_gym.tasks import evaluate_task, run_task


def _assert_solvable(task) -> None:
    if task.static:
        result = evaluate_task(task)
        assert result.success, f"{task.id}: {result.predicate_results}"
        return
    if task.reference_actions:
        result = run_task(task, task.reference_actions)
        assert result.error is None, f"{task.id}: {result.error}"
        assert result.success, f"{task.id}: score={result.score}"
        return
    result = evaluate_task(task)
    assert result.success, f"{task.id}: {result.predicate_results}"


def test_generate_twenty_tasks_all_solvable() -> None:
    config = TaskGeneratorConfig(seed=42, n_tasks=20)
    tasks = generate(config)
    assert len(tasks) == 20
    for task in tasks:
        _assert_solvable(task)


def test_generate_deterministic() -> None:
    config = TaskGeneratorConfig(seed=7, n_tasks=12)
    first = generate(config)
    second = generate(config)
    assert [t.id for t in first] == [t.id for t in second]
    assert [g.model_dump() for g in first[0].goals] == [
        g.model_dump() for g in second[0].goals
    ]
