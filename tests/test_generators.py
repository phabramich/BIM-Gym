from __future__ import annotations

from bim_gym.generators import TaskGeneratorConfig, generate
from bim_gym.tasks import evaluate_task, run_task
from bim_gym.generators.enums import TaskFamilyKind


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


def test_boolean_qa_is_balanced_and_executable() -> None:
    tasks = generate(TaskGeneratorConfig(
        seed=12,
        n_tasks=40,
        families=(TaskFamilyKind.HostingQA, TaskFamilyKind.AlignmentQA),
    ))
    outcomes = [task.goals[0].expected for task in tasks]
    assert outcomes.count(True) == outcomes.count(False)
    assert all(evaluate_task(task).success for task in tasks)


def test_generated_target_ids_are_not_fixed_shortcuts() -> None:
    tasks = generate(TaskGeneratorConfig(seed=4, n_tasks=50))
    target_ids = {task.goals[0].a for task in tasks}
    assert len(target_ids) > 40


def test_clearance_distractor_ids_are_renamed() -> None:
    tasks = generate(TaskGeneratorConfig(
        seed=0,
        n_tasks=8,
        families=(TaskFamilyKind.Clearance,),
    ))
    for task in tasks:
        entity_ids = set(task.scene["entities"])
        assert "column-1" not in entity_ids
        assert "column-2" not in entity_ids
        assert any(eid.endswith("-x") for eid in entity_ids)
