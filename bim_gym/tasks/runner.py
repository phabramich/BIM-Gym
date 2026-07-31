from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from bim_gym.dsl.actions import Action
from bim_gym.dsl.predicates import evaluate_spec
from bim_gym.runtime.episode import Episode
from bim_gym.runtime.predicates import PredicateResult, evaluate as evaluate_predicates
from bim_gym.runtime.scene import Scene
from bim_gym.tasks.handcrafted import _SCENE_BUILDERS
from bim_gym.tasks.spec import TaskSpec

if TYPE_CHECKING:
    from bim_gym.telemetry.models import EpisodeTelemetry


@dataclass
class TaskResult:
    task_id: str
    success: bool
    score: float
    predicate_results: list[PredicateResult] = field(default_factory=list)
    error: str | None = None
    steps: int = 0
    telemetry: EpisodeTelemetry | None = None


def build_scene(task: TaskSpec) -> Scene:
    if task.scene is not None:
        return Scene.from_dict(task.scene)
    if task.scene_factory is None:
        raise ValueError(f"task {task.id} has no scene or scene_factory")
    builder = _SCENE_BUILDERS[task.scene_factory]
    raw = builder(**task.scene_factory_args)
    return Scene.from_dict(raw)


def evaluate_task(task: TaskSpec, scene: Scene | None = None) -> TaskResult:
    scene = scene or build_scene(task)
    results = [evaluate_spec(scene, spec) for spec in task.goals]
    combined = evaluate_predicates(results)
    return TaskResult(
        task_id=task.id,
        success=combined.ok,
        score=combined.score,
        predicate_results=results,
    )


def run_task(
    task: TaskSpec,
    actions: list[Action | dict[str, Any]],
    *,
    record_telemetry: bool = False,
    scene_seed: str | int | None = None,
    model_checkpoint: str | None = None,
) -> TaskResult:
    from bim_gym.telemetry import from_episode

    episode = Episode(build_scene(task))
    last_error: str | None = None

    def _telemetry() -> EpisodeTelemetry | None:
        if not record_telemetry:
            return None
        return from_episode(
            episode,
            task,
            scene_seed=scene_seed,
            model_checkpoint=model_checkpoint,
        )

    for action in actions:
        obs = episode.step(action)
        if not obs.get("ok"):
            last_error = obs.get("error", "step failed")
            break
        if obs.get("done"):
            return TaskResult(
                task_id=task.id,
                success=bool(obs.get("success")),
                score=float(obs.get("result", {}).get("score", 0.0)),
                steps=episode.state._step,
                telemetry=_telemetry(),
            )

    if last_error is not None:
        return TaskResult(
            task_id=task.id,
            success=False,
            score=0.0,
            error=last_error,
            steps=episode.state._step,
            telemetry=_telemetry(),
        )

    result = evaluate_task(task, episode.state.scene)
    result.steps = episode.state._step
    result.telemetry = _telemetry()
    return result
