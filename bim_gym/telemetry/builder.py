from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

from bim_gym.dsl.predicates import evaluate_spec
from bim_gym.runtime.scene import Scene
from bim_gym.telemetry.enums import EpisodePhase
from bim_gym.telemetry.models import (
    EpisodeTelemetry,
    RewardDecomposition,
    TaskSpecSnapshot,
    TurnRecord,
)
from bim_gym.tasks.runner import evaluate_task

if TYPE_CHECKING:
    from bim_gym.runtime.episode import Episode
    from bim_gym.tasks.spec import TaskSpec


def compute_reward_v0(
    *,
    success: bool,
    task: TaskSpec | None,
    scene: Scene,
    n_actions: int,
) -> RewardDecomposition:
    binary = 1.0 if success else 0.0

    dense = 0.0
    if task is not None and task.goals:
        results = [evaluate_spec(scene, spec) for spec in task.goals]
        fraction_ok = sum(1 for r in results if r.ok) / len(results)
        dense = fraction_ok * 0.2

    # штраф v0: -0.02 за каждое действие (без учёта минимального плана)
    penalties = -0.02 * n_actions

    return RewardDecomposition(
        binary=binary,
        dense=dense,
        penalties=penalties,
        geometric_progress=0.0,
    )


def _resolve_success(episode: Episode, task: TaskSpec | None) -> bool:
    if episode.state.success is not None:
        return episode.state.success
    if task is not None:
        return evaluate_task(task, episode.state.scene).success
    return False


def from_episode(
    episode: Episode,
    task: TaskSpec | None = None,
    *,
    phase: EpisodePhase = EpisodePhase.Eval,
    scene_seed: str | int | None = None,
    model_checkpoint: str | None = None,
    reward: RewardDecomposition | None = None,
    episode_id: str | None = None,
) -> EpisodeTelemetry:
    n_actions = episode.state._step
    success = _resolve_success(episode, task)
    reward_decomposition = reward or compute_reward_v0(
        success=success,
        task=task,
        scene=episode.state.scene,
        n_actions=n_actions,
    )
    turns = [
        TurnRecord(
            step=record.step,
            action=record.action,
            observation=record.observation,
        )
        for record in episode.state.history
    ]
    task_snapshot = TaskSpecSnapshot.from_task(task) if task is not None else None

    return EpisodeTelemetry(
        episode_id=episode_id or str(uuid4()),
        phase=phase,
        task_spec=task_snapshot,
        scene_seed=scene_seed,
        turns=turns,
        reward_decomposition=reward_decomposition,
        success=success,
        n_actions=n_actions,
        model_checkpoint=model_checkpoint,
    )


def build_telemetry(
    episode: Episode,
    task: TaskSpec | None = None,
    **kwargs: object,
) -> EpisodeTelemetry:
    return from_episode(episode, task, **kwargs)
