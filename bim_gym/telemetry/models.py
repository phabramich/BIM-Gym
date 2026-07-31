from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from bim_gym.telemetry.constants import VERIFIER_VERSION
from bim_gym.telemetry.enums import EpisodePhase
from bim_gym.tasks.spec import TaskSpec


class RewardDecomposition(BaseModel):
    model_config = ConfigDict(frozen=True)

    binary: float
    dense: float
    penalties: float
    geometric_progress: float = 0.0

    @property
    def total(self) -> float:
        return self.binary + self.dense + self.geometric_progress + self.penalties


class TurnRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    step: int
    action: dict[str, Any]
    observation: dict[str, Any]


class TaskSpecSnapshot(TaskSpec):
    """Замороженная копия TaskSpec на старте эпизода."""

    @classmethod
    def from_task(cls, task: TaskSpec) -> TaskSpecSnapshot:
        return cls.model_validate(task.model_dump(mode="json"))


class EpisodeTelemetry(BaseModel):
    model_config = ConfigDict(frozen=True)

    episode_id: str = Field(default_factory=lambda: str(uuid4()))
    phase: EpisodePhase
    task_spec: TaskSpecSnapshot | None = None
    scene_seed: str | int | None = None
    turns: list[TurnRecord] = Field(default_factory=list)
    reward_decomposition: RewardDecomposition
    success: bool
    n_actions: int
    verifier_version: str = VERIFIER_VERSION
    model_checkpoint: str | None = None


def to_jsonable(telemetry: EpisodeTelemetry) -> dict[str, Any]:
    return telemetry.model_dump(mode="json")
