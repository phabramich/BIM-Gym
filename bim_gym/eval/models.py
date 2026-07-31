from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from bim_gym.generators.config import TaskGeneratorConfig
from bim_gym.generators.enums import TaskFamilyKind
from bim_gym.tasks.spec import TaskSpec


class EvalScore(BaseModel):
    model_config = ConfigDict(frozen=True)

    protocol_valid: bool
    unknown_primitive: bool
    success: bool
    error: str | None = None


class HeldoutMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    n: int
    protocol_valid_rate: float
    success_rate: float
    unknown_primitive_rate: float
    exact_task_overlap_rate: float = 0.0
    structural_overlap_rate: float = 0.0


class TaskEvalResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str
    score: EvalScore


class HeldoutReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    metrics: HeldoutMetrics
    results: list[TaskEvalResult] = Field(default_factory=list)


class HeldoutEvalConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    seed: int = 10_000
    train_seed: int = 0
    n_tasks: int = 20
    families: tuple[TaskFamilyKind, ...] = TaskGeneratorConfig.model_fields[
        "families"
    ].default

    @model_validator(mode="after")
    def _heldout_seed_differs_from_train(self) -> HeldoutEvalConfig:
        if self.seed == self.train_seed:
            raise ValueError("held-out seed must differ from train_seed")
        return self

    def to_generator_config(self) -> TaskGeneratorConfig:
        return TaskGeneratorConfig(
            seed=self.seed,
            n_tasks=self.n_tasks,
            families=self.families,
        )


def aggregate_scores(scores: list[EvalScore]) -> HeldoutMetrics:
    n = len(scores)
    if n == 0:
        return HeldoutMetrics(
            n=0,
            protocol_valid_rate=0.0,
            success_rate=0.0,
            unknown_primitive_rate=0.0,
        )
    return HeldoutMetrics(
        n=n,
        protocol_valid_rate=sum(score.protocol_valid for score in scores) / n,
        success_rate=sum(score.success for score in scores) / n,
        unknown_primitive_rate=sum(score.unknown_primitive for score in scores) / n,
    )


def metrics_to_dict(metrics: HeldoutMetrics) -> dict[str, Any]:
    return {
        "n": float(metrics.n),
        "protocol_valid_rate": metrics.protocol_valid_rate,
        "success_rate": metrics.success_rate,
        "unknown_primitive_rate": metrics.unknown_primitive_rate,
        "exact_task_overlap_rate": metrics.exact_task_overlap_rate,
        "structural_overlap_rate": metrics.structural_overlap_rate,
    }
