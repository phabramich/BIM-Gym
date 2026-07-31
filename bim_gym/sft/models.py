from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from bim_gym.generators.enums import TaskFamilyKind
from bim_gym.sft.enums import SftMessageRole
from bim_gym.telemetry.constants import VERIFIER_VERSION


class SftMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: SftMessageRole
    content: str


class SftMeta(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str
    family: TaskFamilyKind | None = None
    verifier_version: str = VERIFIER_VERSION
    has_error_recovery: bool = False
    generator_version: str | None = None
    scene_seed: int | None = None
    task_spec: dict[str, Any] | None = None
    task_fingerprint: str | None = None
    structural_fingerprint: str | None = None
    difficulty: dict[str, str] = Field(default_factory=dict)


class SftExample(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    messages: list[SftMessage]
    meta: SftMeta


class SftDatasetConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    seed: int = 0
    n_tasks: int = 20
    include_query_prefix: bool = True
    error_recovery_rate: float = 0.0
    strict_self_contained: bool = False

    @model_validator(mode="after")
    def _reject_one_shot_recovery(self) -> SftDatasetConfig:
        if self.n_tasks < 0:
            raise ValueError("n_tasks must be non-negative")
        if self.error_recovery_rate != 0.0:
            raise ValueError(
                "one-shot error recovery is disabled: it requires intervening tool observations"
            )
        return self


def example_to_jsonable(example: SftExample) -> dict[str, Any]:
    return example.model_dump(mode="json")
