from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

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
    error_recovery_rate: float = 0.05


def example_to_jsonable(example: SftExample) -> dict[str, Any]:
    return example.model_dump(mode="json")
