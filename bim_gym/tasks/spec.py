from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from bim_gym.dsl.actions import Action, PredicateSpec


class SceneFactoryName(str, Enum):
    RoomBox = "room_box"
    RoomWithColumnNearWall = "room_with_column_near_wall"
    TwoColumns = "two_columns"


class TaskSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    instruction: str
    goals: list[PredicateSpec]
    scene_factory: SceneFactoryName | None = None
    scene_factory_args: dict[str, Any] = Field(default_factory=dict)
    scene: dict[str, Any] | None = None
    static: bool = False
    reference_actions: list[Action] = Field(default_factory=list)
