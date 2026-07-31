from __future__ import annotations

from typing import Any

from bim_gym.contract.components import Placement
from bim_gym.contract.entities import Entity
from bim_gym.contract.geometry import EulerAngles, Point3D
from bim_gym.contract.model import (
    KULMAN_BIM_MODEL_FORMAT,
    KULMAN_BIM_MODEL_VERSION,
)

_ZERO_ROTATION: EulerAngles = (0.0, 0.0, 0.0)


def default_placement(position: Point3D = (0.0, 0.0, 0.0)) -> Placement:
    return Placement(position=position, rotation=_ZERO_ROTATION)


def new_scene_dict(*, revision: int = 1) -> dict[str, Any]:
    return {
        "format": KULMAN_BIM_MODEL_FORMAT,
        "version": KULMAN_BIM_MODEL_VERSION,
        "revision": revision,
        "roots": [],
        "entities": {},
    }


def dump_entity(entity: Entity) -> dict[str, Any]:
    return entity.model_dump(mode="json", by_alias=True)


def add_entity(scene_dict: dict[str, Any], entity: Entity, *, root: bool = False) -> None:
    scene_dict["entities"][entity.id] = dump_entity(entity)
    if root and entity.id not in scene_dict["roots"]:
        scene_dict["roots"].append(entity.id)
