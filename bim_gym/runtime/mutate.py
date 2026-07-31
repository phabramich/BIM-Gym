from __future__ import annotations

from bim_gym.contract.entities import (
    BeamEntity,
    ColumnEntity,
    Entity,
    EntityId,
    MepCurveEntity,
    OpeningEntity,
    SlabEntity,
    WallEntity,
)
from bim_gym.contract.geometry import Point3D
from bim_gym.contract.model import KulmanBimModel
from bim_gym.runtime.scene import Scene

_PLACED_ENTITY_TYPES = (
    WallEntity,
    SlabEntity,
    ColumnEntity,
    BeamEntity,
    MepCurveEntity,
)


def _replace_entity(model: KulmanBimModel, entity_id: EntityId, entity: Entity) -> KulmanBimModel:
    entities = dict(model.entities)
    entities[entity_id] = entity
    return model.model_copy(update={"entities": entities, "revision": model.revision + 1})


def _with_placement(entity: Entity, position: Point3D) -> Entity:
    if not isinstance(entity, _PLACED_ENTITY_TYPES):
        raise ValueError(f"entity {entity.id} has no Placement")
    props = entity.properties
    placement = props.Placement
    new_placement = placement.model_copy(update={"position": position})
    new_props = props.model_copy(update={"Placement": new_placement})
    return entity.model_copy(update={"properties": new_props})


def move_entity(scene: Scene, entity_id: EntityId, offset_xyz: Point3D) -> Scene:
    entity = scene.get(entity_id)
    if isinstance(entity, OpeningEntity):
        raise ValueError("openings use host_offset; call move_opening instead")
    if not isinstance(entity, _PLACED_ENTITY_TYPES):
        raise ValueError(f"entity {entity_id} cannot be moved")
    ox, oy, oz = offset_xyz
    px, py, pz = entity.properties.Placement.position
    updated = _with_placement(entity, (px + ox, py + oy, pz + oz))
    return Scene(_replace_entity(scene.model, entity_id, updated))


def place_entity(scene: Scene, entity_id: EntityId, position: Point3D) -> Scene:
    entity = scene.get(entity_id)
    if isinstance(entity, OpeningEntity):
        raise ValueError("openings use host_offset; call move_opening instead")
    if not isinstance(entity, _PLACED_ENTITY_TYPES):
        raise ValueError(f"entity {entity_id} cannot be placed")
    return Scene(_replace_entity(scene.model, entity_id, _with_placement(entity, position)))


def move_opening(
    scene: Scene,
    opening_id: EntityId,
    *,
    host_offset_delta: float | None = None,
    absolute_host_offset: float | None = None,
) -> Scene:
    if host_offset_delta is None and absolute_host_offset is None:
        raise ValueError("move_opening requires host_offset_delta or absolute_host_offset")
    if host_offset_delta is not None and absolute_host_offset is not None:
        raise ValueError("move_opening accepts only one of host_offset_delta or absolute_host_offset")

    entity = scene.get(opening_id)
    if not isinstance(entity, OpeningEntity):
        raise ValueError(f"entity {opening_id} is not an opening")

    geom = entity.properties.OpeningGeometry
    if absolute_host_offset is not None:
        new_offset = absolute_host_offset
    else:
        new_offset = geom.host_offset + host_offset_delta  # type: ignore[operator]

    host = scene.get(entity.properties.Hosting.host_id)
    if not isinstance(host, WallEntity):
        raise ValueError(f"opening {opening_id} host is not a wall")
    centerline = host.properties.WallGeometry.centerline
    host_length = sum(
        ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        for (x0, y0), (x1, y1) in zip(centerline, centerline[1:])
    )
    if not 0.0 <= new_offset <= host_length - geom.width:
        raise ValueError(
            f"opening host_offset {new_offset} outside [0, {host_length - geom.width}]"
        )

    new_geom = geom.model_copy(update={"host_offset": new_offset})
    new_props = entity.properties.model_copy(update={"OpeningGeometry": new_geom})
    updated = entity.model_copy(update={"properties": new_props})
    return Scene(_replace_entity(scene.model, opening_id, updated))
