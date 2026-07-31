from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

from bim_gym.contract.enums import EntityType
from bim_gym.contract.entities import (
    BeamEntity,
    ColumnEntity,
    Entity,
    MepCurveEntity,
    OpeningEntity,
    SlabEntity,
    WallEntity,
)
from bim_gym.contract.geometry import (
    CircularColumnGeometry,
    Point3D,
    RectangularColumnGeometry,
)
from bim_gym.runtime.geometry_util import bbox_center as _bbox_center
from bim_gym.contract.model import EntityId, KulmanBimModel


class Scene:
    """Агентский рантайм над KulmanBimModel: исследование без вьюера."""

    def __init__(self, model: KulmanBimModel) -> None:
        self.model = model

    @classmethod
    def from_json_path(cls, path: str | Path) -> Scene:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(KulmanBimModel.from_raw(data))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Scene:
        return cls(KulmanBimModel.from_raw(data))

    def get(self, entity_id: EntityId) -> Entity:
        try:
            return self.model.entities[entity_id]
        except KeyError as exc:
            raise KeyError(f"unknown entity: {entity_id}") from exc

    def list(
        self,
        type: EntityType | None = None,
        parent: EntityId | None = None,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for entity in self.model.entities.values():
            if type is not None and entity.type != type:
                continue
            if parent is not None and entity.parent != parent:
                continue
            rows.append(
                {
                    "id": entity.id,
                    "type": entity.type.value,
                    "name": entity.name,
                    "parent": entity.parent,
                }
            )
        return rows

    def children(self, entity_id: EntityId) -> list[dict[str, Any]]:
        self.get(entity_id)
        return self.list(parent=entity_id)

    def openings_on(self, host_id: EntityId) -> list[dict[str, Any]]:
        self.get(host_id)
        out: list[dict[str, Any]] = []
        for entity in self.model.entities.values():
            if not isinstance(entity, OpeningEntity):
                continue
            if entity.properties.Hosting.host_id != host_id:
                continue
            geom = entity.properties.OpeningGeometry
            out.append(
                {
                    "id": entity.id,
                    "name": entity.name,
                    "kind": entity.properties.OpeningData.kind.value,
                    "width": geom.width,
                    "height": geom.height,
                    "sill_height": geom.sill_height,
                    "host_offset": geom.host_offset,
                }
            )
        return out

    def get_position(self, entity_id: EntityId) -> Point3D | None:
        entity = self.get(entity_id)
        props = getattr(entity, "properties", None)
        placement = getattr(props, "Placement", None)
        if placement is None:
            return None
        return placement.position

    def measure_distance(self, a: EntityId, b: EntityId) -> float:
        pa = self.get_position(a)
        pb = self.get_position(b)
        if pa is None or pb is None:
            raise ValueError("both entities must have Placement to measure distance")
        return _dist(pa, pb)

    def bbox_center(self, entity_id: EntityId) -> Point3D | None:
        box = self.bbox(entity_id)
        if box is None:
            return None
        return _bbox_center(box)

    def wall_centerline_dir(self, entity_id: EntityId) -> Point3D | None:
        entity = self.get(entity_id)
        if not isinstance(entity, WallEntity):
            return None
        cl = entity.properties.WallGeometry.centerline
        if len(cl) < 2:
            return None
        x0, y0 = cl[0]
        x1, y1 = cl[-1]
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        if length < 1e-9:
            return None
        return (dx / length, dy / length, 0.0)

    def bbox(self, entity_id: EntityId) -> dict[str, Point3D] | None:
        entity = self.get(entity_id)
        box = _local_bbox(entity)
        if box is None:
            return None
        placement = getattr(entity.properties, "Placement", None)
        if placement is None:
            return {"min": box[0], "max": box[1]}
        # Пока без полного поворота: только translation (+ scale, если есть).
        ox, oy, oz = placement.position
        sx, sy, sz = placement.scale or (1.0, 1.0, 1.0)
        (x0, y0, z0), (x1, y1, z1) = box
        return {
            "min": (ox + x0 * sx, oy + y0 * sy, oz + z0 * sz),
            "max": (ox + x1 * sx, oy + y1 * sy, oz + z1 * sz),
        }

    def summary(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for entity in self.model.entities.values():
            key = entity.type.value
            counts[key] = counts.get(key, 0) + 1
        return {
            "format": self.model.format,
            "version": self.model.version,
            "revision": self.model.revision,
            "roots": list(self.model.roots),
            "entity_counts": counts,
            "n_entities": len(self.model.entities),
        }


def _dist(a: Point3D, b: Point3D) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _local_bbox(entity: Entity) -> tuple[Point3D, Point3D] | None:
    if isinstance(entity, WallEntity):
        g = entity.properties.WallGeometry
        xs = [p[0] for p in g.centerline]
        ys = [p[1] for p in g.centerline]
        half = g.thickness / 2.0
        z0 = g.base_offset
        z1 = g.base_offset + g.height
        return (min(xs) - half, min(ys) - half, z0), (max(xs) + half, max(ys) + half, z1)

    if isinstance(entity, SlabEntity):
        g = entity.properties.SlabGeometry
        xs = [p[0] for p in g.outline.outer]
        ys = [p[1] for p in g.outline.outer]
        if g.thickness_direction.value == "up":
            z0, z1 = 0.0, g.thickness
        else:
            z0, z1 = -g.thickness, 0.0
        return (min(xs), min(ys), z0), (max(xs), max(ys), z1)

    if isinstance(entity, ColumnEntity):
        g = entity.properties.ColumnGeometry
        z0 = g.base_offset
        z1 = g.base_offset + g.height
        if isinstance(g, RectangularColumnGeometry):
            hx, hy = g.width / 2.0, g.depth / 2.0
            return (-hx, -hy, z0), (hx, hy, z1)
        if isinstance(g, CircularColumnGeometry):
            r = g.diameter / 2.0
            return (-r, -r, z0), (r, r, z1)

    if isinstance(entity, BeamEntity):
        g = entity.properties.BeamGeometry
        xs = [p[0] for p in g.centerline]
        ys = [p[1] for p in g.centerline]
        zs = [p[2] for p in g.centerline]
        hx, hy = g.width / 2.0, g.height / 2.0
        return (
            (min(xs) - hx, min(ys) - hy, min(zs) - hy),
            (max(xs) + hx, max(ys) + hy, max(zs) + hy),
        )

    if isinstance(entity, MepCurveEntity):
        pts: list[Point3D] = []
        for seg in entity.properties.LineGeometry.lines:
            pts.append(seg.start)
            pts.append(seg.end)
        if not pts:
            return None
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        zs = [p[2] for p in pts]
        r = (entity.properties.MepCurveData.diameter or 0.0) / 2.0
        return (min(xs) - r, min(ys) - r, min(zs) - r), (
            max(xs) + r,
            max(ys) + r,
            max(zs) + r,
        )

    return None
