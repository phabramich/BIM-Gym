from __future__ import annotations

import math
from dataclasses import dataclass

from bim_gym.contract.entities import EntityId, OpeningEntity
from bim_gym.contract.geometry import Point3D
from bim_gym.runtime.geometry_util import (
    ADJACENT_TOLERANCE_MM,
    ALIGNMENT_DOT_TOLERANCE,
    BBox,
    DISTANCE_TOLERANCE_MM,
    TOLERANCE_MM,
    bbox_center,
    bbox_gap,
    clamp01,
    dist3,
    face_adjacent,
    point_in_bbox,
)
from bim_gym.runtime.scene import Scene

__all__ = [
    "ADJACENT_TOLERANCE_MM",
    "ALIGNMENT_DOT_TOLERANCE",
    "DISTANCE_TOLERANCE_MM",
    "TOLERANCE_MM",
    "PredicateResult",
    "adjacent",
    "aligned_with",
    "bbox_clear_of",
    "clear_of",
    "distance_between",
    "distance_in_range",
    "evaluate",
    "hosted_by",
    "host_offset_in_range",
    "inside",
    "inside_bbox",
]


@dataclass(frozen=True)
class PredicateResult:
    ok: bool
    score: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score must be in [0, 1], got {self.score}")


def evaluate(results: list[PredicateResult]) -> PredicateResult:
    if not results:
        return PredicateResult(ok=True, score=1.0)
    return PredicateResult(
        ok=all(r.ok for r in results),
        score=sum(r.score for r in results) / len(results),
    )


def hosted_by(scene: Scene, opening_id: EntityId, host_id: EntityId) -> PredicateResult:
    entity = scene.get(opening_id)
    if not isinstance(entity, OpeningEntity):
        return PredicateResult(ok=False, score=0.0)
    actual = entity.properties.Hosting.host_id
    ok = actual == host_id
    return PredicateResult(ok=ok, score=1.0 if ok else 0.0)


def _entity_point(scene: Scene, entity_id: EntityId) -> Point3D:
    pos = scene.get_position(entity_id)
    if pos is not None:
        return pos
    box = scene.bbox(entity_id)
    if box is None:
        raise ValueError(f"entity {entity_id} has no placement or bbox")
    return bbox_center(box)


def distance_between(scene: Scene, a: EntityId, b: EntityId) -> float:
    return dist3(_entity_point(scene, a), _entity_point(scene, b))


def distance_in_range(
    scene: Scene,
    a: EntityId,
    b: EntityId,
    lo: float,
    hi: float,
    tol: float = DISTANCE_TOLERANCE_MM,
) -> PredicateResult:
    d = distance_between(scene, a, b)
    ok = lo - tol <= d <= hi + tol
    if ok:
        return PredicateResult(ok=True, score=1.0)
    if d < lo:
        score = clamp01(1.0 - (lo - d) / max(lo, 1.0))
    else:
        score = clamp01(1.0 - (d - hi) / max(hi, 1.0))
    return PredicateResult(ok=False, score=score)


def _require_bbox(scene: Scene, entity_id: EntityId) -> BBox:
    box = scene.bbox(entity_id)
    if box is None:
        raise ValueError(f"entity {entity_id} has no bbox")
    return box


def bbox_clear_of(a: BBox, b: BBox, margin: float = TOLERANCE_MM) -> PredicateResult:
    gap = bbox_gap(a, b)
    ok = gap >= margin
    if ok:
        score = 1.0
    elif margin <= 0:
        score = 0.0
    else:
        score = clamp01(gap / margin)
    return PredicateResult(ok=ok, score=score)


def clear_of(
    scene: Scene,
    a: EntityId,
    b: EntityId,
    margin: float = TOLERANCE_MM,
) -> PredicateResult:
    return bbox_clear_of(_require_bbox(scene, a), _require_bbox(scene, b), margin)


def adjacent(
    scene: Scene,
    a: EntityId,
    b: EntityId,
    tol: float = ADJACENT_TOLERANCE_MM,
) -> PredicateResult:
    box_a = _require_bbox(scene, a)
    box_b = _require_bbox(scene, b)
    ok = face_adjacent(box_a, box_b, tol)
    gap = bbox_gap(box_a, box_b)
    if ok:
        score = 1.0
    elif tol <= 0:
        score = 0.0
    else:
        score = clamp01(1.0 - gap / tol)
    return PredicateResult(ok=ok, score=score)


def inside_bbox(point: Point3D, box: BBox, margin: float = TOLERANCE_MM) -> PredicateResult:
    ok = point_in_bbox(point, box, margin)
    if ok:
        return PredicateResult(ok=True, score=1.0)
    mn, mx = box["min"], box["max"]
    dx = max(mn[0] - point[0], point[0] - mx[0], 0.0)
    dy = max(mn[1] - point[1], point[1] - mx[1], 0.0)
    dz = max(mn[2] - point[2], point[2] - mx[2], 0.0)
    outside = math.sqrt(dx * dx + dy * dy + dz * dz)
    score = clamp01(1.0 - outside / max(margin, 1.0))
    return PredicateResult(ok=False, score=score)


def inside(
    scene: Scene,
    entity_id: EntityId,
    container_id: EntityId,
    margin: float = TOLERANCE_MM,
) -> PredicateResult:
    box = _require_bbox(scene, container_id)
    point = bbox_center(_require_bbox(scene, entity_id))
    return inside_bbox(point, box, margin)


def host_offset_in_range(
    scene: Scene,
    opening_id: EntityId,
    lo: float,
    hi: float,
    tol: float = DISTANCE_TOLERANCE_MM,
) -> PredicateResult:
    entity = scene.get(opening_id)
    if not isinstance(entity, OpeningEntity):
        return PredicateResult(ok=False, score=0.0)
    value = entity.properties.OpeningGeometry.host_offset
    ok = lo - tol <= value <= hi + tol
    if ok:
        return PredicateResult(ok=True, score=1.0)
    if value < lo:
        score = clamp01(1.0 - (lo - value) / max(abs(lo) + 1.0, 1.0))
    else:
        score = clamp01(1.0 - (value - hi) / max(abs(hi) + 1.0, 1.0))
    return PredicateResult(ok=False, score=score)


def aligned_with(
    scene: Scene,
    a: EntityId,
    b: EntityId,
    tol: float = ALIGNMENT_DOT_TOLERANCE,
) -> PredicateResult:
    da = scene.wall_centerline_dir(a)
    db = scene.wall_centerline_dir(b)
    if da is None or db is None:
        return PredicateResult(ok=False, score=0.0)
    dot = abs(da[0] * db[0] + da[1] * db[1] + da[2] * db[2])
    ok = dot >= tol
    score = clamp01(dot)
    return PredicateResult(ok=ok, score=score)
