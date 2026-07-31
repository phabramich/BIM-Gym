from __future__ import annotations

import math

from bim_gym.contract.geometry import Point3D

BBox = dict[str, Point3D]

# Revit-подобные допуски, мм
TOLERANCE_MM = 1.0
ADJACENT_TOLERANCE_MM = 2.0
DISTANCE_TOLERANCE_MM = 1.0
ALIGNMENT_DOT_TOLERANCE = 0.999  # ~2.5°


def bbox_center(box: BBox) -> Point3D:
    mn, mx = box["min"], box["max"]
    return ((mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2, (mn[2] + mx[2]) / 2)


def point_in_bbox(point: Point3D, box: BBox, margin: float = 0.0) -> bool:
    x, y, z = point
    mn, mx = box["min"], box["max"]
    return (
        mn[0] - margin <= x <= mx[0] + margin
        and mn[1] - margin <= y <= mx[1] + margin
        and mn[2] - margin <= z <= mx[2] + margin
    )


def bbox_gap(a: BBox, b: BBox) -> float:
    dx = max(0.0, max(a["min"][0] - b["max"][0], b["min"][0] - a["max"][0]))
    dy = max(0.0, max(a["min"][1] - b["max"][1], b["min"][1] - a["max"][1]))
    dz = max(0.0, max(a["min"][2] - b["max"][2], b["min"][2] - a["max"][2]))
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def bboxes_overlap(a: BBox, b: BBox, margin: float = 0.0) -> bool:
    return bbox_gap(a, b) <= margin


def axis_overlap(lo1: float, hi1: float, lo2: float, hi2: float, tol: float) -> bool:
    return lo1 <= hi2 + tol and lo2 <= hi1 + tol


def face_adjacent(a: BBox, b: BBox, tol: float) -> bool:
    gap = bbox_gap(a, b)
    if gap > tol:
        return False
    ox = axis_overlap(a["min"][0], a["max"][0], b["min"][0], b["max"][0], tol)
    oy = axis_overlap(a["min"][1], a["max"][1], b["min"][1], b["max"][1], tol)
    oz = axis_overlap(a["min"][2], a["max"][2], b["min"][2], b["max"][2], tol)
    return (ox and oy) or (ox and oz) or (oy and oz)


def dist3(a: Point3D, b: Point3D) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def normalize3(v: Point3D) -> Point3D | None:
    length = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if length < 1e-9:
        return None
    return (v[0] / length, v[1] / length, v[2] / length)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))
