from __future__ import annotations

from enum import Enum
from typing import Any

from bim_gym.contract.enums import OpeningKind, WallKind
from bim_gym.scenes._core import new_scene_dict
from bim_gym.scenes.builders import (
    add_opening_on_wall,
    add_slab,
    add_storey,
    add_wall,
)


class RoomBoxWall(str, Enum):
    South = "south"
    East = "east"
    North = "north"
    West = "west"


def make_room_box(
    origin: tuple[float, float],
    size_xy: tuple[float, float],
    wall_thickness: float,
    height: float,
    *,
    storey_id: str = "storey-1",
    with_door: bool = True,
    door_wall: RoomBoxWall = RoomBoxWall.South,
    door_width: float = 900.0,
    door_height: float = 2100.0,
) -> dict[str, Any]:
    """Прямоугольная комната: этаж, 4 стены, плита пола, опционально дверь."""
    ox, oy = origin
    wx, wy = size_xy
    t = wall_thickness
    half = t / 2.0

    scene = new_scene_dict()
    add_storey(scene, storey_id, height_mm=height)

    wall_specs: list[tuple[str, RoomBoxWall, list[tuple[float, float]]]] = [
        ("wall-s", RoomBoxWall.South, [(ox - half, oy - half), (ox + wx + half, oy - half)]),
        ("wall-e", RoomBoxWall.East, [(ox + wx + half, oy - half), (ox + wx + half, oy + wy + half)]),
        ("wall-n", RoomBoxWall.North, [(ox + wx + half, oy + wy + half), (ox - half, oy + wy + half)]),
        ("wall-w", RoomBoxWall.West, [(ox - half, oy + wy + half), (ox - half, oy - half)]),
    ]

    for wall_id, side, centerline in wall_specs:
        add_wall(
            scene,
            wall_id,
            storey_id,
            centerline=centerline,
            thickness=t,
            height=height,
            kind=WallKind.Exterior,
        )
        if with_door and side == door_wall:
            seg_len = _polyline_length(centerline)
            host_offset = max(0.0, (seg_len - door_width) / 2.0)
            add_opening_on_wall(
                scene,
                f"opening-{wall_id}",
                storey_id,
                wall_id,
                host_offset=host_offset,
                width=door_width,
                height=door_height,
                kind=OpeningKind.Door,
            )

    slab_outline = [
        (ox, oy),
        (ox + wx, oy),
        (ox + wx, oy + wy),
        (ox, oy + wy),
    ]
    add_slab(scene, "slab-floor", storey_id, outline=slab_outline, thickness=t)

    return scene


def _polyline_length(points: list[tuple[float, float]]) -> float:
    total = 0.0
    for i in range(len(points) - 1):
        x0, y0 = points[i]
        x1, y1 = points[i + 1]
        total += ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
    return total
