from __future__ import annotations

import random

from bim_gym.dsl.actions import PredicateSpec, move
from bim_gym.dsl.enums import PredicateKind
from bim_gym.generators._util import pick, rename_scene_entities, round_mm, sample_range
from bim_gym.generators.config import TaskGeneratorConfig
from bim_gym.runtime.predicates import clear_of
from bim_gym.runtime.scene import Scene
from bim_gym.scenes import add_column, default_placement, make_room_box
from bim_gym.scenes.room_box import RoomBoxWall
from bim_gym.tasks.spec import TaskSpec

_WALL_ID = {
    RoomBoxWall.South: "wall-s",
    RoomBoxWall.East: "wall-e",
    RoomBoxWall.North: "wall-n",
    RoomBoxWall.West: "wall-w",
}
_INWARD = {
    RoomBoxWall.South: (0.0, 1.0),
    RoomBoxWall.East: (-1.0, 0.0),
    RoomBoxWall.North: (0.0, -1.0),
    RoomBoxWall.West: (1.0, 0.0),
}
_INSTRUCTIONS = (
    "Move {column} clear of the {wall} wall by at least {margin:.1f} mm.",
    "Reposition {column} away from {wall_id}; required clearance is {margin:.1f} mm.",
    "Resolve the {column}/{wall_id} clash and leave a {margin:.1f} mm gap.",
)


def _clearance_scene(
    config: TaskGeneratorConfig,
    rng: random.Random,
    wall: RoomBoxWall,
) -> tuple[dict, float, tuple[float, float, float]]:
    width = round_mm(sample_range(rng, *config.room_width_range))
    depth = round_mm(sample_range(rng, *config.room_depth_range))
    margin = round_mm(sample_range(rng, *config.clearance_margin_range))
    col = config.column_size
    scene = make_room_box(
        origin=(0.0, 0.0),
        size_xy=(width, depth),
        wall_thickness=config.wall_thickness,
        height=config.storey_height,
        with_door=False,
    )
    wall_id = _WALL_ID[wall]
    box = Scene.from_dict(scene).bbox(wall_id)
    if box is None:
        raise RuntimeError(f"{wall_id} bbox missing")
    ix, iy = _INWARD[wall]
    overlap = round_mm(rng.uniform(15.0, min(60.0, col / 3.0)))
    cx = (box["min"][0] + box["max"][0]) / 2.0
    cy = (box["min"][1] + box["max"][1]) / 2.0
    if ix > 0:
        cx = box["max"][0] + col / 2.0 - overlap
    elif ix < 0:
        cx = box["min"][0] - col / 2.0 + overlap
    elif iy > 0:
        cy = box["max"][1] + col / 2.0 - overlap
    else:
        cy = box["min"][1] - col / 2.0 + overlap
    add_column(
        scene,
        "column-1",
        "storey-1",
        width=col,
        depth=col,
        height=config.storey_height,
        placement=default_placement((cx, cy, 0.0)),
    )
    # A distractor makes entity selection non-trivial without affecting the goal.
    add_column(
        scene,
        "column-2",
        "storey-1",
        width=col * 0.8,
        depth=col * 0.8,
        height=config.storey_height,
        placement=default_placement((width * 0.35, depth * 0.55, 0.0)),
    )
    distance = margin + overlap + round_mm(rng.uniform(30.0, 120.0))
    offset = (round_mm(ix * distance), round_mm(iy * distance), 0.0)
    if clear_of(Scene.from_dict(scene), "column-1", wall_id, margin).ok:
        raise RuntimeError("clearance fixture already satisfies its goal")
    return scene, margin, offset


def generate_clearance(
    config: TaskGeneratorConfig,
    rng: random.Random,
    count: int,
    *,
    id_offset: int = 0,
) -> list[TaskSpec]:
    tasks: list[TaskSpec] = []
    walls = tuple(RoomBoxWall)
    for i in range(count):
        wall = walls[i % len(walls)]
        scene, margin, offset = _clearance_scene(config, rng, wall)
        wall_id = _WALL_ID[wall]
        token = f"{id_offset + i:04d}"
        mapping = {
            wall_id: f"barrier-{token}",
            "column-1": f"member-{token}",
            "column-2": f"member-{token}-x",
        }
        scene = rename_scene_entities(scene, mapping)
        wall_id = mapping[wall_id]
        column_id = mapping["column-1"]
        instruction = pick(rng, _INSTRUCTIONS).format(
            column=column_id, wall=wall.value, wall_id=wall_id, margin=margin
        )
        tasks.append(TaskSpec(
            id=f"gen_clearance_{id_offset + i:04d}",
            instruction=instruction,
            scene=scene,
            goals=[PredicateSpec(
                kind=PredicateKind.ClearOf,
                a=column_id,
                b=wall_id,
                margin=margin,
            )],
            reference_actions=[move(column_id, offset)],
        ))
    return tasks
