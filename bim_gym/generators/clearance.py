from __future__ import annotations

import random

from bim_gym.dsl.actions import PredicateSpec, move
from bim_gym.dsl.enums import PredicateKind
from bim_gym.generators._util import round_mm, sample_range
from bim_gym.generators.config import TaskGeneratorConfig
from bim_gym.scenes import add_column, default_placement, make_room_box
from bim_gym.tasks.spec import TaskSpec

_CLEAR_INSTRUCTIONS: tuple[str, ...] = (
    "Move the column so it is clear of the south wall.",
    "Reposition the column away from the south wall.",
    "Shift the column until it clears the south wall by the required margin.",
)


def _room_with_column_near_south_wall(
    config: TaskGeneratorConfig,
    rng: random.Random,
) -> tuple[dict, float, float, float]:
    width = round_mm(sample_range(rng, *config.room_width_range))
    depth = round_mm(sample_range(rng, *config.room_depth_range))
    margin = round_mm(sample_range(rng, *config.clearance_margin_range))
    col = config.column_size
    overlap = rng.uniform(5.0, 35.0)
    start_y = config.wall_thickness / 2.0 + col / 2.0 - overlap
    move_dy = margin + overlap + 50.0

    scene = make_room_box(
        origin=(0.0, 0.0),
        size_xy=(width, depth),
        wall_thickness=config.wall_thickness,
        height=config.storey_height,
        with_door=False,
    )
    add_column(
        scene,
        "column-1",
        "storey-1",
        width=col,
        depth=col,
        height=config.storey_height,
        placement=default_placement((width / 2.0, start_y, 0.0)),
    )
    return scene, margin, 0.0, move_dy


def generate_clearance(
    config: TaskGeneratorConfig,
    rng: random.Random,
    count: int,
    *,
    id_offset: int = 0,
) -> list[TaskSpec]:
    tasks: list[TaskSpec] = []

    for i in range(count):
        scene, margin, dx, dy = _room_with_column_near_south_wall(config, rng)
        instruction = _CLEAR_INSTRUCTIONS[i % len(_CLEAR_INSTRUCTIONS)]

        tasks.append(
            TaskSpec(
                id=f"gen_clearance_{id_offset + i:04d}",
                instruction=instruction,
                scene=scene,
                goals=[
                    PredicateSpec(
                        kind=PredicateKind.ClearOf,
                        a="column-1",
                        b="wall-s",
                        margin=margin,
                    ),
                ],
                reference_actions=[move("column-1", (dx, dy, 0.0))],
            )
        )

    return tasks
