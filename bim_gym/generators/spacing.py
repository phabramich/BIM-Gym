from __future__ import annotations

import random

from bim_gym.dsl.actions import PredicateSpec, move
from bim_gym.dsl.enums import PredicateKind
from bim_gym.generators._util import pick, round_mm, sample_range
from bim_gym.generators.config import TaskGeneratorConfig
from bim_gym.scenes import add_column, default_placement, make_room_box
from bim_gym.tasks.spec import TaskSpec

_SPACING_MOVE_INSTRUCTIONS: tuple[str, ...] = (
    "Move column-b so the columns are about {target:.0f} mm apart.",
    "Adjust column-b until the column spacing is near {target:.0f} mm.",
    "Reposition column-b to achieve roughly {target:.0f} mm between columns.",
)

_SPACING_QA_INSTRUCTIONS: tuple[str, ...] = (
    "Verify the columns are about {target:.0f} mm apart.",
    "Check whether the distance between the columns is near {target:.0f} mm.",
    "Confirm the column spacing is approximately {target:.0f} mm.",
)


def _two_columns_scene(
    config: TaskGeneratorConfig,
    rng: random.Random,
    *,
    spacing: float,
) -> tuple[dict, float]:
    width = max(
        round_mm(sample_range(rng, *config.room_width_range)),
        spacing + 2.0 * config.column_size + 1000.0,
    )
    depth = round_mm(sample_range(rng, *config.room_depth_range))
    col = config.column_size
    ax = width * 0.25
    ay = depth / 2.0
    bx = ax + spacing

    scene = make_room_box(
        origin=(0.0, 0.0),
        size_xy=(width, depth),
        wall_thickness=config.wall_thickness,
        height=config.storey_height,
        with_door=False,
    )
    add_column(
        scene,
        "column-a",
        "storey-1",
        width=col,
        depth=col,
        height=config.storey_height,
        placement=default_placement((ax, ay, 0.0)),
    )
    add_column(
        scene,
        "column-b",
        "storey-1",
        width=col,
        depth=col,
        height=config.storey_height,
        placement=default_placement((bx, ay, 0.0)),
    )
    return scene, ax


def generate_spacing(
    config: TaskGeneratorConfig,
    rng: random.Random,
    count: int,
    *,
    id_offset: int = 0,
) -> list[TaskSpec]:
    tasks: list[TaskSpec] = []
    half_tol = config.spacing_tolerance / 2.0

    for i in range(count):
        target = round_mm(sample_range(rng, *config.spacing_target_range))
        lo = target - half_tol
        hi = target + half_tol
        is_static = rng.random() < config.spacing_static_fraction

        if is_static:
            scene, _ = _two_columns_scene(config, rng, spacing=target)
            instruction = pick(rng, _SPACING_QA_INSTRUCTIONS).format(target=target)
            reference_actions: list = []
            static = True
        else:
            wrong_spacing = target + rng.choice([-1, 1]) * rng.uniform(400.0, 800.0)
            scene, _ = _two_columns_scene(config, rng, spacing=wrong_spacing)
            move_dx = target - wrong_spacing
            instruction = pick(rng, _SPACING_MOVE_INSTRUCTIONS).format(target=target)
            reference_actions = [move("column-b", (move_dx, 0.0, 0.0))]
            static = False

        tasks.append(
            TaskSpec(
                id=f"gen_spacing_{id_offset + i:04d}",
                instruction=instruction,
                static=static,
                scene=scene,
                goals=[
                    PredicateSpec(
                        kind=PredicateKind.DistanceInRange,
                        a="column-a",
                        b="column-b",
                        lo=lo,
                        hi=hi,
                    ),
                ],
                reference_actions=reference_actions,
            )
        )

    return tasks
