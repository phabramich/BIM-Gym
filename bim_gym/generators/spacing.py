from __future__ import annotations

import random

from bim_gym.dsl.actions import PredicateSpec, move
from bim_gym.dsl.enums import PredicateKind
from bim_gym.generators._util import pick, rename_scene_entities, round_mm, sample_range
from bim_gym.generators.config import TaskGeneratorConfig
from bim_gym.scenes import add_column, default_placement, make_room_box
from bim_gym.tasks.spec import TaskSpec

_SPACING_MOVE_INSTRUCTIONS: tuple[str, ...] = (
    "Move {movable} so {a} and {b} are about {target:.0f} mm apart.",
    "Adjust {movable} until the distance from the other target column is near {target:.0f} mm.",
    "Reposition {movable} to achieve roughly {target:.0f} mm between {a} and {b}.",
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
    axis: str,
) -> dict:
    width = max(
        round_mm(sample_range(rng, *config.room_width_range)),
        spacing + 2.0 * config.column_size + 1000.0,
    )
    depth = max(
        round_mm(sample_range(rng, *config.room_depth_range)),
        spacing + 2.0 * config.column_size + 1000.0,
    )
    col = config.column_size
    ax = width * 0.25
    ay = depth / 2.0
    bx = ax + (spacing if axis == "x" else 0.0)
    by = ay + (spacing if axis == "y" else 0.0)

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
        placement=default_placement((bx, by, 0.0)),
    )
    add_column(
        scene,
        "column-distractor",
        "storey-1",
        width=col * 0.75,
        depth=col * 0.75,
        height=config.storey_height,
        placement=default_placement((width * 0.7, depth * 0.25, 0.0)),
    )
    return scene


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
        static_count = round(count * config.spacing_static_fraction)
        is_static = i < static_count
        axis = "x" if i % 2 == 0 else "y"
        movable = "column-b" if (i // 2) % 2 == 0 else "column-a"
        expected = True

        if is_static:
            expected = i % 4 != 3
            actual_spacing = target if expected else target + rng.uniform(400.0, 800.0)
            scene = _two_columns_scene(config, rng, spacing=actual_spacing, axis=axis)
            instruction = pick(rng, _SPACING_QA_INSTRUCTIONS).format(target=target)
            reference_actions: list = []
            static = True
        else:
            wrong_spacing = target + rng.choice([-1, 1]) * rng.uniform(400.0, 800.0)
            scene = _two_columns_scene(config, rng, spacing=wrong_spacing, axis=axis)
            signed_delta = target - wrong_spacing
            if movable == "column-a":
                signed_delta = -signed_delta
            offset = (signed_delta, 0.0, 0.0) if axis == "x" else (0.0, signed_delta, 0.0)
            instruction = pick(rng, _SPACING_MOVE_INSTRUCTIONS).format(
                target=target, movable=movable, a="column-a", b="column-b"
            )
            reference_actions = [move(movable, offset)]
            static = False

        token = f"{id_offset + i:04d}"
        mapping = {
            "column-a": f"member-{token}-p",
            "column-b": f"member-{token}-q",
            "column-distractor": f"member-{token}-x",
        }
        scene = rename_scene_entities(scene, mapping)
        for old_id, new_id in mapping.items():
            instruction = instruction.replace(old_id, new_id)
        goal_a = mapping["column-a"]
        goal_b = mapping["column-b"]
        if reference_actions:
            original = reference_actions[0]
            reference_actions = [original.model_copy(
                update={"entity_id": mapping[original.entity_id]}
            )]

        tasks.append(
            TaskSpec(
                id=f"gen_spacing_{id_offset + i:04d}",
                instruction=instruction,
                static=static,
                scene=scene,
                goals=[
                    PredicateSpec(
                        kind=PredicateKind.DistanceInRange,
                        a=goal_a,
                        b=goal_b,
                        lo=lo,
                        hi=hi,
                        expected=expected,
                    ),
                ],
                reference_actions=reference_actions,
            )
        )

    return tasks
