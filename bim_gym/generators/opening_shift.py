from __future__ import annotations

import random

from bim_gym.dsl.actions import PredicateSpec, move
from bim_gym.dsl.enums import PredicateKind
from bim_gym.generators._util import pick, rename_scene_entities, round_mm, sample_range
from bim_gym.generators.config import TaskGeneratorConfig
from bim_gym.scenes import make_room_box
from bim_gym.runtime.scene import Scene
from bim_gym.scenes.room_box import RoomBoxWall
from bim_gym.tasks.runner import evaluate_task
from bim_gym.tasks.spec import TaskSpec

_WALL_LABELS: dict[RoomBoxWall, str] = {
    RoomBoxWall.South: "south",
    RoomBoxWall.East: "east",
    RoomBoxWall.North: "north",
    RoomBoxWall.West: "west",
}

_INSTRUCTIONS: tuple[str, ...] = (
    "Slide the door along the {wall} wall to host_offset ≈ {target:.0f} mm.",
    "Move the opening on the {wall} wall so its host_offset is about {target:.0f} mm.",
    "Reposition the door on the {wall} wall toward host_offset {target:.0f} mm.",
    "Shift the {wall}-wall door until host_offset is near {target:.0f} mm.",
)


def generate_opening_shift(
    config: TaskGeneratorConfig,
    rng: random.Random,
    count: int,
    *,
    id_offset: int = 0,
) -> list[TaskSpec]:
    tasks: list[TaskSpec] = []
    walls = tuple(RoomBoxWall)
    tol = 25.0

    for i in range(count):
        door_wall = pick(rng, walls)
        wall_id = f"wall-{door_wall.value[0]}"
        opening_id = f"opening-{wall_id}"
        width = round_mm(sample_range(rng, *config.room_width_range))
        depth = round_mm(sample_range(rng, *config.room_depth_range))
        label = _WALL_LABELS[door_wall]

        scene = make_room_box(
            origin=(0.0, 0.0),
            size_xy=(width, depth),
            wall_thickness=config.wall_thickness,
            height=config.storey_height,
            with_door=True,
            door_wall=door_wall,
        )
        current = float(
            scene["entities"][opening_id]["properties"]["OpeningGeometry"]["hostOffset"]
        )
        target = round_mm(current + rng.choice([-1.0, 1.0]) * rng.uniform(250.0, 700.0))
        runtime_scene = Scene.from_dict(scene)
        opening = runtime_scene.get(opening_id)
        wall = runtime_scene.get(wall_id)
        centerline = wall.properties.WallGeometry.centerline
        wall_length = sum(
            ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
            for (x0, y0), (x1, y1) in zip(centerline, centerline[1:])
        )
        max_offset = wall_length - opening.properties.OpeningGeometry.width
        target = min(max(80.0, target), max_offset - 80.0)
        token = f"{id_offset + i:04d}"
        mapping = {wall_id: f"host-{token}", opening_id: f"opening-{token}"}
        scene = rename_scene_entities(scene, mapping)
        wall_id = mapping[wall_id]
        opening_id = mapping[opening_id]

        goals = [
            PredicateSpec(
                kind=PredicateKind.HostOffsetInRange,
                a=opening_id,
                lo=target - tol,
                hi=target + tol,
                tol=1.0,
            ),
            PredicateSpec(kind=PredicateKind.HostedBy, a=opening_id, b=wall_id),
        ]

        task = TaskSpec(
            id=f"gen_opening_shift_{id_offset + i:04d}",
            instruction=pick(rng, _INSTRUCTIONS).format(wall=label, target=target),
            scene=scene,
            goals=goals,
            reference_actions=[move(opening_id, host_offset=target)],
        )

        if evaluate_task(task).success:
            raise RuntimeError(f"opening_shift leak before move: {task.id}")

        tasks.append(task)

    return tasks
