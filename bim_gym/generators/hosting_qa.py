from __future__ import annotations

import random

from bim_gym.dsl.actions import PredicateSpec
from bim_gym.dsl.enums import PredicateKind
from bim_gym.generators._util import pick, rename_scene_entities, round_mm, sample_range
from bim_gym.generators.config import TaskGeneratorConfig
from bim_gym.scenes import make_room_box
from bim_gym.scenes.room_box import RoomBoxWall
from bim_gym.tasks.spec import TaskSpec

_WALL_LABELS: dict[RoomBoxWall, str] = {
    RoomBoxWall.South: "south",
    RoomBoxWall.East: "east",
    RoomBoxWall.North: "north",
    RoomBoxWall.West: "west",
}

_INSTRUCTION_TEMPLATES: tuple[str, ...] = (
    "Is the door hosted by the {wall} wall?",
    "Verify the door is on the {wall} wall.",
    "Check whether the opening belongs to the {wall} wall.",
    "Confirm the door is hosted by the {wall} wall.",
    "Does the {wall} wall host the door opening?",
    "Report if the door's host is the {wall} wall.",
)


def generate_hosting_qa(
    config: TaskGeneratorConfig,
    rng: random.Random,
    count: int,
    *,
    id_offset: int = 0,
) -> list[TaskSpec]:
    tasks: list[TaskSpec] = []
    walls = tuple(RoomBoxWall)

    for i in range(count):
        door_wall = pick(rng, walls)
        expected = i % 2 == 0
        asked_wall = door_wall
        if not expected:
            alternatives = tuple(wall for wall in walls if wall != door_wall)
            asked_wall = pick(rng, alternatives)
        wall_id = f"wall-{asked_wall.value[0]}"
        opening_id = f"opening-{wall_id}"
        width = round_mm(sample_range(rng, *config.room_width_range))
        depth = round_mm(sample_range(rng, *config.room_depth_range))
        label = _WALL_LABELS[asked_wall]
        instruction = pick(rng, _INSTRUCTION_TEMPLATES).format(wall=label)

        scene = make_room_box(
            origin=(0.0, 0.0),
            size_xy=(width, depth),
            wall_thickness=config.wall_thickness,
            height=config.storey_height,
            with_door=True,
            door_wall=door_wall,
        )
        token = f"{id_offset + i:04d}"
        actual_opening = f"opening-wall-{door_wall.value[0]}"
        mapping = {
            wall_id: f"host-{token}-{asked_wall.value[0]}",
            actual_opening: f"opening-{token}",
        }
        scene = rename_scene_entities(scene, mapping)
        wall_id = mapping[wall_id]
        actual_opening = mapping[actual_opening]

        tasks.append(
            TaskSpec(
                id=f"gen_hosting_qa_{id_offset + i:04d}",
                instruction=instruction,
                static=True,
                scene=scene,
                goals=[PredicateSpec(
                    kind=PredicateKind.HostedBy,
                    a=actual_opening,
                    b=wall_id,
                    expected=expected,
                )],
            )
        )

    return tasks
