from __future__ import annotations

import random

from bim_gym.dsl.actions import PredicateSpec
from bim_gym.dsl.enums import PredicateKind
from bim_gym.generators._util import pick, rename_scene_entities, round_mm, sample_range
from bim_gym.generators.config import TaskGeneratorConfig
from bim_gym.scenes import make_room_box
from bim_gym.tasks.spec import TaskSpec

_PARALLEL_PAIRS = (("wall-s", "wall-n"), ("wall-e", "wall-w"))
_PERPENDICULAR_PAIRS = (("wall-s", "wall-e"), ("wall-e", "wall-n"),
                        ("wall-n", "wall-w"), ("wall-w", "wall-s"))
_INSTRUCTIONS = (
    "Check whether {a} is aligned with {b}.",
    "Verify the alignment relationship between {a} and {b}.",
    "Is {a} parallel-aligned with {b}?",
)


def generate_alignment_qa(
    config: TaskGeneratorConfig,
    rng: random.Random,
    count: int,
    *,
    id_offset: int = 0,
) -> list[TaskSpec]:
    tasks: list[TaskSpec] = []
    for i in range(count):
        expected = i % 2 == 0
        a, b = pick(rng, _PARALLEL_PAIRS if expected else _PERPENDICULAR_PAIRS)
        scene = make_room_box(
            origin=(0.0, 0.0),
            size_xy=(
                round_mm(sample_range(rng, *config.room_width_range)),
                round_mm(sample_range(rng, *config.room_depth_range)),
            ),
            wall_thickness=config.wall_thickness,
            height=config.storey_height,
            with_door=rng.random() < 0.5,
        )
        token = f"{id_offset + i:04d}"
        mapping = {wall: f"face-{token}-{side}" for wall, side in (
            ("wall-s", "q"), ("wall-e", "r"), ("wall-n", "t"), ("wall-w", "u")
        )}
        scene = rename_scene_entities(scene, mapping)
        a, b = mapping[a], mapping[b]
        tasks.append(TaskSpec(
            id=f"gen_alignment_qa_{id_offset + i:04d}",
            instruction=pick(rng, _INSTRUCTIONS).format(a=a, b=b),
            static=True,
            scene=scene,
            goals=[PredicateSpec(
                kind=PredicateKind.AlignedWith,
                a=a,
                b=b,
                expected=expected,
            )],
        ))
    return tasks
