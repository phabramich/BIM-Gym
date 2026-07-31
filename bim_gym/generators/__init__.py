from __future__ import annotations

import random

from bim_gym.generators.clearance import generate_clearance
from bim_gym.generators.config import TaskGeneratorConfig
from bim_gym.generators.enums import TaskFamilyKind
from bim_gym.generators.hosting_qa import generate_hosting_qa
from bim_gym.generators.spacing import generate_spacing
from bim_gym.generators._util import allocate_counts
from bim_gym.tasks.spec import TaskSpec

_FAMILY_GENERATORS = {
    TaskFamilyKind.HostingQA: generate_hosting_qa,
    TaskFamilyKind.Clearance: generate_clearance,
    TaskFamilyKind.Spacing: generate_spacing,
}


def generate(config: TaskGeneratorConfig) -> list[TaskSpec]:
    rng = random.Random(config.seed)
    counts = allocate_counts(config.n_tasks, config.families)
    tasks: list[TaskSpec] = []
    id_offset = 0

    for family in config.families:
        count = counts[family]
        if count == 0:
            continue
        generator = _FAMILY_GENERATORS[family]
        batch = generator(config, rng, count, id_offset=id_offset)
        tasks.extend(batch)
        id_offset += count

    return tasks


__all__ = [
    "TaskFamilyKind",
    "TaskGeneratorConfig",
    "generate",
]
