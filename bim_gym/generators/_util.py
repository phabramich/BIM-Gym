from __future__ import annotations

import random
from typing import Any, TypeVar

from bim_gym.generators.enums import TaskFamilyKind

T = TypeVar("T")


def allocate_counts(
    n_tasks: int,
    families: tuple[TaskFamilyKind, ...],
) -> dict[TaskFamilyKind, int]:
    base, remainder = divmod(n_tasks, len(families))
    counts = {family: base for family in families}
    for family in families[:remainder]:
        counts[family] += 1
    return counts


def pick(rng: random.Random, items: tuple[T, ...]) -> T:
    return items[rng.randrange(len(items))]


def sample_range(rng: random.Random, lo: float, hi: float) -> float:
    return rng.uniform(lo, hi)


def round_mm(value: float) -> float:
    return round(value, 1)


def rename_scene_entities(scene: dict[str, Any], mapping: dict[str, str]) -> dict[str, Any]:
    """Rename entity IDs and every exact ID reference in a raw scene."""
    def _rename(value: Any) -> Any:
        if isinstance(value, dict):
            return {mapping.get(str(key), str(key)): _rename(item) for key, item in value.items()}
        if isinstance(value, list):
            return [_rename(item) for item in value]
        if isinstance(value, tuple):
            return tuple(_rename(item) for item in value)
        if isinstance(value, str):
            return mapping.get(value, value)
        return value

    return _rename(scene)
