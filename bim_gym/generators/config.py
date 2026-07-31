from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from bim_gym.generators.enums import TaskFamilyKind


class TaskGeneratorConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    seed: int = 0
    n_tasks: int = 20
    families: tuple[TaskFamilyKind, ...] = (
        TaskFamilyKind.HostingQA,
        TaskFamilyKind.Clearance,
        TaskFamilyKind.Spacing,
    )
    room_width_range: tuple[float, float] = (3500.0, 5500.0)
    room_depth_range: tuple[float, float] = (2800.0, 4200.0)
    wall_thickness: float = 200.0
    storey_height: float = 3000.0
    column_size: float = 400.0
    clearance_margin_range: tuple[float, float] = (50.0, 80.0)
    spacing_target_range: tuple[float, float] = (1800.0, 2200.0)
    spacing_tolerance: float = 100.0
    spacing_static_fraction: float = 0.4
