from __future__ import annotations

from pydantic import BaseModel, ConfigDict, model_validator

from bim_gym.generators.enums import TaskFamilyKind


class TaskGeneratorConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    seed: int = 0
    n_tasks: int = 20
    families: tuple[TaskFamilyKind, ...] = (
        TaskFamilyKind.HostingQA,
        TaskFamilyKind.Clearance,
        TaskFamilyKind.Spacing,
        TaskFamilyKind.OpeningShift,
        TaskFamilyKind.AlignmentQA,
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

    @model_validator(mode="after")
    def _validate_ranges(self) -> TaskGeneratorConfig:
        if self.n_tasks < 0:
            raise ValueError("n_tasks must be non-negative")
        if not self.families:
            raise ValueError("families must not be empty")
        if len(set(self.families)) != len(self.families):
            raise ValueError("families must be unique")
        for name in (
            "room_width_range", "room_depth_range", "clearance_margin_range",
            "spacing_target_range",
        ):
            lo, hi = getattr(self, name)
            if lo <= 0 or hi < lo:
                raise ValueError(f"invalid {name}: {(lo, hi)}")
        if not 0.0 <= self.spacing_static_fraction <= 1.0:
            raise ValueError("spacing_static_fraction must be in [0, 1]")
        if min(self.wall_thickness, self.storey_height, self.column_size) <= 0:
            raise ValueError("scene dimensions must be positive")
        return self
