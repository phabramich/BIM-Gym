from __future__ import annotations

from typing import Any

from bim_gym.contract.components import (
    BeamData,
    ColumnData,
    Hosting,
    MepCurveData,
    OpeningData,
    Placement,
    SlabData,
    StoreyData,
    WallData,
)
from bim_gym.contract.entities import (
    BeamEntity,
    ColumnEntity,
    MepCurveEntity,
    OpeningEntity,
    SlabEntity,
    StoreyEntity,
    WallEntity,
)
from bim_gym.contract.enums import (
    BeamKind,
    ColumnKind,
    ColumnProfile,
    MepCurveKind,
    OpeningKind,
    SlabKind,
    SlabThicknessDirection,
    StoreyType,
    WallKind,
    ZoningType,
)
from bim_gym.contract.geometry import (
    BeamGeometry,
    LineGeometry,
    LineSegment,
    OpeningGeometry,
    Point2D,
    Point3D,
    PolygonData,
    Polyline2D,
    Polyline3D,
    RectangularColumnGeometry,
    SlabGeometry,
    WallGeometry,
)
from bim_gym.scenes._core import add_entity, default_placement, dump_entity


def build_storey(
    storey_id: str,
    height_mm: float = 3000.0,
    *,
    name: str | None = None,
    storey_index: int = 0,
    storey_type: StoreyType = StoreyType.Typical,
    placement: Placement | None = None,
) -> dict[str, Any]:
    entity = StoreyEntity(
        id=storey_id,
        name=name or f"Storey {storey_index}",
        properties={
            "Placement": placement or default_placement(),
            "StoreyData": StoreyData(
                storey_index=storey_index,
                gross_height=height_mm,
                net_height=height_mm,
                storey_type=storey_type,
                zoning_type=ZoningType.TypicalStorey,
                is_entrance_level=storey_index == 0,
            ),
        },
    )
    return dump_entity(entity)


def add_storey(
    scene_dict: dict[str, Any],
    storey_id: str,
    height_mm: float = 3000.0,
    **kwargs: Any,
) -> str:
    entity = StoreyEntity(
        id=storey_id,
        name=kwargs.pop("name", None) or f"Storey {kwargs.get('storey_index', 0)}",
        properties={
            "Placement": kwargs.pop("placement", None) or default_placement(),
            "StoreyData": StoreyData(
                storey_index=kwargs.pop("storey_index", 0),
                gross_height=height_mm,
                net_height=height_mm,
                storey_type=kwargs.pop("storey_type", StoreyType.Typical),
                zoning_type=kwargs.pop("zoning_type", ZoningType.TypicalStorey),
                is_entrance_level=kwargs.pop("is_entrance_level", False),
            ),
        },
    )
    add_entity(scene_dict, entity, root=True)
    return storey_id


def add_wall(
    scene_dict: dict[str, Any],
    wall_id: str,
    parent_id: str,
    *,
    name: str | None = None,
    centerline: Polyline2D,
    thickness: float,
    height: float,
    kind: WallKind = WallKind.Exterior,
    structural: bool = True,
    base_offset: float = 0.0,
    placement: Placement | None = None,
) -> str:
    entity = WallEntity(
        id=wall_id,
        name=name or wall_id,
        parent=parent_id,
        properties={
            "Placement": placement or default_placement(),
            "WallGeometry": WallGeometry(
                centerline=centerline,
                thickness=thickness,
                height=height,
                base_offset=base_offset,
            ),
            "WallData": WallData(kind=kind, structural=structural),
        },
    )
    add_entity(scene_dict, entity)
    return wall_id


def add_opening_on_wall(
    scene_dict: dict[str, Any],
    opening_id: str,
    parent_id: str,
    host_wall_id: str,
    *,
    name: str | None = None,
    host_offset: float,
    width: float,
    height: float,
    kind: OpeningKind = OpeningKind.Door,
    sill_height: float | None = None,
) -> str:
    if sill_height is None:
        sill_height = 0.0 if kind == OpeningKind.Door else 900.0
    entity = OpeningEntity(
        id=opening_id,
        name=name or opening_id,
        parent=parent_id,
        properties={
            "Hosting": Hosting(host_id=host_wall_id),
            "OpeningGeometry": OpeningGeometry(
                width=width,
                height=height,
                sill_height=sill_height,
                host_offset=host_offset,
            ),
            "OpeningData": OpeningData(kind=kind),
        },
    )
    add_entity(scene_dict, entity)
    return opening_id


def add_slab(
    scene_dict: dict[str, Any],
    slab_id: str,
    parent_id: str,
    *,
    name: str | None = None,
    outline: list[Point2D],
    thickness: float,
    kind: SlabKind = SlabKind.Floor,
    structural: bool = True,
    thickness_direction: SlabThicknessDirection = SlabThicknessDirection.Down,
    placement: Placement | None = None,
) -> str:
    entity = SlabEntity(
        id=slab_id,
        name=name or slab_id,
        parent=parent_id,
        properties={
            "Placement": placement or default_placement(),
            "SlabGeometry": SlabGeometry(
                outline=PolygonData(outer=outline),
                thickness=thickness,
                thickness_direction=thickness_direction,
            ),
            "SlabData": SlabData(kind=kind, structural=structural),
        },
    )
    add_entity(scene_dict, entity)
    return slab_id


def add_column(
    scene_dict: dict[str, Any],
    column_id: str,
    parent_id: str,
    *,
    name: str | None = None,
    width: float,
    depth: float,
    height: float,
    kind: ColumnKind = ColumnKind.Structural,
    structural: bool = True,
    base_offset: float = 0.0,
    placement: Placement | None = None,
) -> str:
    entity = ColumnEntity(
        id=column_id,
        name=name or column_id,
        parent=parent_id,
        properties={
            "Placement": placement or default_placement(),
            "ColumnGeometry": RectangularColumnGeometry(
                profile=ColumnProfile.Rectangular,
                width=width,
                depth=depth,
                height=height,
                base_offset=base_offset,
            ),
            "ColumnData": ColumnData(kind=kind, structural=structural),
        },
    )
    add_entity(scene_dict, entity)
    return column_id


def add_mep_curve(
    scene_dict: dict[str, Any],
    curve_id: str,
    parent_id: str,
    *,
    name: str | None = None,
    centerline: Polyline3D,
    kind: MepCurveKind = MepCurveKind.Pipe,
    diameter: float | None = 100.0,
    placement: Placement | None = None,
) -> str:
    lines = [
        LineSegment(start=centerline[i], end=centerline[i + 1])
        for i in range(len(centerline) - 1)
    ]
    entity = MepCurveEntity(
        id=curve_id,
        name=name or curve_id,
        parent=parent_id,
        properties={
            "Placement": placement or default_placement(),
            "LineGeometry": LineGeometry(lines=lines),
            "MepCurveData": MepCurveData(kind=kind, diameter=diameter),
        },
    )
    add_entity(scene_dict, entity)
    return curve_id


def add_pipe(
    scene_dict: dict[str, Any],
    pipe_id: str,
    parent_id: str,
    *,
    centerline: Polyline3D,
    diameter: float = 100.0,
    name: str | None = None,
    placement: Placement | None = None,
) -> str:
    return add_mep_curve(
        scene_dict,
        pipe_id,
        parent_id,
        name=name,
        centerline=centerline,
        kind=MepCurveKind.Pipe,
        diameter=diameter,
        placement=placement,
    )


def add_beam(
    scene_dict: dict[str, Any],
    beam_id: str,
    parent_id: str,
    *,
    name: str | None = None,
    centerline: Polyline3D,
    width: float,
    height: float,
    kind: BeamKind = BeamKind.Structural,
    structural: bool = True,
    placement: Placement | None = None,
) -> str:
    entity = BeamEntity(
        id=beam_id,
        name=name or beam_id,
        parent=parent_id,
        properties={
            "Placement": placement or default_placement(),
            "BeamGeometry": BeamGeometry(
                centerline=centerline,
                width=width,
                height=height,
            ),
            "BeamData": BeamData(kind=kind, structural=structural),
        },
    )
    add_entity(scene_dict, entity)
    return beam_id
