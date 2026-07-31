from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from bim_gym.contract.enums import ColumnProfile, SlabThicknessDirection

# Длины — мм (для gym предпочтительны int).
Point2D: TypeAlias = tuple[float, float]
Point3D: TypeAlias = tuple[float, float, float]
EulerAngles: TypeAlias = tuple[float, float, float]
PolygonPoints: TypeAlias = Annotated[list[Point2D], Field(min_length=3)]
Polyline2D: TypeAlias = Annotated[list[Point2D], Field(min_length=2)]
Polyline3D: TypeAlias = Annotated[list[Point3D], Field(min_length=2)]


class ContractModel(BaseModel):
    """Листовые поля в JSON — camelCase (совместимость с боевым форматом)."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        alias_generator=to_camel,
    )


class PolygonData(ContractModel):
    outer: PolygonPoints
    holes: list[PolygonPoints] | None = None


class PolygonGeometry(PolygonData):
    pass


class MultiPolygonGeometry(ContractModel):
    polygons: Annotated[list[PolygonData], Field(min_length=1)]


class ExtrudedPolygonGeometry(PolygonData):
    height: float


class ExtrudedMultiPolygonGeometry(MultiPolygonGeometry):
    height: float


class LineSegment(ContractModel):
    start: Point3D
    end: Point3D


class LineGeometry(ContractModel):
    lines: list[LineSegment]


class MeshGeometry(ContractModel):
    vertices: list[Point3D]
    indices: list[int]
    normals: list[Point3D] | None = None
    uvs: list[Point2D] | None = None


class BoxGeometry(ContractModel):
    min: Point3D
    max: Point3D


class WallGeometry(ContractModel):
    """Осевая (location curve) + толщина + высота, как Wall в Revit."""

    centerline: Polyline2D
    thickness: float
    height: float
    base_offset: float = 0.0


class SlabGeometry(ContractModel):
    outline: PolygonData
    thickness: float
    thickness_direction: SlabThicknessDirection = SlabThicknessDirection.Down


class RectangularColumnGeometry(ContractModel):
    profile: Literal[ColumnProfile.Rectangular] = ColumnProfile.Rectangular
    width: float
    depth: float
    height: float
    base_offset: float = 0.0


class CircularColumnGeometry(ContractModel):
    profile: Literal[ColumnProfile.Circular] = ColumnProfile.Circular
    diameter: float
    height: float
    base_offset: float = 0.0


ColumnGeometry = Annotated[
    RectangularColumnGeometry | CircularColumnGeometry,
    Field(discriminator="profile"),
]


class BeamGeometry(ContractModel):
    centerline: Polyline3D
    width: float
    height: float


class OpeningGeometry(ContractModel):
    width: float
    height: float
    sill_height: float
    host_offset: float
