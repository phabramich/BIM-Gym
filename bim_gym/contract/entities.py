from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from bim_gym.contract.components import (
    BeamData,
    ColumnData,
    Hosting,
    MepCurveData,
    ModelRef,
    OpeningData,
    Placement,
    QuarterConfiguration,
    QuarterData,
    SectionData,
    SlabData,
    StoreyData,
    StoreyZoneData,
    WallData,
    WorldPlacement,
)
from bim_gym.contract.enums import EntityType
from bim_gym.contract.geometry import (
    BeamGeometry,
    BoxGeometry,
    CircularColumnGeometry,
    ExtrudedMultiPolygonGeometry,
    ExtrudedPolygonGeometry,
    LineGeometry,
    MeshGeometry,
    MultiPolygonGeometry,
    OpeningGeometry,
    PolygonGeometry,
    RectangularColumnGeometry,
    SlabGeometry,
    WallGeometry,
)

EntityId: TypeAlias = str


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PropertiesModel(BaseModel):
    """Ключи properties — PascalCase, как имена компонентов в формате."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class BaseEntity(ContractModel):
    id: EntityId
    name: str
    parent: EntityId | None = None
    tags: tuple[str, ...] = ()


class SiteProperties(PropertiesModel):
    Placement: Placement
    WorldPlacement: WorldPlacement


class SiteEntity(BaseEntity):
    type: Literal[EntityType.Site] = EntityType.Site
    parent: None = None
    properties: SiteProperties


class ModelRefProperties(PropertiesModel):
    Placement: Placement
    ModelRef: ModelRef


class ModelRefEntity(BaseEntity):
    type: Literal[EntityType.ModelRef] = EntityType.ModelRef
    parent: EntityId
    properties: ModelRefProperties


class QuarterProperties(PropertiesModel):
    Placement: Placement
    QuarterData: QuarterData
    QuarterConfiguration: QuarterConfiguration


class QuarterEntity(BaseEntity):
    type: Literal[EntityType.Quarter] = EntityType.Quarter
    parent: None = None
    properties: QuarterProperties


class SectionProperties(PropertiesModel):
    Placement: Placement
    SectionData: SectionData


class SectionEntity(BaseEntity):
    type: Literal[EntityType.Section] = EntityType.Section
    properties: SectionProperties


class StoreyProperties(PropertiesModel):
    Placement: Placement
    StoreyData: StoreyData


class StoreyEntity(BaseEntity):
    type: Literal[EntityType.Storey] = EntityType.Storey
    properties: StoreyProperties


class StoreyZonePolygonProperties(PropertiesModel):
    Placement: Placement
    StoreyZoneData: StoreyZoneData
    PolygonGeometry: PolygonGeometry


class StoreyZoneMultiPolygonProperties(PropertiesModel):
    Placement: Placement
    StoreyZoneData: StoreyZoneData
    MultiPolygonGeometry: MultiPolygonGeometry


class StoreyZoneEntity(BaseEntity):
    type: Literal[EntityType.StoreyZone] = EntityType.StoreyZone
    parent: EntityId
    properties: StoreyZonePolygonProperties | StoreyZoneMultiPolygonProperties


class WallProperties(PropertiesModel):
    Placement: Placement
    WallGeometry: WallGeometry
    WallData: WallData


class WallEntity(BaseEntity):
    type: Literal[EntityType.Wall] = EntityType.Wall
    parent: EntityId
    properties: WallProperties


class SlabProperties(PropertiesModel):
    Placement: Placement
    SlabGeometry: SlabGeometry
    SlabData: SlabData


class SlabEntity(BaseEntity):
    type: Literal[EntityType.Slab] = EntityType.Slab
    parent: EntityId
    properties: SlabProperties


class ColumnProperties(PropertiesModel):
    Placement: Placement
    ColumnGeometry: RectangularColumnGeometry | CircularColumnGeometry
    ColumnData: ColumnData


class ColumnEntity(BaseEntity):
    type: Literal[EntityType.Column] = EntityType.Column
    parent: EntityId
    properties: ColumnProperties


class BeamProperties(PropertiesModel):
    Placement: Placement
    BeamGeometry: BeamGeometry
    BeamData: BeamData


class BeamEntity(BaseEntity):
    type: Literal[EntityType.Beam] = EntityType.Beam
    parent: EntityId
    properties: BeamProperties


class OpeningProperties(PropertiesModel):
    Hosting: Hosting
    OpeningGeometry: OpeningGeometry
    OpeningData: OpeningData


class OpeningEntity(BaseEntity):
    type: Literal[EntityType.Opening] = EntityType.Opening
    parent: EntityId
    properties: OpeningProperties


class MepCurveProperties(PropertiesModel):
    Placement: Placement
    LineGeometry: LineGeometry
    MepCurveData: MepCurveData


class MepCurveEntity(BaseEntity):
    type: Literal[EntityType.MepCurve] = EntityType.MepCurve
    parent: EntityId
    properties: MepCurveProperties


class GeometryDefinitionPolygon(PropertiesModel):
    PolygonGeometry: PolygonGeometry


class GeometryDefinitionMultiPolygon(PropertiesModel):
    MultiPolygonGeometry: MultiPolygonGeometry


class GeometryDefinitionExtrudedPolygon(PropertiesModel):
    ExtrudedPolygonGeometry: ExtrudedPolygonGeometry


class GeometryDefinitionExtrudedMultiPolygon(PropertiesModel):
    ExtrudedMultiPolygonGeometry: ExtrudedMultiPolygonGeometry


class GeometryDefinitionLine(PropertiesModel):
    LineGeometry: LineGeometry


class GeometryDefinitionMesh(PropertiesModel):
    MeshGeometry: MeshGeometry


class GeometryDefinitionBox(PropertiesModel):
    BoxGeometry: BoxGeometry


class GeometryDefinitionEntity(BaseEntity):
    type: Literal[EntityType.GeometryDefinition] = EntityType.GeometryDefinition
    parent: None = None
    properties: (
        GeometryDefinitionPolygon
        | GeometryDefinitionMultiPolygon
        | GeometryDefinitionExtrudedPolygon
        | GeometryDefinitionExtrudedMultiPolygon
        | GeometryDefinitionLine
        | GeometryDefinitionMesh
        | GeometryDefinitionBox
    )


Entity = Annotated[
    SiteEntity
    | ModelRefEntity
    | QuarterEntity
    | SectionEntity
    | StoreyEntity
    | StoreyZoneEntity
    | WallEntity
    | SlabEntity
    | ColumnEntity
    | BeamEntity
    | OpeningEntity
    | MepCurveEntity
    | GeometryDefinitionEntity,
    Field(discriminator="type"),
]
