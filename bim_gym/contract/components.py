from __future__ import annotations

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel

from bim_gym.contract.enums import (
    ApartmentType,
    BeamKind,
    CardinalDirection,
    ColumnKind,
    ConstructionTechnology,
    LivingClass,
    MepCurveKind,
    OpeningKind,
    PlacementDirection,
    SectionType,
    Side,
    SlabKind,
    StaircaseUnitType,
    StoreyHeightStandard,
    StoreyType,
    WallKind,
    ZoneType,
    ZoningType,
)
from bim_gym.contract.geometry import (
    EulerAngles,
    MultiPolygonGeometry,
    Point3D,
    PolygonGeometry,
)


class ContractModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        alias_generator=to_camel,
    )


class Placement(ContractModel):
    position: Point3D
    # Углы Эйлера (X, Y, Z) в радианах — наследие формата; gym лучше держать градусы int.
    rotation: EulerAngles
    scale: Point3D | None = None


class WorldPlacement(ContractModel):
    latitude: float
    longitude: float
    rotation_angle_true_north: float
    elevation: float


class ModelRef(ContractModel):
    model_id: str


class Hosting(ContractModel):
    host_id: str


class MappedGeometry(ContractModel):
    geometry_definition_id: str


class StoreyHeights(ContractModel):
    underground: float
    first: float
    typical: float
    special: float
    technical: float


class SectionParameters(ContractModel):
    living_class: LivingClass
    construction_technology: ConstructionTechnology
    step_size: float
    frame_element_step: float
    staircase_unit_type: StaircaseUnitType
    storey_height_standard: StoreyHeightStandard
    storey_heights: StoreyHeights


class QuarterData(ContractModel):
    code: str
    climatic_region: str
    collection_name: str
    default_section_parameters: SectionParameters


class ParkingSize(ContractModel):
    width_steps: int
    length_steps: int


class QuarterConfigurationSection(ContractModel):
    section_id: str
    section_configuration: str
    placement_direction: PlacementDirection
    storeys_count: int
    has_joint_after_section: bool


class QuarterArchConfig(ContractModel):
    side: CardinalDirection
    location: Side
    storey_count: int = 2


class QuarterConfiguration(ContractModel):
    quarter_name: str
    climatic_region: str
    collection_name: str
    living_class: LivingClass
    construction_technology: ConstructionTechnology
    frame_element_step: float
    staircase_unit_type: StaircaseUnitType
    storey_height_standard: StoreyHeightStandard
    parking_size: ParkingSize
    sections: list[QuarterConfigurationSection]
    arch: QuarterArchConfig | None = None


class SectionShapeParameters(ContractModel):
    size_a_steps: int
    depth_a_steps: int
    size_b_steps: int | None = None
    depth_b_steps: int | None = None


class SectionCoreTemplate(ContractModel):
    id: str
    name: str
    geometry: PolygonGeometry | MultiPolygonGeometry
    placement: Placement
    parts: list[PolygonGeometry]


class SectionData(ContractModel):
    section_type: SectionType
    section_configuration: str
    mirrored: bool
    shape: SectionShapeParameters
    parameters: SectionParameters
    core_template: SectionCoreTemplate | None = None


class StoreyData(ContractModel):
    storey_index: int
    gross_height: float
    net_height: float
    storey_type: StoreyType
    zoning_type: ZoningType
    is_entrance_level: bool
    zoning_variant_id: str | None = None


class StoreyZoneData(ContractModel):
    zone_type: ZoneType
    is_external: bool
    apartment_type: ApartmentType | None = None


class WallData(ContractModel):
    kind: WallKind
    structural: bool


class SlabData(ContractModel):
    kind: SlabKind
    structural: bool


class ColumnData(ContractModel):
    kind: ColumnKind
    structural: bool


class BeamData(ContractModel):
    kind: BeamKind
    structural: bool


class OpeningData(ContractModel):
    kind: OpeningKind


class MepCurveData(ContractModel):
    kind: MepCurveKind
    diameter: float | None = None
    width: float | None = None
    height: float | None = None
    system_id: str | None = None
