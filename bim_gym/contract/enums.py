from enum import Enum


class EntityType(str, Enum):
    Site = "Site"
    ModelRef = "ModelRef"
    Quarter = "Quarter"
    Section = "Section"
    Storey = "Storey"
    StoreyZone = "StoreyZone"
    Wall = "Wall"
    Slab = "Slab"
    Column = "Column"
    Beam = "Beam"
    Opening = "Opening"
    MepCurve = "MepCurve"
    GeometryDefinition = "GeometryDefinition"


class CommonPropertyType(str, Enum):
    Placement = "Placement"
    WorldPlacement = "WorldPlacement"
    ModelRef = "ModelRef"
    Hosting = "Hosting"
    PolygonGeometry = "PolygonGeometry"
    MultiPolygonGeometry = "MultiPolygonGeometry"
    ExtrudedPolygonGeometry = "ExtrudedPolygonGeometry"
    ExtrudedMultiPolygonGeometry = "ExtrudedMultiPolygonGeometry"
    MappedGeometry = "MappedGeometry"
    LineGeometry = "LineGeometry"
    MeshGeometry = "MeshGeometry"
    BoxGeometry = "BoxGeometry"
    WallGeometry = "WallGeometry"
    SlabGeometry = "SlabGeometry"
    ColumnGeometry = "ColumnGeometry"
    BeamGeometry = "BeamGeometry"
    OpeningGeometry = "OpeningGeometry"


class EntitySpecificPropertyType(str, Enum):
    QuarterData = "QuarterData"
    QuarterConfiguration = "QuarterConfiguration"
    SectionData = "SectionData"
    StoreyData = "StoreyData"
    StoreyZoneData = "StoreyZoneData"
    WallData = "WallData"
    SlabData = "SlabData"
    ColumnData = "ColumnData"
    BeamData = "BeamData"
    OpeningData = "OpeningData"
    MepCurveData = "MepCurveData"


class SectionType(str, Enum):
    Row = "row"
    Corner = "corner"
    RectangularCorner = "rectangular_corner"
    Tower = "tower"
    CentralCoreTower = "central_core_tower"
    CentralCoreSingleLinkTower = "central_core_single_link_tower"
    CentralCoreUrbanVilla = "central_core_urban_villa"
    Gap = "gap"
    Custom = "custom"


class PlacementDirection(str, Enum):
    N = "N"
    NE = "NE"
    E = "E"
    ES = "ES"
    S = "S"
    SW = "SW"
    W = "W"
    WN = "WN"


class CardinalDirection(str, Enum):
    N = "N"
    E = "E"
    S = "S"
    W = "W"


class StoreyType(str, Enum):
    Underground = "underground"
    First = "first"
    Typical = "typical"
    Technical = "technical"
    TechnicalWithTerrace = "technical_with_terrace"
    Special = "special"
    Roof = "roof"
    Custom = "custom"


class ZoningType(str, Enum):
    BasementStorey = "basement_storey"
    FirstStorey = "first_storey"
    TypicalStorey = "typical_storey"
    SpecialStorey = "special_storey"
    ParkingAdjacent = "parking_adjacent"
    Custom = "custom"


class ZoneType(str, Enum):
    Apartment = "apartment"
    Common = "common"
    Core = "core"
    Mep = "mep"
    Retail = "retail"
    Technical = "technical"
    Subzone = "subzone"
    Draft = "draft"
    DraftCore = "draft_core"


class ApartmentType(str, Enum):
    Studio = "0C"
    OneBedroom = "1C"
    TwoBedrooms = "2C"
    ThreeBedrooms = "3C"
    FourBedrooms = "4C"


class WallKind(str, Enum):
    Interior = "interior"
    Exterior = "exterior"
    Retaining = "retaining"
    Generic = "generic"


class SlabKind(str, Enum):
    Floor = "floor"
    Roof = "roof"
    Landing = "landing"
    Generic = "generic"


class ColumnKind(str, Enum):
    Architectural = "architectural"
    Structural = "structural"
    Generic = "generic"


class BeamKind(str, Enum):
    Architectural = "architectural"
    Structural = "structural"
    Generic = "generic"


class OpeningKind(str, Enum):
    Door = "door"
    Window = "window"
    Void = "void"


class MepCurveKind(str, Enum):
    Pipe = "pipe"
    Duct = "duct"
    CableTray = "cable_tray"
    Conduit = "conduit"


class ColumnProfile(str, Enum):
    Rectangular = "rectangular"
    Circular = "circular"


class SlabThicknessDirection(str, Enum):
    Down = "down"
    Up = "up"


class LivingClass(str, Enum):
    A = "A"
    B = "B"
    C = "C"


class ConstructionTechnology(str, Enum):
    Monolith = "monolith"
    Prefab = "prefab"


class StaircaseUnitType(str, Enum):
    Monolithic = "monolithic"
    Prefabricated = "prefabricated"


class StoreyHeightStandard(str, Enum):
    New = "new"
    Old = "old"


class Side(str, Enum):
    Left = "left"
    Right = "right"
    Bottom = "bottom"
    Top = "top"
