from __future__ import annotations

from enum import Enum


class ActionKind(str, Enum):
    Query = "query"
    Note = "note"
    Recall = "recall"
    Move = "move"
    Place = "place"
    Rotate = "rotate"
    Assert = "assert"
    NoOp = "noop"


class QueryOp(str, Enum):
    List = "list"
    Get = "get"
    Children = "children"
    OpeningsOn = "openings_on"
    MeasureDistance = "measure_distance"
    Bbox = "bbox"
    Summary = "summary"


class PredicateKind(str, Enum):
    HostedBy = "hosted_by"
    ClearOf = "clear_of"
    Adjacent = "adjacent"
    Inside = "inside"
    AlignedWith = "aligned_with"
    DistanceInRange = "distance_in_range"
