from bim_gym.runtime.predicates import (
    ADJACENT_TOLERANCE_MM,
    ALIGNMENT_DOT_TOLERANCE,
    DISTANCE_TOLERANCE_MM,
    TOLERANCE_MM,
    PredicateResult,
    adjacent,
    aligned_with,
    bbox_clear_of,
    clear_of,
    distance_between,
    distance_in_range,
    evaluate,
    hosted_by,
    inside,
    inside_bbox,
)
from bim_gym.runtime.scene import Scene

__all__ = [
    "ADJACENT_TOLERANCE_MM",
    "ALIGNMENT_DOT_TOLERANCE",
    "DISTANCE_TOLERANCE_MM",
    "TOLERANCE_MM",
    "PredicateResult",
    "Scene",
    "adjacent",
    "aligned_with",
    "bbox_clear_of",
    "clear_of",
    "distance_between",
    "distance_in_range",
    "Episode",
    "EpisodeState",
    "evaluate",
    "hosted_by",
    "inside",
    "inside_bbox",
    "StepRecord",
]


def __getattr__(name: str):
    if name in {"Episode", "EpisodeState", "StepRecord"}:
        from bim_gym.runtime.episode import Episode, EpisodeState, StepRecord

        return {"Episode": Episode, "EpisodeState": EpisodeState, "StepRecord": StepRecord}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
