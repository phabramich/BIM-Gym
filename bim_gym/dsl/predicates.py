from __future__ import annotations

from bim_gym.dsl.actions import PredicateSpec
from bim_gym.dsl.enums import PredicateKind
from bim_gym.runtime.predicates import (
    PredicateResult,
    adjacent,
    aligned_with,
    clear_of,
    distance_in_range,
    hosted_by,
    inside,
)
from bim_gym.runtime.scene import Scene


def evaluate_spec(scene: Scene, spec: PredicateSpec) -> PredicateResult:
    match spec.kind:
        case PredicateKind.HostedBy:
            if spec.b is None:
                raise ValueError("hosted_by requires b (host_id)")
            return hosted_by(scene, spec.a, spec.b)
        case PredicateKind.ClearOf:
            if spec.b is None:
                raise ValueError("clear_of requires b")
            kwargs: dict[str, float] = {}
            if spec.margin is not None:
                kwargs["margin"] = spec.margin
            return clear_of(scene, spec.a, spec.b, **kwargs)
        case PredicateKind.Adjacent:
            if spec.b is None:
                raise ValueError("adjacent requires b")
            kwargs = {}
            if spec.tol is not None:
                kwargs["tol"] = spec.tol
            return adjacent(scene, spec.a, spec.b, **kwargs)
        case PredicateKind.Inside:
            if spec.b is None:
                raise ValueError("inside requires b (container_id)")
            kwargs = {}
            if spec.margin is not None:
                kwargs["margin"] = spec.margin
            return inside(scene, spec.a, spec.b, **kwargs)
        case PredicateKind.AlignedWith:
            if spec.b is None:
                raise ValueError("aligned_with requires b")
            kwargs = {}
            if spec.tol is not None:
                kwargs["tol"] = spec.tol
            return aligned_with(scene, spec.a, spec.b, **kwargs)
        case PredicateKind.DistanceInRange:
            if spec.b is None or spec.lo is None or spec.hi is None:
                raise ValueError("distance_in_range requires b, lo, hi")
            kwargs = {}
            if spec.tol is not None:
                kwargs["tol"] = spec.tol
            return distance_in_range(scene, spec.a, spec.b, spec.lo, spec.hi, **kwargs)
