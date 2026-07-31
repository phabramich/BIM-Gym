from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from bim_gym.dsl.enums import ActionKind, PredicateKind, QueryOp


class PredicateSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: PredicateKind
    a: str
    b: str | None = None
    lo: float | None = None
    hi: float | None = None
    margin: float | None = None
    tol: float | None = None


class Action(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: ActionKind
    op: QueryOp | None = None
    entity_id: str | None = None
    entity_type: str | None = None
    parent: str | None = None
    a: str | None = None
    b: str | None = None
    key: str | None = None
    value: Any | None = None
    items: list[PredicateSpec] = Field(default_factory=list)
    offset: tuple[float, float, float] | None = None
    position: tuple[float, float, float] | None = None
    host_offset: float | None = None
    host_offset_delta: float | None = None


def query(op: QueryOp, **kwargs: Any) -> Action:
    return Action(kind=ActionKind.Query, op=op, **kwargs)


def note(key: str, value: Any) -> Action:
    return Action(kind=ActionKind.Note, key=key, value=value)


def recall(key: str) -> Action:
    return Action(kind=ActionKind.Recall, key=key)


def assert_predicates(items: list[PredicateSpec]) -> Action:
    return Action(kind=ActionKind.Assert, items=items)


def noop() -> Action:
    return Action(kind=ActionKind.NoOp)


def move(
    entity_id: str,
    offset: tuple[float, float, float],
) -> Action:
    return Action(kind=ActionKind.Move, entity_id=entity_id, offset=offset)


def place(entity_id: str, position: tuple[float, float, float]) -> Action:
    return Action(kind=ActionKind.Place, entity_id=entity_id, position=position)
