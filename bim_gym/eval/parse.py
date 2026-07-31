from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from bim_gym.dsl.actions import Action
from bim_gym.dsl.enums import ActionKind, QueryOp
from bim_gym.sft.enums import AssistantContentFormat

_KNOWN_KINDS = {kind.value for kind in ActionKind}
_KNOWN_QUERY_OPS = {op.value for op in QueryOp}


def _raw_actions(payload: dict[str, Any]) -> list[Any] | None:
    actions = payload.get("actions")
    if not isinstance(actions, list):
        return None
    return actions


def detect_unknown_primitive(text: str) -> bool:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return False
    if not isinstance(payload, dict):
        return False
    raw_actions = _raw_actions(payload)
    if raw_actions is None:
        return False

    for item in raw_actions:
        if not isinstance(item, dict):
            continue
        kind = item.get("kind")
        if kind is not None and kind not in _KNOWN_KINDS:
            return True
        if kind == ActionKind.Query.value:
            op = item.get("op")
            if op is not None and op not in _KNOWN_QUERY_OPS:
                return True
    return False


def parse_assistant_actions(text: str) -> list[Action] | None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("format") != AssistantContentFormat.BimGymActionsV1.value:
        return None

    raw_actions = _raw_actions(payload)
    if raw_actions is None:
        return None

    actions: list[Action] = []
    for item in raw_actions:
        if not isinstance(item, dict):
            return None
        try:
            actions.append(Action.model_validate(item))
        except ValidationError:
            return None
    return actions
