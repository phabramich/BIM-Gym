from __future__ import annotations

from enum import Enum


class SftMessageRole(str, Enum):
    System = "system"
    User = "user"
    Assistant = "assistant"


class AssistantContentFormat(str, Enum):
    BimGymActionsV1 = "bim_gym_actions_v1"
