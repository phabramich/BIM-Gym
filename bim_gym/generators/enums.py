from __future__ import annotations

from enum import Enum


class TaskFamilyKind(str, Enum):
    HostingQA = "hosting_qa"
    Clearance = "clearance"
    Spacing = "spacing"
