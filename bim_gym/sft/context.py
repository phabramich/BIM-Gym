from __future__ import annotations

import hashlib
import json
from typing import Any

from bim_gym.tasks.runner import build_scene
from bim_gym.tasks.spec import TaskSpec

CONTEXT_SCHEMA = "bim_gym_task_context_v1"
GENERATOR_VERSION = "0.2.0"


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def task_to_jsonable(task: TaskSpec) -> dict[str, Any]:
    return task.model_dump(mode="json")


def task_from_jsonable(value: dict[str, Any]) -> TaskSpec:
    return TaskSpec.model_validate(value)


def task_context(task: TaskSpec) -> dict[str, Any]:
    scene = build_scene(task)
    entities: list[dict[str, Any]] = []
    for entity_id in sorted(scene.model.entities):
        entity = scene.model.entities[entity_id]
        row = entity.model_dump(mode="json")
        row["bbox"] = scene.bbox(entity_id)
        entities.append(row)
    return {
        "schema": CONTEXT_SCHEMA,
        "units": "mm",
        "coordinates": "right-handed XYZ; XY plan; +Z up",
        "scene": {
            "format": scene.model.format,
            "version": scene.model.version,
            "revision": scene.model.revision,
            "roots": sorted(scene.model.roots),
            "entities": entities,
        },
        "goals": [goal.model_dump(mode="json") for goal in task.goals],
    }


def task_fingerprint(task: TaskSpec) -> str:
    semantic = task_to_jsonable(task)
    semantic.pop("id", None)
    semantic.pop("instruction", None)
    return hashlib.sha256(canonical_json(semantic).encode("utf-8")).hexdigest()


def structural_fingerprint(task: TaskSpec) -> str:
    scene = build_scene(task)
    entity_types: dict[str, int] = {}
    parent_edges: dict[tuple[str, str], int] = {}
    for entity in scene.model.entities.values():
        entity_type = entity.type.value
        entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
        if entity.parent and entity.parent in scene.model.entities:
            parent_type = scene.model.entities[entity.parent].type.value
            edge = (parent_type, entity_type)
            parent_edges[edge] = parent_edges.get(edge, 0) + 1
    def _entity_type(entity_id: str | None) -> str | None:
        if entity_id is None or entity_id not in scene.model.entities:
            return None
        return scene.model.entities[entity_id].type.value

    goal_rows = []
    for goal in task.goals:
        goal_rows.append({
            "kind": goal.kind.value,
            "a_type": _entity_type(goal.a),
            "b_type": _entity_type(goal.b),
            "expected": goal.expected,
        })
    action_rows = []
    goal_a = {goal.a for goal in task.goals}
    goal_b = {goal.b for goal in task.goals if goal.b}
    for action in task.reference_actions:
        direction = None
        if action.offset is not None:
            direction = tuple(0 if abs(v) < 1e-9 else (1 if v > 0 else -1) for v in action.offset)
        role = "a" if action.entity_id in goal_a else ("b" if action.entity_id in goal_b else "other")
        action_rows.append({
            "kind": action.kind.value,
            "target_type": _entity_type(action.entity_id),
            "target_role": role,
            "direction": direction,
            "host_offset_mode": "absolute" if action.host_offset is not None else (
                "delta" if action.host_offset_delta is not None else None
            ),
        })
    signature = {
        "static": task.static,
        "entity_types": sorted(entity_types.items()),
        "parent_edges": sorted((a, b, n) for (a, b), n in parent_edges.items()),
        "goals": sorted(goal_rows, key=canonical_json),
        "actions": action_rows,
    }
    return hashlib.sha256(canonical_json(signature).encode("utf-8")).hexdigest()


def difficulty_for(task: TaskSpec) -> dict[str, str]:
    scene = build_scene(task)
    return {
        "entities": "small" if len(scene.model.entities) < 10 else "medium",
        "goals": "single" if len(task.goals) == 1 else "multi",
        "actions": "static" if not task.reference_actions else (
            "single" if len(task.reference_actions) == 1 else "multi"
        ),
    }
