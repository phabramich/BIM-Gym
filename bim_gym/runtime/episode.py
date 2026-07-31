from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from bim_gym.contract.entities import OpeningEntity
from bim_gym.contract.enums import EntityType
from bim_gym.dsl.actions import Action
from bim_gym.dsl.enums import ActionKind, QueryOp
from bim_gym.dsl.predicates import evaluate_spec
from bim_gym.runtime.mutate import move_entity, move_opening, place_entity
from bim_gym.runtime.predicates import PredicateResult, evaluate as evaluate_predicates
from bim_gym.runtime.scene import Scene


@dataclass
class StepRecord:
    step: int
    action: dict[str, Any]
    observation: dict[str, Any]


@dataclass
class EpisodeState:
    scene: Scene
    memory: dict[str, Any] = field(default_factory=dict)
    history: list[StepRecord] = field(default_factory=list)
    done: bool = False
    success: bool | None = None
    _step: int = 0


class Episode:
    def __init__(self, scene: Scene) -> None:
        self.state = EpisodeState(scene=scene)

    def step(self, action: Action | dict[str, Any]) -> dict[str, Any]:
        if self.state.done:
            return self._obs(ok=False, error="episode already finished")

        try:
            parsed = action if isinstance(action, Action) else Action.model_validate(action)
        except ValidationError as exc:
            return self._record_error(action, f"invalid action: {exc}")

        try:
            observation = self._dispatch(parsed)
        except Exception as exc:  # noqa: BLE001 — агенту нужен структурированный ответ
            return self._record_error(parsed.model_dump(), str(exc))

        self.state._step += 1
        self.state.history.append(
            StepRecord(
                step=self.state._step,
                action=parsed.model_dump(),
                observation=observation,
            )
        )
        return observation

    def _dispatch(self, action: Action) -> dict[str, Any]:
        match action.kind:
            case ActionKind.Query:
                return self._handle_query(action)
            case ActionKind.Note:
                return self._handle_note(action)
            case ActionKind.Recall:
                return self._handle_recall(action)
            case ActionKind.Assert:
                return self._handle_assert(action)
            case ActionKind.NoOp:
                return self._obs(ok=True, result=None)
            case ActionKind.Move:
                return self._handle_move(action)
            case ActionKind.Place:
                return self._handle_place(action)
            case ActionKind.Rotate:
                return self._obs(ok=False, error=f"{action.kind.value} not implemented")
        raise AssertionError(f"unhandled action kind: {action.kind}")

    def _handle_query(self, action: Action) -> dict[str, Any]:
        if action.op is None:
            return self._obs(ok=False, error="query requires op")
        scene = self.state.scene
        match action.op:
            case QueryOp.List:
                entity_type = EntityType(action.entity_type) if action.entity_type else None
                result = scene.list(type=entity_type, parent=action.parent)
            case QueryOp.Get:
                if action.entity_id is None:
                    return self._obs(ok=False, error="get requires entity_id")
                entity = scene.get(action.entity_id)
                result = {
                    "id": entity.id,
                    "type": entity.type.value,
                    "name": entity.name,
                    "parent": entity.parent,
                }
            case QueryOp.Children:
                if action.entity_id is None:
                    return self._obs(ok=False, error="children requires entity_id")
                result = scene.children(action.entity_id)
            case QueryOp.OpeningsOn:
                if action.entity_id is None:
                    return self._obs(ok=False, error="openings_on requires entity_id")
                result = scene.openings_on(action.entity_id)
            case QueryOp.MeasureDistance:
                if action.a is None or action.b is None:
                    return self._obs(ok=False, error="measure_distance requires a and b")
                result = scene.measure_distance(action.a, action.b)
            case QueryOp.Bbox:
                if action.entity_id is None:
                    return self._obs(ok=False, error="bbox requires entity_id")
                result = scene.bbox(action.entity_id)
            case QueryOp.Summary:
                result = scene.summary()
            case _:
                return self._obs(ok=False, error=f"unknown query op: {action.op}")
        return self._obs(ok=True, result=result)

    def _handle_note(self, action: Action) -> dict[str, Any]:
        if action.key is None:
            return self._obs(ok=False, error="note requires key")
        self.state.memory[action.key] = action.value
        return self._obs(ok=True, result={"key": action.key})

    def _handle_recall(self, action: Action) -> dict[str, Any]:
        if action.key is None:
            return self._obs(ok=False, error="recall requires key")
        if action.key not in self.state.memory:
            return self._obs(ok=False, error=f"unknown memory key: {action.key}")
        return self._obs(ok=True, result=self.state.memory[action.key])

    def _handle_assert(self, action: Action) -> dict[str, Any]:
        if not action.items:
            return self._obs(ok=False, error="assert requires items")
        results: list[dict[str, Any]] = []
        predicate_results: list[PredicateResult] = []
        for spec in action.items:
            pr = evaluate_spec(self.state.scene, spec)
            predicate_results.append(pr)
            results.append(
                {
                    "kind": spec.kind.value,
                    "a": spec.a,
                    "b": spec.b,
                    "ok": pr.ok,
                    "score": pr.score,
                }
            )
        combined = evaluate_predicates(predicate_results)
        self.state.done = True
        self.state.success = combined.ok
        return self._obs(
            ok=True,
            result={"predicates": results, "score": combined.score},
            done=True,
            success=combined.ok,
        )

    def _handle_move(self, action: Action) -> dict[str, Any]:
        if action.entity_id is None:
            return self._obs(ok=False, error="move requires entity_id")
        entity = self.state.scene.get(action.entity_id)
        if isinstance(entity, OpeningEntity):
            if action.host_offset is None and action.host_offset_delta is None:
                return self._obs(
                    ok=False,
                    error="move on opening requires host_offset or host_offset_delta",
                )
            self.state.scene = move_opening(
                self.state.scene,
                action.entity_id,
                host_offset_delta=action.host_offset_delta,
                absolute_host_offset=action.host_offset,
            )
            opening = self.state.scene.get(action.entity_id)
            assert isinstance(opening, OpeningEntity)
            return self._obs(
                ok=True,
                result={"host_offset": opening.properties.OpeningGeometry.host_offset},
            )
        if action.offset is None:
            return self._obs(ok=False, error="move requires offset")
        self.state.scene = move_entity(self.state.scene, action.entity_id, action.offset)
        pos = self.state.scene.get_position(action.entity_id)
        return self._obs(ok=True, result={"position": pos})

    def _handle_place(self, action: Action) -> dict[str, Any]:
        if action.entity_id is None:
            return self._obs(ok=False, error="place requires entity_id")
        if action.position is None:
            return self._obs(ok=False, error="place requires position")
        entity = self.state.scene.get(action.entity_id)
        if entity.type == EntityType.Opening:
            return self._obs(ok=False, error="place on opening is not supported; use host_offset")
        self.state.scene = place_entity(self.state.scene, action.entity_id, action.position)
        return self._obs(ok=True, result={"position": action.position})

    def _record_error(self, action: Any, error: str) -> dict[str, Any]:
        payload = action if isinstance(action, dict) else (
            action.model_dump() if isinstance(action, Action) else {"raw": action}
        )
        observation = self._obs(ok=False, error=error)
        self.state._step += 1
        self.state.history.append(
            StepRecord(step=self.state._step, action=payload, observation=observation)
        )
        return observation

    def _obs(
        self,
        *,
        ok: bool,
        result: Any = None,
        error: str | None = None,
        done: bool | None = None,
        success: bool | None = None,
    ) -> dict[str, Any]:
        obs: dict[str, Any] = {
            "ok": ok,
            "step": self.state._step + 1,
            "done": self.state.done if done is None else done,
            "success": self.state.success if success is None else success,
        }
        if result is not None:
            obs["result"] = result
        if error is not None:
            obs["error"] = error
        return obs
