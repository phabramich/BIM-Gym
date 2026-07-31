from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from bim_gym.contract.entities import Entity, EntityId

KULMAN_BIM_MODEL_FORMAT = "kulman-bim-model"
KULMAN_BIM_MODEL_VERSION = "0.2.0"

_entity_adapter: TypeAdapter[Entity] = TypeAdapter(Entity)


class KulmanBimModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Annotated[str, Field(pattern=r"^kulman-bim-model$")] = KULMAN_BIM_MODEL_FORMAT
    version: str = KULMAN_BIM_MODEL_VERSION
    revision: int
    roots: tuple[EntityId, ...]
    entities: dict[EntityId, Entity]

    @classmethod
    def from_raw(cls, data: dict) -> KulmanBimModel:
        raw_entities = data.get("entities", {})
        entities = {
            eid: _entity_adapter.validate_python(payload)
            for eid, payload in raw_entities.items()
        }
        return cls(
            format=data.get("format", KULMAN_BIM_MODEL_FORMAT),
            version=data.get("version", KULMAN_BIM_MODEL_VERSION),
            revision=data["revision"],
            roots=tuple(data["roots"]),
            entities=entities,
        )
