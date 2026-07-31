from __future__ import annotations

from bim_gym.contract.enums import EntityType
from bim_gym.dsl import PredicateSpec, assert_predicates, query
from bim_gym.dsl.enums import PredicateKind, QueryOp
from bim_gym.runtime.episode import Episode
from bim_gym.runtime.scene import Scene
from bim_gym.scenes import make_room_box


def _room_episode() -> Episode:
    return Episode(Scene.from_dict(make_room_box(
        origin=(0.0, 0.0),
        size_xy=(4000.0, 3000.0),
        wall_thickness=200.0,
        height=3000.0,
        with_door=True,
    )))


def test_episode_query_walls_and_openings() -> None:
    ep = _room_episode()

    walls_obs = ep.step(query(QueryOp.List, entity_type=EntityType.Wall.value, parent="storey-1"))
    assert walls_obs["ok"] is True
    assert len(walls_obs["result"]) == 4

    openings_obs = ep.step(query(QueryOp.OpeningsOn, entity_id="wall-s"))
    assert openings_obs["ok"] is True
    assert len(openings_obs["result"]) == 1
    assert openings_obs["result"][0]["kind"] == "door"
    assert ep.state.done is False


def test_episode_assert_hosted_by_succeeds() -> None:
    ep = _room_episode()
    ep.step(query(QueryOp.List, entity_type=EntityType.Wall.value))
    ep.step(query(QueryOp.OpeningsOn, entity_id="wall-s"))

    obs = ep.step(
        assert_predicates(
            [PredicateSpec(kind=PredicateKind.HostedBy, a="opening-wall-s", b="wall-s")]
        )
    )

    assert obs["ok"] is True
    assert obs["done"] is True
    assert obs["success"] is True
    assert obs["result"]["predicates"][0]["ok"] is True
    assert ep.state.success is True


def test_episode_assert_hosted_by_fails() -> None:
    ep = _room_episode()

    obs = ep.step(
        assert_predicates(
            [PredicateSpec(kind=PredicateKind.HostedBy, a="opening-wall-s", b="wall-n")]
        )
    )

    assert obs["ok"] is True
    assert obs["done"] is True
    assert obs["success"] is False
    assert obs["result"]["predicates"][0]["ok"] is False
    assert ep.state.success is False


def test_episode_invalid_action_returns_error() -> None:
    ep = _room_episode()

    obs = ep.step({"kind": "query", "op": "get"})

    assert obs["ok"] is False
    assert "error" in obs
    assert ep.state.done is False


def test_episode_note_and_recall() -> None:
    ep = _room_episode()

    from bim_gym.dsl import note, recall

    note_obs = ep.step(note("door_host", "wall-s"))
    assert note_obs["ok"] is True

    recall_obs = ep.step(recall("door_host"))
    assert recall_obs["ok"] is True
    assert recall_obs["result"] == "wall-s"

    missing_obs = ep.step(recall("missing"))
    assert missing_obs["ok"] is False
