from __future__ import annotations

import pytest

from bim_gym.dsl import PredicateSpec, assert_predicates, move
from bim_gym.dsl.enums import PredicateKind
from bim_gym.runtime.episode import Episode
from bim_gym.runtime.mutate import move_entity, move_opening
from bim_gym.runtime.scene import Scene
from bim_gym.scenes import add_column, default_placement, make_room_box
from bim_gym.tasks import (
    COLUMN_CLEAR_OF_WALL,
    DOOR_ON_SOUTH_WALL,
    evaluate_task,
    run_task,
)


def _column_scene() -> Scene:
    raw = make_room_box(
        origin=(0.0, 0.0),
        size_xy=(4000.0, 3000.0),
        wall_thickness=200.0,
        height=3000.0,
        with_door=False,
    )
    add_column(
        raw,
        "column-1",
        "storey-1",
        width=400.0,
        depth=400.0,
        height=3000.0,
        placement=default_placement((2000.0, 50.0, 0.0)),
    )
    return Scene.from_dict(raw)


def test_move_entity_translates_column() -> None:
    scene = _column_scene()
    before = scene.get_position("column-1")
    assert before == (2000.0, 50.0, 0.0)

    moved = move_entity(scene, "column-1", (0.0, 300.0, 0.0))
    after = moved.get_position("column-1")
    assert after == (2000.0, 350.0, 0.0)


def test_move_opening_updates_host_offset() -> None:
    scene = Scene.from_dict(
        make_room_box(
            origin=(0.0, 0.0),
            size_xy=(4000.0, 3000.0),
            wall_thickness=200.0,
            height=3000.0,
            with_door=True,
        )
    )
    before = scene.openings_on("wall-s")[0]["host_offset"]
    moved = move_opening(scene, "opening-wall-s", host_offset_delta=500.0)
    after = moved.openings_on("wall-s")[0]["host_offset"]
    assert after == pytest.approx(before + 500.0)


def test_episode_move_action() -> None:
    ep = Episode(_column_scene())
    obs = ep.step(move("column-1", (0.0, 400.0, 0.0)))
    assert obs["ok"] is True
    assert obs["result"]["position"] == (2000.0, 450.0, 0.0)


def test_evaluate_task_door_on_south_wall() -> None:
    result = evaluate_task(DOOR_ON_SOUTH_WALL)
    assert result.success is True
    assert result.score == 1.0


def test_run_task_column_clear_of_wall() -> None:
    result = run_task(
        COLUMN_CLEAR_OF_WALL,
        [
            move("column-1", (0.0, 300.0, 0.0)),
            assert_predicates(
                [PredicateSpec(kind=PredicateKind.ClearOf, a="column-1", b="wall-s", margin=50.0)]
            ),
        ],
    )
    assert result.error is None
    assert result.success is True
    assert result.score == 1.0
