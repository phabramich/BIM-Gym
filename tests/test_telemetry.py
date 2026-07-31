from __future__ import annotations

from bim_gym.dsl import PredicateSpec, assert_predicates, move
from bim_gym.dsl.enums import PredicateKind
from bim_gym.runtime.episode import Episode
from bim_gym.tasks import COLUMN_CLEAR_OF_WALL, build_scene, run_task
from bim_gym.telemetry import VERIFIER_VERSION, build_telemetry, to_jsonable


def test_build_telemetry_from_handcrafted_task() -> None:
    result = run_task(
        COLUMN_CLEAR_OF_WALL,
        [
            move("column-1", (0.0, 300.0, 0.0)),
            assert_predicates(
                [PredicateSpec(kind=PredicateKind.ClearOf, a="column-1", b="wall-s", margin=50.0)]
            ),
        ],
        record_telemetry=True,
        scene_seed=42,
        model_checkpoint="test-checkpoint",
    )

    assert result.success is True
    assert result.telemetry is not None
    telemetry = result.telemetry

    assert telemetry.verifier_version == VERIFIER_VERSION
    assert telemetry.success is True
    assert telemetry.n_actions == len(telemetry.turns)
    assert telemetry.n_actions == 2
    assert telemetry.scene_seed == 42
    assert telemetry.model_checkpoint == "test-checkpoint"
    assert telemetry.task_spec is not None
    assert telemetry.task_spec.id == COLUMN_CLEAR_OF_WALL.id

    reward = telemetry.reward_decomposition
    assert reward.binary == 1.0
    assert reward.dense == 0.2
    assert reward.penalties == -0.04
    assert reward.geometric_progress == 0.0


def test_build_telemetry_helper_matches_episode_history() -> None:
    episode = Episode(build_scene(COLUMN_CLEAR_OF_WALL))
    episode.step(move("column-1", (0.0, 300.0, 0.0)))
    episode.step(
        assert_predicates(
            [PredicateSpec(kind=PredicateKind.ClearOf, a="column-1", b="wall-s", margin=50.0)]
        )
    )

    telemetry = build_telemetry(episode, COLUMN_CLEAR_OF_WALL, scene_seed="seed-a")
    payload = to_jsonable(telemetry)

    assert set(payload) == {
        "episode_id",
        "phase",
        "task_spec",
        "scene_seed",
        "turns",
        "reward_decomposition",
        "success",
        "n_actions",
        "verifier_version",
        "model_checkpoint",
    }
    assert payload["verifier_version"] == VERIFIER_VERSION
    assert payload["success"] is True
    assert len(payload["turns"]) == payload["n_actions"]
    assert payload["n_actions"] == len(episode.state.history)
    assert set(payload["reward_decomposition"]) == {
        "binary",
        "dense",
        "penalties",
        "geometric_progress",
    }
