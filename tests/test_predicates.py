from __future__ import annotations

import pytest

from bim_gym.runtime.predicates import (
    adjacent,
    aligned_with,
    clear_of,
    distance_between,
    distance_in_range,
    evaluate,
    hosted_by,
    inside,
)
from bim_gym.runtime.scene import Scene


def _wall_storey_model() -> dict:
    return {
        "format": "kulman-bim-model",
        "version": "0.2.0",
        "revision": 1,
        "roots": ["site-1"],
        "entities": {
            "site-1": {
                "id": "site-1",
                "type": "Site",
                "name": "Site",
                "properties": {
                    "Placement": {"position": [0, 0, 0], "rotation": [0, 0, 0]},
                    "WorldPlacement": {
                        "latitude": 0,
                        "longitude": 0,
                        "rotationAngleTrueNorth": 0,
                        "elevation": 0,
                    },
                },
            },
            "storey-1": {
                "id": "storey-1",
                "type": "Storey",
                "name": "L1",
                "parent": "site-1",
                "properties": {
                    "Placement": {"position": [0, 0, 0], "rotation": [0, 0, 0]},
                    "StoreyData": {
                        "storeyIndex": 0,
                        "grossHeight": 3000,
                        "netHeight": 2700,
                        "storeyType": "typical",
                        "zoningType": "typical_storey",
                        "isEntranceLevel": False,
                    },
                },
            },
            "wall-1": {
                "id": "wall-1",
                "type": "Wall",
                "name": "Wall A",
                "parent": "storey-1",
                "properties": {
                    "Placement": {"position": [0, 0, 0], "rotation": [0, 0, 0]},
                    "WallGeometry": {
                        "centerline": [[0, 0], [5000, 0]],
                        "thickness": 200,
                        "height": 3000,
                        "baseOffset": 0,
                    },
                    "WallData": {"kind": "interior", "structural": True},
                },
            },
            "wall-2": {
                "id": "wall-2",
                "type": "Wall",
                "name": "Wall B",
                "parent": "storey-1",
                "properties": {
                    "Placement": {"position": [0, 200, 0], "rotation": [0, 0, 0]},
                    "WallGeometry": {
                        "centerline": [[0, 0], [5000, 0]],
                        "thickness": 200,
                        "height": 3000,
                        "baseOffset": 0,
                    },
                    "WallData": {"kind": "interior", "structural": True},
                },
            },
            "wall-3": {
                "id": "wall-3",
                "type": "Wall",
                "name": "Wall far",
                "parent": "storey-1",
                "properties": {
                    "Placement": {"position": [0, 5000, 0], "rotation": [0, 0, 0]},
                    "WallGeometry": {
                        "centerline": [[0, 0], [5000, 0]],
                        "thickness": 200,
                        "height": 3000,
                        "baseOffset": 0,
                    },
                    "WallData": {"kind": "interior", "structural": True},
                },
            },
            "opening-1": {
                "id": "opening-1",
                "type": "Opening",
                "name": "Door",
                "parent": "storey-1",
                "properties": {
                    "Hosting": {"hostId": "wall-1"},
                    "OpeningGeometry": {
                        "width": 900,
                        "height": 2100,
                        "sillHeight": 0,
                        "hostOffset": 2000,
                    },
                    "OpeningData": {"kind": "door"},
                },
            },
            "column-1": {
                "id": "column-1",
                "type": "Column",
                "name": "Column",
                "parent": "storey-1",
                "properties": {
                    "Placement": {"position": [2500, 100, 0], "rotation": [0, 0, 0]},
                    "ColumnGeometry": {
                        "profile": "rectangular",
                        "width": 400,
                        "depth": 400,
                        "height": 3000,
                        "baseOffset": 0,
                    },
                    "ColumnData": {"kind": "structural", "structural": True},
                },
            },
        },
    }


@pytest.fixture
def scene() -> Scene:
    return Scene.from_dict(_wall_storey_model())


def test_hosted_by(scene: Scene) -> None:
    ok = hosted_by(scene, "opening-1", "wall-1")
    assert ok.ok is True
    assert ok.score == 1.0

    bad = hosted_by(scene, "opening-1", "wall-2")
    assert bad.ok is False
    assert bad.score == 0.0


def test_distance_between_and_in_range(scene: Scene) -> None:
    d = distance_between(scene, "wall-1", "wall-2")
    assert d == pytest.approx(200.0)

    in_range = distance_in_range(scene, "wall-1", "wall-2", 190, 210)
    assert in_range.ok is True
    assert in_range.score == 1.0

    out_range = distance_in_range(scene, "wall-1", "wall-3", 0, 100)
    assert out_range.ok is False
    assert out_range.score < 1.0


def test_clear_of(scene: Scene) -> None:
    separated = clear_of(scene, "wall-1", "wall-3")
    assert separated.ok is True
    assert separated.score == 1.0

    touching = clear_of(scene, "wall-1", "wall-2", margin=1.0)
    assert touching.ok is False


def test_adjacent(scene: Scene) -> None:
    result = adjacent(scene, "wall-1", "wall-2")
    assert result.ok is True
    assert result.score == 1.0

    far = adjacent(scene, "wall-1", "wall-3")
    assert far.ok is False


def test_aligned_with(scene: Scene) -> None:
    result = aligned_with(scene, "wall-1", "wall-2")
    assert result.ok is True
    assert result.score == pytest.approx(1.0, abs=1e-6)


def test_inside_bbox_center(scene: Scene) -> None:
    result = inside(scene, "column-1", "wall-1")
    assert result.ok is True
    assert result.score == 1.0


def test_evaluate_ands_ok_and_averages_scores(scene: Scene) -> None:
    combined = evaluate(
        [
            hosted_by(scene, "opening-1", "wall-1"),
            adjacent(scene, "wall-1", "wall-2"),
            hosted_by(scene, "opening-1", "wall-2"),
        ]
    )
    assert combined.ok is False
    assert combined.score == pytest.approx(2.0 / 3.0)
