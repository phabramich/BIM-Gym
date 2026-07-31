from bim_gym.contract.enums import EntityType
from bim_gym.contract.model import KulmanBimModel
from bim_gym.runtime.scene import Scene
from bim_gym.scenes import make_room_box


def test_make_room_box_loads_and_openings_on() -> None:
    raw = make_room_box(
        origin=(0.0, 0.0),
        size_xy=(4000.0, 3000.0),
        wall_thickness=200.0,
        height=3000.0,
        with_door=True,
    )

    model = KulmanBimModel.from_raw(raw)
    scene = Scene(model)

    summary = scene.summary()
    counts = summary["entity_counts"]
    assert counts[EntityType.Storey.value] == 1
    assert counts[EntityType.Wall.value] == 4
    assert counts[EntityType.Slab.value] == 1
    assert counts[EntityType.Opening.value] == 1
    assert summary["n_entities"] == 7

    south_openings = scene.openings_on("wall-s")
    assert len(south_openings) == 1
    assert south_openings[0]["kind"] == "door"
    assert south_openings[0]["width"] == 900.0
    assert south_openings[0]["height"] == 2100.0

    walls = scene.list(type=EntityType.Wall, parent="storey-1")
    assert len(walls) == 4
    assert all(row["parent"] == "storey-1" for row in walls)


def test_make_room_box_from_dict() -> None:
    raw = make_room_box((1000.0, 2000.0), (5000.0, 4000.0), 250.0, 3200.0, with_door=False)
    scene = Scene.from_dict(raw)
    assert scene.summary()["entity_counts"].get(EntityType.Opening.value, 0) == 0
    assert scene.summary()["entity_counts"][EntityType.Wall.value] == 4
