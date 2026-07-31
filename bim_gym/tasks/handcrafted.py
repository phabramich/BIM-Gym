from __future__ import annotations

from typing import Any

from bim_gym.dsl.actions import PredicateSpec
from bim_gym.dsl.enums import PredicateKind
from bim_gym.scenes import add_column, default_placement, make_room_box
from bim_gym.scenes.room_box import RoomBoxWall
from bim_gym.tasks.spec import SceneFactoryName, TaskSpec

DOOR_ON_SOUTH_WALL = TaskSpec(
    id="door_on_south_wall",
    instruction="Is the door hosted by the south wall?",
    static=True,
    scene_factory=SceneFactoryName.RoomBox,
    scene_factory_args={
        "origin": (0.0, 0.0),
        "size_xy": (4000.0, 3000.0),
        "wall_thickness": 200.0,
        "height": 3000.0,
        "with_door": True,
        "door_wall": RoomBoxWall.South,
    },
    goals=[
        PredicateSpec(kind=PredicateKind.HostedBy, a="opening-wall-s", b="wall-s"),
    ],
)

COLUMN_CLEAR_OF_WALL = TaskSpec(
    id="column_clear_of_wall",
    instruction="Move the column so it is clear of the south wall.",
    scene_factory=SceneFactoryName.RoomWithColumnNearWall,
    goals=[
        PredicateSpec(kind=PredicateKind.ClearOf, a="column-1", b="wall-s", margin=50.0),
    ],
)

WALLS_DISTANCE_IN_RANGE = TaskSpec(
    id="walls_distance_in_range",
    instruction="Verify the south and north walls are the expected room depth apart.",
    static=True,
    scene_factory=SceneFactoryName.RoomBox,
    scene_factory_args={
        "origin": (0.0, 0.0),
        "size_xy": (4000.0, 3000.0),
        "wall_thickness": 200.0,
        "height": 3000.0,
        "with_door": False,
    },
    goals=[
        PredicateSpec(
            kind=PredicateKind.DistanceInRange,
            a="wall-s",
            b="wall-n",
            lo=3100.0,
            hi=3300.0,
            tol=100.0,
        ),
    ],
)

TWO_COLUMNS_SPACING = TaskSpec(
    id="two_columns_spacing",
    instruction="Move column-b so the columns are about 2000 mm apart.",
    scene_factory=SceneFactoryName.TwoColumns,
    goals=[
        PredicateSpec(
            kind=PredicateKind.DistanceInRange,
            a="column-a",
            b="column-b",
            lo=1900.0,
            hi=2100.0,
        ),
    ],
)

OPENING_SHIFT_ALONG_WALL = TaskSpec(
    id="opening_shift_along_wall",
    instruction="Slide the door 500 mm along the south wall.",
    scene_factory=SceneFactoryName.RoomBox,
    scene_factory_args={
        "origin": (0.0, 0.0),
        "size_xy": (4000.0, 3000.0),
        "wall_thickness": 200.0,
        "height": 3000.0,
        "with_door": True,
        "door_wall": RoomBoxWall.South,
    },
    goals=[
        PredicateSpec(
            kind=PredicateKind.HostOffsetInRange,
            a="opening-wall-s",
            lo=2125.0,
            hi=2175.0,
        ),
        PredicateSpec(kind=PredicateKind.HostedBy, a="opening-wall-s", b="wall-s"),
    ],
)

ALL_TASKS: tuple[TaskSpec, ...] = (
    DOOR_ON_SOUTH_WALL,
    COLUMN_CLEAR_OF_WALL,
    WALLS_DISTANCE_IN_RANGE,
    TWO_COLUMNS_SPACING,
    OPENING_SHIFT_ALONG_WALL,
)


def _room_with_column_near_wall(**_kwargs: Any) -> dict[str, Any]:
    scene = make_room_box(
        origin=(0.0, 0.0),
        size_xy=(4000.0, 3000.0),
        wall_thickness=200.0,
        height=3000.0,
        with_door=False,
    )
    add_column(
        scene,
        "column-1",
        "storey-1",
        width=400.0,
        depth=400.0,
        height=3000.0,
        placement=default_placement((2000.0, 50.0, 0.0)),
    )
    return scene


def _two_columns(**_kwargs: Any) -> dict[str, Any]:
    scene = make_room_box(
        origin=(0.0, 0.0),
        size_xy=(6000.0, 3000.0),
        wall_thickness=200.0,
        height=3000.0,
        with_door=False,
    )
    add_column(
        scene,
        "column-a",
        "storey-1",
        width=400.0,
        depth=400.0,
        height=3000.0,
        placement=default_placement((1500.0, 1500.0, 0.0)),
    )
    add_column(
        scene,
        "column-b",
        "storey-1",
        width=400.0,
        depth=400.0,
        height=3000.0,
        placement=default_placement((4000.0, 1500.0, 0.0)),
    )
    return scene


_SCENE_BUILDERS: dict[SceneFactoryName, Any] = {
    SceneFactoryName.RoomBox: make_room_box,
    SceneFactoryName.RoomWithColumnNearWall: _room_with_column_near_wall,
    SceneFactoryName.TwoColumns: _two_columns,
}
