import math
from dataclasses import dataclass
from typing import Literal, Optional

Coord = tuple[float, float]
Channel = Literal["red", "ir"]


@dataclass(frozen=True)
class BoardConfig:
    width: float
    height: float
    n_ring_leds: int
    counter_shape: tuple[int, int]
    counter_spacing: float
    radius_red: float
    radius_ir: float
    counter_origin_red: Coord
    counter_origin_ir: Coord
    corner_leds_red: tuple[Coord, ...]
    corner_leds_ir: tuple[Coord, ...]
    aruco_size: float

    def radius(self, ch: Channel) -> float:
        return self.radius_red if ch == "red" else self.radius_ir

    def counter_origin(self, ch: Channel) -> Coord:
        return self.counter_origin_red if ch == "red" else self.counter_origin_ir

    def corner_leds(self, ch: Channel) -> tuple[Coord, ...]:
        return self.corner_leds_red if ch == "red" else self.corner_leds_ir


REV_1 = BoardConfig(
    width=250.0,
    height=250.0,
    n_ring_leds=100,
    counter_shape=(1, 16),
    counter_spacing=8.0,
    radius_red=115.0,
    radius_ir=110.0,
    counter_origin_red=(65.0, 53.0),
    counter_origin_ir=(65.0, 48.0),
    corner_leds_red=((5.0, 20.0), (5.0, 245.0), (245.0, 5.0), (245.0, 245.0)),
    corner_leds_ir=((20.0, 20.0), (20.0, 230.0), (230.0, 20.0), (230.0, 230.0)),
    aruco_size=91.2,
)

REV_2 = BoardConfig(
    width=250.0,
    height=250.0,
    n_ring_leds=100,
    counter_shape=(2, 10),
    counter_spacing=12.0,
    radius_red=115.0,
    radius_ir=110.0,
    counter_origin_red=(71.0, 41.0),
    counter_origin_ir=(71.0, 47.0),
    corner_leds_red=((5.0, 20.0), (5.0, 245.0), (245.0, 5.0), (245.0, 245.0)),
    corner_leds_ir=((20.0, 20.0), (20.0, 230.0), (230.0, 20.0), (230.0, 230.0)),
    aruco_size=90.0,
)


class Geometry:
    def __init__(self, config: BoardConfig) -> None:
        self.c = config
        self.cx = config.width / 2
        self.cy = config.height / 2

    def corner_leds(self, ch: Channel) -> list[Coord]:
        return list(self.c.corner_leds(ch))

    def counter_leds(self, ch: Channel) -> list[Coord]:
        ox, oy = self.c.counter_origin(ch)
        rows, cols = self.c.counter_shape
        s = self.c.counter_spacing
        return [
            (ox + col * s, oy + row * s) for row in range(rows) for col in range(cols)
        ][::-1]

    def ring_leds(self, ch: Channel) -> list[Coord]:
        n, r = self.c.n_ring_leds, self.c.radius(ch)
        return [
            (self.cx + r * math.cos(a), self.cy + r * math.sin(a))
            for i in range(n)
            for a in [-(i / n + 0.25) * 2 * math.pi]
        ]

    def aruco_corners(self) -> list[Coord]:
        half = self.c.aruco_size / 2
        return [
            (self.cx - half, self.cy - half),
            (self.cx + half, self.cy - half),
            (self.cx + half, self.cy + half),
            (self.cx - half, self.cy + half),
        ]

    def period(self) -> int:
        return self.c.n_ring_leds


ARUCO_REVISIONS: dict[int, BoardConfig] = {0: REV_1, 21: REV_2}
FTK_REVISIONS: dict[int, BoardConfig] = {240: REV_1, 241: REV_2}


def geometry_from_aruco_marker(marker_id: int) -> Optional[Geometry]:
    config = ARUCO_REVISIONS.get(marker_id)
    return Geometry(config) if config else None


def geometry_from_ftk_marker(marker_id: int) -> Optional[Geometry]:
    config = FTK_REVISIONS.get(marker_id)
    return Geometry(config) if config else None
