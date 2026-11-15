import unittest

from examples.castle_global_playground import (
    CastleTile,
    _build_compatibility,
    _road_is_connected,
    _tile_edges,
)
from src.wfc_tiles import Direction


def castle_tile(name, road_ports, pixels=None):
    rows = pixels or (("#98c415", "#98c415"), ("#98c415", "#98c415"))
    return CastleTile(
        name=name,
        pixels=rows,
        edges=_tile_edges(rows),
        road_ports=frozenset(road_ports),
    )


class CastleExercisePlaygroundTests(unittest.TestCase):
    def test_pixel_edges_determine_local_compatibility(self):
        road = castle_tile(
            "road",
            (Direction.E, Direction.W),
            (("#98c415", "#fff200"), ("#98c415", "#fff200")),
        )
        matching = castle_tile(
            "matching",
            (Direction.E, Direction.W),
            (("#fff200", "#98c415"), ("#fff200", "#98c415")),
        )
        incompatible = castle_tile(
            "incompatible",
            (),
            (("#00a2e8", "#00a2e8"), ("#00a2e8", "#00a2e8")),
        )

        compatibility = _build_compatibility([road, matching, incompatible])

        self.assertIn(1, compatibility[Direction.E][0])
        self.assertNotIn(2, compatibility[Direction.E][0])

    def test_global_road_connectivity_follows_matching_tile_ports(self):
        tiles = [
            castle_tile("east_west", (Direction.E, Direction.W)),
            castle_tile("north_south", (Direction.N, Direction.S)),
        ]
        connected = {
            (0, 0, 0): 0,
            (0, 1, 0): 0,
            (0, 2, 0): 0,
        }
        broken = {
            (0, 0, 0): 0,
            (0, 1, 0): 1,
            (0, 2, 0): 0,
        }

        self.assertTrue(
            _road_is_connected(connected, tiles, (0, 0, 0), (0, 2, 0))
        )
        self.assertFalse(
            _road_is_connected(broken, tiles, (0, 0, 0), (0, 2, 0))
        )


if __name__ == "__main__":
    unittest.main()
