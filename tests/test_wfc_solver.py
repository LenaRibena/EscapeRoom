import unittest

from src.wfc_solver import solve_wfc
from src.wfc_tiles import ContradictionError, Direction, PortKind, Tile


def tile(name, **ports):
    all_ports = {direction: PortKind.NONE for direction in Direction}
    all_ports.update(
        {Direction[direction]: PortKind[kind] for direction, kind in ports.items()}
    )
    return Tile(name, all_ports)


class SolveWfcTests(unittest.TestCase):
    def test_fixed_tile_propagates_to_compatible_neighbor(self):
        tiles = [
            tile("empty"),
            tile("east_walk", E="WALK"),
            tile("west_walk", W="WALK"),
        ]
        cells = [(0, 0, 0), (0, 1, 0)]

        result = solve_wfc(
            cells,
            tiles,
            fixed_tiles={(0, 0, 0): 1},
            seed=7,
        )

        self.assertEqual(result[(0, 0, 0)], 1)
        self.assertEqual(result[(0, 1, 0)], 2)

    def test_same_seed_produces_the_same_layout(self):
        tiles = [tile("first"), tile("second"), tile("third")]
        cells = [(0, x, 0) for x in range(8)]

        first = solve_wfc(cells, tiles, seed=19)
        second = solve_wfc(cells, tiles, seed=19)

        self.assertEqual(first, second)

    def test_trace_is_emitted_for_the_successful_run(self):
        tiles = [tile("first"), tile("second"), tile("third")]
        steps = []

        result = solve_wfc(
            [(0, 0, 0), (0, 2, 0)],
            tiles,
            seed=7,
            on_step=steps.append,
        )

        self.assertGreater(len(steps), 1)
        self.assertEqual(steps[-1], {cell: frozenset({index}) for cell, index in result.items()})

    def test_incompatible_fixed_tiles_raise_after_bounded_attempts(self):
        tiles = [tile("empty"), tile("east_walk", E="WALK")]
        cells = [(0, 0, 0), (0, 1, 0)]

        with self.assertRaisesRegex(ContradictionError, "after 2 attempts"):
            solve_wfc(
                cells,
                tiles,
                fixed_tiles={(0, 0, 0): 1, (0, 1, 0): 0},
                max_attempts=2,
            )

    def test_fixed_cell_must_exist(self):
        with self.assertRaisesRegex(ValueError, "must exist"):
            solve_wfc(
                [(0, 0, 0)],
                [tile("empty")],
                fixed_tiles={(0, 1, 0): 0},
            )

    def test_empty_grid_returns_empty_layout(self):
        self.assertEqual(solve_wfc([], [], seed=1), {})


if __name__ == "__main__":
    unittest.main()
