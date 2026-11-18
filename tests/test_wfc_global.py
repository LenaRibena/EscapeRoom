import unittest

from src.wfc_global import validate_global_constraints
from src.wfc_solver import solve_wfc
from src.wfc_tiles import ContradictionError, Direction, PortKind, Tile


def tile(name, **ports):
    all_ports = {direction: PortKind.NONE for direction in Direction}
    all_ports.update(
        {Direction[direction]: PortKind[kind] for direction, kind in ports.items()}
    )
    return Tile(name, all_ports)


class GlobalConstraintTests(unittest.TestCase):
    def test_required_pair_must_have_a_continuous_port_path(self):
        tiles = [
            tile("empty"),
            tile("east_end", E="WALK"),
            tile("west_end", W="WALK"),
        ]
        layout = {(0, 0, 0): 1, (0, 2, 0): 2}

        with self.assertRaisesRegex(ValueError, "not connected"):
            validate_global_constraints(
                layout,
                tiles,
                required_pairs=[((0, 0, 0), (0, 2, 0))],
            )

    def test_global_path_through_collapsed_layout_is_accepted(self):
        tiles = [
            tile("empty"),
            tile("east_end", E="WALK"),
            tile("corridor", W="WALK", E="WALK"),
            tile("west_end", W="WALK"),
        ]
        cells = [(0, x, 0) for x in range(3)]

        layout = solve_wfc(
            cells,
            tiles,
            fixed_tiles={(0, 0, 0): 1, (0, 2, 0): 3},
            seed=1,
            required_pairs=[((0, 0, 0), (0, 2, 0))],
            require_closed_boundaries=True,
        )

        self.assertEqual(layout, {(0, 0, 0): 1, (0, 1, 0): 2, (0, 2, 0): 3})

    def test_solver_rejects_fixed_open_ports_at_boundaries(self):
        with self.assertRaisesRegex(ValueError, "open port outside the layout"):
            solve_wfc(
                [(0, 0, 0)],
                [tile("east_end", E="WALK")],
                fixed_tiles={(0, 0, 0): 0},
                require_closed_boundaries=True,
            )

    def test_solver_filters_boundary_incompatible_tiles_before_collapse(self):
        tiles = [tile("east_open", E="WALK"), tile("closed")]

        result = solve_wfc(
            [(0, 0, 0)],
            tiles,
            seed=1,
            require_closed_boundaries=True,
        )

        self.assertEqual(result[(0, 0, 0)], 1)

    def test_solver_retries_when_global_connectivity_fails(self):
        tiles = [
            tile("east_end", E="WALK"),
            tile("west_end", W="WALK"),
        ]
        cells = [(0, 0, 0), (0, 2, 0)]

        with self.assertRaisesRegex(ContradictionError, "after 2 attempts"):
            solve_wfc(
                cells,
                tiles,
                fixed_tiles={(0, 0, 0): 0, (0, 2, 0): 1},
                required_pairs=[(cells[0], cells[1])],
                max_attempts=2,
            )

    def test_global_constraint_rejects_mismatched_neighbor_ports(self):
        tiles = [tile("east_end", E="WALK"), tile("empty")]

        with self.assertRaisesRegex(ValueError, "Mismatched ports"):
            validate_global_constraints(
                {(0, 0, 0): 0, (0, 1, 0): 1},
                tiles,
            )


if __name__ == "__main__":
    unittest.main()
