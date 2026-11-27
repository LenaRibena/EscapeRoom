import unittest

from src.wfc_global import (
    has_tile_supported_route,
    prune_unreachable_route_domains,
    validate_global_constraints,
)
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
        layout = {(0, 0, 0): 1, (0, 1, 0): 2, (0, 2, 0): 3}
        validate_global_constraints(
            layout, tiles, required_pairs=[((0, 0, 0), (0, 2, 0))]
        )

    def test_global_constraint_rejects_mismatched_neighbor_ports(self):
        tiles = [tile("east_end", E="WALK"), tile("empty")]

        with self.assertRaisesRegex(ValueError, "Mismatched ports"):
            validate_global_constraints(
                {(0, 0, 0): 0, (0, 1, 0): 1},
                tiles,
            )

    def test_horizontal_slide_segments_connect_head_to_tail(self):
        tiles = [
            tile("head", E="SLIDE"),
            tile("middle", W="SLIDE", E="SLIDE"),
            tile("tail", W="SLIDE"),
        ]
        cells = [(1, 0, 0), (1, 1, 0), (1, 2, 0)]
        layout = {cell: index for index, cell in enumerate(cells)}

        validate_global_constraints(
            layout,
            tiles,
            required_pairs=[(cells[0], cells[-1])],
        )

    def test_reachability_pruning_preserves_tiles_on_a_valid_route(self):
        start, middle, end, stray = (
            (0, 0, 0),
            (0, 1, 0),
            (0, 2, 0),
            (0, 0, 3),
        )
        tiles = [
            tile("slide_start", E="SLIDE"),
            tile("slide_middle", W="SLIDE", E="SLIDE"),
            tile("slide_end", W="SLIDE"),
            tile("empty"),
            tile("stray_slide", E="SLIDE"),
        ]
        domains = {
            start: {0},
            middle: {1, 3},
            end: {2},
            stray: {3, 4},
        }

        changed = prune_unreachable_route_domains(
            domains,
            tiles,
            [(start, end, PortKind.SLIDE)],
        )

        self.assertIn(stray, changed)
        self.assertEqual(domains[start], {0})
        self.assertEqual(domains[middle], {1, 3})
        self.assertEqual(domains[end], {2})
        self.assertEqual(domains[stray], {3})

    def test_tile_aware_pruning_rejects_ports_split_across_tile_candidates(self):
        start, middle, end = (0, 0, 0), (0, 1, 0), (0, 2, 0)
        tiles = [
            tile("start", E="SLIDE"),
            tile("end", W="SLIDE"),
            tile("middle_west_only", W="SLIDE"),
            tile("middle_east_only", E="SLIDE"),
        ]
        domains = {
            start: {0},
            middle: {2, 3},
            end: {1},
        }

        with self.assertRaisesRegex(ContradictionError, "No tile-compatible"):
            prune_unreachable_route_domains(
                domains,
                tiles,
                [(start, end, PortKind.SLIDE)],
                tile_aware=True,
            )

    def test_tile_aware_pruning_preserves_a_valid_joint_tile_assignment(self):
        start, middle, end, stray = (
            (0, 0, 0),
            (0, 1, 0),
            (0, 2, 0),
            (0, 0, 3),
        )
        tiles = [
            tile("start", E="SLIDE"),
            tile("end", W="SLIDE"),
            tile("middle_straight", W="SLIDE", E="SLIDE"),
            tile("middle_west_only", W="SLIDE"),
            tile("middle_east_only", E="SLIDE"),
            tile("stray_slide", N="SLIDE"),
            tile("empty"),
        ]
        domains = {
            start: {0},
            middle: {2, 3, 4},
            end: {1},
            stray: {5, 6},
        }

        changed = prune_unreachable_route_domains(
            domains,
            tiles,
            [(start, end, PortKind.SLIDE)],
            tile_aware=True,
        )

        self.assertIn(middle, changed)
        self.assertEqual(domains[start], {0})
        self.assertEqual(domains[middle], {2})
        self.assertEqual(domains[end], {1})
        self.assertEqual(domains[stray], {6})

    def test_tile_aware_pruning_keeps_all_supported_alternative_endpoint_tiles(self):
        start, middle, end = (0, 0, 0), (0, 1, 0), (0, 2, 0)
        tiles = [
            tile("start_variant_a", E="SLIDE"),
            tile("start_variant_b", E="SLIDE"),
            tile("middle", W="SLIDE", E="SLIDE"),
            tile("end", W="SLIDE"),
        ]
        domains = {start: {0, 1}, middle: {2}, end: {3}}

        prune_unreachable_route_domains(
            domains,
            tiles,
            [(start, end, PortKind.SLIDE)],
            tile_aware=True,
        )

        self.assertEqual(domains[start], {0, 1})
        self.assertEqual(domains[middle], {2})
        self.assertEqual(domains[end], {3})

    def test_tile_supported_feasibility_respects_required_horizontal_steps(self):
        start, end = (1, 0, 0), (0, 0, 0)
        tiles = [
            tile("slide_down", D="SLIDE"),
            tile("slide_up", U="SLIDE"),
        ]
        domains = {start: {0}, end: {1}}

        self.assertTrue(
            has_tile_supported_route(
                domains,
                tiles,
                start,
                end,
                PortKind.SLIDE,
            )
        )
        self.assertFalse(
            has_tile_supported_route(
                domains,
                tiles,
                start,
                end,
                PortKind.SLIDE,
                required_horizontal_steps=1,
            )
        )

    def test_tile_aware_pruning_preserves_upward_stair_direction(self):
        start, end = (0, 0, 0), (1, 0, 0)
        tiles = [
            tile("stair_up", U="STAIR"),
            tile("stair_down", D="STAIR"),
        ]
        domains = {start: {0}, end: {1}}

        self.assertFalse(
            prune_unreachable_route_domains(
                domains,
                tiles,
                [(start, end, PortKind.STAIR)],
                tile_aware=True,
            )
        )
        validate_global_constraints(
            {start: 0, end: 1},
            tiles,
            required_typed_pairs=[(start, end, PortKind.STAIR)],
        )


if __name__ == "__main__":
    unittest.main()
