import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.wfc_building import WFCBuilding, build_route_tiles
from src.hybrid_wfc import (
    HybridBudgetExhausted, HybridGeometryRejected, _horizontal_turn_cost,
    _length_lower_bound,
    relative_cell_budget, route_length_lower_bound, solve_length_bounded_hybrid,
    validate_hybrid_clearance,
)
from src.wfc_global import validate_global_constraints
from src.wfc_tiles import (
    ContradictionError, Direction, PortKind,
)


class HybridTests(unittest.TestCase):
    def setUp(self):
        self.tiles = build_route_tiles(PortKind.SLIDE)
        self.cells = [(0, x, y) for x in range(4) for y in range(4)]
        self.start, self.end = (0, 0, 0), (0, 3, 3)
        self.allowed = {cell: {0, *self.tiles.segment_indices} for cell in self.cells}
        self.allowed[self.start] = {self.tiles.connector_indices[Direction.E]}
        self.allowed[self.end] = {self.tiles.connector_indices[Direction.W]}

    def test_unbounded_hybrid_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "requires exactly one relative or absolute cell budget"):
            WFCBuilding(4, 10, 10, difficulty=2, seed=10, grid_resolution=3,
                        routing_strategy="hybrid", hybrid_relative_budget=False)

    def test_building_defaults_to_final_hybrid_settings(self):
        building = WFCBuilding(4, 10, 10, difficulty=2, seed=10, grid_resolution=3,
                               generate_routes=False)
        self.assertEqual(building.routing_strategy, "hybrid")
        self.assertTrue(building.hybrid_relative_budget)
        self.assertEqual(building.hybrid_weighting, "compact")
        self.assertEqual(building.hybrid_minimum_clearance, 0.25)
        self.assertEqual(building.hybrid_geometry_preference, 1.5)
        self.assertTrue(building.hybrid_geometry_backtracking)
        with self.assertRaisesRegex(ValueError, "routing_strategy must be 'hybrid'"):
            WFCBuilding(4, 10, 10, routing_strategy="dfs_guided")

    def test_length_lower_bound_counts_prefix_and_horizontal_requirement(self):
        prefix = [(2, 0, 0), (1, 0, 0)]
        self.assertEqual(_length_lower_bound(prefix, (1, 1, 0), (0, 3, 0), 3), 6)
        self.assertEqual(_length_lower_bound(prefix, (1, 1, 0), (0, 3, 0), 5), 8)

    def test_port_aware_lower_bound_and_relative_budget(self):
        bound = route_length_lower_bound(
            self.cells, self.tiles.tiles, self.allowed, self.start, self.end, 0
        )
        self.assertEqual(bound, 7)
        self.assertEqual(relative_cell_budget(bound, .25, 4), 9)
        blocked = dict(self.allowed)
        blocked[(0, 1, 0)] = {0}
        with self.assertRaises(ContradictionError):
            route_length_lower_bound(
                self.cells, self.tiles.tiles, blocked, self.start, self.end, 0
            )
        with self.assertRaises(ValueError):
            relative_cell_budget(bound, float("nan"), 4)

    def test_clearance_rejects_close_routes_but_excludes_shared_endpoints(self):
        def route(route_id, source, target, points):
            return SimpleNamespace(route_id=route_id, source=source, target=target,
                                   accepted_geometry=points)
        a = route("a", (0, 0, 0), (1, 0, 0), ((0, 0, 0), (1, 0, 0)))
        b = route("b", (0, 0.1, 0), (1, 0.1, 0),
                  ((0, .1, 0), (1, .1, 0)))
        with self.assertRaisesRegex(HybridGeometryRejected, "clearance"):
            validate_hybrid_clearance(a, [b], .25)
        shared = route("c", (0, 0, 0), (0, 1, 0), ((0, 0, 0), (0, 1, 0)))
        validate_hybrid_clearance(a, [shared], .25)

    def test_bounded_route_samples_valid_choices(self):
        events = []
        result = solve_length_bounded_hybrid(
            self.cells, self.tiles.tiles, start=self.start, end=self.end,
            allowed_tiles=self.allowed, seed=10, max_cells=9,
            required_horizontal_steps=0, empty_index=0, max_seconds=5,
            trace_callback=events.append,
        )
        self.assertEqual(len(result), len(self.cells))
        choices = [event for event in events if event["event"] == "hybrid_length_choice"]
        self.assertTrue(any(len(event["eligible"]) > 1 for event in choices))
        self.assertTrue(all(abs(sum(item[3] for item in event["eligible"]) - 1) < 1e-10
                            for event in choices))
        repeat = solve_length_bounded_hybrid(
            self.cells, self.tiles.tiles, start=self.start, end=self.end,
            allowed_tiles=self.allowed, seed=10, max_cells=9,
            required_horizontal_steps=0, empty_index=0, max_seconds=5,
        )
        self.assertEqual(result, repeat)

    def test_bounded_route_rejects_too_short_limit(self):
        with self.assertRaises(ContradictionError):
            solve_length_bounded_hybrid(
                self.cells, self.tiles.tiles, start=self.start, end=self.end,
                allowed_tiles=self.allowed, seed=10, max_cells=4,
                required_horizontal_steps=0, empty_index=0, max_seconds=5,
            )

    def test_backtracking_restores_and_tries_another_choice(self):
        events = []
        calls = 0

        def fail_first(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ValueError("synthetic validation conflict")
            return validate_global_constraints(*args, **kwargs)

        with patch("src.wfc_global.validate_global_constraints", side_effect=fail_first):
            solve_length_bounded_hybrid(
                self.cells, self.tiles.tiles, start=self.start, end=self.end,
                allowed_tiles=self.allowed, seed=10, max_cells=9,
                required_horizontal_steps=0, empty_index=0, max_seconds=5,
                trace_callback=events.append,
            )
        self.assertTrue(any(event["event"] == "hybrid_backtrack" for event in events))
        self.assertGreater(calls, 1)

    def test_exhausted_budget_is_unknown(self):
        with self.assertRaisesRegex(HybridBudgetExhausted, "UNKNOWN"):
            solve_length_bounded_hybrid(
                self.cells, self.tiles.tiles, start=self.start, end=self.end,
                allowed_tiles=self.allowed, seed=10, max_cells=9,
                required_horizontal_steps=0, empty_index=0, max_seconds=1e-12,
            )

    def test_horizontal_reversal_cost_includes_vertical_steps(self):
        prefix = [(3, 0, 0), (3, 1, 0), (2, 1, 0)]
        reversal = _horizontal_turn_cost(prefix, (2, 0, 0))
        turn = _horizontal_turn_cost(prefix, (2, 1, 1))
        self.assertGreater(reversal, turn)
        self.assertGreater(turn, 0)
        self.assertEqual(_horizontal_turn_cost(prefix, (1, 1, 0)), 0)

    def test_geometry_rejection_backtracks_without_mutating_previous_state(self):
        events = []
        seen = []

        def validate(layout, path):
            seen.append(path)
            if len(seen) == 1:
                raise HybridGeometryRejected("synthetic radius failure")

        result = solve_length_bounded_hybrid(
            self.cells, self.tiles.tiles, start=self.start, end=self.end,
            allowed_tiles=self.allowed, seed=10, max_cells=9,
            required_horizontal_steps=0, empty_index=0, max_seconds=5,
            max_backtracks=20, geometry_preference=1.5, validate_complete=validate,
            trace_callback=events.append,
        )
        self.assertTrue(result)
        self.assertGreaterEqual(len(seen), 2)
        self.assertNotEqual(seen[0], seen[1])
        self.assertTrue(any(event["event"] == "hybrid_geometry_rejection" for event in events))
        self.assertTrue(any(event["event"] == "hybrid_backtrack" for event in events))
        selections = [event for event in events if event["event"] == "hybrid_length_choice"]
        self.assertTrue(all(weight > 0 and probability > 0
                            for event in selections
                            for _, _, weight, probability in event["eligible"]))


if __name__ == "__main__":
    unittest.main()
