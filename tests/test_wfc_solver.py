from collections import Counter
import random
import unittest
from unittest.mock import patch

from src.wfc_building import build_route_tiles
from src.wfc_solver import _select_route_tile, solve_wfc
from src.wfc_tiles import (
    ContradictionError,
    Direction,
    PortKind,
    Tile,
    build_compatibility,
)


def tile(name, **ports):
    all_ports = {direction: PortKind.NONE for direction in Direction}
    all_ports.update(
        {Direction[direction]: PortKind[kind] for direction, kind in ports.items()}
    )
    return Tile(name, all_ports)


class SolveWfcTests(unittest.TestCase):
    def test_removed_routing_strategies_are_rejected(self):
        for strategy in ("pure_dfs", "incremental_frontier"):
            with self.subTest(strategy=strategy):
                with self.assertRaisesRegex(ValueError, "dfs_guided"):
                    solve_wfc([], [], routing_strategy=strategy)

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

    def test_optional_trace_does_not_change_seeded_layout(self):
        tiles = [tile("first"), tile("second"), tile("third")]
        cells = [(0, x, 0) for x in range(8)]
        events = []

        baseline = solve_wfc(cells, tiles, seed=19)
        traced = solve_wfc(cells, tiles, seed=19, trace_callback=events.append)

        self.assertEqual(traced, baseline)
        event_types = {event["event"] for event in events}
        self.assertIn("solver_attempt_start", event_types)
        self.assertIn("collapse_selected", event_types)
        self.assertIn("solver_solution", event_types)

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

    def test_allowed_tiles_restricts_endpoint_domains(self):
        tiles = [
            tile("empty"),
            tile("east_connector", E="SLIDE"),
            tile("west_connector", W="SLIDE"),
        ]
        cells = [(0, 0, 0), (0, 1, 0)]

        result = solve_wfc(
            cells,
            tiles,
            allowed_tiles={
                cells[0]: {1},
                cells[1]: {2},
            },
            required_pairs=[(cells[0], cells[1])],
            seed=1,
        )

        self.assertEqual(result, {cells[0]: 1, cells[1]: 2})

    def test_empty_grid_returns_empty_layout(self):
        self.assertEqual(solve_wfc([], [], seed=1), {})

    def test_global_pruning_failure_reports_phase_and_endpoint_domain(self):
        start = (0, 0, 0)
        end = (0, 2, 0)
        tiles = [
            tile("east_slide", E="SLIDE"),
            tile("west_slide", W="SLIDE"),
        ]

        with self.assertRaises(ContradictionError) as context:
            solve_wfc(
                [start, end],
                tiles,
                fixed_tiles={start: 0, end: 1},
                required_typed_pairs=[(start, end, PortKind.SLIDE)],
                max_attempts=1,
            )

        error = context.exception
        self.assertEqual(error.failure.phase, "global_reachability_pruning")
        self.assertEqual(error.failure.cell, start)
        self.assertEqual(error.failure.domain, (0,))
        self.assertEqual(error.failure.attempt, 1)

    def test_route_selector_failure_reports_collapse_selection_phase(self):
        start = (1, 0, 0)
        end = (0, 0, 0)
        isolated = (4, 4, 4)
        tiles = [
            tile("empty"),
            tile("empty_variant"),
            tile("down_slide", D="SLIDE"),
            tile("up_slide", U="SLIDE"),
        ]

        with self.assertRaises(ContradictionError) as context:
            solve_wfc(
                [start, end, isolated],
                tiles,
                fixed_tiles={start: 2, end: 3},
                required_typed_pairs=[(start, end, PortKind.SLIDE)],
                required_horizontal_steps=1,
                max_attempts=1,
            )

        error = context.exception
        self.assertEqual(error.failure.phase, "cell_collapse_selection")
        self.assertEqual(error.failure.cell, start)
        self.assertEqual(error.failure.domain, (2,))

    def test_preferred_empty_tile_is_used_after_required_route_is_collapsed(self):
        start, middle, end, isolated = (
            (0, 0, 0),
            (0, 1, 0),
            (0, 2, 0),
            (4, 4, 4),
        )
        tiles = [
            tile("empty"),
            tile("empty_variant"),
            tile("slide_start", E="SLIDE"),
            tile("slide_middle", W="SLIDE", E="SLIDE"),
            tile("slide_end", W="SLIDE"),
        ]

        result = solve_wfc(
            [start, middle, end, isolated],
            tiles,
            fixed_tiles={start: 2, end: 4},
            required_typed_pairs=[(start, end, PortKind.SLIDE)],
            preferred_tile_index=0,
            seed=1,
        )

        self.assertEqual(result[start], 2)
        self.assertEqual(result[middle], 3)
        self.assertEqual(result[end], 4)
        self.assertEqual(result[isolated], 0)

    def test_tile_aware_connectivity_preserves_a_valid_wfc_solution(self):
        start, middle, end = (0, 0, 0), (0, 1, 0), (0, 2, 0)
        tiles = [
            tile("start", E="SLIDE"),
            tile("middle", W="SLIDE", E="SLIDE"),
            tile("end", W="SLIDE"),
            tile("empty"),
        ]

        result = solve_wfc(
            [start, middle, end],
            tiles,
            fixed_tiles={start: 0, end: 2},
            required_typed_pairs=[(start, end, PortKind.SLIDE)],
            tile_aware_connectivity=True,
            seed=4,
        )

        self.assertEqual(result, {start: 0, middle: 1, end: 2})

    def test_endpoint_frontier_selects_the_first_unresolved_route_cell(self):
        start, first, second, end = (
            (0, 0, 0),
            (0, 1, 0),
            (0, 2, 0),
            (0, 3, 0),
        )
        tiles = [
            tile("start", E="SLIDE"),
            tile("end", W="SLIDE"),
            tile("middle_a", W="SLIDE", E="SLIDE"),
            tile("middle_b", W="SLIDE", E="SLIDE"),
            tile("middle_c", W="SLIDE", E="SLIDE"),
        ]
        domains = {
            start: {0},
            first: {2, 3, 4},
            second: {2, 3},
            end: {1},
        }
        arguments = (
            domains,
            tiles,
            [(start, end, PortKind.SLIDE)],
            random.Random(5),
            0,
        )
        keywords = {
            "compatibility": build_compatibility(tiles),
            "phase_counts": Counter(),
        }

        choice = _select_route_tile(
            *arguments,
            **keywords,
            endpoint_frontier=True,
        )

        self.assertEqual(choice[0], first)

    def test_one_step_lookahead_skips_an_immediately_contradictory_candidate(self):
        start, middle, end = (0, 0, 0), (0, 1, 0), (0, 2, 0)
        tiles = [
            tile("start", E="SLIDE"),
            tile("end", W="SLIDE"),
            tile("middle_a", W="SLIDE", E="SLIDE"),
            tile("middle_b", W="SLIDE", E="SLIDE"),
        ]
        domains = {
            start: {0},
            middle: {2, 3},
            end: {1},
        }
        rejected_candidate = []

        def propagate_with_one_bad_choice(trial_domains, compatibility, changed):
            _ = compatibility, changed
            candidate = next(iter(trial_domains[middle]))
            if not rejected_candidate:
                rejected_candidate.append(candidate)
                raise ContradictionError("injected local lookahead contradiction")

        with patch(
            "src.wfc_solver.propagate",
            side_effect=propagate_with_one_bad_choice,
        ) as propagation:
            choice = _select_route_tile(
                domains,
                tiles,
                [(start, end, PortKind.SLIDE)],
                random.Random(0),
                0,
                compatibility=build_compatibility(tiles),
                endpoint_frontier=True,
                candidate_lookahead=True,
                phase_counts=Counter(),
            )

        self.assertEqual(propagation.call_count, 2)
        self.assertEqual(choice[0], middle)
        self.assertNotEqual(choice[1], rejected_candidate[0])

    def test_lookahead_preserves_rng_state_for_single_dfs_candidate(self):
        start, middle, end = (0, 0, 0), (0, 1, 0), (0, 2, 0)
        tiles = [
            tile("start", E="SLIDE"),
            tile("end", W="SLIDE"),
            tile("middle", W="SLIDE", E="SLIDE"),
        ]
        domains = {start: {0}, middle: {2}, end: {1}}
        baseline_rng = random.Random(11)
        lookahead_rng = random.Random(11)
        arguments = (
            domains,
            tiles,
            [(start, end, PortKind.SLIDE)],
        )
        keywords = {
            "required_horizontal_steps": 0,
            "compatibility": build_compatibility(tiles),
        }

        baseline = _select_route_tile(
            *arguments,
            baseline_rng,
            phase_counts=Counter(),
            **keywords,
        )
        lookahead = _select_route_tile(
            *arguments,
            lookahead_rng,
            candidate_lookahead=True,
            phase_counts=Counter(),
            **keywords,
        )

        self.assertEqual(lookahead, baseline)
        self.assertEqual(lookahead_rng.getstate(), baseline_rng.getstate())

    def test_local_propagation_failure_reports_initial_stage(self):
        cells = [(0, 0, 0), (0, 1, 0)]
        tiles = [
            tile("east_walk", E="WALK"),
            tile("empty"),
        ]

        with self.assertRaises(ContradictionError) as context:
            solve_wfc(
                cells,
                tiles,
                fixed_tiles={cells[0]: 0, cells[1]: 1},
                max_attempts=1,
            )

        error = context.exception
        self.assertEqual(error.failure.phase, "initial_propagation")
        self.assertIn(error.failure.cell, cells)

    def test_final_validation_failure_reports_its_stage(self):
        with patch(
            "src.wfc_solver.validate_global_constraints",
            side_effect=ValueError("injected validation failure"),
        ):
            with self.assertRaises(ContradictionError) as context:
                solve_wfc(
                    [(0, 0, 0)],
                    [tile("empty")],
                    max_attempts=1,
                )

        self.assertEqual(
            context.exception.failure.phase,
            "final_validation",
        )


if __name__ == "__main__":
    unittest.main()
