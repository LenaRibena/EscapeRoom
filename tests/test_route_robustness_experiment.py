import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from archive.legacy_graph_wfc.src.legacy_building import Building as LegacyBuilding
from src.wfc_building import WFCBuilding

from examples.route_robustness_experiment import (
    _plot_results,
    _write_csv,
    run_controlled_comparison,
    run_experiments,
    run_solver_heuristic_ablation,
)


class RouteRobustnessExperimentTests(unittest.TestCase):
    def setUp(self):
        class LegacyGraphFixture(LegacyBuilding):
            def __init__(self, floors, width, length, difficulty, *, max_graph_attempts=100):
                super().__init__(floors, width, length, difficulty)

        original_bases = WFCBuilding.__bases__
        WFCBuilding.__bases__ = (LegacyGraphFixture,)
        self.addCleanup(setattr, WFCBuilding, "__bases__", original_bases)

    def test_seed_resolution_run_records_route_types_lengths_and_attempts(self):
        result = run_experiments((0,), (2,))[0]

        self.assertTrue(result.success)
        self.assertTrue(result.first_route_success)
        self.assertFalse(result.tile_aware_connectivity)
        self.assertTrue(result.endpoint_frontier)
        self.assertFalse(result.candidate_lookahead)
        self.assertEqual(result.graph_feasibility, "feasible")
        self.assertEqual(result.discrete_wfc_feasibility, "solution_found")
        self.assertEqual(result.smoothed_geometry_feasibility, "accepted")
        self.assertEqual(result.route_count, 9)
        self.assertGreater(result.generation_seconds, 0)
        self.assertGreaterEqual(
            result.wfc_attempts,
            sum(route["attempts"] for route in result.route_metrics),
        )
        self.assertTrue(
            all(
                route["kind"] in ("stair", "slide")
                and (
                    (route["kind"] == "slide" and route["cell_count"] > 0
                     and route["attempts"] > 0)
                    or (route["kind"] == "stair" and route["cell_count"] == 0
                        and route["attempts"] == 0)
                )
                and route["recovery_count"] >= 0
                for route in result.route_metrics
            )
        )
        self.assertGreaterEqual(result.recovery_attempts, 0)

    def test_inherited_graph_failure_is_labeled_separately_from_wfc(self):
        result = run_experiments((7,), (2,))[0]

        self.assertFalse(result.success)
        self.assertEqual(result.failure_stage, "graph generation")
        self.assertEqual(result.graph_feasibility, "no_solution_found")
        self.assertEqual(result.discrete_wfc_feasibility, "not_tested")
        self.assertEqual(result.smoothed_geometry_feasibility, "not_tested")
        self.assertIsNone(result.first_route_success)
        self.assertIn("building graph generation failed", result.error)

    def test_bounded_wfc_failure_is_not_reported_as_proven_infeasibility(self):
        result = run_experiments(
            (2,),
            (2,),
            max_route_attempts=1,
            recovery_strategy="none",
            minimum_curvature_radius=0.0,
        )[0]

        self.assertEqual(result.graph_feasibility, "feasible")
        self.assertEqual(
            result.discrete_wfc_feasibility,
            "solution_candidate_found",
        )
        self.assertEqual(
            result.smoothed_geometry_feasibility,
            "all_candidates_rejected",
        )

    def test_summary_plot_can_be_saved(self):
        results = run_experiments((0, 7), (2,))
        with TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "summary.png"
            _plot_results(results, path)

            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 0)

    def test_controlled_comparison_uses_matched_seeds_and_exports_json_details(self):
        results = run_controlled_comparison((7,), (2,))

        self.assertEqual(len(results), 4)
        self.assertEqual(
            {result.recovery_strategy for result in results},
            {"none", "recency", "conflict-aware"},
        )
        self.assertEqual({result.seed for result in results}, {7})
        self.assertEqual({result.grid_resolution for result in results}, {2})
        self.assertEqual(
            {result.minimum_curvature_radius for result in results},
            {0.0, 0.02},
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "comparison.csv"
            _write_csv(path, results)
            contents = path.read_text(encoding="utf-8")
            self.assertIn("route_failure_details", contents)
            self.assertIn("recovery_details", contents)

    def test_solver_heuristic_ablation_keeps_each_option_independent(self):
        results = run_solver_heuristic_ablation((7,), (2,))

        self.assertEqual(len(results), 6)
        self.assertEqual(
            {
                (
                    result.tile_aware_connectivity,
                    result.endpoint_frontier,
                    result.candidate_lookahead,
                )
                for result in results
            },
            {
                (False, False, False),
                (True, False, False),
                (False, True, False),
                (False, False, True),
                (True, True, False),
                (True, True, True),
            },
        )
        self.assertTrue(all(result.graph_feasibility == "no_solution_found" for result in results))


if __name__ == "__main__":
    unittest.main()
