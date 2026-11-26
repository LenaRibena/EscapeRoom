import unittest
from unittest.mock import patch

from src.building_generation import (
    BuildingRegenerationError,
    _candidate_seed,
    generate_building,
)
from src.json_export import building_to_dict
from src.wfc_building import RouteGenerationError, WFCBuilding


class BuildingRegenerationTests(unittest.TestCase):
    def test_seed10_regeneration_is_deterministic_and_exported(self):
        options = dict(floors=4, width=10, length=10, difficulty=2, seed=10,
                       grid_resolution=3, recovery_strategy="none",
                       max_building_retries=3)
        first = generate_building(**options)
        second = generate_building(**options)
        self.assertEqual([a.candidate_seed for a in first.attempts],
                         [a.candidate_seed for a in second.attempts])
        self.assertEqual(first.attempts[0].stage, "accepted")
        self.assertEqual(
            first.building.building_retry_count,
            len(first.attempts) - 1,
        )
        accepted_index = first.building.building_retry_count
        self.assertEqual(first.building.requested_seed, 10)
        self.assertEqual(
            first.building.accepted_seed, _candidate_seed(10, accepted_index)
        )
        first_json = building_to_dict(first.building)
        self.assertEqual(first_json, building_to_dict(second.building))
        self.assertEqual(first_json["generation"]["seed"], 10)
        self.assertEqual(first_json["generation"]["requested_seed"], 10)
        self.assertEqual(
            first_json["generation"]["accepted_seed"],
            _candidate_seed(10, accepted_index),
        )
        self.assertEqual(
            first_json["generation"]["building_retry_count"], accepted_index
        )
        self.assertEqual([r["geometry"] for r in first_json["routes"]],
                         [r.accepted_geometry for r in first.building.routes])
        reproduced = WFCBuilding(
            floors=4, width=10, length=10, difficulty=2, seed=first.building.accepted_seed,
            grid_resolution=3, recovery_strategy="none",
        )
        reproduced_json = building_to_dict(reproduced)
        self.assertEqual(first_json["graph"], reproduced_json["graph"])
        self.assertEqual(first_json["routes"], reproduced_json["routes"])

    def test_zero_retries_preserves_original_seeded_building(self):
        options = dict(floors=4, width=10, length=10, difficulty=2, seed=10,
                       grid_resolution=3, recovery_strategy="none")
        direct = WFCBuilding(**options)
        result = generate_building(**options)
        self.assertEqual(building_to_dict(direct), building_to_dict(result.building))
        self.assertEqual(result.attempts[0].candidate_seed, 10)
        self.assertEqual(len(result.attempts), 1)

    def test_exhaustion_preserves_attempts_and_original_error(self):
        class FakeBuilding:
            def __init__(self, *, seed, **kwargs):
                self.seed = seed
                self.total_wfc_attempts = 0
                self.route_failures = [{"route_type": "stair"}]

            def generate_routes(self):
                raise RouteGenerationError(
                    "stair collision", failure_classification="geometry_collision",
                    failure_details={"obstacle": "stair"},
                )

        with patch("src.building_generation.WFCBuilding", FakeBuilding):
            with self.assertRaises(BuildingRegenerationError) as caught:
                generate_building(seed=10, max_building_retries=3)
        self.assertEqual(len(caught.exception.building_attempts), 4)
        self.assertEqual([a.candidate_seed for a in caught.exception.building_attempts],
                         [_candidate_seed(10, i) for i in range(4)])
        self.assertTrue(all(a.stage == "direct-stair" for a in caught.exception.building_attempts))
        self.assertEqual(caught.exception.building_attempts[0].failure_details["obstacle"], "stair")
        self.assertIsInstance(caught.exception.__cause__, RouteGenerationError)

    def test_graph_generation_error_is_not_retried(self):
        with patch("src.building_generation.WFCBuilding",
                   side_effect=RuntimeError("original graph failed")) as factory:
            with self.assertRaisesRegex(RuntimeError, "original graph failed") as caught:
                generate_building(seed=2, max_building_retries=3)
            self.assertEqual(factory.call_count, 1)
            self.assertEqual(
                [attempt.stage for attempt in caught.exception.building_attempts],
                ["graph-generation"],
            )

        class FakeBuilding:
            def __init__(self, **kwargs):
                self.total_wfc_attempts = 1
                self.route_failures = [{"route_type": "slide"}]

            def generate_routes(self):
                raise RouteGenerationError("bad endpoint", failure_classification="invalid_endpoint")

        with patch("src.building_generation.WFCBuilding", FakeBuilding):
            with self.assertRaisesRegex(RouteGenerationError, "bad endpoint") as caught:
                generate_building(seed=2, max_building_retries=3)
        self.assertEqual(len(caught.exception.building_attempts), 1)
        self.assertEqual(caught.exception.building_attempts[0].stage, "slide-routing")

    def test_graph_generation_error_after_route_rejection_is_not_retried(self):
        class FakeBuilding:
            total_wfc_attempts = 0
            route_failures = [{"route_type": "stair"}]

            def generate_routes(self):
                raise RouteGenerationError(
                    "stair collision", failure_classification="geometry_collision"
                )

        with patch("src.building_generation.WFCBuilding",
                   side_effect=[FakeBuilding(), RuntimeError("original graph failed")]) as factory:
            with self.assertRaisesRegex(RuntimeError, "original graph failed") as caught:
                generate_building(seed=2, max_building_retries=3)
        self.assertEqual(factory.call_count, 2)
        self.assertEqual(
            [attempt.stage for attempt in caught.exception.building_attempts],
            ["direct-stair", "graph-generation"],
        )

    def test_budget_validation_and_seed_sequence(self):
        self.assertEqual(_candidate_seed(10, 0), 10)
        self.assertEqual(len({_candidate_seed(10, i) for i in range(4)}), 4)
        for invalid in (-1, True, 1.5, 2**32):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    generate_building(max_building_retries=invalid)

    def test_slide_geometry_rejection_retries_but_missing_geometry_propagates(self):
        created = []

        class FakeBuilding:
            def __init__(self, *, seed, **kwargs):
                self.seed = seed
                self.total_wfc_attempts = 0
                self.route_failures = []
                self.routes = [object()]
                created.append(self)

            def generate_routes(self):
                if len(created) == 1:
                    self.total_wfc_attempts = 30
                    self.route_failures = [{"route_type": "slide"}]
                    raise RouteGenerationError(
                        "geometry rejected",
                        failure_classification="exhausted_retries",
                        failure_details={"geometry_rejections": 30},
                    )

            def validate_accepted_geometry(self, route):
                raise RuntimeError("accepted route lacks stored geometry")

        with patch("src.building_generation.WFCBuilding", FakeBuilding):
            with self.assertRaisesRegex(RuntimeError, "lacks stored geometry"):
                generate_building(seed=1, max_building_retries=3)
        self.assertEqual(len(created), 2)

    def test_final_geometry_failure_has_a_distinct_stage(self):
        created = []

        class FakeBuilding:
            def __init__(self, **kwargs):
                self.total_wfc_attempts = 0
                self.route_failures = []
                self.routes = [object()]
                created.append(self)

            def generate_routes(self):
                pass

            def validate_accepted_geometry(self, route):
                raise RouteGenerationError(
                    "collision on revalidation",
                    failure_classification="geometry_collision",
                )

        with patch("src.building_generation.WFCBuilding", FakeBuilding):
            with self.assertRaises(BuildingRegenerationError) as caught:
                generate_building(seed=1, max_building_retries=1)
        self.assertEqual(len(created), 2)
        self.assertTrue(all(attempt.stage == "geometry-validation"
                            for attempt in caught.exception.building_attempts))


if __name__ == "__main__":
    unittest.main()
