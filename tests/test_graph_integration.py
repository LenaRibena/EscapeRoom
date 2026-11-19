import unittest
from unittest.mock import patch

import networkx as nx
import numpy as np

from examples import route_robustness_experiment
from src.graph import Building, GraphGenerationError
from src.wfc_building import WFCBuilding


class GraphIntegrationTests(unittest.TestCase):
    def test_seeded_graph_has_valid_solution_and_dead_ends(self):
        state = np.random.get_state()
        try:
            np.random.seed(10)
            building = Building(5, 5, 5, difficulty=2)
        finally:
            np.random.set_state(state)
        self.assertEqual(building.path[0], building.start_pos)
        self.assertEqual(building.path[-1], building.exit_pos)
        self.assertTrue(all(building.graph.has_edge(a, b)
                            for a, b in zip(building.path, building.path[1:])))
        self.assertEqual(building.validate_graph(), ([], []))
        self.assertTrue(1 <= building.graph_attempts <= 100)

    def test_impossible_solution_is_rejected_with_attempt_limit(self):
        with self.assertRaisesRegex(GraphGenerationError, "after 3 attempts.*no solution path"):
            Building(1, 1, 1, difficulty=2, max_graph_attempts=3)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            Building(2, 2, 2, difficulty=1, max_graph_attempts=0)

    def test_graph_retries_then_raises_without_accepting_invalid_candidate(self):
        empty_path = np.empty((0, 3), dtype=int)
        with patch.object(Building, "generate_path", return_value=empty_path) as search:
            with self.assertRaisesRegex(GraphGenerationError, "after 4 attempts"):
                Building(2, 2, 2, difficulty=1, max_graph_attempts=4)
        self.assertEqual(search.call_count, 4)

    def test_population_no_path_is_retried_and_reported(self):
        state = np.random.get_state()
        try:
            np.random.seed(10)
            with patch.object(Building, "populate_graph",
                              side_effect=nx.NetworkXNoPath("unreachable")) as populate:
                with self.assertRaisesRegex(GraphGenerationError,
                                            "after 2 attempts.*unreachable"):
                    Building(5, 5, 5, difficulty=2, max_graph_attempts=2)
            self.assertEqual(populate.call_count, 2)
        finally:
            np.random.set_state(state)

    def test_population_failure_can_regenerate_a_valid_graph(self):
        state = np.random.get_state()
        original = Building.populate_graph
        calls = 0

        def fail_once(building, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise nx.NetworkXNoPath("first candidate is invalid")
            return original(building, *args, **kwargs)

        try:
            np.random.seed(10)
            with patch.object(Building, "populate_graph", fail_once):
                building = Building(5, 5, 5, difficulty=2)
            self.assertGreaterEqual(building.graph_attempts, 2)
            self.assertEqual(building.validate_graph(), ([], []))
        finally:
            np.random.set_state(state)

    def test_floor_link_preserves_existing_stair_head_type(self):
        building = Building.__new__(Building)
        building.exit_pos = (0, 0, 0)
        graph = nx.DiGraph()
        source, head, tail = (0, 0, 1), (1, 0, 1), (1, 0, 2)
        for node in (source, head, tail):
            graph.add_node(node, type="wall", dead_end=True)
        building.update_graph(graph, [head, tail])
        building.update_graph(graph, [source, head])
        self.assertEqual(graph.nodes[head]["type"], "stair_head")
        self.assertEqual(graph.edges[head, tail]["type"], "stair")
        self.assertFalse(graph.has_edge(head, source))

    def test_stale_dead_end_flags_are_corrected_before_validation(self):
        building = Building.__new__(Building)
        building.start_pos, building.exit_pos = (0, 0, 1), (0, 0, 0)
        leaf = (1, 0, 1)
        graph = nx.DiGraph()
        for node in (building.start_pos, building.exit_pos, leaf):
            graph.add_node(node, type="floor", dead_end=False)
        graph.add_edges_from(((building.start_pos, building.exit_pos),
                              (building.start_pos, leaf)))
        building.graph = graph
        self.assertEqual(building.validate_graph(), ([leaf], []))
        self.assertTrue(graph.nodes[leaf]["dead_end"])

    def test_dead_end_flag_tracks_expansion_of_a_floor_leaf(self):
        building = Building.__new__(Building)
        building.exit_pos = (0, 0, 0)
        graph = nx.DiGraph()
        start, leaf, extension = (0, 0, 1), (1, 0, 1), (2, 0, 1)
        for node in (start, leaf, extension):
            graph.add_node(node, type="wall", dead_end=True)
        graph.nodes[start]["type"] = "floor"
        building.update_graph(graph, [start, leaf])
        self.assertTrue(graph.nodes[start]["dead_end"])
        self.assertTrue(graph.nodes[leaf]["dead_end"])
        building.update_graph(graph, [leaf, extension])
        self.assertFalse(graph.nodes[leaf]["dead_end"])
        self.assertTrue(graph.nodes[extension]["dead_end"])

    def test_non_dead_end_cycle_policy_is_not_strengthened(self):
        building = Building.__new__(Building)
        building.start_pos, building.exit_pos = (0, 0, 1), (0, 0, 0)
        a, b, c = (1, 0, 1), (2, 0, 1), (3, 0, 1)
        graph = nx.DiGraph()
        for node in (building.start_pos, building.exit_pos, a, b, c):
            graph.add_node(node, type="floor", dead_end=False)
        graph.add_edges_from(((building.start_pos, building.exit_pos),
                              (building.start_pos, a), (a, b), (b, c), (c, a)))
        building.graph = graph
        self.assertEqual(building.validate_graph(), ([], []))
        self.assertFalse(nx.has_path(graph, a, building.exit_pos))

    def test_directed_sink_with_multiple_predecessors_is_a_dead_end(self):
        building = Building.__new__(Building)
        building.start_pos, building.exit_pos = (0, 0, 1), (0, 0, 0)
        other, sink = (1, 0, 1), (2, 0, 1)
        graph = nx.DiGraph()
        for node in (building.start_pos, building.exit_pos, other, sink):
            graph.add_node(node, type="floor", dead_end=False)
        graph.add_edges_from(((building.start_pos, building.exit_pos),
                              (building.start_pos, sink),
                              (building.start_pos, other), (other, sink)))
        building.graph = graph
        self.assertEqual(building.validate_graph(), ([sink], []))

    def test_unreachable_non_wall_node_is_rejected(self):
        building = Building.__new__(Building)
        building.start_pos, building.exit_pos = (0, 0, 1), (0, 0, 0)
        disconnected = (1, 0, 1)
        graph = nx.DiGraph()
        for node in (building.start_pos, building.exit_pos, disconnected):
            graph.add_node(node, type="floor", dead_end=False)
        graph.add_edge(building.start_pos, building.exit_pos)
        building.graph = graph
        self.assertEqual(building.validate_graph(), ([disconnected], [disconnected]))

    def test_wfc_constructor_exposes_graph_attempt_limit_and_restores_rng(self):
        state = np.random.get_state()
        try:
            np.random.seed(123)
            before = np.random.get_state()
            with self.assertRaises(GraphGenerationError):
                WFCBuilding(1, 1, 1, difficulty=2, seed=10,
                            max_graph_attempts=2, generate_routes=False)
            after = np.random.get_state()
            self.assertEqual(before[0], after[0])
            self.assertTrue(np.array_equal(before[1], after[1]))
            self.assertEqual(before[2:], after[2:])
        finally:
            np.random.set_state(state)

    def test_experiment_reports_bounded_graph_failure_and_timing(self):
        with patch.object(route_robustness_experiment, "WFCBuilding",
                          side_effect=GraphGenerationError("attempt limit")) as factory:
            result, = route_robustness_experiment.run_experiments(
                [10], [2], max_graph_attempts=3,
            )
        self.assertEqual(factory.call_args.kwargs["max_graph_attempts"], 3)
        self.assertEqual(result.failure_stage, "graph generation")
        self.assertEqual(result.graph_feasibility, "no_solution_found_within_attempt_budget")
        self.assertEqual(result.routing_seconds, 0)
        self.assertGreaterEqual(result.graph_seconds, 0)
        self.assertEqual(result.graph_seconds, result.generation_seconds)

    def test_experiment_comparison_modes_forward_graph_attempt_limit(self):
        with patch.object(route_robustness_experiment, "run_experiments",
                          return_value=()) as run:
            route_robustness_experiment.run_controlled_comparison([10], max_graph_attempts=3)
            route_robustness_experiment.run_solver_heuristic_ablation([10], max_graph_attempts=3)
        self.assertEqual(run.call_count, 10)
        self.assertTrue(all(call.kwargs["max_graph_attempts"] == 3
                            for call in run.call_args_list))


if __name__ == "__main__":
    unittest.main()
