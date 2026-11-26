from dataclasses import replace
from math import cos, hypot, pi, sin
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import networkx as nx

from src.wfc_building import (
    RouteGenerationError,
    ConnectorReservation,
    WFCBuilding,
    WFCPath,
    _classify_route_failure,
    _edge_name,
    build_route_tiles,
)
from src.route_diagnostics import diagnose_routes
from src.wfc_tiles import Direction, PortKind


def small_building(
    source=(1, 1, 1),
    target=(1, 1, 0),
    kind="slide",
    resolution=1,
    vertical_resolution=None,
    recovery_strategy="recency",
    minimum_curvature_radius=0.0,
):
    if vertical_resolution is None:
        vertical_resolution = resolution
    building = object.__new__(WFCBuilding)
    building.floors = 2
    building.width = 3
    building.length = 3
    building.grid_resolution = resolution
    building.vertical_grid_resolution = vertical_resolution
    building.grid_shape = (
        vertical_resolution + 1,
        2 * resolution + 1,
        2 * resolution + 1,
    )
    building.seed = 7
    building.max_route_attempts = 20
    building.max_route_backtrack_depth = 2
    building.max_route_recovery_attempts = 1
    building.minimum_slide_horizontal_travel = 0.5
    building.maximum_slide_grade = 1.5
    building.minimum_turning_radius = 0.02
    building.minimum_curvature_radius = minimum_curvature_radius
    building.tile_aware_connectivity = False
    building.endpoint_frontier = False
    building.candidate_lookahead = False
    building.recovery_strategy = recovery_strategy
    building.forbidden_cells = set()
    building.occupancy = {}
    building._reserved_connector_cells = set()
    building._reserved_connector_owners = {}
    building._connector_reservations_by_cell = {}
    building.total_wfc_attempts = 0
    building.recovery_attempts = 0
    building.routes_retried = 0
    building.routes = []
    building.recovery_events = []
    building.route_failures = []
    building.graph = nx.DiGraph()
    building.graph.add_edge(source, target, kind=kind)
    building.floor_grid = frozenset(
        (x, y, floor)
        for x in range(3)
        for y in range(3)
        for floor in range(2)
    )
    return building


class WFCBuildingTests(unittest.TestCase):
    def test_embedding_repair_preserves_topology_and_solution(self):
        building = object.__new__(WFCBuilding)
        building.length = 5
        building.width = 5
        building.grid_resolution = 2
        building.start_pos = (4, 4, 4)
        building.exit_pos = (0, 0, 0)
        source, target = (4, 4, 2), (2, 4, 4)
        blocker, other = (3, 4, 3), (3, 3, 2)
        building.graph = nx.DiGraph()
        building.graph.add_edge(source, target, type="stair")
        building.graph.add_edge(blocker, other, type="slide")
        building.graph.add_nodes_from((building.start_pos, building.exit_pos))
        building.path = [building.start_pos, source, target, building.exit_pos]
        building.all_valid_notes = list(building.graph)
        building.floor_grid = frozenset(building.graph)
        building.embedding_repair = None

        self.assertEqual(
            building._stair_endpoint_conflict(building.graph),
            (source, target, blocker),
        )
        original_edges = {
            (start, end, attributes["type"])
            for start, end, attributes in building.graph.edges(data=True)
        }
        building._repair_graph_embedding()

        self.assertEqual(building.embedding_repair["floor"], 3)
        self.assertIsNone(building._stair_endpoint_conflict(building.graph))
        mapping = {
            blocker: (1, 0, 3),
        }
        self.assertEqual(
            {
                (start, end, attributes["type"])
                for start, end, attributes in building.graph.edges(data=True)
            },
            {
                (mapping.get(start, start), mapping.get(end, end), kind)
                for start, end, kind in original_edges
            },
        )
        self.assertEqual(
            building.path,
            [mapping.get(node, node) for node in (
                building.start_pos, source, target, building.exit_pos
            )],
        )
        self.assertEqual(building.floor_grid, frozenset(building.graph))

    def test_direct_stair_failures_are_reported(self):
        building = WFCBuilding(
            floors=4, width=10, length=10, difficulty=2, seed=10,
            grid_resolution=3, recovery_strategy="none", generate_routes=False,
        )
        with patch.object(
            building, "_direct_stair",
            side_effect=RouteGenerationError(
                "test collision", failure_classification="geometry_collision"
            ),
        ):
            with self.assertRaises(RouteGenerationError) as caught:
                building.generate_routes()
        self.assertTrue(caught.exception.failure_classification)
        self.assertTrue(building.route_failures)
        self.assertTrue(caught.exception.failure_details)
        self.assertTrue(any(
            failure.get("route_type") in {"stair", "slide"}
            for failure in building.route_failures
        ))

    def test_duplicate_rejected_paths_skip_repeated_geometry_validation(self):
        building = small_building(resolution=2)
        with patch.object(
            building, "_smooth_route_candidate",
            side_effect=RouteGenerationError("rejected geometry"),
        ) as smooth:
            with self.assertRaises(RouteGenerationError) as caught:
                building._generate_route("slide-0", (1, 1, 1), (1, 1, 0), "slide")
        details = caught.exception.failure_details
        self.assertGreater(details["duplicate_candidates"], 0)
        self.assertEqual(smooth.call_count, details["rejected_route_signatures"])
        self.assertEqual(
            details["geometry_rejections"], details["discrete_route_candidates"]
        )

    def test_reverse_stair_clearance_conflict_identifies_candidate_cell(self):
        building = small_building(
            (0, 0, 0), (2, 2, 1), "stair", resolution=2, vertical_resolution=2
        )
        stair = building._direct_stair("stair", (0, 0, 0), (2, 2, 1))
        building._add_route(stair)
        cell = (1, 2, 2)
        candidate = WFCPath(
            "slide", PortKind.SLIDE, (2, 0, 1), (2, 1, 0), (cell,),
            accepted_geometry=((2.0, 0.0, 1.0), (2.0, 1.0, 0.0)),
        )
        with self.assertRaises(RouteGenerationError) as caught:
            building._validate_existing_routes_with_candidate(candidate)
        self.assertEqual(caught.exception.failure_details["blocking_stair_route_id"], "stair")
        self.assertEqual(
            building._stair_conflict_exclusions(
                candidate, caught.exception, {cell: frozenset((0, 1))}, 0, set()
            ),
            (cell,),
        )
        self.assertEqual(tuple(building.routes), (stair,))

    def test_graph_endpoint_validation_preserves_shared_stair_endpoint(self):
        building = small_building(
            (0, 0, 0), (2, 2, 1), "stair", resolution=2, vertical_resolution=2
        )
        building.graph.add_edge((0, 0, 0), (0, 2, 1), kind="stair")
        building.graph.add_edge((2, 0, 1), (2, 2, 1), kind="slide")
        self.assertIsNone(building._stair_endpoint_conflict(building.graph))

    def test_stair_conflict_exclusions_never_remove_connectors_or_endpoints(self):
        building = small_building(
            (0, 0, 0), (2, 2, 1), "stair", resolution=2, vertical_resolution=2
        )
        stair = building._direct_stair("stair", (0, 0, 0), (2, 2, 1))
        building._add_route(stair)
        cells = ((2, 4, 0), (1, 4, 0), (1, 3, 0), (0, 3, 0))
        route = WFCPath("slide", PortKind.SLIDE, (2, 0, 1), (1, 0, 0), cells)
        error = RouteGenerationError(
            "intersects",
            failure_details={
                "other_route_id": "stair",
                "candidate_segment_points": ((1.5, 0.0, 0.5), (1.0, 0.0, 0.0)),
            },
        )
        allowed = {cell: frozenset((0, 1)) for cell in cells}
        protected = {cells[0], cells[1], cells[-1]}
        self.assertEqual(
            building._stair_conflict_exclusions(route, error, allowed, 0, protected),
            (cells[2],),
        )
        self.assertEqual(allowed, {cell: frozenset((0, 1)) for cell in cells})

    def test_direct_diagonal_stair_does_not_invoke_wfc(self):
        source, target = (0, 0, 0), (2, 2, 1)
        building = small_building(source, target, "stair", resolution=2, vertical_resolution=2)
        with patch("src.wfc_building.solve_wfc", side_effect=AssertionError("stair used WFC")):
            routes = building.generate_routes()
        self.assertEqual(len(routes), 1)
        self.assertTrue(routes[0].direct_stair)
        self.assertEqual(routes[0].accepted_geometry, (source, target))
        self.assertEqual(routes[0].cells, ())
        self.assertEqual(building.total_wfc_attempts, 0)
        self.assertEqual(diagnose_routes(building).logical_errors, ())

    def test_direct_stair_rejects_forbidden_cell_along_segment(self):
        source, target = (0, 0, 0), (2, 2, 1)
        building = small_building(source, target, "stair", resolution=2, vertical_resolution=2)
        building.forbidden_cells.add((1, 2, 2))
        with self.assertRaisesRegex(RouteGenerationError, "Direct stair.*infeasible") as caught:
            building.generate_routes()
        self.assertEqual(caught.exception.failure_details["blocked_cell"], (1, 2, 2))
        self.assertEqual(building.total_wfc_attempts, 0)

    def test_stair_reservation_allows_shared_endpoint_in_slide_grid(self):
        source, target = (0, 0, 0), (2, 2, 1)
        building = small_building(source, target, "stair", resolution=2, vertical_resolution=2)
        stair = building._direct_stair("stair", source, target)
        building._add_route(stair)
        start = building.graph_node_to_cell(source)
        available = building._route_grid_cells(
            start,
            building.graph_node_to_cell((2, 0, 1)),
            shared_endpoints={start},
            route_id="slide",
        )
        self.assertIn(start, available)

    def test_direct_stairs_reject_crossings_and_unrelated_graph_endpoints(self):
        building = small_building((0, 0, 0), (2, 2, 1), "stair", resolution=2, vertical_resolution=2)
        first = building._direct_stair("first", (0, 0, 0), (2, 2, 1))
        building._add_route(first)
        with self.assertRaisesRegex(RouteGenerationError, "intersects route first"):
            building._direct_stair("crossing", (0, 2, 0), (2, 0, 1))
        building._restore_routes(())
        building.graph.add_edge((1, 1, 0), (1, 1, 1), kind="slide")
        with self.assertRaisesRegex(RouteGenerationError, "route-grid cell"):
            building._direct_stair("blocked", (0, 0, 0), (2, 2, 2))

    def test_direct_stair_and_slide_intersection_and_shared_endpoint(self):
        building = small_building((0, 0, 0), (2, 2, 1), "stair", resolution=2, vertical_resolution=2)
        stair = building._direct_stair("stair", (0, 0, 0), (2, 2, 1))
        building._add_route(stair)
        crossing = WFCPath(
            "crossing", PortKind.SLIDE, (0, 2, 1), (2, 0, 0), (),
            accepted_geometry=((0.0, 2.0, 1.0), (2.0, 0.0, 0.0)),
        )
        with self.assertRaisesRegex(RouteGenerationError, "intersects route stair"):
            building._validate_route_intersections(crossing, crossing.accepted_geometry)
        shared = WFCPath(
            "shared", PortKind.SLIDE, (0, 2, 1), (0, 0, 0), (),
            accepted_geometry=((0.0, 2.0, 1.0), (0.0, 0.0, 0.0)),
        )
        building._validate_route_intersections(shared, shared.accepted_geometry)
        overlapping = replace(
            shared, accepted_geometry=((2.0, 2.0, 1.0), (0.0, 0.0, 0.0))
        )
        with self.assertRaisesRegex(RouteGenerationError, "intersects route stair"):
            building._validate_route_intersections(overlapping, overlapping.accepted_geometry)

    def test_direct_stair_rejects_crossing_reserved_connector(self):
        building = small_building((0, 0, 0), (2, 2, 1), "stair", resolution=2, vertical_resolution=2)
        cell = (1, 2, 2)
        building._reserved_connector_owners[cell] = {"slide"}
        building._connector_reservations_by_cell[cell] = {
            "slide": ConnectorReservation((1, 1, 0), PortKind.SLIDE, Direction.U, False)
        }
        with self.assertRaisesRegex(RouteGenerationError, "reserved connector"):
            building._direct_stair("stair", (0, 0, 0), (2, 2, 1))


    def test_seed10_routes_stairs_and_exact_plotted_geometry(self):
        for resolution in (2, 3):
            with self.subTest(resolution=resolution):
                building = WFCBuilding(
                    floors=4, width=10, length=10, difficulty=2,
                    grid_resolution=resolution, seed=10,
                    endpoint_frontier=True, recovery_strategy="none",
                    minimum_curvature_radius=0.02, max_route_attempts=30,
                )
                stairs = [
                    (source, target)
                    for source, target, attributes in building.graph.edges(data=True)
                    if _edge_name(attributes) == "stair"
                ]
                expected_stairs = set(stairs)
                self.assertTrue(expected_stairs)
                self.assertTrue(all(target[2] > source[2] for source, target in stairs))
                for route in building.routes:
                    self.assertIsNotNone(route.accepted_geometry)
                    self.assertEqual(
                        building.smooth_path(route), route.accepted_geometry
                    )
                    if route.kind is PortKind.STAIR:
                        self.assertIn((route.source, route.target), expected_stairs)
                        self.assertEqual(route.accepted_geometry, (route.source, route.target))
                        self.assertTrue(route.direct_stair)
                        self.assertEqual(route.cells, ())
                with patch.object(
                    building, "_smooth_path_geometry",
                    side_effect=AssertionError("plot regenerated geometry"),
                ):
                    self.assertEqual(diagnose_routes(building).logical_errors, ())
                    figure, (graph_axes, route_axes) = building.plot()
                    for route, line in zip(building.routes, route_axes.lines):
                        plotted = tuple(zip(*line.get_data_3d()))
                        self.assertEqual(plotted, route.accepted_geometry)
                    stair_lines = [
                        line for line in graph_axes.lines
                        if line.get_color() == "#e58a22"
                    ]
                    self.assertEqual(
                        {tuple(zip(*line.get_data_3d())) for line in stair_lines},
                        expected_stairs,
                    )
                    plt.close(figure)
                    figure, _ = building.plot(show_diagnostics=True)
                    plt.close(figure)

                building.routes[0] = replace(
                    building.routes[0], accepted_geometry=None
                )
                with self.assertRaisesRegex(RuntimeError, "lacks stored geometry"):
                    diagnose_routes(building)
                with self.assertRaisesRegex(RuntimeError, "lacks stored geometry"):
                    building.plot()
                plt.close("all")

    def test_isolated_route_filter_keeps_stairs_as_fixed_occupancy(self):
        route_id = "0:(0, 4, 3)->(0, 0, 0):slide"
        building = WFCBuilding(
            floors=4,
            width=10,
            length=10,
            difficulty=2,
            grid_resolution=3,
            seed=10,
            recovery_strategy="none",
            only_route_id=route_id,
        )

        self.assertIn(route_id, {route.route_id for route in building.routes})
        self.assertTrue(
            all(
                route.direct_stair or route.route_id == route_id
                for route in building.routes
            )
        )
        self.assertEqual(
            sum(route.kind is PortKind.SLIDE for route in building.routes),
            1,
        )

    def test_later_candidate_cannot_invalidate_earlier_smoothed_route(self):
        building = small_building(
            (0, 0, 0), (2, 2, 1), "stair", resolution=2, vertical_resolution=2
        )
        stair = building._direct_stair("stair", (0, 0, 0), (2, 2, 1))
        building._add_route(stair)
        candidate = WFCPath(
            "candidate", PortKind.SLIDE, (2, 0, 1), (2, 1, 0),
            ((1, 2, 2),), vertical_resolution=2,
            accepted_geometry=((2.0, 0.0, 1.0), (2.0, 1.0, 0.0)),
        )
        original_routes = tuple(building.routes)
        original_occupancy = {cell: set(owners) for cell, owners in building.occupancy.items()}
        with self.assertRaisesRegex(RouteGenerationError, "collides near"):
            building._validate_existing_routes_with_candidate(candidate)
        self.assertEqual(tuple(building.routes), original_routes)
        self.assertEqual(building.occupancy, original_occupancy)
        building._add_route(candidate)
        report = diagnose_routes(building)
        self.assertTrue(any("collides near" in error for error in report.logical_errors))
        self.assertEqual(report.routes[0].geometry_points, stair.accepted_geometry)


    def test_seed10_preserves_constrained_slide_geometry_after_generation(self):
        building = WFCBuilding(
            floors=4,
            width=10,
            length=10,
            difficulty=2,
            grid_resolution=3,
            seed=10,
            endpoint_frontier=True,
            recovery_strategy="none",
            minimum_curvature_radius=0.02,
            max_route_attempts=30,
        )

        routes = building.generate_routes()

        slide_routes = [route for route in routes if route.kind is PortKind.SLIDE]
        self.assertTrue(slide_routes)
        for route in slide_routes:
            self.assertEqual(
                building.smooth_path(route), route.accepted_geometry
            )
            self.assertEqual(
                route.accepted_geometry[0],
                tuple(float(value) for value in route.source),
            )
            self.assertEqual(
                route.accepted_geometry[-1],
                tuple(float(value) for value in route.target),
            )

        diagnostics = diagnose_routes(building, routes)
        self.assertEqual(diagnostics.logical_errors, ())
        self.assertEqual(
            diagnostics.warnings,
            ("17:(0, 9, 2)->(0, 9, 1):slide: route has a turn sharper than 100.0 degrees",),
        )

    def test_routes_at_one_graph_node_can_share_a_typed_landing(self):
        building = small_building(vertical_resolution=3)
        shared_node = (1, 1, 0)
        stair_target = (0, 0, 1)
        slide_source = (2, 2, 1)
        building.graph.clear()
        building.graph.add_edge(shared_node, stair_target, kind="stair")
        building.graph.add_edge(slide_source, shared_node, kind="slide")
        building.floor_grid = frozenset(building.graph.nodes)
        shared_cell = building._endpoint_neighbor(
            building.graph_node_to_cell(shared_node),
            PortKind.STAIR,
            True,
        )
        self.assertIsNotNone(shared_cell)

        def fake_generate(
            route_id,
            source,
            target,
            edge_name,
            seed_offset=0,
            recovery_count=0,
        ):
            kind = PortKind(edge_name)
            connector = building._endpoint_neighbor(
                building.graph_node_to_cell(shared_node),
                kind,
                source == shared_node,
            )
            cells = (
                building.graph_node_to_cell(source),
                connector,
                building.graph_node_to_cell(target),
            )
            domains = building._route_grid_cells(
                cells[0],
                cells[-1],
                route_connector_cells={connector},
                route_id=route_id,
            )
            self.assertIn(shared_cell, domains)
            return WFCPath(
                route_id,
                kind,
                source,
                target,
                cells,
                vertical_resolution=building.vertical_grid_resolution,
            )

        with patch.object(building, "_generate_route", side_effect=fake_generate):
            routes = building.generate_routes()

        self.assertEqual(len(routes), 2)
        self.assertEqual(building.occupancy[shared_cell], {next(
            route.route_id for route in routes if route.kind is PortKind.SLIDE
        )})
        reservations = building._connector_reservations_by_cell[shared_cell]
        self.assertEqual(
            {reservation.kind for reservation in reservations.values()},
            {PortKind.SLIDE},
        )
        self.assertEqual(
            {reservation.node for reservation in reservations.values()},
            {shared_node},
        )

    def test_landing_sharing_rejects_mismatched_kind_or_direction(self):
        building = small_building(vertical_resolution=3)
        node = (1, 1, 0)
        connector = building._endpoint_neighbor(
            building.graph_node_to_cell(node),
            PortKind.STAIR,
            True,
        )
        self.assertIsNotNone(connector)
        building._connector_reservations_by_cell[connector] = {
            "stair": ConnectorReservation(
                node,
                PortKind.STAIR,
                Direction.U,
                True,
            ),
            "slide": ConnectorReservation(
                node,
                PortKind.SLIDE,
                Direction.D,
                False,
            ),
        }

        self.assertFalse(
            building._can_share_landing(
                connector,
                "stair",
                {"slide"},
                expected_kind=PortKind.SLIDE,
            )
        )
        self.assertFalse(
            building._can_share_landing(
                connector,
                "stair",
                {"slide"},
                expected_kind=PortKind.STAIR,
            )
        )
        building._connector_reservations_by_cell[connector]["slide"] = (
            ConnectorReservation(
                node,
                PortKind.NONE,
                Direction.U,
                False,
            )
        )
        self.assertFalse(
            building._can_share_landing(
                connector,
                "stair",
                {"slide"},
                expected_kind=PortKind.STAIR,
            )
        )

    def test_landing_sharing_does_not_allow_unrelated_route_overlap(self):
        building = small_building(vertical_resolution=3)
        node = (1, 1, 0)
        start = building.graph_node_to_cell(node)
        connector = building._endpoint_neighbor(start, PortKind.STAIR, True)
        other_node = (2, 2, 0)
        building._connector_reservations_by_cell[connector] = {
            "current": ConnectorReservation(
                node,
                PortKind.STAIR,
                Direction.U,
                True,
            ),
            "unrelated": ConnectorReservation(
                other_node,
                PortKind.STAIR,
                Direction.U,
                True,
            ),
        }
        building._reserved_connector_owners[connector] = {
            "current",
            "unrelated",
        }
        building.occupancy[connector] = {"unrelated"}

        cells = building._route_grid_cells(
            start,
            building.graph_node_to_cell((0, 0, 1)),
            route_connector_cells={connector},
            route_id="current",
        )

        self.assertNotIn(connector, cells)


    def test_resolution_two_shared_landing_seeds_keep_typed_graph_routes_valid(self):
        from src.route_diagnostics import diagnose_routes

        for seed in (6, 15, 19):
            with self.subTest(seed=seed):
                building = WFCBuilding(
                    floors=4,
                    width=10,
                    length=10,
                    difficulty=2,
                    grid_resolution=2,
                    seed=seed,
                    max_route_attempts=30,
                    recovery_strategy="none",
                    minimum_curvature_radius=0.02,
                    endpoint_frontier=True,
                    generate_routes=False,
                )
                try:
                    routes = building.generate_routes()
                except RouteGenerationError as error:
                    self.assertTrue(error.failure_classification)
                    self.assertTrue(building.route_failures)
                    continue

                self.assertGreater(len(building.routes), 0)
                route_edges = {
                    (source, target)
                    for source, target, attributes
                    in building.graph.edges(data=True)
                    if _edge_name(attributes) in {"stair", "slide"}
                }
                routed_edges = {
                    (route.source, route.target) for route in building.routes
                }
                if seed == 6:
                    self.assertEqual(route_edges, routed_edges)
                else:
                    self.assertEqual(route_edges, routed_edges)

                overlapping_cells = {
                    cell: owners
                    for cell, owners in building.occupancy.items()
                    if len(owners) > 1
                }
                for cell, owners in overlapping_cells.items():
                    owner_routes = {
                        route.route_id: route
                        for route in building.routes
                        if route.route_id in owners
                    }
                    common_nodes = {
                        node
                        for route in owner_routes.values()
                        for node in (route.source, route.target)
                    }
                    shared_graph_endpoint = any(
                        cell == building.graph_node_to_cell(node)
                        and all(
                            node in (route.source, route.target)
                            for route in owner_routes.values()
                        )
                        for node in common_nodes
                    )
                    shared_connector = all(
                        building._can_share_landing(
                            cell,
                            route_id,
                            owners - {route_id},
                        )
                        for route_id in owners
                    )
                    self.assertTrue(shared_graph_endpoint or shared_connector)

                for route in building.routes:
                    floor_deltas = [
                        next_cell[0] - cell[0]
                        for cell, next_cell in zip(
                            route.cells,
                            route.cells[1:],
                        )
                    ]
                    self.assertTrue(
                        all(delta >= 0 for delta in floor_deltas)
                        if route.kind is PortKind.STAIR
                        else all(delta <= 0 for delta in floor_deltas)
                    )
                self.assertEqual(diagnose_routes(building).logical_errors, ())

    def test_route_tiles_include_empty_connectors_straights_and_turns(self):
        catalogue = build_route_tiles(PortKind.SLIDE)
        names = {tile.name for tile in catalogue.tiles}

        self.assertIn("empty", names)
        self.assertIn("slide_connector_D", names)
        self.assertIn("slide_straight_U_D", names)
        self.assertIn("slide_turn_N_E", names)
        self.assertEqual(catalogue.tiles[catalogue.empty_index].ports[Direction.U], PortKind.NONE)

    def test_route_failure_classifier_distinguishes_solver_and_reachability_failures(self):
        self.assertEqual(
            _classify_route_failure(
                "No solution found after 1 attempts; last contradiction: no tiles"
            ),
            "tile_contradiction",
        )
        self.assertEqual(
            _classify_route_failure("No tile-supported slide path from start to end"),
            "unreachable_endpoints",
        )
        self.assertEqual(
            _classify_route_failure("Smoothed slide has curvature radius 0.01"),
            "geometric_rejection",
        )

    def test_adjacent_aligned_slide_rejects_vertical_only_route(self):
        building = small_building(resolution=2)

        with self.assertRaisesRegex(RouteGenerationError, "horizontal travel"):
            building._generate_route(
                "slide-0",
                (1, 1, 1),
                (1, 1, 0),
                "slide",
            )

    def test_aligned_slide_uses_added_vertical_samples_for_horizontal_travel(self):
        building = small_building(resolution=2, vertical_resolution=3)
        route = building._generate_route(
            "aligned-slide",
            (1, 1, 1),
            (1, 1, 0),
            "slide",
        )

        self.assertEqual(route.cells[0], building.graph_node_to_cell(route.source))
        self.assertEqual(route.cells[-1], building.graph_node_to_cell(route.target))
        self.assertGreater(
            sum(
                first[1:] != second[1:]
                for first, second in zip(route.cells, route.cells[1:])
            ),
            0,
        )
        self.assertEqual(route.vertical_resolution, 3)

    def test_default_vertical_resolution_leaves_room_for_aligned_routes(self):
        building = WFCBuilding(
            floors=4,
            width=10,
            length=10,
            difficulty=2,
            grid_resolution=2,
            seed=0,
            generate_routes=False,
        )

        self.assertEqual(building.vertical_grid_resolution, 3)
        self.assertEqual(
            building.grid_shape[0],
            (building.floors - 1) * building.vertical_grid_resolution + 1,
        )

    def test_failed_route_reopens_and_regenerates_one_previous_route(self):
        building = small_building(
            source=(0, 0, 1),
            target=(0, 0, 0),
        )
        previous_edge = ((0, 0, 1), (0, 0, 0), "slide")
        failing_edge = ((2, 2, 1), (2, 2, 0), "slide")
        building.graph.clear()
        building.graph.add_edge(previous_edge[0], previous_edge[1], kind="slide")
        building.graph.add_edge(failing_edge[0], failing_edge[1], kind="slide")
        building.graph.add_edge((1, 1, 0), (1, 1, 1), kind="stair")
        calls = {}

        def fake_generate(
            route_id,
            source,
            target,
            edge_name,
            seed_offset=0,
            recovery_count=0,
        ):
            calls[route_id] = calls.get(route_id, 0) + 1
            if source == failing_edge[0] and calls[route_id] == 1:
                raise RouteGenerationError("occupied route", attempts=1)
            kind = PortKind(edge_name)
            return WFCPath(
                route_id,
                kind,
                source,
                target,
                (
                    building.graph_node_to_cell(source),
                    building.graph_node_to_cell(target),
                ),
                vertical_resolution=building.vertical_grid_resolution,
                recovery_count=recovery_count,
            )

        with patch.object(building, "_generate_route", side_effect=fake_generate):
            routes = building.generate_routes()

        self.assertEqual(len(routes), 3)
        self.assertEqual(
            calls,
            {
                "0:(0, 0, 1)->(0, 0, 0):slide": 2,
                "1:(2, 2, 1)->(2, 2, 0):slide": 2,
            },
        )
        self.assertEqual(building.recovery_attempts, 1)
        self.assertEqual(building.routes_retried, 1)
        self.assertEqual(routes[0].recovery_count, 1)
        self.assertEqual(routes[1].recovery_count, 1)
        self.assertTrue(routes[2].direct_stair)
        self.assertEqual(routes[2].recovery_count, 0)
        self.assertEqual(
            set(building.occupancy),
            {cell for route in routes for cell in route.cells},
        )

    def test_recovery_can_reopen_two_previous_routes(self):
        building = small_building()
        edges = (
            ((0, 0, 1), (0, 0, 0), "slide"),
            ((2, 2, 1), (2, 2, 0), "slide"),
            ((2, 0, 1), (2, 0, 0), "slide"),
        )
        building.graph.clear()
        for source, target, kind in edges:
            building.graph.add_edge(source, target, kind=kind)
        calls = {}
        unaffected_route = None

        def fake_generate(
            route_id,
            source,
            target,
            edge_name,
            seed_offset=0,
            recovery_count=0,
        ):
            calls[route_id] = calls.get(route_id, 0) + 1
            if route_id.startswith("2:") and calls[route_id] == 1:
                raise RouteGenerationError("later route conflict", attempts=1)
            if route_id.startswith("1:") and seed_offset == 80:
                raise RouteGenerationError("one-route recovery conflict", attempts=1)
            return WFCPath(
                route_id,
                PortKind(edge_name),
                source,
                target,
                (
                    building.graph_node_to_cell(source),
                    building.graph_node_to_cell(target),
                ),
                vertical_resolution=building.vertical_grid_resolution,
                recovery_count=recovery_count,
            )

        with patch.object(building, "_generate_route", side_effect=fake_generate):
            routes = building.generate_routes()

        self.assertEqual(len(routes), 3)
        self.assertEqual(building.recovery_attempts, 2)
        self.assertEqual(building.routes_retried, 3)
        self.assertEqual(calls["0:(0, 0, 1)->(0, 0, 0):slide"], 2)
        self.assertEqual(calls["1:(2, 2, 1)->(2, 2, 0):slide"], 3)
        self.assertEqual(calls["2:(2, 0, 1)->(2, 0, 0):slide"], 2)
        self.assertTrue(all(route.recovery_count == 1 for route in routes))

    def test_conflict_candidates_restore_reachability_by_freeing_owned_cells(self):
        source = (0, 0, 1)
        target = (0, 0, 0)
        building = small_building(
            source=source,
            target=target,
            vertical_resolution=3,
            recovery_strategy="conflict-aware",
        )
        building.floor_grid = frozenset((source, target))
        blocked_cells = tuple(
            (1, x, y) for x in range(3) for y in range(3)
        )
        blocker = WFCPath(
            "blocker",
            PortKind.SLIDE,
            source,
            target,
            blocked_cells,
            vertical_resolution=3,
        )
        building.routes = [blocker]
        building.occupancy = {cell: {"blocker"} for cell in blocked_cells}

        candidates = building._blocking_route_sets(
            building.graph_node_to_cell(source),
            building.graph_node_to_cell(target),
            set(),
            "failed-route",
        )

        self.assertEqual(candidates, (("blocker",),))

    def test_conflict_aware_rerouting_is_deterministic_and_preserves_unaffected_route(
        self,
    ):
        def run_once():
            building = small_building(recovery_strategy="conflict-aware")
            edges = (
                ((0, 0, 1), (0, 0, 0), "slide"),
                ((2, 2, 1), (2, 2, 0), "slide"),
                ((1, 1, 1), (1, 1, 0), "slide"),
            )
            building.graph.clear()
            for source, target, kind in edges:
                building.graph.add_edge(source, target, kind=kind)
            calls = {}

            def fake_generate(
                route_id,
                source,
                target,
                edge_name,
                seed_offset=0,
                recovery_count=0,
            ):
                calls[route_id] = calls.get(route_id, 0) + 1
                if route_id.startswith("2:") and calls[route_id] == 1:
                    raise RouteGenerationError("occupied route", attempts=1)
                path = WFCPath(
                    route_id,
                    PortKind(edge_name),
                    source,
                    target,
                    (
                        building.graph_node_to_cell(source),
                        building.graph_node_to_cell(target),
                    ),
                    vertical_resolution=building.vertical_grid_resolution,
                    recovery_count=recovery_count,
                )
                if route_id.startswith("1:") and recovery_count == 0:
                    nonlocal unaffected_route
                    unaffected_route = path
                return path

            with patch.object(
                building,
                "_blocking_route_sets",
                return_value=(("0:(0, 0, 1)->(0, 0, 0):slide",),),
            ), patch.object(
                building,
                "_generate_route",
                side_effect=fake_generate,
            ):
                routes = building.generate_routes()
            return building, unaffected_route, routes, calls

        first, unaffected_route, routes, calls = run_once()
        second, _, repeated_routes, _ = run_once()

        self.assertEqual(
            tuple(route.cells for route in routes),
            tuple(route.cells for route in repeated_routes),
        )
        self.assertEqual(first.recovery_events, second.recovery_events)
        self.assertEqual(routes[1].route_id, unaffected_route.route_id)
        self.assertEqual(calls[unaffected_route.route_id], 1)
        self.assertEqual(routes[0].recovery_count, 1)
        self.assertEqual(routes[2].recovery_count, 1)
        self.assertEqual(first.recovery_attempts, 1)
        self.assertEqual(first.routes_retried, 1)
        self.assertEqual(
            sum(len(owners) > 1 for owners in first.occupancy.values()),
            0,
        )
        self.assertEqual(first.route_failures[0]["recovery_succeeded"], True)
        self.assertEqual(
            first.route_failures[0]["routes_selected_for_reopening"],
            ("0:(0, 0, 1)->(0, 0, 0):slide",),
        )

    def test_conflict_aware_recovery_fails_and_restores_occupancy_when_unfeasible(self):
        building = small_building(recovery_strategy="conflict-aware")
        edges = (
            ((0, 0, 1), (0, 0, 0), "slide"),
            ((2, 2, 1), (2, 2, 0), "slide"),
            ((1, 1, 1), (1, 1, 0), "slide"),
        )
        building.graph.clear()
        for source, target, kind in edges:
            building.graph.add_edge(source, target, kind=kind)

        def fake_generate(
            route_id,
            source,
            target,
            edge_name,
            seed_offset=0,
            recovery_count=0,
        ):
            if route_id.startswith("2:"):
                raise RouteGenerationError("occupied route", attempts=1)
            return WFCPath(
                route_id,
                PortKind(edge_name),
                source,
                target,
                (
                    building.graph_node_to_cell(source),
                    building.graph_node_to_cell(target),
                ),
                vertical_resolution=building.vertical_grid_resolution,
                recovery_count=recovery_count,
            )

        with patch.object(building, "_blocking_route_sets", return_value=()), patch.object(
            building, "_generate_route", side_effect=fake_generate
        ):
            with self.assertRaisesRegex(RouteGenerationError, "occupied route"):
                building.generate_routes()

        self.assertEqual(len(building.routes), 2)
        self.assertEqual(
            building.occupancy,
            {
                cell: {route.route_id}
                for route in building.routes
                for cell in route.cells
            },
        )
        self.assertFalse(building.route_failures[0]["recovery_succeeded"])

    def test_adjacent_floor_stair_ascends_in_graph_edge_direction(self):
        source = (1, 1, 0)
        target = (1, 1, 1)
        building = small_building(source, target, "stair")

        route = building._generate_route(
            "stair-0",
            source,
            target,
            "stair",
        )

        self.assertEqual(route.cells[0], (0, 1, 1))
        self.assertEqual(route.cells[-1], (1, 1, 1))
        self.assertTrue(all(
            route.cells[index + 1][0] >= route.cells[index][0]
            for index in range(len(route.cells) - 1)
        ))

    def test_forbidden_cells_are_not_used_by_route(self):
        building = small_building()
        building.forbidden_cells.add((1, 1, 1))

        with self.assertRaisesRegex(RouteGenerationError, "forbidden endpoint"):
            building._generate_route(
                "slide-0",
                (1, 1, 1),
                (1, 1, 0),
                "slide",
            )

    def test_wrong_graph_edge_direction_is_reported(self):
        building = small_building((1, 1, 0), (1, 1, 1), "slide")

        with self.assertRaisesRegex(RouteGenerationError, "must descend"):
            building._generate_route(
                "slide-0",
                (1, 1, 0),
                (1, 1, 1),
                "slide",
            )

    def test_graph_coordinates_map_to_floor_x_y_wfc_order(self):
        building = small_building(resolution=2)

        self.assertEqual(
            building.graph_node_to_cell((2, 1, 1)),
            (2, 4, 2),
        )

    def test_slide_smoothing_preserves_endpoints_and_strict_descent(self):
        building = small_building(
            source=(0, 0, 1),
            target=(2, 2, 0),
            resolution=2,
        )
        route = building._generate_route(
            "slide-curve",
            (0, 0, 1),
            (2, 2, 0),
            "slide",
        )

        smoothed = building.smooth_path(route)

        self.assertEqual(smoothed[0], route.world_points(2)[0])
        self.assertEqual(smoothed[-1], route.world_points(2)[-1])
        self.assertTrue(
            all(
                smoothed[index + 1][2] < smoothed[index][2]
                for index in range(len(smoothed) - 1)
            )
        )

        original = route.world_points(2)
        first_horizontal = next(
            point
            for point in original[1:]
            if point[:2] != original[0][:2]
        )
        smoothed_first = next(
            point
            for point in smoothed[1:]
            if point[:2] != smoothed[0][:2]
        )
        last_horizontal = next(
            point
            for point in reversed(original[:-1])
            if point[:2] != original[-1][:2]
        )
        smoothed_last = next(
            point
            for point in reversed(smoothed[:-1])
            if point[:2] != smoothed[-1][:2]
        )
        self.assertAlmostEqual(
            (first_horizontal[0] - original[0][0])
            * (smoothed_first[0] - smoothed[0][0])
            + (first_horizontal[1] - original[0][1])
            * (smoothed_first[1] - smoothed[0][1]),
            (
                (first_horizontal[0] - original[0][0]) ** 2
                + (first_horizontal[1] - original[0][1]) ** 2
            )
            ** 0.5
            * (
                (smoothed_first[0] - smoothed[0][0]) ** 2
                + (smoothed_first[1] - smoothed[0][1]) ** 2
            )
            ** 0.5,
        )
        self.assertAlmostEqual(
            (original[-1][0] - last_horizontal[0])
            * (smoothed[-1][0] - smoothed_last[0])
            + (original[-1][1] - last_horizontal[1])
            * (smoothed[-1][1] - smoothed_last[1]),
            (
                (original[-1][0] - last_horizontal[0]) ** 2
                + (original[-1][1] - last_horizontal[1]) ** 2
            )
            ** 0.5
            * (
                (smoothed[-1][0] - smoothed_last[0]) ** 2
                + (smoothed[-1][1] - smoothed_last[1]) ** 2
            )
            ** 0.5,
        )

    def test_slide_smoothing_retries_local_strengths_before_rejecting(self):
        building = small_building(
            resolution=2,
            vertical_resolution=3,
            minimum_curvature_radius=0.13,
        )
        building.floors = 5
        building.length = building.width = 5
        building.grid_shape = (13, 9, 9)
        xy_cells = (
            (0, 0),
            (0, 3),
            (1, 3),
            (1, 5),
            (3, 5),
            (3, 7),
            (5, 7),
            (5, 8),
            (8, 8),
        )
        route = WFCPath(
            "local-smoothing",
            PortKind.SLIDE,
            (0, 0, 4),
            (4, 4, 0),
            tuple(
                (12 if index < len(xy_cells) - 1 else 0, x, y)
                for index, (x, y) in enumerate(xy_cells)
            ),
            vertical_resolution=3,
        )
        with self.assertRaisesRegex(RouteGenerationError, "curvature radius"):
            building.smooth_path(route, smoothing_strength=1.0)

        recovered = building._smooth_route_candidate(route)

        self.assertEqual(recovered.cells, route.cells)
        self.assertEqual(recovered.smoothing_strength, 0.75)
        smoothed = building.smooth_path(recovered)
        self.assertEqual(recovered.accepted_geometry, smoothed)
        self.assertEqual(smoothed[0], route.world_points(2, 3)[0])
        self.assertEqual(smoothed[-1], route.world_points(2, 3)[-1])

    def test_slide_with_no_valid_local_smoothing_variant_is_rejected(self):
        building = small_building()
        route = WFCPath(
            "unrecoverable-smoothing",
            PortKind.SLIDE,
            (1, 1, 1),
            (1, 1, 0),
            ((2, 2, 2), (1, 2, 2), (0, 2, 2)),
        )
        strengths = []

        def reject(candidate):
            strengths.append(candidate.smoothing_strength)
            raise RouteGenerationError(
                f"invalid at strength {candidate.smoothing_strength}"
            )

        with patch.object(building, "smooth_path", side_effect=reject):
            with self.assertRaisesRegex(RouteGenerationError, "strength 0.5"):
                building._smooth_route_candidate(route)

        self.assertEqual(strengths, [1.0, 0.75, 0.5])

    def test_smoothed_route_must_clear_unrelated_graph_route_endpoints(self):
        building = small_building(
            source=(0, 0, 0),
            target=(2, 2, 1),
            kind="stair",
            resolution=1,
            vertical_resolution=2,
        )
        building.graph.add_edge((1, 1, 0), (1, 1, 1), kind="slide")
        route = WFCPath(
            "graph-endpoint-clearance",
            PortKind.STAIR,
            (0, 0, 0),
            (2, 2, 1),
            ((0, 0, 0), (0, 1, 1), (2, 2, 2)),
            vertical_resolution=2,
        )

        with self.assertRaisesRegex(
            RouteGenerationError,
            "collides near route-grid cell",
        ):
            building._validate_smoothed_path(
                route,
                ((0.0, 0.0, 0.0), (1.0, 1.0, 0.0), (2.0, 2.0, 1.0)),
            )

    def test_smoothed_route_must_clear_unrelated_reserved_connector_segments(self):
        building = small_building(
            source=(0, 0, 0),
            target=(2, 2, 1),
            kind="stair",
            resolution=1,
            vertical_resolution=2,
        )
        node = (1, 1, 1)
        connector = (1, 1, 1)
        building._reserved_connector_owners[connector] = {"future-route"}
        building._connector_reservations_by_cell[connector] = {
            "future-route": ConnectorReservation(
                node,
                PortKind.SLIDE,
                Direction.D,
                True,
            )
        }
        route = WFCPath(
            "reserved-connector-clearance",
            PortKind.STAIR,
            (0, 0, 0),
            (2, 2, 1),
            ((0, 0, 0), (1, 1, 1), (2, 2, 2)),
            vertical_resolution=2,
        )

        with self.assertRaisesRegex(
            RouteGenerationError,
            "approaches reserved connector",
        ) as context:
            building._validate_reserved_connector_clearance(
                route,
                ((1.0, 1.0, 0.25), (1.0, 1.0, 1.25)),
            )
        self.assertLess(
            context.exception.failure_details["minimum_clearance"],
            context.exception.failure_details["collision_tolerance"],
        )
        self.assertEqual(
            context.exception.failure_details["blocking_connector_segment"],
            ((1.0, 1.0, 1.0), (1.0, 1.0, 0.5)),
        )

    def test_smoothed_routes_may_not_intersect_away_from_shared_endpoints(self):
        building = small_building(
            source=(0, 0, 0),
            target=(2, 2, 1),
            kind="stair",
            resolution=1,
            vertical_resolution=2,
        )
        other = WFCPath(
            "unrelated-crossing",
            PortKind.STAIR,
            (0, 2, 0),
            (2, 0, 1),
            ((0, 0, 2), (1, 1, 1), (2, 2, 0)),
            vertical_resolution=2,
        )
        candidate = WFCPath(
            "crossing-candidate",
            PortKind.STAIR,
            (0, 0, 0),
            (2, 2, 1),
            ((0, 0, 0), (1, 1, 1), (2, 2, 2)),
            vertical_resolution=2,
        )
        building.routes.append(other)

        with self.assertRaisesRegex(
            RouteGenerationError,
            "intersects route unrelated-crossing",
        ):
            building._validate_route_intersections(
                candidate,
                candidate.world_points(1, 2),
            )

    def test_vertically_aligned_slide_without_horizontal_travel_is_rejected(self):
        building = small_building(resolution=2)
        route = WFCPath(
            "vertical-slide",
            PortKind.SLIDE,
            (1, 1, 1),
            (1, 1, 0),
            ((2, 2, 2), (1, 2, 2), (0, 2, 2)),
        )

        with self.assertRaisesRegex(RouteGenerationError, "horizontal travel"):
            building.smooth_path(route)

    def test_vertically_aligned_endpoints_can_use_feasible_horizontal_detour(self):
        building = small_building(
            source=(1, 1, 1),
            target=(1, 1, 0),
            resolution=3,
        )
        route = WFCPath(
            "aligned-detour",
            PortKind.SLIDE,
            (1, 1, 1),
            (1, 1, 0),
            (
                (3, 3, 3),
                (2, 3, 3),
                (2, 4, 3),
                (2, 5, 3),
                (1, 5, 3),
                (1, 4, 3),
                (1, 3, 3),
                (0, 3, 3),
            ),
        )

        smooth = building.smooth_path(route)

        self.assertEqual(smooth[0], route.world_points(3)[0])
        self.assertEqual(smooth[-1], route.world_points(3)[-1])
        self.assertGreater(
            sum(
                hypot(second[0] - first[0], second[1] - first[1])
                for first, second in zip(smooth, smooth[1:])
            ),
            building.minimum_slide_horizontal_travel,
        )
        self.assertTrue(
            all(
                second[2] < first[2]
                for first, second in zip(smooth, smooth[1:])
            )
        )

    def test_wfc_failure_reports_number_of_attempts(self):
        building = small_building(resolution=2)

        with self.assertRaises(RouteGenerationError) as context:
            building._generate_route(
                "slide-0",
                (1, 1, 1),
                (1, 1, 0),
                "slide",
            )

        self.assertEqual(context.exception.attempts, building.max_route_attempts)
        details = context.exception.failure_details
        self.assertEqual(details["route_id"], "slide-0")
        self.assertEqual(details["route_type"], "slide")
        self.assertEqual(details["grid_dimensions"], building.grid_shape)
        self.assertEqual(details["feasibility_stage"], "smoothed_geometry")
        self.assertGreater(details["available_cell_count"], 0)
        endpoint_constraints = details["endpoint_constraints"]
        self.assertIn("start_connectors", endpoint_constraints)
        self.assertIn("end_connectors", endpoint_constraints)

    def test_slide_exceeding_maximum_grade_is_rejected(self):
        building = small_building(
            source=(0, 0, 2),
            target=(1, 0, 0),
            resolution=2,
        )
        route = WFCPath(
            "steep-slide",
            PortKind.SLIDE,
            (0, 0, 2),
            (1, 0, 0),
            (
                (4, 0, 0),
                (3, 0, 0),
                (2, 0, 0),
                (1, 0, 0),
                (1, 1, 0),
                (0, 2, 0),
            ),
        )

        with self.assertRaisesRegex(RouteGenerationError, "horizontal travel"):
            building.smooth_path(route)

    def test_feasible_curved_slide_has_bounded_arc_length_grade(self):
        building = small_building(
            source=(0, 0, 1),
            target=(2, 2, 0),
            resolution=2,
        )
        route = WFCPath(
            "curved-slide",
            PortKind.SLIDE,
            (0, 0, 1),
            (2, 2, 0),
            (
                (2, 0, 0),
                (1, 0, 0),
                (1, 1, 0),
                (1, 2, 0),
                (1, 3, 0),
                (1, 4, 0),
                (1, 4, 1),
                (1, 4, 2),
                (1, 4, 3),
                (1, 4, 4),
                (0, 4, 4),
            ),
        )

        smooth = building.smooth_path(route)

        self.assertEqual(smooth[0], route.world_points(2)[0])
        self.assertEqual(smooth[-1], route.world_points(2)[-1])
        self.assertTrue(
            all(
                (second[2] - first[2]) < 0
                for first, second in zip(smooth, smooth[1:])
            )
        )
        self.assertTrue(
            all(
                abs(second[2] - first[2])
                <= building.maximum_slide_grade
                * (
                    (second[0] - first[0]) ** 2
                    + (second[1] - first[1]) ** 2
                )
                ** 0.5
                + 1e-9
                for first, second in zip(smooth, smooth[1:])
            )
        )

    def test_slide_turning_radius_is_enforced(self):
        building = small_building(
            source=(0, 0, 1),
            target=(2, 2, 0),
            resolution=2,
        )
        building.minimum_turning_radius = 10.0
        route = WFCPath(
            "tight-turn",
            PortKind.SLIDE,
            (0, 0, 1),
            (2, 2, 0),
            (
                (2, 0, 0),
                (1, 0, 0),
                (1, 1, 0),
                (1, 2, 0),
                (1, 3, 0),
                (1, 4, 0),
                (1, 4, 1),
                (1, 4, 2),
                (1, 4, 3),
                (1, 4, 4),
                (0, 4, 4),
            ),
        )

        with self.assertRaisesRegex(RouteGenerationError, "minimum is"):
            building.smooth_path(route)

    def test_slide_minimum_curvature_radius_rejects_tight_curve_and_accepts_feasible_curve(
        self,
    ):
        source = (1, 1, 1)
        target = (1, 1, 0)
        route = WFCPath(
            "curvature-test",
            PortKind.SLIDE,
            source,
            target,
            (
                (1, 1, 1),
                (0, 1, 1),
            ),
        )

        def descending_circle(radius):
            return tuple(
                (
                    1 + radius * cos(index * 2 * pi / 32),
                    1 + radius * sin(index * 2 * pi / 32),
                    1 - index / 32,
                )
                for index in range(33)
            )

        tight = small_building(minimum_curvature_radius=0.6)
        with self.assertRaisesRegex(RouteGenerationError, "curvature radius"):
            tight._validate_smoothed_path(route, descending_circle(0.5))

        feasible = small_building(minimum_curvature_radius=0.4)
        feasible._validate_smoothed_path(route, descending_circle(0.5))

    def test_inherited_graph_routes_are_generated_without_collisions(self):
        building = WFCBuilding(
            floors=4,
            width=10,
            length=10,
            difficulty=2,
            grid_resolution=3,
            seed=10,
        )
        expected_edges = {
            (source, target, attributes.get("kind", attributes.get("type")))
            for source, target, attributes in building.graph.edges(data=True)
            if attributes.get("kind", attributes.get("type")) in ("stair", "slide")
        }
        generated_edges = {
            (route.source, route.target, route.kind.value)
            for route in building.routes
        }

        self.assertEqual(generated_edges, expected_edges)
        for route in building.routes:
            points = building.smooth_path(route)
            if route.kind is PortKind.SLIDE:
                self.assertGreaterEqual(route.attempts, 1)
                self.assertLessEqual(route.attempts, building.max_route_attempts)
                self.assertIn(route.route_id, building.occupancy[route.cells[0]])
                horizontal_travel = sum(
                    hypot(second[0] - first[0], second[1] - first[1])
                    for first, second in zip(points, points[1:])
                )
                height_drop = points[0][2] - points[-1][2]
                self.assertGreaterEqual(
                    horizontal_travel,
                    max(
                        building.minimum_slide_horizontal_travel,
                        height_drop / building.maximum_slide_grade,
                    )
                    - 1e-9,
                )
                self.assertTrue(
                    all(
                        points[index + 1][2] < points[index][2]
                        for index in range(len(points) - 1)
                    )
                )
                self.assertTrue(
                    all(
                        abs(second[2] - first[2])
                        <= building.maximum_slide_grade
                        * hypot(second[0] - first[0], second[1] - first[1])
                        + 1e-9
                        for first, second in zip(points, points[1:])
                    )
                )
            else:
                self.assertEqual(route.attempts, 0)
                self.assertEqual(points, (route.source, route.target))

        for first_index, first in enumerate(building.routes):
            for second in building.routes[first_index + 1 :]:
                intersections = set(first.cells) & set(second.cells)
                shared_nodes = {first.source, first.target} & {
                    second.source,
                    second.target,
                }
                allowed_shared_cells = {
                    building.graph_node_to_cell(node)
                    for node in shared_nodes
                }
                self.assertLessEqual(intersections, allowed_shared_cells)


if __name__ == "__main__":
    unittest.main()
