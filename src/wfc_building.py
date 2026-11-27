from __future__ import annotations

from collections import Counter, deque
from collections.abc import Callable, Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, replace
from itertools import combinations
from math import ceil, hypot, isfinite
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from src.graph import Building
from src.hybrid_wfc import HybridBudgetExhausted
from src.wfc_solver import solve_wfc
from src.wfc_tiles import (
    Cell,
    ContradictionError,
    Direction,
    PortKind,
    Tile,
    _DIRECTION_OFFSETS,
)

if TYPE_CHECKING:
    from src.route_diagnostics import BuildingRouteDiagnostics


GraphNode = tuple[int, int, int]
_HORIZONTAL_DIRECTIONS = (
    Direction.N,
    Direction.E,
    Direction.S,
    Direction.W,
)
_EDGE_PORTS = {
    "stair": PortKind.STAIR,
    "slide": PortKind.SLIDE,
}
_EDGE_COLORS = {
    "stair": "#e58a22",
    "slide": "#277da1",
}
_SLIDE_SMOOTHING_STRENGTHS = (1.0, 0.75, 0.5)


class RouteGenerationError(RuntimeError):
    """Raised when a graph-defined inter-floor edge cannot be routed."""

    def __init__(
        self,
        message: str,
        attempts: int = 0,
        failure_classification: str = "unknown",
        failure_details: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.failure_classification = failure_classification
        self.failure_details = dict(failure_details or {})


@dataclass(frozen=True)
class RouteTileSet:
    tiles: tuple[Tile, ...]
    empty_index: int
    connector_indices: Mapping[Direction, int]
    segment_indices: tuple[int, ...]
    vertical_straight_index: int


@dataclass(frozen=True)
class ConnectorReservation:
    node: GraphNode
    kind: PortKind
    direction: Direction
    is_start: bool


@dataclass(frozen=True)
class WFCPath:
    route_id: str
    kind: PortKind
    source: GraphNode
    target: GraphNode
    cells: tuple[Cell, ...]
    attempts: int = 1
    vertical_resolution: int | None = None
    recovery_count: int = 0
    smoothing_strength: float = 1.0
    height_profile_anchor: tuple[float, float] | None = None
    accepted_geometry: tuple[tuple[float, float, float], ...] | None = None
    direct_stair: bool = False
    rejected_route_signatures: int = 0
    retry_exclusion: Cell | None = None

    def world_points(
        self,
        resolution: int,
        vertical_resolution: int | None = None,
    ) -> tuple[tuple[float, float, float], ...]:
        """Return points as (x, y, floor), scaled back to graph coordinates."""
        vertical_scale = vertical_resolution or self.vertical_resolution or resolution
        return tuple(
            (x / resolution, y / resolution, floor / vertical_scale)
            for floor, x, y in self.cells
        )


def build_route_tiles(kind: PortKind) -> RouteTileSet:
    """Build empty, connector, straight, and turning patterns for one route kind."""
    if kind not in (PortKind.STAIR, PortKind.SLIDE):
        raise ValueError("Inter-floor routes must be stairs or slides")

    tiles = [
        Tile(
            "empty",
            {direction: PortKind.NONE for direction in Direction},
        )
    ]
    empty_index = 0
    connector_indices = {}
    for direction in Direction:
        ports = {item: PortKind.NONE for item in Direction}
        ports[direction] = kind
        connector_indices[direction] = len(tiles)
        tiles.append(Tile(f"{kind.value}_connector_{direction.value}", ports))

    segment_indices = []
    vertical_straight_index = -1
    for first, second in combinations(Direction, 2):
        ports = {direction: PortKind.NONE for direction in Direction}
        ports[first] = kind
        ports[second] = kind
        segment_indices.append(len(tiles))
        category = _segment_category(first, second, kind)
        if {first, second} == {Direction.U, Direction.D}:
            vertical_straight_index = len(tiles)
        tiles.append(
            Tile(
                f"{kind.value}_{category}_{first.value}_{second.value}",
                ports,
            )
        )

    return RouteTileSet(
        tuple(tiles),
        empty_index,
        connector_indices,
        tuple(segment_indices),
        vertical_straight_index,
    )


def _segment_category(
    first: Direction,
    second: Direction,
    kind: PortKind,
) -> str:
    if (
        first in _HORIZONTAL_DIRECTIONS
        and second in _HORIZONTAL_DIRECTIONS
    ):
        return "straight" if _opposite(first) is second else "turn"
    if _opposite(first) is second:
        return "straight"
    return "descending" if kind is PortKind.SLIDE else "ascending"


class WFCBuilding(Building):
    """Reserve graph-defined stairs and route slides through a shared WFC grid.

    The inherited floor graph remains untouched. WFC cells use
    ``(floor_step, x_step, y_step)``. Horizontal resolution is
    configurable; the default vertical resolution is at least three cells per
    floor interval so a route can turn while descending between aligned nodes.
    """

    def __init__(
        self,
        floors: int,
        width: int,
        length: int,
        difficulty: int = 3,
        grid_resolution: int = 2,
        seed: int = 0,
        max_route_attempts: int = 30,
        forbidden_cells: Iterable[Cell] = (),
        generate_routes: bool = True,
        minimum_slide_horizontal_travel: float = 0.5,
        maximum_slide_grade: float = 1.5,
        minimum_turning_radius: float = 0.02,
        vertical_grid_resolution: int | None = None,
        max_route_backtrack_depth: int = 2,
        max_route_recovery_attempts: int = 1,
        recovery_strategy: str = "recency",
        minimum_curvature_radius: float = 0.02,
        tile_aware_connectivity: bool = False,
        endpoint_frontier: bool = True,
        candidate_lookahead: bool = False,
        routing_strategy: str = "dfs_guided",
        routing_seed: int | None = None,
        trace_callback: Callable[[Mapping[str, object]], None] | None = None,
        trace_route_id: str | None = None,
        trace_wfc: bool = False,
        trace_states: bool = False,
        only_route_id: str | None = None,
        hybrid_weighting: str = "uniform",
        hybrid_compactness_strength: float = 0.12,
    ) -> None:
        if not isinstance(grid_resolution, int) or grid_resolution < 1:
            raise ValueError("grid_resolution must be a positive integer")
        if vertical_grid_resolution is not None and (
            not isinstance(vertical_grid_resolution, int)
            or vertical_grid_resolution < 1
        ):
            raise ValueError("vertical_grid_resolution must be a positive integer")
        if not isinstance(max_route_attempts, int) or max_route_attempts < 1:
            raise ValueError("max_route_attempts must be a positive integer")
        if (
            not isinstance(max_route_backtrack_depth, int)
            or max_route_backtrack_depth < 0
        ):
            raise ValueError("max_route_backtrack_depth must be a non-negative integer")
        if (
            not isinstance(max_route_recovery_attempts, int)
            or max_route_recovery_attempts < 0
        ):
            raise ValueError(
                "max_route_recovery_attempts must be a non-negative integer"
            )
        if recovery_strategy not in ("none", "recency", "conflict-aware"):
            raise ValueError(
                "recovery_strategy must be 'none', 'recency', or 'conflict-aware'"
            )
        if not all(
            isinstance(option, bool)
            for option in (
                tile_aware_connectivity,
                endpoint_frontier,
                candidate_lookahead,
                trace_wfc,
                trace_states,
            )
        ):
            raise ValueError("WFC heuristic options must be booleans")
        if routing_strategy not in ("dfs_guided", "hybrid"):
            raise ValueError("routing_strategy must be 'dfs_guided' or 'hybrid'")
        if hybrid_weighting not in ("uniform", "compact"):
            raise ValueError("hybrid_weighting must be 'uniform' or 'compact'")
        if not isfinite(hybrid_compactness_strength) or hybrid_compactness_strength < 0:
            raise ValueError("hybrid_compactness_strength must be finite and non-negative")
        if (
            not isfinite(minimum_slide_horizontal_travel)
            or minimum_slide_horizontal_travel <= 0
        ):
            raise ValueError("minimum_slide_horizontal_travel must be positive")
        if not isfinite(maximum_slide_grade) or maximum_slide_grade <= 0:
            raise ValueError("maximum_slide_grade must be positive")
        if not isfinite(minimum_turning_radius) or minimum_turning_radius < 0:
            raise ValueError("minimum_turning_radius cannot be negative")
        if not isfinite(minimum_curvature_radius) or minimum_curvature_radius < 0:
            raise ValueError("minimum_curvature_radius cannot be negative")

        self._suppress_graph_plots = True
        random_state = np.random.get_state()
        np.random.seed(seed)
        try:
            super().__init__(difficulty)
        finally:
            np.random.set_state(random_state)
        self._suppress_graph_plots = False
        if (floors, width, length) != (self.floors, self.width, self.length):
            raise ValueError(
                "Requested WFC dimensions do not match the upstream graph dimensions"
            )

        self.grid_resolution = grid_resolution
        self.vertical_grid_resolution = (
            vertical_grid_resolution
            if vertical_grid_resolution is not None
            else max(grid_resolution, 3)
        )
        self.seed = seed
        self.routing_seed = seed if routing_seed is None else routing_seed
        self.trace_callback = trace_callback
        self.trace_route_id = trace_route_id
        self.trace_wfc = trace_wfc
        self.trace_states = trace_states
        self.only_route_id = only_route_id
        self.requested_seed = seed
        self.accepted_seed = seed
        self.building_retry_count = 0
        self.max_route_attempts = max_route_attempts
        self.max_route_backtrack_depth = max_route_backtrack_depth
        self.max_route_recovery_attempts = max_route_recovery_attempts
        self.recovery_strategy = recovery_strategy
        self.minimum_slide_horizontal_travel = minimum_slide_horizontal_travel
        self.maximum_slide_grade = maximum_slide_grade
        self.minimum_turning_radius = minimum_turning_radius
        self.minimum_curvature_radius = minimum_curvature_radius
        self.tile_aware_connectivity = tile_aware_connectivity
        self.endpoint_frontier = endpoint_frontier
        self.candidate_lookahead = candidate_lookahead
        self.routing_strategy = routing_strategy
        self.hybrid_weighting = hybrid_weighting
        self.hybrid_compactness_strength = hybrid_compactness_strength
        self.floor_grid = frozenset(self.graph.nodes)
        self.grid_shape = (
            (floors - 1) * self.vertical_grid_resolution + 1,
            (length - 1) * grid_resolution + 1,
            (width - 1) * grid_resolution + 1,
        )
        self.forbidden_cells = set(forbidden_cells)
        self.occupancy: dict[Cell, set[str]] = {}
        self._reserved_connector_cells: set[Cell] = set()
        self._reserved_connector_owners: dict[Cell, set[str]] = {}
        self._connector_reservations_by_cell: dict[
            Cell, dict[str, ConnectorReservation]
        ] = {}
        self.routes: list[WFCPath] = []
        self.recovery_events: list[dict[str, object]] = []
        self.route_failures: list[dict[str, object]] = []
        self.total_wfc_attempts = 0
        self.recovery_attempts = 0
        self.routes_retried = 0
        self._validate_graph_coordinates()
        self._validate_forbidden_cells()
        self.embedding_repair: dict[str, object] | None = None
        self._repair_graph_embedding()
        if generate_routes:
            self.generate_routes()

    def _stair_endpoint_conflict(
        self, graph: nx.DiGraph
    ) -> tuple[GraphNode, GraphNode, GraphNode] | None:
        from src.route_diagnostics import _segment_distance

        edges = [
            (source, target, _edge_name(attributes))
            for source, target, attributes in graph.edges(data=True)
            if _edge_name(attributes) in _EDGE_PORTS
        ]
        endpoints = {node for source, target, _ in edges for node in (source, target)}
        tolerance = 0.2 / self.grid_resolution
        for source, target, kind in edges:
            if kind != "stair":
                continue
            for node in sorted(endpoints - {source, target}):
                if _segment_distance(source, target, node, node) < tolerance:
                    return source, target, node
        return None

    def _repair_graph_embedding(self) -> None:
        conflict = self._stair_endpoint_conflict(self.graph)
        if conflict is None:
            return
        source, target, endpoint = conflict
        floor = endpoint[2]
        if floor in (self.start_pos[2], self.exit_pos[2]):
            return
        transforms = (
            ("rotate_180", lambda x, y: (self.length - 1 - x, self.width - 1 - y)),
            ("reflect_y", lambda x, y: (x, self.width - 1 - y)),
            ("reflect_x", lambda x, y: (self.length - 1 - x, y)),
        )
        for name, transform in transforms:
            mapping = {
                node: (*transform(node[0], node[1]), node[2])
                for node in self.graph.nodes
                if node[2] == floor
            }
            if any(mapping.get(node, node) != node for node in (self.start_pos, self.exit_pos)):
                continue
            graph = nx.relabel_nodes(self.graph, mapping, copy=True)
            if self._stair_endpoint_conflict(graph) is not None:
                continue
            self.graph = graph
            self.path = [mapping.get(node, node) for node in self.path]
            if hasattr(self, "all_valid_notes"):
                self.all_valid_notes = [
                    mapping.get(node, node) for node in self.all_valid_notes
                ]
            self.floor_grid = frozenset(graph.nodes)
            self.embedding_repair = {
                "floor": floor,
                "transform": name,
                "original_stair": (source, target),
                "original_blocking_endpoint": endpoint,
                "mapped_endpoint": mapping.get(endpoint, endpoint),
            }
            return

    def display_graph(self, graph: nx.DiGraph) -> None:
        """Suppress inherited constructor popups; plots are shown explicitly."""
        if not getattr(self, "_suppress_graph_plots", True):
            super().display_graph(graph)

    def generate_routes(self) -> tuple[WFCPath, ...]:
        """Reserve graph-defined stairs, then route slides with bounded recovery."""
        self.routes.clear()
        self.occupancy.clear()
        self.total_wfc_attempts = 0
        self.recovery_attempts = 0
        self.routes_retried = 0
        self.recovery_events.clear()
        self.route_failures.clear()
        routable_edges: list[
            tuple[int, int, int, GraphNode, GraphNode, str]
        ] = []
        for original_order, (source, target, attributes) in enumerate(
            self.graph.edges(data=True)
        ):
            edge_name = _edge_name(attributes)
            if edge_name not in _EDGE_PORTS:
                continue
            vertical_span = abs(target[2] - source[2])
            horizontal_span = abs(target[0] - source[0]) + abs(target[1] - source[1])
            routable_edges.append(
                (
                    -vertical_span,
                    -horizontal_span,
                    original_order,
                    source,
                    target,
                    edge_name,
                )
            )

        routable_edges.sort(key=lambda edge: edge[:3])
        route_specs = tuple(
            (
                f"{route_number}:{source}->{target}:{edge_name}",
                source,
                target,
                edge_name,
            )
            for route_number, (_, _, _, source, target, edge_name) in enumerate(
                routable_edges
            )
        )
        only_route_id = getattr(self, "only_route_id", None)
        if only_route_id is not None and all(
            route_id != only_route_id
            for route_id, _, _, _ in route_specs
        ):
            raise ValueError(f"Unknown route ID {only_route_id!r}")
        self._reserved_connector_owners = {}
        self._connector_reservations_by_cell = {}
        for route_id, source, target, edge_name in route_specs:
            if edge_name == "stair":
                continue
            kind = _EDGE_PORTS[edge_name]
            for endpoint, is_start in (
                (source, True),
                (target, False),
            ):
                graph_node = endpoint
                endpoint_cell = self.graph_node_to_cell(endpoint)
                neighbor = self._endpoint_neighbor(
                    endpoint_cell,
                    kind,
                    is_start,
                )
                if neighbor is not None:
                    self._reserved_connector_owners.setdefault(
                        neighbor, set()
                    ).add(route_id)
                    self._connector_reservations_by_cell.setdefault(
                        neighbor, {}
                    )[route_id] = ConnectorReservation(
                        graph_node,
                        kind,
                        self._endpoint_direction(kind, is_start),
                        is_start,
                    )
        self._reserved_connector_cells = set(self._reserved_connector_owners)
        stair_specs = tuple(spec for spec in route_specs if spec[3] == "stair")
        slide_specs = tuple(spec for spec in route_specs if spec[3] == "slide")
        if only_route_id is not None and all(
            route_id != only_route_id
            for route_id, _, _, _ in slide_specs
        ):
            raise ValueError(
                f"Isolated routing requires a slide route ID, got {only_route_id!r}"
            )
        slides_to_generate = (
            tuple(spec for spec in slide_specs if spec[0] == only_route_id)
            if only_route_id is not None
            else slide_specs
        )
        for route_id, source, target, _ in stair_specs:
            trace_callback = getattr(self, "trace_callback", None)
            trace_route_id = getattr(self, "trace_route_id", None)
            trace_enabled = (
                trace_callback is not None
                and (
                    trace_route_id is None
                    or trace_route_id == route_id
                )
            )
            if trace_enabled and trace_callback is not None:
                start = self.graph_node_to_cell(source)
                end = self.graph_node_to_cell(target)
                trace_callback({
                    "event": "route_summary",
                    "route_id": route_id,
                    "route_kind": "stair",
                    "source": source,
                    "target": target,
                    "start_cell": start,
                    "end_cell": end,
                    "grid_dimensions": self.grid_shape,
                    "available_cells": (start, end),
                    "available_cell_count": 2,
                    "excluded_cells_by_reason": {},
                    "initial_domain_size_distribution": {},
                    "initial_pre_collapsed_cells": (start, end),
                    "initial_unresolved_cells": (),
                    "initial_fixed_cell_count": 2,
                    "direct_graph_route": True,
                })
            try:
                path = self._direct_stair(route_id, source, target)
                self._validate_existing_routes_with_candidate(path)
            except RouteGenerationError as error:
                self._record_route_failure(
                    route_id, source, target, "stair", error, (), False
                )
                raise RouteGenerationError(
                    f"Direct stair {source} -> {target} is infeasible: {error}",
                    failure_classification=error.failure_classification,
                    failure_details={"route_id": route_id, **error.failure_details},
                ) from error
            if trace_enabled and trace_callback is not None:
                trace_callback({
                    "event": "route_attempt_success",
                    "route_id": route_id,
                    "route_attempt": 1,
                    "route_retries_used": 0,
                    "direct_graph_route": True,
                })
            self._add_route(path)
        route_indices = {spec[0]: index for index, spec in enumerate(route_specs)}
        for route_index, route_spec in enumerate(slides_to_generate):
            route_id, source, target, edge_name = route_spec
            try:
                path = self._generate_route(
                    route_id,
                    source,
                    target,
                    edge_name,
                    seed_offset=route_indices[route_id] * self.max_route_attempts,
                )
            except RouteGenerationError as initial_error:
                self._record_route_failure(
                    route_id,
                    source,
                    target,
                    edge_name,
                    initial_error,
                    (),
                    False,
                )
                recovered = self._recover_routes(
                    slides_to_generate,
                    route_index,
                    source,
                    target,
                    edge_name,
                )
                self._finalize_route_failure(route_id, recovered)
                if not recovered:
                    recovery_note = ""
                    if self.recovery_strategy != "none":
                        recovery_note = (
                            " Bounded recovery exhausted "
                            f"(depth={self.max_route_backtrack_depth}, "
                            f"attempts={self.max_route_recovery_attempts})."
                        )
                    raise RouteGenerationError(
                        f"{initial_error}{recovery_note}",
                        attempts=initial_error.attempts,
                        failure_classification=initial_error.failure_classification,
                        failure_details=initial_error.failure_details,
                    ) from initial_error
                continue
            self._add_route(path)
        by_id = {route.route_id: route for route in self.routes}
        self._restore_routes(
            tuple(by_id[spec[0]] for spec in route_specs if spec[0] in by_id)
        )
        return tuple(self.routes)

    def _direct_stair(
        self, route_id: str, source: GraphNode, target: GraphNode
    ) -> WFCPath:
        if target[2] <= source[2]:
            raise RouteGenerationError(
                f"Direct stair {source} -> {target} must ascend",
                failure_classification="invalid_movement_direction",
            )
        points = tuple(
            tuple(float(value) for value in node) for node in (source, target)
        )
        path = WFCPath(
            route_id, PortKind.STAIR, source, target, (),
            attempts=0,
            vertical_resolution=self.vertical_grid_resolution,
            direct_stair=True,
        )
        self._validate_smoothed_path(path, points)
        return replace(path, accepted_geometry=points)

    def _recover_routes(
        self,
        route_specs: Sequence[tuple[str, GraphNode, GraphNode, str]],
        failed_index: int,
        failed_source: GraphNode,
        failed_target: GraphNode,
        failed_edge_name: str,
    ) -> bool:
        if self.recovery_strategy == "none":
            return False
        if self.recovery_strategy == "conflict-aware":
            start = self.graph_node_to_cell(failed_source)
            end = self.graph_node_to_cell(failed_target)
            connector_cells = {
                neighbor
                for endpoint, is_start in ((start, True), (end, False))
                if (
                    neighbor := self._endpoint_neighbor(
                        endpoint,
                        _EDGE_PORTS[failed_edge_name],
                        is_start,
                    )
                ) is not None
            }
            candidate_sets = self._blocking_route_sets(
                start,
                end,
                connector_cells,
                route_specs[failed_index][0],
            )
            failure = self._latest_failure(route_specs[failed_index][0])
            if failure is not None:
                failure["blocking_route_candidates"] = candidate_sets
                failure["recovery_strategy_attempted"] = (
                    "conflict-aware + recency fallback"
                )
            for attempt, candidate_ids in enumerate(
                candidate_sets[: self.max_route_recovery_attempts],
                start=1,
            ):
                if self._attempt_route_subset(
                    route_specs,
                    failed_index,
                    candidate_ids,
                    attempt,
                    "conflict-aware",
                ):
                    return True
            return self._recover_routes_by_recency(route_specs, failed_index)
        return self._recover_routes_by_recency(route_specs, failed_index)

    def _blocking_route_sets(
        self,
        start: Cell,
        end: Cell,
        route_connector_cells: set[Cell],
        route_id: str,
    ) -> tuple[tuple[str, ...], ...]:
        """Rank route sets that restore untyped free-space endpoint reachability.

        This is a counterfactual spatial heuristic, not evidence that typed WFC
        propagation or geometry constraints admit a route.
        """
        endpoint_nodes = {
            node
            for path in self.routes
            for node in (path.source, path.target)
            if self.graph_node_to_cell(node) in (start, end)
        }
        shared_endpoints = {
            self.graph_node_to_cell(node) for node in endpoint_nodes
        }
        available = set(
            self._route_grid_cells(
                start,
                end,
                shared_endpoints,
                route_connector_cells,
                route_id=route_id,
            )
        )
        if _spatially_reachable(start, end, available):
            return ()

        unconstrained = set(
            self._route_grid_cells(
                start,
                end,
                shared_endpoints,
                route_connector_cells,
                route_id=route_id,
                ignore_occupancy=True,
                ignore_reserved=True,
            )
        )
        if not _spatially_reachable(start, end, unconstrained):
            return ()

        ranked: list[tuple[int, int, tuple[int, ...], tuple[str, ...]]] = []
        route_order = {route.route_id: index for index, route in enumerate(self.routes)}
        reroutable = tuple(route for route in self.routes if not route.direct_stair)
        maximum_size = min(self.max_route_backtrack_depth, len(reroutable))
        for size in range(1, maximum_size + 1):
            for route_group in combinations(reroutable, size):
                route_ids = frozenset(route.route_id for route in route_group)
                candidate_cells = set(
                    self._route_grid_cells(
                        start,
                        end,
                        shared_endpoints,
                        route_connector_cells,
                        route_id=route_id,
                        freed_route_ids=route_ids,
                    )
                )
                if _spatially_reachable(start, end, candidate_cells):
                    ids = tuple(route.route_id for route in route_group)
                    ranked.append(
                        (
                            size,
                            sum(len(route.cells) for route in route_group),
                            tuple(route_order[route_id] for route_id in ids),
                            ids,
                        )
                    )
        ranked.sort()
        return tuple(candidate[3] for candidate in ranked)

    def _attempt_route_subset(
        self,
        route_specs: Sequence[tuple[str, GraphNode, GraphNode, str]],
        failed_index: int,
        selected_route_ids: Sequence[str],
        attempt: int,
        strategy: str,
    ) -> bool:
        original_routes = tuple(self.routes)
        original_by_id = {route.route_id: route for route in original_routes}
        selected = set(selected_route_ids)
        selected_indices = [
            index
            for index in range(failed_index)
            if route_specs[index][0] in selected
        ]
        preserve = tuple(
            route for route in original_routes if route.route_id not in selected
        )
        self._restore_routes(preserve)
        event: dict[str, object] = {
            "failed_route_id": route_specs[failed_index][0],
            "failed_source": route_specs[failed_index][1],
            "failed_target": route_specs[failed_index][2],
            "strategy": strategy,
            "selected_routes": tuple(selected_route_ids),
            "reopened_route_snapshots": tuple(
                {
                    "route_id": route.route_id,
                    "kind": route.kind.value,
                    "cells": route.cells,
                    "points": route.world_points(
                        self.grid_resolution,
                        self.vertical_grid_resolution,
                    ),
                }
                for route in original_routes
                if route.route_id in selected
            ),
            "attempt": attempt,
            "succeeded": False,
        }
        self.recovery_events.append(event)
        self.recovery_attempts += 1
        try:
            for route_index in (*selected_indices, failed_index):
                route_id, source, target, edge_name = route_specs[route_index]
                previous = original_by_id.get(route_id)
                if previous is not None:
                    self.routes_retried += 1
                try:
                    path = self._generate_route(
                        route_id,
                        source,
                        target,
                        edge_name,
                        seed_offset=(
                            route_index * self.max_route_attempts
                            + attempt
                            * len(route_specs)
                            * self.max_route_attempts
                        ),
                        recovery_count=(
                            previous.recovery_count + 1 if previous else 1
                        ),
                    )
                except RouteGenerationError as error:
                    self._record_route_failure(
                        route_id,
                        source,
                        target,
                        edge_name,
                        error,
                        selected_route_ids,
                        False,
                    )
                    raise
                self._add_route(path)
            rebuilt_by_id = {route.route_id: route for route in self.routes}
            ordered = tuple(route for route in preserve if route.direct_stair) + tuple(
                rebuilt_by_id[route_specs[index][0]]
                for index in range(failed_index + 1)
            )
            self._restore_routes(ordered)
            event["succeeded"] = True
            trace_callback = getattr(self, "trace_callback", None)
            if trace_callback is not None:
                trace_callback({
                    "event": "global_route_recovery_attempt",
                    **event,
                })
            return True
        except RouteGenerationError:
            self._restore_routes(original_routes)
            trace_callback = getattr(self, "trace_callback", None)
            if trace_callback is not None:
                trace_callback({
                    "event": "global_route_recovery_attempt",
                    **event,
                })
            return False

    def _recover_routes_by_recency(
        self,
        route_specs: Sequence[tuple[str, GraphNode, GraphNode, str]],
        failed_index: int,
    ) -> bool:
        maximum_depth = min(self.max_route_backtrack_depth, failed_index)
        original_routes = tuple(self.routes)
        for depth in range(1, maximum_depth + 1):
            for retry in range(self.max_route_recovery_attempts):
                first_index = failed_index - depth
                selected_ids = tuple(
                    route_specs[index][0]
                    for index in range(first_index, failed_index)
                )
                attempt = (depth - 1) * self.max_route_recovery_attempts + retry + 1
                if self.recovery_strategy == "conflict-aware":
                    attempt += self.max_route_recovery_attempts
                if self._attempt_route_subset(
                    route_specs,
                    failed_index,
                    selected_ids,
                    attempt,
                    "recency-fallback"
                    if self.recovery_strategy == "conflict-aware"
                    else "recency",
                ):
                    return True
        self._restore_routes(original_routes)
        return False

    def _record_route_failure(
        self,
        route_id: str,
        source: GraphNode,
        target: GraphNode,
        edge_name: str,
        error: RouteGenerationError,
        selected_routes: Sequence[str],
        recovery_succeeded: bool,
    ) -> None:
        self.route_failures.append(
            {
                "route_id": route_id,
                "route_type": edge_name,
                "source": source,
                "target": target,
                "grid_resolution": self.grid_resolution,
                "seed": self.seed,
                "wfc_attempts": error.attempts,
                "failure_classification": (
                    error.failure_classification
                    if error.failure_classification != "unknown"
                    else _classify_route_failure(str(error))
                ),
                "occupied_cells": len(self.occupancy),
                "routes_already_accepted": len(self.routes),
                "recovery_strategy": self.recovery_strategy,
                "recovery_strategy_attempted": self.recovery_strategy,
                "routes_selected_for_reopening": tuple(selected_routes),
                "recovery_succeeded": recovery_succeeded,
                "error": str(error),
                "failure_details": error.failure_details,
            }
        )

    def _latest_failure(self, route_id: str) -> dict[str, object] | None:
        return next(
            (
                failure
                for failure in self.route_failures
                if failure["route_id"] == route_id
            ),
            None,
        )

    def _finalize_route_failure(self, route_id: str, recovered: bool) -> None:
        failure = self._latest_failure(route_id)
        if failure is None:
            return
        failure["recovery_succeeded"] = recovered
        route_events = tuple(
            event
            for event in self.recovery_events
            if event["failed_route_id"] == route_id
        )
        successful_event = next(
            (event for event in reversed(route_events) if event["succeeded"]),
            None,
        )
        failure["recovery_routes_attempted"] = tuple(
            event["selected_routes"] for event in route_events
        )
        failure["recovery_strategies_attempted"] = tuple(
            dict.fromkeys(str(event["strategy"]) for event in route_events)
        )
        if successful_event is not None:
            failure["routes_selected_for_reopening"] = successful_event[
                "selected_routes"
            ]
            failure["successful_recovery_strategy"] = successful_event["strategy"]
        elif route_events:
            failure["routes_selected_for_reopening"] = tuple(
                dict.fromkeys(
                    route_id
                    for event in route_events
                    for route_id in event["selected_routes"]
                )
            )

    def _add_route(self, path: WFCPath) -> None:
        self.routes.append(path)
        for cell in path.cells:
            self.occupancy.setdefault(cell, set()).add(path.route_id)

    def _smooth_route_candidate(self, path: WFCPath) -> WFCPath:
        strengths = (
            _SLIDE_SMOOTHING_STRENGTHS
            if path.kind is PortKind.SLIDE
            else (path.smoothing_strength,)
        )
        failures: list[RouteGenerationError] = []
        for strength in strengths:
            candidate = replace(path, smoothing_strength=strength)
            try:
                points = self.smooth_path(candidate)
            except RouteGenerationError as error:
                error.failure_details.setdefault("smoothing_strength", strength)
                failures.append(error)
                constrained = self._constrained_endpoint_smoothing(candidate, error)
                if constrained is not None:
                    return constrained
            else:
                return replace(candidate, accepted_geometry=points)
        raise failures[-1]

    def _constrained_endpoint_smoothing(
        self,
        path: WFCPath,
        error: RouteGenerationError,
    ) -> WFCPath | None:
        blocked_cell = error.failure_details.get("blocked_cell")
        if (
            path.kind is not PortKind.SLIDE
            or not isinstance(blocked_cell, tuple)
            or blocked_cell not in self._other_graph_route_endpoint_cells(path)
        ):
            return None

        base = self._smooth_path_geometry(
            path,
            8,
            path.smoothing_strength,
        )
        tolerance = 0.2 / self.grid_resolution
        blocked_point = (
            blocked_cell[1] / self.grid_resolution,
            blocked_cell[2] / self.grid_resolution,
            blocked_cell[0] / self.vertical_grid_resolution,
        )
        distances = [0.0]
        for first, second in zip(base, base[1:]):
            distances.append(
                distances[-1]
                + hypot(second[0] - first[0], second[1] - first[1])
            )
        total_distance = distances[-1]
        if total_distance <= 1e-12:
            return None

        near_obstacle = sorted(
            (
                (
                    (point[0] - blocked_point[0]) ** 2
                    + (point[1] - blocked_point[1]) ** 2
                    + (point[2] - blocked_point[2]) ** 2,
                    index,
                )
                for index, point in enumerate(base[1:-1], 1)
                if (
                    hypot(
                        point[0] - blocked_point[0],
                        point[1] - blocked_point[1],
                    )
                    < tolerance
                    and abs(point[2] - blocked_point[2]) < tolerance
                )
            )
        )
        if not near_obstacle:
            return None

        for _, index in near_obstacle:
            fraction = distances[index] / total_distance
            endpoints = path.world_points(
                self.grid_resolution,
                self.vertical_grid_resolution,
            )
            for anchor_height in (
                blocked_point[2] + 2.5 * tolerance,
                blocked_point[2] - 2.5 * tolerance,
            ):
                if not endpoints[0][2] > anchor_height > endpoints[-1][2]:
                    continue
                candidate = replace(
                    path,
                    height_profile_anchor=(fraction, anchor_height),
                )
                try:
                    points = self.smooth_path(candidate)
                except RouteGenerationError:
                    continue
                return replace(candidate, accepted_geometry=points)
        return None

    def _validate_existing_routes_with_candidate(self, candidate: WFCPath) -> None:
        original_routes = tuple(self.routes)
        self._add_route(candidate)
        try:
            for existing in original_routes:
                if existing.accepted_geometry is None:
                    raise RuntimeError(
                        f"Accepted route {existing.route_id} lacks stored geometry"
                    )
                try:
                    self._validate_smoothed_path(existing, existing.accepted_geometry)
                except RouteGenerationError as error:
                    if existing.direct_stair:
                        error.failure_details.setdefault(
                            "blocking_stair_route_id", existing.route_id
                        )
                    raise
        finally:
            self._restore_routes(original_routes)

    def _restore_routes(self, routes: Sequence[WFCPath]) -> None:
        self.routes = list(routes)
        self.occupancy = {}
        for path in self.routes:
            for cell in path.cells:
                self.occupancy.setdefault(cell, set()).add(path.route_id)

    def _stair_conflict_exclusions(
        self,
        path: WFCPath,
        error: RouteGenerationError,
        allowed: Mapping[Cell, frozenset[int]],
        empty_index: int,
        protected: set[Cell],
    ) -> tuple[Cell, ...]:
        other_id = error.failure_details.get(
            "blocking_stair_route_id", error.failure_details.get("other_route_id")
        )
        if not any(
            route.route_id == other_id and route.direct_stair
            for route in self.routes
        ):
            return ()
        blocked_cell = error.failure_details.get("blocked_cell")
        if (
            isinstance(blocked_cell, tuple)
            and blocked_cell in path.cells
            and blocked_cell not in protected
            and blocked_cell in allowed
            and empty_index in allowed[blocked_cell]
            and len(allowed[blocked_cell]) > 1
        ):
            return (blocked_cell,)
        segment = error.failure_details.get("candidate_segment_points")
        if not (
            isinstance(segment, tuple)
            and len(segment) == 2
            and all(isinstance(point, tuple) and len(point) == 3 for point in segment)
        ):
            return ()

        from src.route_diagnostics import _segment_distance

        first, second = segment
        first_xy = (first[0], first[1], 0.0)
        second_xy = (second[0], second[1], 0.0)

        def distance(cell: Cell) -> float:
            point = (cell[1] / self.grid_resolution, cell[2] / self.grid_resolution, 0.0)
            return _segment_distance(first_xy, second_xy, point, point)

        candidates = [
            cell for cell in path.cells
            if cell not in protected
            and cell in allowed
            and empty_index in allowed[cell]
            and len(allowed[cell]) > 1
        ]
        return tuple(sorted(candidates, key=distance)[:4])

    def _generate_route(
        self,
        route_id: str,
        source: GraphNode,
        target: GraphNode,
        edge_name: str,
        seed_offset: int = 0,
        recovery_count: int = 0,
    ) -> WFCPath:
        trace_callback = getattr(self, "trace_callback", None)
        trace_route_id = getattr(self, "trace_route_id", None)
        trace_wfc = getattr(self, "trace_wfc", False)
        trace_states = getattr(self, "trace_states", False)
        routing_seed = getattr(self, "routing_seed", self.seed)
        kind = _EDGE_PORTS[edge_name]
        start = self.graph_node_to_cell(source)
        end = self.graph_node_to_cell(target)
        if start in self.forbidden_cells or end in self.forbidden_cells:
            raise RouteGenerationError(
                f"{edge_name} edge {source} -> {target} has a forbidden endpoint",
                failure_classification="invalid_endpoint",
            )
        if not _has_expected_vertical_direction(start, end, kind):
            direction = "ascend" if kind is PortKind.STAIR else "descend"
            raise RouteGenerationError(
                f"{edge_name} edge {source} -> {target} must {direction}",
                failure_classification="invalid_movement_direction",
            )

        tile_set = build_route_tiles(kind)
        route_connector_cells = {
            neighbor
            for endpoint, is_start in ((start, True), (end, False))
            if (
                neighbor := self._endpoint_neighbor(endpoint, kind, is_start)
            ) is not None
        }
        trace_enabled = (
            trace_callback is not None
            and (
                trace_route_id is None
                or trace_route_id == route_id
            )
        )
        shared_endpoints = {
            self.graph_node_to_cell(node)
            for path in self.routes
            for node in (path.source, path.target)
            if node in (source, target)
        }
        cells = self._route_grid_cells(
            start,
            end,
            shared_endpoints,
            route_connector_cells,
            route_id=route_id,
            excluded_reasons=(excluded_reasons := {}) if trace_enabled else None,
        )
        if start not in cells or end not in cells:
            raise RouteGenerationError(
                f"{edge_name} endpoints {source} -> {target} are outside the route grid",
                failure_classification="unreachable_endpoints",
            )
        allowed = self._route_domains(
            cells,
            start,
            end,
            kind,
            tile_set,
        )
        def emit_trace(event: str, **details: object) -> None:
            if trace_enabled and trace_callback is not None:
                trace_callback({
                    "event": event,
                    "route_id": route_id,
                    "source": source,
                    "target": target,
                    **details,
                })

        if trace_enabled:
            domain_distribution = Counter(len(domain) for domain in allowed.values())
            emit_trace(
                "route_summary",
                route_kind=edge_name,
                start_cell=start,
                end_cell=end,
                grid_dimensions=self.grid_shape,
                available_cells=tuple(cells),
                available_cell_count=len(cells),
                empty_tile_index=tile_set.empty_index,
                excluded_cells_by_reason={
                    reason: tuple(sorted(excluded))
                    for reason, excluded in excluded_reasons.items()
                },
                initial_domain_size_distribution=dict(sorted(domain_distribution.items())),
                initial_pre_collapsed_cells=tuple(
                    cell for cell, domain in allowed.items() if len(domain) == 1
                ),
                initial_unresolved_cells=tuple(
                    cell for cell, domain in allowed.items() if len(domain) > 1
                ),
                initial_fixed_cell_count=sum(
                    len(domain) == 1 for domain in allowed.values()
                ),
                heuristics={
                    "tile_aware_connectivity": self.tile_aware_connectivity,
                    "endpoint_frontier": self.endpoint_frontier,
                    "candidate_lookahead": self.candidate_lookahead,
                    "routing_strategy": getattr(
                        self, "routing_strategy", "dfs_guided"
                    ),
                },
            )
        if not allowed[start] or not allowed[end]:
            classification = (
                "occupancy_conflict"
                if self._endpoint_connector_is_blocked(
                    start,
                    end,
                    kind,
                    route_id,
                )
                else "unreachable_endpoints"
            )
            emit_trace(
                "route_attempt_failure",
                route_attempt=0,
                failure_phase="initial_endpoint_domains",
                failure_classification=classification,
                error="No connector orientation fits endpoint domain",
            )
            raise RouteGenerationError(
                f"No connector orientation fits {edge_name} edge "
                f"{source} -> {target}",
                failure_classification=classification,
            )

        last_error = "no continuous directed route"
        last_feasibility_stage = "discrete_wfc"
        discrete_route_candidates = 0
        geometry_rejections = 0
        last_geometry_failure_details: dict[str, object] = {}
        solver_stage_counts: dict[str, int] = {}
        solver_attempt_failures: list[dict[str, object]] = []
        rejected_signatures: set[tuple[Cell, ...]] = set()
        exclusions: deque[Cell] = deque()
        queued_exclusions: set[Cell] = set()
        tried_exclusions: set[Cell] = set()
        duplicate_candidates = 0
        all_candidate_paths: list[dict[str, object]] = []
        for attempt in range(self.max_route_attempts):
            last_feasibility_stage = "discrete_wfc"
            self.total_wfc_attempts += 1
            excluded = exclusions.popleft() if exclusions else None
            attempt_allowed = allowed
            if excluded is not None:
                tried_exclusions.add(excluded)
                attempt_allowed = dict(allowed)
                attempt_allowed[excluded] = frozenset((tile_set.empty_index,))
            emit_trace(
                "route_attempt_start",
                route_attempt=attempt + 1,
                routing_seed=routing_seed + seed_offset + attempt,
                excluded_cell=excluded,
                excluded_cell_count=len(tried_exclusions),
            )
            candidate_paths: list[dict[str, object]] = []

            def solver_trace(event: Mapping[str, object]) -> None:
                if event.get("event") == "route_candidate":
                    candidate_path = {
                        "attempt": event.get("attempt"),
                        "route_attempt": attempt + 1,
                        "candidate_path": event.get("candidate_path"),
                    }
                    candidate_paths.append(candidate_path)
                    all_candidate_paths.append(candidate_path)
                emit_trace(
                    "solver_trace",
                    route_attempt=attempt + 1,
                    solver_event=dict(event),
                )
            solve_route = solve_wfc
            if (
                kind is PortKind.SLIDE
                and getattr(self, "routing_strategy", "dfs_guided") == "hybrid"
                and getattr(self, "hybrid_max_cells", None) is None
                and not getattr(self, "hybrid_relative_budget", False)
            ):
                raise ValueError("hybrid routing requires a relative or absolute cell budget")
            if (
                kind is PortKind.SLIDE
                and getattr(self, "routing_strategy", "dfs_guided") == "hybrid"
                and (getattr(self, "hybrid_max_cells", None) is not None
                     or getattr(self, "hybrid_relative_budget", False))
            ):
                from src.hybrid_wfc import (
                    HybridGeometryRejected, relative_cell_budget,
                    route_length_lower_bound, solve_length_bounded_hybrid,
                    validate_hybrid_clearance,
                )

                def validate_hybrid_geometry(layout, path):
                    extracted = self._extract_route(
                        layout, tile_set, start, end, kind, route_id
                    )
                    if extracted != path:
                        raise HybridGeometryRejected("Extracted route differs from hybrid prefix")
                    candidate = WFCPath(
                        route_id, kind, source, target, path,
                        attempts=attempt + 1,
                        vertical_resolution=self.vertical_grid_resolution,
                        recovery_count=recovery_count,
                    )
                    try:
                        candidate = self._smooth_route_candidate(candidate)
                        self._validate_existing_routes_with_candidate(candidate)
                    except RouteGenerationError as error:
                        raise HybridGeometryRejected(str(error)) from error
                    minimum_clearance = getattr(self, "hybrid_minimum_clearance", None)
                    if minimum_clearance is not None:
                        validate_hybrid_clearance(candidate, self.routes, minimum_clearance)

                def solve_route(route_cells, route_tiles, **options):
                    max_cells = getattr(self, "hybrid_max_cells", None)
                    if getattr(self, "hybrid_relative_budget", False):
                        lower_bound = route_length_lower_bound(
                            route_cells, route_tiles, options["allowed_tiles"], start, end,
                            options["required_horizontal_steps"],
                        )
                        max_cells = relative_cell_budget(
                            lower_bound, self.hybrid_detour_fraction, self.hybrid_detour_cap
                        )
                        emit_trace(
                            "hybrid_route_budget", route_attempt=attempt + 1,
                            lower_bound_cells=lower_bound, max_cells=max_cells,
                        )
                    return solve_length_bounded_hybrid(
                        route_cells, route_tiles, start=start, end=end,
                        allowed_tiles=options["allowed_tiles"], seed=options["seed"],
                        max_cells=max_cells,
                        required_horizontal_steps=options["required_horizontal_steps"],
                        empty_index=tile_set.empty_index,
                        tile_aware_connectivity=options["tile_aware_connectivity"],
                        weighting=self.hybrid_weighting,
                        compactness_strength=self.hybrid_compactness_strength,
                        max_backtracks=self.hybrid_max_backtracks,
                        max_seconds=self.hybrid_max_seconds,
                        trace_callback=options["trace_callback"],
                        trace_states=options["trace_states"],
                        geometry_preference=getattr(self, "hybrid_geometry_preference", 0.0),
                        validate_complete=(
                            validate_hybrid_geometry
                            if (getattr(self, "hybrid_geometry_backtracking", False)
                                or getattr(self, "hybrid_minimum_clearance", None) is not None)
                            else None
                        ),
                    )
            try:
                layout = solve_route(
                    cells,
                    tile_set.tiles,
                    seed=routing_seed + seed_offset + attempt,
                    max_attempts=1,
                    required_pairs=[(start, end)],
                    required_typed_pairs=[(start, end, kind)],
                    required_horizontal_steps=(
                        self._required_slide_horizontal_steps(start, end)
                        if kind is PortKind.SLIDE
                        else 0
                    ),
                    preferred_tile_index=tile_set.empty_index,
                    tile_aware_connectivity=self.tile_aware_connectivity,
                    endpoint_frontier=self.endpoint_frontier,
                    candidate_lookahead=self.candidate_lookahead,
                    require_closed_boundaries=True,
                    allowed_tiles=attempt_allowed,
                    trace_callback=(
                        solver_trace
                        if trace_enabled and trace_wfc
                        else None
                    ),
                    trace_states=trace_states,
                )
            except HybridBudgetExhausted as error:
                last_error = str(error)
                emit_trace(
                    "route_attempt_failure", route_attempt=attempt + 1,
                    failure_phase="hybrid_budget_exhausted", failure_message=last_error,
                )
                continue
            except ContradictionError as error:
                last_error = str(error)
                failure = getattr(error, "failure", None)
                if failure is not None:
                    solver_attempt_failures.append(
                        {
                            "attempt": attempt + 1,
                            "phase": failure.phase,
                            "cell": failure.cell,
                            "domain": failure.domain,
                            "domain_tile_names": tuple(
                                tile_set.tiles[tile_index].name
                                for tile_index in failure.domain
                                if 0 <= tile_index < len(tile_set.tiles)
                            ),
                            "message": failure.message,
                        }
                    )
                    for phase, count in error.phase_counts.items():
                        solver_stage_counts[phase] = (
                            solver_stage_counts.get(phase, 0) + count
                        )
                    emit_trace(
                        "route_attempt_failure",
                        route_attempt=attempt + 1,
                        solver_attempt=failure.attempt,
                        failure_phase=failure.phase,
                        failure_message=failure.message,
                        phase_counts=dict(error.phase_counts),
                        route_retries_used=attempt,
                        dfs_candidate_paths=tuple(candidate_paths),
                    )
                continue

            path = self._extract_route(
                layout,
                tile_set,
                start,
                end,
                kind,
                route_id,
            )
            if path is not None:
                path_set = set(path)
                path_comparisons = tuple(
                    {
                        "candidate_index": index,
                        "attempt": candidate_path["attempt"],
                        "path": candidate_path["candidate_path"],
                        "path_length": len(candidate_path["candidate_path"]),
                        "exact_match": tuple(candidate_path["candidate_path"]) == path,
                        "jaccard_similarity": (
                            len(set(candidate_path["candidate_path"]) & path_set)
                            / len(set(candidate_path["candidate_path"]) | path_set)
                            if set(candidate_path["candidate_path"]) | path_set
                            else 1.0
                        ),
                    }
                    for index, candidate_path in enumerate(all_candidate_paths)
                )
                emit_trace(
                    "route_extracted",
                    route_attempt=attempt + 1,
                    first_dfs_candidate_path=(
                        all_candidate_paths[0]["candidate_path"]
                        if all_candidate_paths else None
                    ),
                    subsequent_dfs_candidate_paths=tuple(all_candidate_paths[1:]),
                    final_discrete_route=path,
                    final_discrete_route_length=len(path),
                    path_comparisons=path_comparisons,
                )
                discrete_route_candidates += 1
                last_feasibility_stage = "smoothed_geometry"
                if path in rejected_signatures:
                    duplicate_candidates += 1
                    geometry_rejections += 1
                    continue
                candidate = WFCPath(
                    route_id,
                    kind,
                    source,
                    target,
                    path,
                    attempts=attempt + 1,
                    vertical_resolution=self.vertical_grid_resolution,
                    recovery_count=recovery_count,
                )
                try:
                    candidate = self._smooth_route_candidate(candidate)
                    self._validate_existing_routes_with_candidate(candidate)
                except RouteGenerationError as error:
                    geometry_rejections += 1
                    rejected_signatures.add(path)
                    last_error = str(error)
                    candidate_exclusions = self._stair_conflict_exclusions(
                        candidate,
                        error,
                        allowed,
                        tile_set.empty_index,
                        {start, end, *route_connector_cells},
                    )
                    emit_trace(
                        "route_attempt_geometry_rejection",
                        route_attempt=attempt + 1,
                        route_retries_used=attempt,
                        failure_classification=error.failure_classification,
                        error=str(error),
                        discrete_candidate_path=path,
                        next_exclusions=tuple(sorted(candidate_exclusions)),
                    )
                    last_geometry_failure_details = {
                        **error.failure_details,
                        "discrete_candidate_cells": tuple(path),
                    }
                    last_geometry_failure_details.setdefault(
                        "smoothing_strength",
                        candidate.smoothing_strength,
                    )
                    if "candidate_points" in error.failure_details:
                        last_geometry_failure_details[
                            "smoothed_candidate_points"
                        ] = error.failure_details["candidate_points"]
                    for cell in candidate_exclusions:
                        if cell not in queued_exclusions:
                            queued_exclusions.add(cell)
                            exclusions.append(cell)
                    continue
                emit_trace(
                    "route_attempt_success",
                    route_attempt=attempt + 1,
                    route_retries_used=attempt,
                    solver_failures=len(solver_attempt_failures),
                    candidate_path_count=len(candidate_paths),
                    total_dfs_candidate_paths=len(all_candidate_paths),
                )
                return replace(
                    candidate,
                    rejected_route_signatures=len(rejected_signatures),
                    retry_exclusion=excluded,
                )
            last_error = "collapsed route has disconnected cells or a wrong direction"
            emit_trace(
                "route_attempt_no_route",
                route_attempt=attempt + 1,
                route_retries_used=attempt,
                candidate_path_count=len(candidate_paths),
                total_dfs_candidate_paths=len(all_candidate_paths),
                message=last_error,
            )

        raise RouteGenerationError(
            f"Could not route {edge_name} edge {source} -> {target} after "
            f"{self.max_route_attempts} attempts: {last_error}",
            attempts=self.max_route_attempts,
            failure_classification=_classify_route_failure(last_error),
            failure_details={
                "route_id": route_id,
                "route_type": edge_name,
                "source": source,
                "target": target,
                "start_cell": start,
                "end_cell": end,
                "grid_dimensions": self.grid_shape,
                "grid_resolution": self.grid_resolution,
                "vertical_grid_resolution": self.vertical_grid_resolution,
                "available_cell_count": len(cells),
                "endpoint_constraints": {
                    "start_directions": tuple(
                        direction.name
                        for direction in self._endpoint_directions(
                            start, kind, True, set(cells)
                        )
                    ),
                    "end_directions": tuple(
                        direction.name
                        for direction in self._endpoint_directions(
                            end, kind, False, set(cells)
                        )
                    ),
                    "connector_neighbors": tuple(
                        neighbor
                        for neighbor in sorted(route_connector_cells)
                    ),
                    "start_connectors": self._connector_constraint_details(
                        start,
                        kind,
                        True,
                        tile_set,
                        set(cells),
                    ),
                    "end_connectors": self._connector_constraint_details(
                        end,
                        kind,
                        False,
                        tile_set,
                        set(cells),
                    ),
                },
                "solver_phase_counts": solver_stage_counts,
                "solver_attempt_failures": tuple(solver_attempt_failures),
                "last_solver_failure": (
                    solver_attempt_failures[-1]
                    if solver_attempt_failures
                    else None
                ),
                "feasibility_stage": last_feasibility_stage,
                "discrete_route_candidates": discrete_route_candidates,
                "geometry_rejections": geometry_rejections,
                "rejected_route_signatures": len(rejected_signatures),
                "duplicate_candidates": duplicate_candidates,
                "retry_exclusions": tuple(sorted(tried_exclusions)),
                "last_geometry_failure_details": last_geometry_failure_details,
                "heuristics": {
                    "tile_aware_connectivity": self.tile_aware_connectivity,
                    "endpoint_frontier": self.endpoint_frontier,
                    "candidate_lookahead": self.candidate_lookahead,
                    "routing_strategy": getattr(
                        self, "routing_strategy", "dfs_guided"
                    ),
                },
            },
        )

    def _connector_constraint_details(
        self,
        endpoint: Cell,
        kind: PortKind,
        is_start: bool,
        tile_set: RouteTileSet,
        cells: set[Cell],
    ) -> tuple[dict[str, object], ...]:
        details = []
        for direction in self._endpoint_directions(
            endpoint,
            kind,
            is_start,
            cells,
        ):
            offset = _DIRECTION_OFFSETS[direction]
            connector_cell = (
                endpoint[0] + offset[0],
                endpoint[1] + offset[1],
                endpoint[2] + offset[2],
            )
            tile_index = tile_set.connector_indices[direction]
            details.append(
                {
                    "direction": direction.name,
                    "connector_cell": connector_cell,
                    "tile_index": tile_index,
                    "tile_name": tile_set.tiles[tile_index].name,
                    "port_kind": kind.name,
                }
            )
        return tuple(details)

    def _endpoint_connector_is_blocked(
        self,
        start: Cell,
        end: Cell,
        kind: PortKind,
        route_id: str,
    ) -> bool:
        for endpoint, is_start in ((start, True), (end, False)):
            neighbor = self._endpoint_neighbor(endpoint, kind, is_start)
            if neighbor is None:
                continue
            other_occupants = self.occupancy.get(neighbor, set()) - {route_id}
            other_reservations = (
                self._reserved_connector_owners.get(neighbor, set())
                - {route_id}
            )
            can_share = self._can_share_landing(
                neighbor,
                route_id,
                other_occupants | other_reservations,
                expected_kind=kind,
            )
            if (
                (other_occupants and not can_share)
                or (other_reservations and not can_share)
                or neighbor in self.forbidden_cells
            ):
                return True
        return False

    def _can_share_landing(
        self,
        cell: Cell,
        route_id: str | None,
        other_route_ids: set[str],
        *,
        expected_kind: PortKind | None = None,
    ) -> bool:
        """Allow connector overlap only for typed ports at one graph-node landing."""
        if route_id is None or not other_route_ids:
            return False
        reservations = self._connector_reservations_by_cell.get(cell, {})
        current = reservations.get(route_id)
        if current is None or current.kind not in (
            PortKind.STAIR,
            PortKind.SLIDE,
        ) or (
            expected_kind is not None and current.kind is not expected_kind
        ):
            return False
        if (
            self._endpoint_direction(current.kind, current.is_start)
            is not current.direction
            or self._endpoint_neighbor(
                self.graph_node_to_cell(current.node),
                current.kind,
                current.is_start,
            )
            != cell
        ):
            return False

        for other_route_id in other_route_ids:
            other = reservations.get(other_route_id)
            if (
                other is None
                or other.kind not in (PortKind.STAIR, PortKind.SLIDE)
                or other.node != current.node
                or other.direction is not current.direction
                or self._endpoint_direction(other.kind, other.is_start)
                is not other.direction
                or self._endpoint_neighbor(
                    self.graph_node_to_cell(other.node),
                    other.kind,
                    other.is_start,
                )
                != cell
            ):
                return False
        return True

    def _route_grid_cells(
        self,
        start: Cell,
        end: Cell,
        shared_endpoints: set[Cell] | None = None,
        route_connector_cells: set[Cell] | None = None,
        *,
        route_id: str | None = None,
        ignore_occupancy: bool = False,
        ignore_reserved: bool = False,
        freed_route_ids: set[str] | frozenset[str] = frozenset(),
        excluded_reasons: MutableMapping[str, set[Cell]] | None = None,
    ) -> list[Cell]:
        shared_endpoints = shared_endpoints or set()
        route_connector_cells = route_connector_cells or set()
        lower_floor = min(start[0], end[0])
        upper_floor = max(start[0], end[0])
        blocked = set(self.forbidden_cells)
        reason_by_cell: dict[Cell, str] = {}
        if excluded_reasons is not None:
            for cell in self.forbidden_cells:
                reason_by_cell[cell] = "forbidden"
        if not ignore_occupancy:
            for cell, owners in self.occupancy.items():
                remaining_owners = owners - freed_route_ids
                is_shared_landing = (
                    cell in route_connector_cells
                    and self._can_share_landing(
                        cell,
                        route_id,
                        remaining_owners,
                    )
                )
                if (
                    cell not in shared_endpoints
                    and not is_shared_landing
                    and (not owners or remaining_owners)
                ):
                    blocked.add(cell)
                    reason_by_cell.setdefault(cell, "occupied_by_existing_route")
        graph_floor_cells = {
            self.graph_node_to_cell(node)
            for node in self.floor_grid
        }
        blocked.update(graph_floor_cells - {start, end})
        for cell in graph_floor_cells - {start, end}:
            reason_by_cell.setdefault(cell, "graph_floor_cell")
        if not ignore_reserved:
            for cell, owners in self._reserved_connector_owners.items():
                remaining_owners = owners - freed_route_ids - (
                    {route_id} if route_id else set()
                )
                if not owners or (
                    remaining_owners
                    and (
                        cell not in route_connector_cells
                        or not self._can_share_landing(
                            cell,
                            route_id,
                            remaining_owners,
                        )
                    )
                ):
                    blocked.add(cell)
                    reason_by_cell.setdefault(cell, "reserved_connector")
        if excluded_reasons is None:
            return [
                (floor, x, y)
                for floor in range(lower_floor, upper_floor + 1)
                for x in range(self.grid_shape[1])
                for y in range(self.grid_shape[2])
                if (floor, x, y) not in blocked
                and (
                    (floor, x, y) in (start, end)
                    or (
                        floor not in (start[0], end[0])
                        and floor % self.vertical_grid_resolution == 0
                        and (
                            x % self.grid_resolution != 0
                            or y % self.grid_resolution != 0
                        )
                    )
                    or floor % self.vertical_grid_resolution != 0
                )
            ]
        available = []
        excluded_for_route: dict[str, set[Cell]] = {}
        for floor in range(lower_floor, upper_floor + 1):
            for x in range(self.grid_shape[1]):
                for y in range(self.grid_shape[2]):
                    cell = (floor, x, y)
                    lies_on_routing_grid = (
                        cell in (start, end)
                        or (
                            floor not in (start[0], end[0])
                            and floor % self.vertical_grid_resolution == 0
                            and (
                                x % self.grid_resolution != 0
                                or y % self.grid_resolution != 0
                            )
                        )
                        or floor % self.vertical_grid_resolution != 0
                    )
                    if cell in blocked:
                        reason = reason_by_cell.get(cell, "blocked")
                    elif not lies_on_routing_grid:
                        reason = "outside_refined_routing_grid"
                    else:
                        available.append(cell)
                        continue
                    excluded_for_route.setdefault(reason, set()).add(cell)
        if excluded_reasons is not None:
            excluded_reasons.update(excluded_for_route)
        return available
    def _route_domains(
        self,
        cells: Sequence[Cell],
        start: Cell,
        end: Cell,
        kind: PortKind,
        tile_set: RouteTileSet,
    ) -> dict[Cell, frozenset[int]]:
        cell_set = set(cells)
        endpoint_direction = {
            start: self._endpoint_directions(start, kind, is_start=True, cells=cell_set),
            end: self._endpoint_directions(end, kind, is_start=False, cells=cell_set),
        }
        segment_or_empty = frozenset(
            (tile_set.empty_index, *tile_set.segment_indices)
        )
        allowed = {
            cell: segment_or_empty
            for cell in cells
        }
        for cell in cells:
            if (
                cell[0] % self.vertical_grid_resolution == 0
                and cell not in (start, end)
            ):
                allowed[cell] = frozenset(
                    (tile_set.empty_index, tile_set.vertical_straight_index)
                )
        for endpoint, directions in endpoint_direction.items():
            allowed[endpoint] = frozenset(
                tile_set.connector_indices[direction]
                for direction in directions
            )
        return allowed

    def _endpoint_directions(
        self,
        cell: Cell,
        kind: PortKind,
        is_start: bool,
        cells: set[Cell],
    ) -> tuple[Direction, ...]:
        if kind is PortKind.STAIR:
            direction = Direction.U if is_start else Direction.D
        else:
            direction = Direction.D if is_start else Direction.U
        neighbor = tuple(
            cell[index] + _DIRECTION_OFFSETS[direction][index]
            for index in range(3)
        )
        return (direction,) if neighbor in cells else ()

    def _endpoint_neighbor(
        self,
        cell: Cell,
        kind: PortKind,
        is_start: bool,
    ) -> Cell | None:
        direction = self._endpoint_direction(kind, is_start)
        offset = _DIRECTION_OFFSETS[direction]
        neighbor = tuple(cell[index] + offset[index] for index in range(3))
        if all(0 <= coordinate < size for coordinate, size in zip(neighbor, self.grid_shape)):
            return neighbor
        return None

    @staticmethod
    def _endpoint_direction(kind: PortKind, is_start: bool) -> Direction:
        if kind is PortKind.STAIR:
            return Direction.U if is_start else Direction.D
        return Direction.D if is_start else Direction.U

    def _extract_route(
        self,
        layout: Mapping[Cell, int],
        tile_set: RouteTileSet,
        start: Cell,
        end: Cell,
        kind: PortKind,
        route_id: str | None = None,
    ) -> tuple[Cell, ...] | None:
        active = {
            cell
            for cell, tile_index in layout.items()
            if tile_index != tile_set.empty_index
        }
        if start not in active or end not in active:
            return None

        adjacency: dict[Cell, set[Cell]] = {cell: set() for cell in active}
        for cell in active:
            tile = tile_set.tiles[layout[cell]]
            for direction, offset in _DIRECTION_OFFSETS.items():
                if tile.ports[direction] is not kind:
                    continue
                neighbor = tuple(cell[index] + offset[index] for index in range(3))
                if neighbor not in active:
                    continue
                if tile_set.tiles[layout[neighbor]].ports[_opposite(direction)] is not kind:
                    continue
                if _can_travel(direction, kind):
                    adjacency[cell].add(neighbor)

        previous: dict[Cell, Cell | None] = {start: None}
        pending = deque([start])
        while pending and end not in previous:
            current = pending.popleft()
            for neighbor in adjacency[current]:
                if neighbor not in previous:
                    previous[neighbor] = current
                    pending.append(neighbor)
        if end not in previous or set(previous) != active:
            return None

        path = []
        current: Cell | None = end
        while current is not None:
            path.append(current)
            current = previous[current]
        path.reverse()

        floor_deltas = [
            path[index + 1][0] - path[index][0]
            for index in range(len(path) - 1)
        ]
        if kind is PortKind.STAIR and any(delta < 0 for delta in floor_deltas):
            return None
        if kind is PortKind.SLIDE and any(delta > 0 for delta in floor_deltas):
            return None
        if any(cell in self.forbidden_cells for cell in path):
            return None
        if any(
            cell in self.occupancy
            and cell not in (start, end)
            and not self._can_share_landing(
                cell,
                route_id,
                self.occupancy[cell] - {route_id} if route_id else set(),
                expected_kind=kind,
            )
            for cell in path
        ):
            return None
        return tuple(path)

    def graph_node_to_cell(self, node: GraphNode) -> Cell:
        """Map graph (x, y, floor) into WFC (floor, x, y) grid coordinates."""
        x, y, floor = node
        return (
            floor * self.vertical_grid_resolution,
            x * self.grid_resolution,
            y * self.grid_resolution,
        )

    def smooth_path(
        self,
        path: WFCPath,
        samples_per_segment: int = 8,
        *,
        smoothing_strength: float | None = None,
    ) -> tuple[tuple[float, float, float], ...]:
        """Smooth XY and assign slide height by horizontal arc length."""
        if samples_per_segment < 1:
            raise ValueError("samples_per_segment must be positive")
        strength = (
            path.smoothing_strength
            if smoothing_strength is None
            else smoothing_strength
        )
        if not isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError("smoothing_strength must be between zero and one")
        sampled = self._smooth_path_geometry(
            path,
            samples_per_segment,
            strength,
        )
        self._validate_smoothed_path(path, sampled)
        return tuple(sampled)

    def validate_accepted_geometry(
        self, path: WFCPath
    ) -> tuple[tuple[float, float, float], ...]:
        """Validate and return the geometry saved when the route was accepted."""
        if path.accepted_geometry is None:
            raise RuntimeError(f"Accepted route {path.route_id} lacks stored geometry")
        if path.direct_stair and path.accepted_geometry != (
            path.source, path.target
        ):
            raise RouteGenerationError(
                f"Direct stair {path.route_id} must match its graph endpoints",
                failure_classification="invalid_endpoint",
            )
        self._validate_smoothed_path(path, path.accepted_geometry)
        return path.accepted_geometry

    def _smooth_path_geometry(
        self,
        path: WFCPath,
        samples_per_segment: int,
        smoothing_strength: float,
    ) -> list[tuple[float, float, float]]:
        points = path.world_points(
            self.grid_resolution,
            self.vertical_grid_resolution,
        )
        if path.direct_stair:
            sampled = tuple(
                tuple(float(value) for value in node)
                for node in (path.source, path.target)
            )
        elif path.kind is PortKind.STAIR:
            sampled = points
        elif len(points) < 3:
            sampled_xy = [(point[0], point[1]) for point in points]
            sampled = self._slide_elevation_profile(points, sampled_xy)
        else:
            controls = _simplify_xy_controls(points)
            if len(controls) < 2:
                sampled = self._slide_elevation_profile(
                    points,
                    [(points[0][0], points[0][1]), (points[-1][0], points[-1][1])],
                )
            else:
                sampled_xy: list[tuple[float, float]] = []
                for index in range(len(controls) - 1):
                    p0 = controls[max(0, index - 1)]
                    p1 = controls[index]
                    p2 = controls[index + 1]
                    p3 = controls[min(len(controls) - 1, index + 2)]
                    for step in range(samples_per_segment):
                        t = step / samples_per_segment
                        t2 = t * t
                        t3 = t2 * t
                        terminal_segment = index in (
                            0,
                            len(controls) - 2,
                        )
                        catmull = tuple(
                            0.5
                            * (
                                2 * p1[axis]
                                + (-p0[axis] + p2[axis]) * t
                                + (
                                    2 * p0[axis]
                                    - 5 * p1[axis]
                                    + 4 * p2[axis]
                                    - p3[axis]
                                )
                                * t2
                                + (
                                    -p0[axis]
                                    + 3 * p1[axis]
                                    - 3 * p2[axis]
                                    + p3[axis]
                                )
                                * t3
                            )
                            for axis in range(2)
                        )
                        position = tuple(
                            catmull[axis] * (
                                0.0 if terminal_segment else smoothing_strength
                            )
                            + (p1[axis] + (p2[axis] - p1[axis]) * t)
                            * (
                                1.0
                                if terminal_segment
                                else 1.0 - smoothing_strength
                            )
                            for axis in range(2)
                        )
                        sampled_xy.append(
                            (
                                min(max(position[0], 0.0), self.length - 1),
                                min(max(position[1], 0.0), self.width - 1),
                            )
                        )
                sampled_xy.append(controls[-1])
                sampled = self._slide_elevation_profile(points, sampled_xy)

        if path.height_profile_anchor is not None:
            sampled = self._apply_height_profile_anchor(
                path,
                sampled,
                path.height_profile_anchor,
            )
        return sampled

    def _apply_height_profile_anchor(
        self,
        path: WFCPath,
        points: Sequence[tuple[float, float, float]],
        anchor: tuple[float, float],
    ) -> list[tuple[float, float, float]]:
        fraction, anchor_height = anchor
        if not 0 < fraction < 1:
            raise RouteGenerationError(
                "slide height-profile anchor must be inside the route"
            )
        distances = [0.0]
        for first, second in zip(points, points[1:]):
            distances.append(
                distances[-1]
                + hypot(second[0] - first[0], second[1] - first[1])
            )
        anchor_distance = fraction * distances[-1]
        if not distances[-1] > 0:
            raise RouteGenerationError(
                "slide height-profile anchor needs horizontal travel"
            )
        if not points[0][2] > anchor_height > points[-1][2]:
            raise RouteGenerationError(
                "slide height-profile anchor must preserve descent"
            )

        anchored = []
        for point, distance in zip(points, distances):
            height = (
                points[0][2]
                + (anchor_height - points[0][2])
                * (distance / anchor_distance)
                if distance <= anchor_distance
                else anchor_height
                + (points[-1][2] - anchor_height)
                * (
                    (distance - anchor_distance)
                    / (distances[-1] - anchor_distance)
                )
            )
            anchored.append((point[0], point[1], height))
        return anchored

    def _slide_elevation_profile(
        self,
        endpoints: Sequence[tuple[float, float, float]],
        xy_points: Sequence[tuple[float, float]],
    ) -> list[tuple[float, float, float]]:
        if len(endpoints) < 2 or len(xy_points) < 2:
            raise RouteGenerationError("slide needs distinct fixed endpoints")

        cumulative = [0.0]
        for first, second in zip(xy_points, xy_points[1:]):
            cumulative.append(
                cumulative[-1] + hypot(second[0] - first[0], second[1] - first[1])
            )
        horizontal_travel = cumulative[-1]
        height_drop = endpoints[0][2] - endpoints[-1][2]
        required_travel = max(
            self.minimum_slide_horizontal_travel,
            height_drop / self.maximum_slide_grade,
        )
        if horizontal_travel + 1e-9 < required_travel:
            raise RouteGenerationError(
                f"slide has {horizontal_travel:.3g} units of horizontal travel; "
                f"at least {required_travel:.3g} are required for its "
                f"{height_drop:.3g}-unit drop"
            )

        sampled = [
            (
                point[0],
                point[1],
                endpoints[0][2]
                + (endpoints[-1][2] - endpoints[0][2])
                * (distance / horizontal_travel),
            )
            for point, distance in zip(xy_points, cumulative)
        ]
        if self.minimum_turning_radius:
            too_tight = [
                (index, radius)
                for index, radius in enumerate(_turning_radii(sampled))
                if radius + 1e-9 < self.minimum_turning_radius
            ]
            if too_tight:
                segment, radius = min(too_tight, key=lambda item: item[1])
                raise RouteGenerationError(
                    f"slide turn near segment {segment} has radius {radius:.3g}; "
                    f"minimum is {self.minimum_turning_radius:.3g}"
                )
        return _remove_duplicate_points(sampled)

    def _required_slide_horizontal_steps(
        self,
        start: Cell,
        end: Cell,
    ) -> int:
        height_drop = (start[0] - end[0]) / self.vertical_grid_resolution
        travel = max(
            self.minimum_slide_horizontal_travel,
            height_drop / self.maximum_slide_grade,
        )
        return ceil(travel * self.grid_resolution - 1e-12)

    def _validate_smoothed_path(
        self,
        path: WFCPath,
        points: Sequence[tuple[float, float, float]],
    ) -> None:
        from src.route_diagnostics import _segment_distance

        if (
            path.kind is PortKind.SLIDE
            and self.minimum_curvature_radius > 0
        ):
            from src.route_diagnostics import _curvature_radii

            radii = _curvature_radii(points)
            violating = [
                (index, radius)
                for index, radius in enumerate(radii)
                if radius < self.minimum_curvature_radius
            ]
            if violating:
                segment, radius = min(violating, key=lambda item: item[1])
                raise RouteGenerationError(
                    f"Smoothed slide {path.route_id} has curvature radius "
                    f"{radius:.3g} near segment {segment}; minimum is "
                    f"{self.minimum_curvature_radius:.3g}",
                    failure_classification="geometric_rejection",
                )

        tolerance = 0.2 / self.grid_resolution
        other_cells = {
            cell
            for cell, owners in self.occupancy.items()
            if any(owner != path.route_id for owner in owners)
            and cell not in self._shared_graph_endpoint_cells(path)
            and not self._can_share_landing(
                cell,
                path.route_id,
                owners - {path.route_id},
                expected_kind=path.kind,
            )
        }
        other_cells.update(self._other_graph_route_endpoint_cells(path))
        forbidden = self.forbidden_cells
        allowed_shared_endpoints = self._shared_graph_endpoint_cells(path)
        blocked_cells = (*forbidden, *other_cells)
        for index, point in enumerate(points):
            x, y, floor = point
            if not (
                0 <= x <= self.length - 1
                and 0 <= y <= self.width - 1
                and min(path.source[2], path.target[2]) <= floor
                <= max(path.source[2], path.target[2])
            ):
                raise RouteGenerationError(
                    f"Smoothed route {path.route_id} leaves the building grid"
                )
            if path.kind is PortKind.SLIDE and len(points) > 1:
                next_point = points[index + 1] if index + 1 < len(points) else None
                delta = next_point[2] - floor if next_point is not None else 0
                if delta >= 0 and next_point is not None:
                    raise RouteGenerationError(
                        f"Smoothed slide {path.route_id} is not strictly descending"
                    )
                if next_point is not None:
                    horizontal = hypot(next_point[0] - x, next_point[1] - y)
                    grade = abs(delta) / horizontal if horizontal else (
                        float("inf") if delta else 0.0
                    )
                    if grade > self.maximum_slide_grade + 1e-9:
                        raise RouteGenerationError(
                            f"Smoothed slide {path.route_id} has grade "
                            f"{grade:.3g}, exceeding maximum "
                            f"{self.maximum_slide_grade:.3g}"
                        )

            for blocked_cell in (() if path.direct_stair else blocked_cells):
                if (
                    index in (0, len(points) - 1)
                    and blocked_cell in allowed_shared_endpoints
                ):
                    continue
                blocked_point = (
                    blocked_cell[1] / self.grid_resolution,
                    blocked_cell[2] / self.grid_resolution,
                    blocked_cell[0] / self.vertical_grid_resolution,
                )
                distance_squared = sum(
                    (point[axis] - blocked_point[axis]) ** 2
                    for axis in range(3)
                )
                if distance_squared < tolerance**2:
                    raise RouteGenerationError(
                        f"Smoothed route {path.route_id} collides near "
                        f"route-grid cell {blocked_cell}",
                        failure_classification="geometry_collision",
                        failure_details={"blocked_cell": blocked_cell},
                    )
        if path.direct_stair:
            for blocked_cell in blocked_cells:
                blocked_point = (
                    blocked_cell[1] / self.grid_resolution,
                    blocked_cell[2] / self.grid_resolution,
                    blocked_cell[0] / self.vertical_grid_resolution,
                )
                if (
                    blocked_cell in allowed_shared_endpoints
                    and blocked_point in (points[0], points[-1])
                    and blocked_cell not in forbidden
                ):
                    continue
                distance = _segment_distance(
                    points[0], points[-1], blocked_point, blocked_point
                )
                if distance < tolerance:
                    raise RouteGenerationError(
                        f"Direct stair {path.route_id} collides near route-grid "
                        f"cell {blocked_cell}",
                        failure_classification="geometry_collision",
                        failure_details={
                            "blocked_cell": blocked_cell,
                            "minimum_clearance": distance,
                            "collision_tolerance": tolerance,
                        },
                    )
        self._validate_reserved_connector_clearance(path, points)
        self._validate_route_intersections(path, points)

    def _validate_reserved_connector_clearance(
        self,
        path: WFCPath,
        points: Sequence[tuple[float, float, float]],
    ) -> None:
        from src.route_diagnostics import _segment_distance

        tolerance = 0.2 / self.grid_resolution
        path_endpoints = {path.source, path.target}
        for cell, owners in self._reserved_connector_owners.items():
            for owner in owners - {path.route_id}:
                if self._can_share_landing(
                    cell,
                    path.route_id,
                    {owner},
                    expected_kind=path.kind,
                ):
                    continue
                reservation = self._connector_reservations_by_cell[cell][owner]
                node = self.graph_node_to_cell(reservation.node)
                node_point = (
                    node[1] / self.grid_resolution,
                    node[2] / self.grid_resolution,
                    node[0] / self.vertical_grid_resolution,
                )
                connector_point = (
                    cell[1] / self.grid_resolution,
                    cell[2] / self.grid_resolution,
                    cell[0] / self.vertical_grid_resolution,
                )
                shared_nodes = (
                    {reservation.node}
                    if reservation.node in path_endpoints
                    else set()
                )
                for segment_index, (start, end) in enumerate(
                    zip(points, points[1:])
                ):
                    distance = _segment_distance(
                        start,
                        end,
                        node_point,
                        connector_point,
                    )
                    if shared_nodes and distance > 1e-9:
                        continue
                    if distance >= tolerance:
                        continue
                    if distance <= 1e-9 and self._shared_endpoint_contact(
                        start,
                        end,
                        node_point,
                        connector_point,
                        shared_nodes,
                    ):
                        continue
                    raise RouteGenerationError(
                        f"Smoothed route {path.route_id} approaches reserved "
                        f"connector for route {owner} near segment "
                        f"{segment_index}",
                        failure_classification="occupancy_conflict",
                        failure_details={
                            "other_route_id": owner,
                            "candidate_segment": segment_index,
                            "connector_cell": cell,
                            "minimum_clearance": distance,
                            "collision_tolerance": tolerance,
                            "candidate_segment_points": (start, end),
                            "blocking_connector_segment": (
                                node_point,
                                connector_point,
                            ),
                            "candidate_points": tuple(points),
                        },
                    )

    def _other_graph_route_endpoint_cells(self, path: WFCPath) -> set[Cell]:
        route_nodes = {
            node
            for source, target, attributes in self.graph.edges(data=True)
            if _edge_name(attributes) in _EDGE_PORTS
            for node in (source, target)
        }
        return {
            self.graph_node_to_cell(node)
            for node in route_nodes - {path.source, path.target}
        }

    def _shared_graph_endpoint_cells(self, path: WFCPath) -> set[Cell]:
        endpoints = {path.source, path.target}
        shared_nodes = {
            endpoint
            for other in self.routes
            if other.route_id != path.route_id
            for endpoint in endpoints
            if endpoint in (other.source, other.target)
        }
        return {self.graph_node_to_cell(node) for node in shared_nodes}

    def _validate_route_intersections(
        self,
        path: WFCPath,
        points: Sequence[tuple[float, float, float]],
    ) -> None:
        from src.route_diagnostics import _segment_distance

        for other in self.routes:
            if other.route_id == path.route_id:
                continue
            other_points = other.accepted_geometry
            if other_points is None:
                other_points = self._smooth_path_geometry(
                    other,
                    8,
                    other.smoothing_strength,
                )
            shared_nodes = {
                endpoint
                for endpoint in (path.source, path.target)
                if endpoint in (other.source, other.target)
            }
            shared_connector_segments = self._shared_connector_segments(
                path.route_id,
                other.route_id,
            )
            for first_index, (first_start, first_end) in enumerate(
                zip(points, points[1:])
            ):
                for second_index, (second_start, second_end) in enumerate(
                    zip(other_points, other_points[1:])
                ):
                    if _segment_distance(
                        first_start,
                        first_end,
                        second_start,
                        second_end,
                    ) > 1e-9:
                        continue
                    if (
                        self._shared_endpoint_contact(
                            first_start,
                            first_end,
                            second_start,
                            second_end,
                            shared_nodes,
                        )
                        or self._shared_connector_segment_contact(
                            first_start,
                            first_end,
                            second_start,
                            second_end,
                            shared_connector_segments,
                        )
                    ):
                        continue
                    raise RouteGenerationError(
                        f"Smoothed route {path.route_id} intersects route "
                        f"{other.route_id} near segments "
                        f"{first_index}/{second_index}",
                        failure_classification="occupancy_conflict",
                        failure_details={
                            "other_route_id": other.route_id,
                            "candidate_segment": first_index,
                            "other_segment": second_index,
                            "minimum_clearance": 0.0,
                            "candidate_segment_points": (
                                first_start,
                                first_end,
                            ),
                            "other_segment_points": (
                                second_start,
                                second_end,
                            ),
                            "candidate_points": tuple(points),
                        },
                    )

    def _shared_connector_segments(
        self,
        route_id: str,
        other_route_id: str,
    ) -> set[frozenset[tuple[float, float, float]]]:
        segments = set()
        for cell, reservations in self._connector_reservations_by_cell.items():
            other_ids = set(reservations) - {route_id}
            if other_route_id not in other_ids or not self._can_share_landing(
                cell,
                route_id,
                {other_route_id},
            ):
                continue
            reservation = reservations[route_id]
            node = self.graph_node_to_cell(reservation.node)
            connector_point = (
                cell[1] / self.grid_resolution,
                cell[2] / self.grid_resolution,
                cell[0] / self.vertical_grid_resolution,
            )
            node_point = (
                node[1] / self.grid_resolution,
                node[2] / self.grid_resolution,
                node[0] / self.vertical_grid_resolution,
            )
            segments.add(frozenset((node_point, connector_point)))
        return segments

    def _shared_endpoint_contact(
        self,
        first_start: tuple[float, float, float],
        first_end: tuple[float, float, float],
        second_start: tuple[float, float, float],
        second_end: tuple[float, float, float],
        shared_nodes: set[GraphNode],
    ) -> bool:
        for node in shared_nodes:
            cell = self.graph_node_to_cell(node)
            point = (
                cell[1] / self.grid_resolution,
                cell[2] / self.grid_resolution,
                cell[0] / self.vertical_grid_resolution,
            )
            if point not in (first_start, first_end) or point not in (
                second_start,
                second_end,
            ):
                continue
            first_other = first_end if first_start == point else first_start
            second_other = second_end if second_start == point else second_start
            first_vector = tuple(first_other[i] - point[i] for i in range(3))
            second_vector = tuple(second_other[i] - point[i] for i in range(3))
            cross = (
                first_vector[1] * second_vector[2]
                - first_vector[2] * second_vector[1],
                first_vector[2] * second_vector[0]
                - first_vector[0] * second_vector[2],
                first_vector[0] * second_vector[1]
                - first_vector[1] * second_vector[0],
            )
            dot = sum(
                first_vector[index] * second_vector[index]
                for index in range(3)
            )
            if sum(value * value for value in cross) > 1e-18 or dot <= 0:
                return True
        return False

    @staticmethod
    def _shared_connector_segment_contact(
        first_start: tuple[float, float, float],
        first_end: tuple[float, float, float],
        second_start: tuple[float, float, float],
        second_end: tuple[float, float, float],
        allowed_segments: set[frozenset[tuple[float, float, float]]],
    ) -> bool:
        first = frozenset((first_start, first_end))
        second = frozenset((second_start, second_end))
        return first == second and first in allowed_segments

    def plot(self, show_diagnostics: bool = False) -> tuple[Figure, tuple[Axes, Axes]]:
        """Plot the inherited graph and routed inter-floor grid side-by-side."""
        figure = plt.figure(figsize=(14, 7), constrained_layout=True)
        graph_axes = figure.add_subplot(121, projection="3d")
        route_axes = figure.add_subplot(122, projection="3d")
        diagnostics = None
        if show_diagnostics:
            from src.route_diagnostics import diagnose_routes

            diagnostics = diagnose_routes(self)
        self._plot_original_graph(graph_axes)
        self._plot_route_grid(route_axes, diagnostics)
        return figure, (graph_axes, route_axes)

    def _plot_original_graph(self, axes: Axes) -> None:
        edges_seen = set()
        for source, target, attributes in self.graph.edges(data=True):
            edge_name = _edge_name(attributes)
            key = (
                frozenset((source, target))
                if edge_name == "floor"
                else (source, target, edge_name)
            )
            if key in edges_seen:
                continue
            edges_seen.add(key)
            axes.plot(
                [source[0], target[0]],
                [source[1], target[1]],
                [source[2], target[2]],
                color=_EDGE_COLORS.get(edge_name, "#64748b"),
                linewidth=1.0 if edge_name == "floor" else 2.5,
                alpha=0.75,
            )
            if edge_name == "stair":
                for node, prefix in ((source, "S"), (target, "T")):
                    axes.text(
                        node[0], node[1], node[2],
                        f"{prefix}{node[2]}", color=_EDGE_COLORS["stair"],
                        fontsize=8, weight="bold",
                    )
        for node, attributes in self.graph.nodes(data=True):
            node_kind = attributes.get("kind", attributes.get("type", "wall"))
            color = {
                "floor": "#334155",
                "stair_head": "#e58a22",
                "stair_tail": "#f4a261",
                "slide_head": "#277da1",
                "slide_tail": "#4ea8de",
            }.get(node_kind, "#cbd5e1")
            size = 48 if node_kind != "wall" else 10
            axes.scatter(
                [node[0]],
                [node[1]],
                [node[2]],
                color=color,
                s=size,
                alpha=0.9 if node_kind != "wall" else 0.25,
                depthshade=False,
            )
        self._mark_graph_endpoint(axes, self.start_pos, "START", "#2a9d55")
        self._mark_graph_endpoint(axes, self.exit_pos, "EXIT", "#d1495b")
        axes.set_title("Inherited Building graph")
        axes.set_xlabel("Graph x")
        axes.set_ylabel("Graph y")
        axes.set_zlabel("Floor")

    def _plot_route_grid(
        self,
        axes: Axes,
        diagnostics: BuildingRouteDiagnostics | None = None,
    ) -> None:
        floor_count, x_count, y_count = self.grid_shape
        floor, x, y = np.indices((floor_count, x_count, y_count))
        axes.scatter(
            (x.ravel() / self.grid_resolution),
            (y.ravel() / self.grid_resolution),
            (floor.ravel() / self.vertical_grid_resolution),
            color="#cbd5e1",
            s=4,
            alpha=0.12,
            depthshade=False,
        )
        reports = (
            {report.route_id: report for report in diagnostics.routes}
            if diagnostics is not None
            else {}
        )
        if diagnostics is not None:
            reopened_route_ids: set[str] = set()
            failed_route_ids: set[str] = set()
            for event in self.recovery_events:
                for snapshot in event["reopened_route_snapshots"]:
                    route_id = str(snapshot["route_id"])
                    if route_id in reopened_route_ids:
                        continue
                    reopened_route_ids.add(route_id)
                    points = snapshot["points"]
                    axes.plot(
                        [point[0] for point in points],
                        [point[1] for point in points],
                        [point[2] for point in points],
                        color="#64748b",
                        linewidth=1.5,
                        linestyle=":",
                        alpha=0.65,
                    )
                route_id = str(event["failed_route_id"])
                if route_id in failed_route_ids:
                    continue
                failed_route_ids.add(route_id)
                source = event["failed_source"]
                target = event["failed_target"]
                axes.plot(
                    [source[0], target[0]],
                    [source[1], target[1]],
                    [source[2], target[2]],
                    color="#ef476f",
                    linewidth=2.0,
                    linestyle=":",
                    alpha=0.8,
                )
        for path in self.routes:
            report = reports.get(path.route_id)
            points = path.accepted_geometry
            if points is None:
                raise RuntimeError(f"Accepted route {path.route_id} lacks stored geometry")
            color = _EDGE_COLORS[path.kind.value]
            axes.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                [point[2] for point in points],
                color=color,
                linewidth=3.5 if path.recovery_count else 2.5,
                linestyle="--" if path.recovery_count else "-",
            )
            if report is not None:
                for segment in report.problem_segments:
                    if segment + 1 >= len(points):
                        continue
                    axes.plot(
                        [points[segment][0], points[segment + 1][0]],
                        [points[segment][1], points[segment + 1][1]],
                        [points[segment][2], points[segment + 1][2]],
                        color="#e63946",
                        linewidth=5,
                        alpha=0.85,
                    )
            self._mark_graph_endpoint(
                axes, path.source, "H", "#2a9d55"
            )
            self._mark_graph_endpoint(
                axes, path.target, "T", "#d1495b"
            )
        if diagnostics is not None:
            for pair in diagnostics.route_pairs:
                if (
                    not pair.warnings and not pair.logical_errors
                ) or pair.closest_segments is None:
                    continue
                first_report = reports[pair.route_ids[0]]
                second_report = reports[pair.route_ids[1]]
                for report, segment in zip(
                    (first_report, second_report),
                    pair.closest_segments,
                ):
                    points = report.geometry_points
                    if segment + 1 >= len(points):
                        continue
                    axes.plot(
                        [points[segment][0], points[segment + 1][0]],
                        [points[segment][1], points[segment + 1][1]],
                        [points[segment][2], points[segment + 1][2]],
                        color="#f72585",
                        linewidth=6,
                        alpha=0.9,
                    )
        axes.set_title("3D inter-floor WFC routes")
        notes = []
        if any(path.recovery_count for path in self.routes):
            notes.append("dashed = regenerated during recovery")
        if diagnostics is not None:
            notes.extend(
                (
                    "red = curvature/route issue; magenta = close route pair",
                    "gray dotted = pre-recovery discrete route cells",
                    "red dotted = failed endpoint chord, not a generated route",
                    "dashed regenerated routes were accepted after recovery",
                )
            )
        if notes:
            axes.text2D(
                0.02,
                0.98,
                "\n".join(notes),
                transform=axes.transAxes,
                va="top",
                fontsize=8,
            )
        axes.set_xlabel("Graph x")
        axes.set_ylabel("Graph y")
        axes.set_zlabel("Floor")

    @staticmethod
    def _mark_graph_endpoint(
        axes: Axes,
        node: GraphNode,
        label: str,
        color: str,
    ) -> None:
        x, y, floor = node
        axes.scatter([x], [y], [floor], color=color, s=90, edgecolor="white")
        axes.text(x, y, floor + 0.08, label, color=color, weight="bold")

    def _validate_graph_coordinates(self) -> None:
        for node in self.graph.nodes:
            if (
                not isinstance(node, tuple)
                or len(node) != 3
                or any(not isinstance(value, int) for value in node)
            ):
                raise ValueError(
                    f"Building graph node {node!r} is not an integer (x, y, floor)"
                )
            x, y, floor = node
            if not (
                0 <= x < self.length
                and 0 <= y < self.width
                and 0 <= floor < self.floors
            ):
                raise ValueError(
                    f"Building graph node {node!r} is outside declared "
                    "(x, y, floor) dimensions"
                )

    def _validate_forbidden_cells(self) -> None:
        floor_count, x_count, y_count = self.grid_shape
        for cell in self.forbidden_cells:
            if (
                not isinstance(cell, tuple)
                or len(cell) != 3
                or any(not isinstance(value, int) for value in cell)
            ):
                raise ValueError(
                    "Forbidden route cells must be (floor_step, x_step, y_step)"
                )
            if not (
                0 <= cell[0] < floor_count
                and 0 <= cell[1] < x_count
                and 0 <= cell[2] < y_count
            ):
                raise ValueError(f"Forbidden route cell {cell} is outside the grid")


def _edge_name(attributes: Mapping[str, object]) -> str:
    kind = attributes.get("kind")
    edge_type = attributes.get("type")
    if kind is not None and edge_type is not None and kind != edge_type:
        raise ValueError(f"Conflicting graph edge labels {kind!r} and {edge_type!r}")
    value = kind if kind is not None else edge_type
    return value if isinstance(value, str) else ""


def _has_expected_vertical_direction(
    start: Cell,
    end: Cell,
    kind: PortKind,
) -> bool:
    if kind is PortKind.STAIR:
        return end[0] > start[0]
    return end[0] < start[0]


def _spatially_reachable(start: Cell, end: Cell, cells: set[Cell]) -> bool:
    """Check untyped six-neighbor free-space connectivity for recovery ranking."""
    if start not in cells or end not in cells:
        return False
    pending = deque([start])
    visited = {start}
    while pending:
        cell = pending.popleft()
        if cell == end:
            return True
        for offset in _DIRECTION_OFFSETS.values():
            neighbor = tuple(cell[axis] + offset[axis] for axis in range(3))
            if neighbor in cells and neighbor not in visited:
                visited.add(neighbor)
                pending.append(neighbor)
    return False


def _classify_route_failure(message: str) -> str:
    lowered = message.lower()
    if any(
        phrase in lowered
        for phrase in ("curvature radius", "minimum is", "grade", "horizontal travel")
    ):
        return "geometric_rejection"
    if any(
        phrase in lowered
        for phrase in (
            "occupied",
            "occupancy",
            "reserved connector",
            "collision",
            "collides",
            "overlap",
        )
    ):
        return "occupancy_conflict"
    if any(
        phrase in lowered
        for phrase in (
            "contradiction",
            "no solution",
            "no compatible tiles",
            "no possible tiles",
        )
    ):
        return "tile_contradiction"
    if any(
        phrase in lowered
        for phrase in ("cannot connect", "not connected", "no tile-supported")
    ):
        return "unreachable_endpoints"
    if "connector orientation" in lowered:
        return "unreachable_endpoints"
    if "disconnected cells" in lowered or "wrong direction" in lowered:
        return "unreachable_endpoints"
    return "exhausted_retries"


def _can_travel(
    direction: Direction,
    kind: PortKind,
) -> bool:
    if direction in _HORIZONTAL_DIRECTIONS:
        return True
    if kind is PortKind.STAIR:
        return direction is Direction.U
    return direction is Direction.D


def _opposite(direction: Direction) -> Direction:
    return {
        Direction.N: Direction.S,
        Direction.E: Direction.W,
        Direction.S: Direction.N,
        Direction.W: Direction.E,
        Direction.U: Direction.D,
        Direction.D: Direction.U,
    }[direction]


def _turning_radii(
    points: Sequence[tuple[float, float, float]],
) -> tuple[float, ...]:
    radii = []
    for first, middle, last in zip(points, points[1:], points[2:]):
        first_length = hypot(middle[0] - first[0], middle[1] - first[1])
        second_length = hypot(last[0] - middle[0], last[1] - middle[1])
        chord_length = hypot(last[0] - first[0], last[1] - first[1])
        cross = abs(
            (middle[0] - first[0]) * (last[1] - middle[1])
            - (middle[1] - first[1]) * (last[0] - middle[0])
        )
        if min(first_length, second_length, chord_length) <= 1e-12:
            continue
        radii.append(
            first_length * second_length * chord_length / (2 * cross)
            if cross > 1e-12
            else float("inf")
        )
    return tuple(radii)


def _remove_duplicate_points(
    points: Sequence[tuple[float, float, float]],
) -> list[tuple[float, float, float]]:
    unique = []
    for point in points:
        if not unique or any(
            abs(point[axis] - unique[-1][axis]) > 1e-12
            for axis in range(3)
        ):
            unique.append(point)
    return unique


def _simplify_xy_controls(
    points: Sequence[tuple[float, float, float]],
) -> tuple[tuple[float, float], ...]:
    controls: list[tuple[float, float]] = []
    for point in points:
        xy = (point[0], point[1])
        if not controls or hypot(
            xy[0] - controls[-1][0], xy[1] - controls[-1][1]
        ) > 1e-12:
            controls.append(xy)
            while len(controls) >= 3:
                first, middle, last = controls[-3:]
                incoming = (middle[0] - first[0], middle[1] - first[1])
                outgoing = (last[0] - middle[0], last[1] - middle[1])
                cross = incoming[0] * outgoing[1] - incoming[1] * outgoing[0]
                dot = incoming[0] * outgoing[0] + incoming[1] * outgoing[1]
                if abs(cross) > 1e-12 or dot < 0:
                    break
                controls.pop(-2)
    return tuple(controls)
