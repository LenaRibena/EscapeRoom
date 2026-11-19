"""Logical and geometric diagnostics for graph-constrained WFC routes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import acos, inf, isfinite, pi, sqrt
from typing import Protocol

from src.wfc_tiles import Cell, PortKind

Point3D = tuple[float, float, float]


class RouteLike(Protocol):
    route_id: str
    kind: PortKind
    source: tuple[int, int, int]
    target: tuple[int, int, int]
    cells: tuple[Cell, ...]

    def world_points(self, resolution: int) -> tuple[Point3D, ...]: ...


class BuildingLike(Protocol):
    grid_resolution: int
    routes: Sequence[RouteLike]

    def graph_node_to_cell(self, node: tuple[int, int, int]) -> Cell: ...

    def smooth_path(self, path: RouteLike) -> tuple[Point3D, ...]: ...

    def validate_accepted_geometry(self, path: RouteLike) -> tuple[Point3D, ...]: ...


@dataclass(frozen=True)
class DiagnosticThresholds:
    """Heuristic thresholds in graph-coordinate units, not Blender dimensions."""

    minimum_slide_slope: float = 0.05
    maximum_slide_slope: float = 1.5
    maximum_turn_angle_degrees: float = 100.0
    minimum_curvature_radius: float = 0.02
    minimum_clearance: float = 0.25
    shared_endpoint_exclusion_radius: float = 0.25

    def __post_init__(self) -> None:
        if self.minimum_slide_slope < 0:
            raise ValueError("minimum_slide_slope cannot be negative")
        if self.maximum_slide_slope <= 0:
            raise ValueError("maximum_slide_slope must be positive")
        if self.minimum_slide_slope > self.maximum_slide_slope:
            raise ValueError("minimum_slide_slope cannot exceed maximum_slide_slope")
        if not 0 < self.maximum_turn_angle_degrees <= 180:
            raise ValueError("maximum_turn_angle_degrees must be in (0, 180]")
        if (
            not isfinite(self.minimum_curvature_radius)
            or self.minimum_curvature_radius < 0
        ):
            raise ValueError("minimum_curvature_radius must be finite and non-negative")
        if self.minimum_clearance < 0:
            raise ValueError("minimum_clearance cannot be negative")
        if self.shared_endpoint_exclusion_radius < 0:
            raise ValueError("shared_endpoint_exclusion_radius cannot be negative")


@dataclass(frozen=True)
class RouteDiagnostic:
    route_id: str
    kind: PortKind
    endpoint_errors: tuple[float, float] | None
    continuity_valid: bool
    direction_valid: bool
    logical_errors: tuple[str, ...]
    warnings: tuple[str, ...]
    minimum_slope: float | None
    maximum_slope: float | None
    turn_angles_degrees: tuple[float, ...]
    maximum_curvature: float | None
    minimum_curvature_radius: float | None
    curvature_radii: tuple[float, ...]
    problem_segments: tuple[int, ...]
    geometry_points: tuple[Point3D, ...]


@dataclass(frozen=True)
class RoutePairDiagnostic:
    route_ids: tuple[str, str]
    overlapping_cells: tuple[Cell, ...]
    minimum_clearance: float | None
    closest_segments: tuple[int, int] | None
    logical_errors: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class BuildingRouteDiagnostics:
    routes: tuple[RouteDiagnostic, ...]
    route_pairs: tuple[RoutePairDiagnostic, ...]

    @property
    def logical_errors(self) -> tuple[str, ...]:
        return tuple(
            f"{route.route_id}: {error}"
            for route in self.routes
            for error in route.logical_errors
        ) + tuple(
            f"{pair.route_ids[0]} / {pair.route_ids[1]}: {error}"
            for pair in self.route_pairs
            for error in pair.logical_errors
        )

    @property
    def warnings(self) -> tuple[str, ...]:
        return tuple(
            f"{route.route_id}: {warning}"
            for route in self.routes
            for warning in route.warnings
        ) + tuple(
            f"{pair.route_ids[0]} / {pair.route_ids[1]}: {warning}"
            for pair in self.route_pairs
            for warning in pair.warnings
        )


def diagnose_routes(
    building: BuildingLike,
    routes: Sequence[RouteLike] | None = None,
    thresholds: DiagnosticThresholds = DiagnosticThresholds(),
) -> BuildingRouteDiagnostics:
    """Report graph/route correctness separately from geometric warnings."""
    route_list = tuple(building.routes if routes is None else routes)
    reports = []
    geometries: dict[str, tuple[Point3D, ...]] = {}
    for route in route_list:
        report = _diagnose_route(building, route, thresholds)
        reports.append(report)
        geometries[route.route_id] = report.geometry_points

    pairs = tuple(
        _diagnose_pair(
            building,
            first,
            second,
            geometries[first.route_id],
            geometries[second.route_id],
            thresholds,
        )
        for index, first in enumerate(route_list)
        for second in route_list[index + 1 :]
    )
    return BuildingRouteDiagnostics(tuple(reports), pairs)


def _diagnose_route(
    building: BuildingLike,
    route: RouteLike,
    thresholds: DiagnosticThresholds,
) -> RouteDiagnostic:
    errors = []
    warnings = []
    problem_segments: set[int] = set()
    expected_start = tuple(float(value) for value in route.source)
    expected_end = tuple(float(value) for value in route.target)
    discrete_points = route.world_points(building.grid_resolution)
    endpoint_errors = None

    if getattr(route, "direct_stair", False):
        geometry = getattr(route, "accepted_geometry", None)
        if geometry is None:
            raise RuntimeError(f"Accepted route {route.route_id} lacks stored geometry")
        endpoint_errors = (
            _point_distance(geometry[0], expected_start),
            _point_distance(geometry[-1], expected_end),
        )
        continuity_valid = len(geometry) == 2
        direction_valid = route.kind is PortKind.STAIR and route.target[2] > route.source[2]
        if any(error > 1e-9 for error in endpoint_errors):
            errors.append("route endpoints do not match its graph-defined source and target")
        if not continuity_valid:
            errors.append("direct stair must consist of one straight segment")
        if not direction_valid:
            errors.append("direct stair must ascend")
    elif not route.cells:
        errors.append("route has no cells")
        continuity_valid = False
        direction_valid = False
    else:
        actual_start = discrete_points[0]
        actual_end = discrete_points[-1]
        endpoint_errors = (
            _point_distance(actual_start, expected_start),
            _point_distance(actual_end, expected_end),
        )
        if any(error > 1e-9 for error in endpoint_errors):
            errors.append(
                "route endpoints do not match its graph-defined source and target"
            )

        continuity_errors = [
            index
            for index, (first, second) in enumerate(
                zip(route.cells, route.cells[1:])
            )
            if sum(abs(first[axis] - second[axis]) for axis in range(3)) != 1
        ]
        continuity_valid = not continuity_errors
        if continuity_errors:
            errors.append("route contains non-adjacent consecutive grid cells")
            problem_segments.update(continuity_errors)

        floor_deltas = [
            route.cells[index + 1][0] - route.cells[index][0]
            for index in range(len(route.cells) - 1)
        ]
        direction_valid = (
            all(delta >= 0 for delta in floor_deltas)
            if route.kind is PortKind.STAIR
            else all(delta <= 0 for delta in floor_deltas)
        )
        if not direction_valid:
            errors.append(
                "stair route must not descend"
                if route.kind is PortKind.STAIR
                else "slide route must not ascend"
            )
            problem_segments.update(
                index
                for index, delta in enumerate(floor_deltas)
                if (
                    delta < 0
                    if route.kind is PortKind.STAIR
                    else delta > 0
                )
            )

    from src.wfc_building import RouteGenerationError

    try:
        if hasattr(building, "validate_accepted_geometry") and (
            getattr(route, "accepted_geometry", None) is not None
            or any(route is accepted for accepted in building.routes)
        ):
            geometry_points = building.validate_accepted_geometry(route)
        else:
            geometry_points = building.smooth_path(route)
    except RouteGenerationError as error:
        geometry_points = getattr(route, "accepted_geometry", None)
        if geometry_points is None:
            geometry_points = discrete_points
        errors.append(f"smoothed geometry is invalid: {error}")
        problem_segments.update(range(max(0, len(geometry_points) - 1)))

    slopes = (
        _slide_slopes(geometry_points)
        if route.kind is PortKind.SLIDE and len(geometry_points) > 1
        else ()
    )
    minimum_slope = min(slopes) if slopes else None
    maximum_slope = max(slopes) if slopes else None
    if minimum_slope is not None and minimum_slope < thresholds.minimum_slide_slope:
        warnings.append(
            f"slide slope {minimum_slope:.3g} is below "
            f"{thresholds.minimum_slide_slope:.3g}"
        )
        problem_segments.update(
            index
            for index, slope in enumerate(slopes)
            if slope < thresholds.minimum_slide_slope
        )
    if maximum_slope is not None and maximum_slope > thresholds.maximum_slide_slope:
        warnings.append(
            f"slide slope {maximum_slope:.3g} exceeds "
            f"{thresholds.maximum_slide_slope:.3g}"
        )
        problem_segments.update(
            index
            for index, slope in enumerate(slopes)
            if slope > thresholds.maximum_slide_slope
        )

    turn_angles, curvatures = _turn_metrics(geometry_points)
    curvature_radii = _curvature_radii(geometry_points)
    for index, angle in enumerate(turn_angles):
        if angle > thresholds.maximum_turn_angle_degrees:
            problem_segments.update((index, index + 1))
    if any(angle > thresholds.maximum_turn_angle_degrees for angle in turn_angles):
        warnings.append(
            "route has a turn sharper than "
            f"{thresholds.maximum_turn_angle_degrees:.1f} degrees"
        )
    minimum_curvature_radius = (
        min(curvature_radii) if curvature_radii else None
    )
    if (
        route.kind is PortKind.SLIDE
        and thresholds.minimum_curvature_radius > 0
        and any(
            radius < thresholds.minimum_curvature_radius
            for radius in curvature_radii
        )
    ):
        warnings.append(
            "minimum curvature radius "
            f"{minimum_curvature_radius:.3g} is below "
            f"{thresholds.minimum_curvature_radius:.3g}"
        )
        problem_segments.update(
            segment
            for index, radius in enumerate(curvature_radii)
            if radius < thresholds.minimum_curvature_radius
            for segment in (index, index + 1)
        )

    return RouteDiagnostic(
        route.route_id,
        route.kind,
        endpoint_errors,
        continuity_valid,
        direction_valid,
        tuple(errors),
        tuple(warnings),
        minimum_slope,
        maximum_slope,
        turn_angles,
        max(curvatures) if curvatures else None,
        minimum_curvature_radius,
        curvature_radii,
        tuple(sorted(problem_segments)),
        tuple(geometry_points),
    )


def _diagnose_pair(
    building: BuildingLike,
    first: RouteLike,
    second: RouteLike,
    first_points: tuple[Point3D, ...],
    second_points: tuple[Point3D, ...],
    thresholds: DiagnosticThresholds,
) -> RoutePairDiagnostic:
    shared_nodes = {first.source, first.target} & {second.source, second.target}
    shared_cells = {
        building.graph_node_to_cell(node)
        for node in shared_nodes
    }
    overlaps = tuple(sorted(set(first.cells) & set(second.cells)))
    errors = []
    warnings = []
    landing_checker = getattr(building, "_can_share_landing", None)
    unexpected_overlaps = tuple(
        cell
        for cell in overlaps
        if cell not in shared_cells
        and not (
            callable(landing_checker)
            and landing_checker(cell, first.route_id, {second.route_id})
            and landing_checker(cell, second.route_id, {first.route_id})
        )
    )
    if unexpected_overlaps:
        errors.append(f"routes overlap at non-shared cells {unexpected_overlaps}")

    clearance, closest = _polyline_clearance(
        first_points,
        second_points,
        tuple(tuple(float(value) for value in node) for node in shared_nodes),
        thresholds.shared_endpoint_exclusion_radius,
    )
    if clearance is not None and clearance < thresholds.minimum_clearance:
        warnings.append(
            f"minimum route clearance {clearance:.3g} is below "
            f"{thresholds.minimum_clearance:.3g}"
        )
    return RoutePairDiagnostic(
        (first.route_id, second.route_id),
        overlaps,
        clearance,
        closest,
        tuple(errors),
        tuple(warnings),
    )


def _slide_slopes(points: Sequence[Point3D]) -> tuple[float, ...]:
    slopes = []
    for first, second in zip(points, points[1:]):
        horizontal = sqrt((second[0] - first[0]) ** 2 + (second[1] - first[1]) ** 2)
        vertical = abs(second[2] - first[2])
        slopes.append(vertical / horizontal if horizontal else (inf if vertical else 0.0))
    return tuple(slopes)


def _turn_metrics(
    points: Sequence[Point3D],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    angles = []
    curvatures = []
    segments = [
        tuple(second[axis] - first[axis] for axis in range(3))
        for first, second in zip(points, points[1:])
    ]
    lengths = [_vector_length(segment) for segment in segments]
    for index, (first, second) in enumerate(zip(segments, segments[1:])):
        if lengths[index] == 0 or lengths[index + 1] == 0:
            angles.append(180.0)
            curvatures.append(inf)
            continue
        cosine = sum(first[axis] * second[axis] for axis in range(3))
        cosine /= lengths[index] * lengths[index + 1]
        angle_radians = acos(min(1.0, max(-1.0, cosine)))
        angles.append(angle_radians * 180 / pi)
        radius = _discrete_curvature_radius(points, index + 1)
        curvatures.append(0.0 if radius == inf else (inf if radius == 0 else 1 / radius))
    return tuple(angles), tuple(curvatures)


def _curvature_radii(points: Sequence[Point3D]) -> tuple[float, ...]:
    return tuple(
        _discrete_curvature_radius(points, center)
        for center in range(1, len(points) - 1)
    )


def _discrete_curvature_radius(
    points: Sequence[Point3D],
    center: int,
) -> float:
    middle = points[center]
    window = min(2, center, len(points) - center - 1)
    first = points[center - window]
    last = points[center + window]
    incoming = tuple(middle[axis] - first[axis] for axis in range(3))
    outgoing = tuple(last[axis] - middle[axis] for axis in range(3))
    incoming_length = _vector_length(incoming)
    outgoing_length = _vector_length(outgoing)
    if min(incoming_length, outgoing_length) <= 1e-12:
        return 0.0
    cosine = sum(
        incoming[axis] * outgoing[axis] for axis in range(3)
    ) / (incoming_length * outgoing_length)
    angle = acos(min(1.0, max(-1.0, cosine)))
    if angle <= 1e-9:
        return inf
    return (incoming_length + outgoing_length) / (2 * angle)


def _polyline_clearance(
    first: Sequence[Point3D],
    second: Sequence[Point3D],
    shared_endpoints: Sequence[Point3D],
    exclusion_radius: float,
) -> tuple[float | None, tuple[int, int] | None]:
    minimum = inf
    closest = None
    for first_index, (first_start, first_end) in enumerate(zip(first, first[1:])):
        if _near_shared_endpoint(
            first_start, first_end, shared_endpoints, exclusion_radius
        ):
            continue
        for second_index, (second_start, second_end) in enumerate(
            zip(second, second[1:])
        ):
            if _near_shared_endpoint(
                second_start, second_end, shared_endpoints, exclusion_radius
            ):
                continue
            distance = _segment_distance(
                first_start,
                first_end,
                second_start,
                second_end,
            )
            if distance < minimum:
                minimum = distance
                closest = (first_index, second_index)
    return (minimum, closest) if closest is not None else (None, None)


def _near_shared_endpoint(
    start: Point3D,
    end: Point3D,
    endpoints: Sequence[Point3D],
    radius: float,
) -> bool:
    return any(
        min(_point_distance(start, endpoint), _point_distance(end, endpoint))
        <= radius
        for endpoint in endpoints
    )


def _segment_distance(
    first_start: Point3D,
    first_end: Point3D,
    second_start: Point3D,
    second_end: Point3D,
) -> float:
    first = _subtract(first_end, first_start)
    second = _subtract(second_end, second_start)
    between = _subtract(first_start, second_start)
    first_length = _dot(first, first)
    second_length = _dot(second, second)
    cross = _dot(first, second)
    first_projection = _dot(first, between)
    second_projection = _dot(second, between)

    if first_length <= 1e-18 and second_length <= 1e-18:
        return _vector_length(between)
    if first_length <= 1e-18:
        return _point_segment_distance(first_start, second_start, second_end)
    if second_length <= 1e-18:
        return _point_segment_distance(second_start, first_start, first_end)

    denominator = first_length * second_length - cross * cross
    numerator_first = 0.0
    denominator_first = denominator
    numerator_second = 0.0
    denominator_second = denominator
    if denominator <= 1e-18:
        numerator_first = 0.0
        denominator_first = 1.0
        numerator_second = second_projection
        denominator_second = second_length
    else:
        numerator_first = cross * second_projection - first_projection * second_length
        numerator_second = first_length * second_projection - cross * first_projection
        if numerator_first < 0:
            numerator_first = 0.0
            numerator_second = second_projection
            denominator_second = second_length
        elif numerator_first > denominator_first:
            numerator_first = denominator_first
            numerator_second = second_projection + cross
            denominator_second = second_length

    if numerator_second < 0:
        numerator_second = 0.0
        if -first_projection < 0:
            numerator_first = 0.0
        elif -first_projection > first_length:
            numerator_first = denominator_first
        else:
            numerator_first = -first_projection
            denominator_first = first_length
    elif numerator_second > denominator_second:
        numerator_second = denominator_second
        if -first_projection + cross < 0:
            numerator_first = 0.0
        elif -first_projection + cross > first_length:
            numerator_first = denominator_first
        else:
            numerator_first = -first_projection + cross
            denominator_first = first_length

    first_fraction = numerator_first / denominator_first if numerator_first else 0.0
    second_fraction = (
        numerator_second / denominator_second if numerator_second else 0.0
    )
    difference = tuple(
        between[axis]
        + first_fraction * first[axis]
        - second_fraction * second[axis]
        for axis in range(3)
    )
    return _vector_length(difference)


def _point_segment_distance(
    point: Point3D,
    start: Point3D,
    end: Point3D,
) -> float:
    segment = _subtract(end, start)
    denominator = _dot(segment, segment)
    if denominator == 0:
        return _point_distance(point, start)
    fraction = min(
        1.0,
        max(0.0, _dot(_subtract(point, start), segment) / denominator),
    )
    closest = tuple(start[axis] + fraction * segment[axis] for axis in range(3))
    return _point_distance(point, closest)


def _subtract(first: Point3D, second: Point3D) -> Point3D:
    return tuple(first[index] - second[index] for index in range(3))


def _dot(first: Point3D, second: Point3D) -> float:
    return sum(first[index] * second[index] for index in range(3))


def _vector_length(vector: Point3D) -> float:
    return sqrt(_dot(vector, vector))


def _point_distance(first: Point3D, second: Point3D) -> float:
    return _vector_length(_subtract(first, second))
