"""Opt-in bounded regeneration of complete graph-constrained buildings."""

from dataclasses import dataclass
from time import perf_counter
from collections.abc import Callable, Mapping

from src.route_diagnostics import diagnose_routes
from src.wfc_building import RouteGenerationError, WFCBuilding


@dataclass(frozen=True)
class BuildingAttempt:
    candidate_seed: int
    stage: str
    classification: str
    error: str
    graph_seconds: float
    routing_seconds: float
    wfc_attempts: int
    failure_details: dict[str, object]


@dataclass(frozen=True)
class BuildingGenerationResult:
    building: WFCBuilding
    attempts: tuple[BuildingAttempt, ...]


class BuildingRegenerationError(RouteGenerationError):
    """All permitted whole-building candidates failed geometric validation."""

    def __init__(self, attempts: tuple[BuildingAttempt, ...]) -> None:
        self.building_attempts = attempts
        last = attempts[-1]
        super().__init__(
            f"Whole-building regeneration exhausted after {len(attempts)} "
            f"candidates; last {last.stage} failure (seed {last.candidate_seed}): "
            f"{last.error}",
            attempts=sum(attempt.wfc_attempts for attempt in attempts),
            failure_classification=last.classification,
        )


def _candidate_seed(seed: int, index: int) -> int:
    if index == 0:
        return seed
    return (seed + index * 0x9E3779B9) % (2**32)


def generate_building(
    *,
    seed: int = 0,
    max_building_retries: int = 0,
    trace_callback: Callable[[Mapping[str, object]], None] | None = None,
    **options,
) -> BuildingGenerationResult:
    """Generate a validated building, optionally retrying geometric failures.

    ``max_building_retries`` counts additional complete candidates. Graph
    generation errors and non-geometric route failures propagate without retry.
    """
    if (not isinstance(max_building_retries, int)
            or isinstance(max_building_retries, bool)
            or not 0 <= max_building_retries < 2**32):
        raise ValueError("max_building_retries must be a non-negative integer below 2**32")
    if "generate_routes" in options:
        raise ValueError("generate_building always generates routes")
    if "forbidden_cells" in options:
        options["forbidden_cells"] = tuple(options["forbidden_cells"])
    if trace_callback is not None:
        options["trace_callback"] = trace_callback

    attempts = []
    for index in range(max_building_retries + 1):
        candidate_seed = _candidate_seed(seed, index)
        if trace_callback is not None:
            trace_callback({
                "event": "building_attempt_start",
                "building_attempt": index + 1,
                "candidate_seed": candidate_seed,
            })
        graph_start = perf_counter()
        try:
            building = WFCBuilding(
                seed=candidate_seed, generate_routes=False, **options
            )
        except Exception as error:
            if max_building_retries:
                attempts.append(BuildingAttempt(
                    candidate_seed, "graph-generation", "graph-generation",
                    str(error), perf_counter() - graph_start, 0.0, 0, {},
                ))
                error.building_attempts = tuple(attempts)
            if trace_callback is not None:
                trace_callback({
                    "event": "building_attempt_failure",
                    "building_attempt": index + 1,
                    "candidate_seed": candidate_seed,
                    "stage": "graph-generation",
                    "classification": "graph-generation",
                    "error": str(error),
                    "wfc_attempts": 0,
                })
            raise
        graph_seconds = perf_counter() - graph_start
        route_start = perf_counter()
        try:
            building.generate_routes()
        except RouteGenerationError as error:
            failure = building.route_failures[-1] if building.route_failures else {}
            kind = failure.get("route_type")
            stage = (
                "direct-stair" if kind == "stair" else
                "slide-routing" if kind == "slide" else "route-generation"
            )
            geometric = (
                error.failure_classification in {
                    "geometry_collision", "geometric_rejection", "occupancy_conflict"
                }
                or bool(error.failure_details.get("geometry_rejections"))
            )
            attempts.append(BuildingAttempt(
                candidate_seed, stage, error.failure_classification, str(error),
                graph_seconds, perf_counter() - route_start,
                building.total_wfc_attempts, dict(error.failure_details),
            ))
            if trace_callback is not None:
                trace_callback({
                    "event": "building_attempt_failure",
                    "building_attempt": index + 1,
                    "candidate_seed": candidate_seed,
                    "stage": stage,
                    "classification": error.failure_classification,
                    "error": str(error),
                    "wfc_attempts": building.total_wfc_attempts,
                    "route_count": len(building.routes),
                })
            if not geometric or max_building_retries == 0:
                if max_building_retries:
                    error.building_attempts = tuple(attempts)
                raise
            if index == max_building_retries:
                raise BuildingRegenerationError(tuple(attempts)) from error
            continue

        try:
            if max_building_retries:
                for route in building.routes:
                    building.validate_accepted_geometry(route)
        except RouteGenerationError as error:
            attempts.append(BuildingAttempt(
                candidate_seed, "geometry-validation", error.failure_classification,
                str(error), graph_seconds, perf_counter() - route_start,
                building.total_wfc_attempts, dict(error.failure_details),
            ))
            if trace_callback is not None:
                trace_callback({
                    "event": "building_attempt_failure",
                    "building_attempt": index + 1,
                    "candidate_seed": candidate_seed,
                    "stage": "geometry-validation",
                    "classification": error.failure_classification,
                    "error": str(error),
                    "wfc_attempts": building.total_wfc_attempts,
                })
            if (max_building_retries == 0 or error.failure_classification not in {
                "geometry_collision", "geometric_rejection", "occupancy_conflict"
            }):
                if max_building_retries:
                    error.building_attempts = tuple(attempts)
                raise
            if index == max_building_retries:
                raise BuildingRegenerationError(tuple(attempts)) from error
            continue
        if max_building_retries:
            diagnostics = diagnose_routes(building)
            if diagnostics.logical_errors:
                raise RuntimeError(
                    "Generated building failed logical validation:\n"
                    + "\n".join(diagnostics.logical_errors)
                )
        attempts.append(BuildingAttempt(
            candidate_seed, "accepted", "", "", graph_seconds,
            perf_counter() - route_start, building.total_wfc_attempts, {},
        ))
        if trace_callback is not None:
            trace_callback({
                "event": "building_attempt_accepted",
                "building_attempt": index + 1,
                "candidate_seed": candidate_seed,
                "wfc_attempts": building.total_wfc_attempts,
                "route_count": len(building.routes),
                "building_retries_used": index,
                "global_route_recovery_attempts": building.recovery_attempts,
                "individual_route_retries": building.routes_retried,
            })
        building.requested_seed = seed
        building.accepted_seed = candidate_seed
        building.building_retry_count = index
        return BuildingGenerationResult(building, tuple(attempts))
    raise AssertionError("unreachable whole-building retry state")
