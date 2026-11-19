"""Measure WFC route-generation success, retries, lengths, and runtime."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Sequence

import networkx as nx
import matplotlib.pyplot as plt

from src.route_diagnostics import DiagnosticThresholds, diagnose_routes
from src.building_generation import generate_building
from src.graph import GraphGenerationError
from src.wfc_building import RouteGenerationError, WFCBuilding


@dataclass(frozen=True)
class ExperimentResult:
    seed: int
    grid_resolution: int
    success: bool
    recovery_strategy: str
    minimum_curvature_radius: float
    tile_aware_connectivity: bool
    endpoint_frontier: bool
    candidate_lookahead: bool
    graph_feasibility: str
    discrete_wfc_feasibility: str
    smoothed_geometry_feasibility: str
    first_route_success: bool | None
    route_count: int
    wfc_attempts: int
    generation_seconds: float
    graph_seconds: float
    routing_seconds: float
    failure_stage: str
    recovery_attempts: int
    routes_retried: int
    logical_errors: int
    geometry_warnings: int
    route_failure_details: tuple[dict[str, object], ...]
    recovery_details: tuple[dict[str, object], ...]
    route_metrics: tuple[dict[str, object], ...]
    error: str
    accepted_seed: int | None
    building_retry_count: int
    building_attempts: tuple[dict[str, object], ...]


def run_experiments(
    seeds: Sequence[int],
    grid_resolutions: Sequence[int],
    *,
    floors: int = 5,
    width: int = 5,
    length: int = 5,
    difficulty: int = 2,
    max_graph_attempts: int = 100,
    max_building_retries: int = 0,
    max_route_attempts: int = 30,
    max_route_backtrack_depth: int = 2,
    max_route_recovery_attempts: int = 1,
    recovery_strategy: str = "recency",
    minimum_curvature_radius: float = 0.02,
    tile_aware_connectivity: bool = False,
    endpoint_frontier: bool = True,
    candidate_lookahead: bool = False,
) -> tuple[ExperimentResult, ...]:
    """Run each requested deterministic seed/resolution combination."""
    results = []
    for grid_resolution in grid_resolutions:
        if grid_resolution < 1:
            raise ValueError("grid resolutions must be positive integers")
        for seed in seeds:
            started = perf_counter()
            building = None
            error_message = ""
            failure_stage = ""
            graph_feasibility = "not_tested"
            discrete_feasibility = "not_tested"
            geometry_feasibility = "not_tested"
            failed_attempts = 0
            success = False
            graph_finished = None
            building_attempts = ()
            try:
                options = dict(
                    floors=floors,
                    width=width,
                    length=length,
                    difficulty=difficulty,
                    max_graph_attempts=max_graph_attempts,
                    grid_resolution=grid_resolution,
                    seed=seed,
                    max_route_attempts=max_route_attempts,
                    max_route_backtrack_depth=max_route_backtrack_depth,
                    max_route_recovery_attempts=max_route_recovery_attempts,
                    recovery_strategy=recovery_strategy,
                    minimum_curvature_radius=minimum_curvature_radius,
                    tile_aware_connectivity=tile_aware_connectivity,
                    endpoint_frontier=endpoint_frontier,
                    candidate_lookahead=candidate_lookahead,
                )
                if max_building_retries:
                    result = generate_building(
                        max_building_retries=max_building_retries, **options
                    )
                    building = result.building
                    building_attempts = result.attempts
                    graph_finished = started + sum(a.graph_seconds for a in building_attempts)
                else:
                    building = WFCBuilding(generate_routes=False, **options)
                    graph_finished = perf_counter()
                    building.generate_routes()
                graph_feasibility = "feasible"
                success = True
                discrete_feasibility = "solution_found"
                geometry_feasibility = "accepted"
            except RouteGenerationError as error:
                building_attempts = getattr(error, "building_attempts", ())
                direct_stair_failure = bool(
                    (building_attempts and building_attempts[-1].stage == "direct-stair")
                    or (building is not None and building.route_failures
                        and building.route_failures[-1]["route_type"] == "stair")
                )
                failure_stage = (
                    "direct stair geometry" if direct_stair_failure else
                    "geometric validation" if building_attempts and
                    building_attempts[-1].stage == "geometry-validation" else
                    "WFC routing"
                )
                error_message = str(error)
                failed_attempts = (
                    sum(a.wfc_attempts for a in building_attempts)
                    if building_attempts else error.attempts
                )
                graph_feasibility = "feasible"
                failure_details = (
                    building_attempts[-1].failure_details if building_attempts else
                    building.route_failures[-1].get("failure_details", {})
                    if building is not None and building.route_failures
                    else error.failure_details
                )
                if building_attempts:
                    graph_finished = started + sum(a.graph_seconds for a in building_attempts)
                discrete_candidates = int(
                    failure_details.get("discrete_route_candidates", 0)
                )
                geometry_rejections = int(
                    failure_details.get("geometry_rejections", 0)
                )
                if direct_stair_failure:
                    geometry_feasibility = "direct_stair_rejected"
                elif discrete_candidates:
                    discrete_feasibility = "solution_candidate_found"
                    geometry_feasibility = (
                        "all_candidates_rejected"
                        if geometry_rejections == discrete_candidates
                        else "candidate_rejections"
                    )
                else:
                    discrete_feasibility = (
                        "no_solution_found_within_attempt_budget"
                    )
            except nx.NetworkXNoPath as error:
                failure_stage = "graph generation"
                error_message = f"building graph generation failed: {error}"
                graph_feasibility = "no_solution_found"
            except GraphGenerationError as error:
                building_attempts = getattr(error, "building_attempts", ())
                failure_stage = "graph generation"
                error_message = str(error)
                graph_feasibility = "no_solution_found_within_attempt_budget"
            duration = perf_counter() - started
            graph_seconds = (graph_finished - started) if graph_finished is not None else duration
            routing_seconds = duration - graph_seconds if graph_finished is not None else 0.0
            if failure_stage == "graph generation" and building_attempts:
                routing_seconds = sum(a.routing_seconds for a in building_attempts)
                graph_seconds = duration - routing_seconds
            routes = tuple(building.routes) if building is not None else ()
            route_metrics = tuple(
                {
                    "route_id": route.route_id,
                    "kind": route.kind.value,
                    "cell_count": len(route.cells),
                    "attempts": route.attempts,
                    "recovery_count": route.recovery_count,
                    "geometry": "direct_stair" if route.direct_stair else "wfc",
                    "rejected_route_signatures": route.rejected_route_signatures,
                    "retry_exclusion": route.retry_exclusion,
                }
                for route in routes
            )
            diagnostics = (
                diagnose_routes(
                    building,
                    thresholds=DiagnosticThresholds(
                        minimum_curvature_radius=minimum_curvature_radius
                    ),
                )
                if success and building is not None
                else None
            )
            total_attempts = (
                sum(a.wfc_attempts for a in building_attempts) if building_attempts
                else building.total_wfc_attempts if building is not None
                else failed_attempts
            )
            failures = (
                tuple(building.route_failures) if building is not None else ()
            )
            first_route_success = (
                None
                if building is None or failure_stage in ("graph generation", "direct stair geometry")
                else (
                    any(route.kind.value == "slide" for route in routes)
                    if success
                    else any(route.kind.value == "slide" for route in routes)
                )
            )
            results.append(
                ExperimentResult(
                    seed=seed,
                    grid_resolution=grid_resolution,
                    success=success,
                    recovery_strategy=recovery_strategy,
                    minimum_curvature_radius=minimum_curvature_radius,
                    tile_aware_connectivity=tile_aware_connectivity,
                    endpoint_frontier=endpoint_frontier,
                    candidate_lookahead=candidate_lookahead,
                    graph_feasibility=graph_feasibility,
                    discrete_wfc_feasibility=discrete_feasibility,
                    smoothed_geometry_feasibility=geometry_feasibility,
                    first_route_success=first_route_success,
                    route_count=len(routes),
                    wfc_attempts=total_attempts,
                    generation_seconds=duration,
                    graph_seconds=graph_seconds,
                    routing_seconds=routing_seconds,
                    failure_stage=failure_stage,
                    recovery_attempts=(
                        building.recovery_attempts if building is not None else 0
                    ),
                    routes_retried=(
                        building.routes_retried if building is not None else 0
                    ),
                    logical_errors=(
                        len(diagnostics.logical_errors) if diagnostics else 0
                    ),
                    geometry_warnings=(
                        len(diagnostics.warnings) if diagnostics else 0
                    ),
                    route_failure_details=failures,
                    recovery_details=(
                        tuple(building.recovery_events) if building is not None else ()
                    ),
                    route_metrics=route_metrics,
                    error=error_message,
                    accepted_seed=building.accepted_seed if success and building is not None else None,
                    building_retry_count=max(0, len(building_attempts) - 1),
                    building_attempts=tuple(asdict(a) for a in building_attempts),
                )
            )
    return tuple(results)


def _write_csv(path: Path, results: Sequence[ExperimentResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=(
                "seed",
                "grid_resolution",
                "success",
                "recovery_strategy",
                "minimum_curvature_radius",
                "tile_aware_connectivity",
                "endpoint_frontier",
                "candidate_lookahead",
                "graph_feasibility",
                "discrete_wfc_feasibility",
                "smoothed_geometry_feasibility",
                "first_route_success",
                "route_count",
                "wfc_attempts",
                "generation_seconds",
                "graph_seconds",
                "routing_seconds",
                "failure_stage",
                "recovery_attempts",
                "routes_retried",
                "logical_errors",
                "geometry_warnings",
                "route_failure_details",
                "recovery_details",
                "route_metrics",
                "error",
                "accepted_seed",
                "building_retry_count",
                "building_attempts",
            ),
        )
        writer.writeheader()
        for result in results:
            row = asdict(result)
            row["route_failure_details"] = json.dumps(
                result.route_failure_details
            )
            row["recovery_details"] = json.dumps(result.recovery_details)
            row["route_metrics"] = json.dumps(result.route_metrics)
            row["building_attempts"] = json.dumps(result.building_attempts)
            writer.writerow(row)


def _print_results(results: Sequence[ExperimentResult]) -> None:
    print(
        "resolution seed status first-route routes WFC-attempts generation-seconds "
        "recoveries retried-routes building-retries accepted-seed route-lengths/types"
    )
    for result in results:
        details = ", ".join(
            f"{route['kind']}:{route['cell_count']} cells/"
            f"{route['attempts']} tries"
            for route in result.route_metrics
        )
        print(
            f"{result.grid_resolution:>10} {result.seed:>4} "
            f"{'success' if result.success else 'failure':>7} "
            f"{str(result.first_route_success):>11} "
            f"{result.route_count:>6} {result.wfc_attempts:>12} "
            f"{result.generation_seconds:>18.4f} "
            f"{result.recovery_attempts:>10} {result.routes_retried:>13} "
            f"{result.building_retry_count:>16} {str(result.accepted_seed):>13} "
            f"{result.failure_stage or '-'} {details}"
        )
        if result.error:
            print(f"  failure: {result.error}")

    groups = dict.fromkeys(
        (result.grid_resolution, result.recovery_strategy,
         result.minimum_curvature_radius, result.tile_aware_connectivity,
         result.endpoint_frontier, result.candidate_lookahead)
        for result in results
    )
    for resolution, strategy, radius, tile_aware, frontier, lookahead in groups:
        matching = [
            result for result in results
            if (
                result.grid_resolution == resolution
                and result.recovery_strategy == strategy
                and result.minimum_curvature_radius == radius
                and result.tile_aware_connectivity == tile_aware
                and result.endpoint_frontier == frontier
                and result.candidate_lookahead == lookahead
            )
        ]
        successes = sum(result.success for result in matching)
        first_route_successes = sum(
            result.first_route_success is True for result in matching
        )
        graph_valid = [
            result for result in matching
            if result.graph_feasibility == "feasible"
        ]
        conditional_successes = sum(result.success for result in graph_valid)
        conditional_rate = (
            f"{conditional_successes}/{len(graph_valid)} graph-valid "
            f"({conditional_successes / len(graph_valid):.0%})"
            if graph_valid
            else "n/a (no graph-valid seeds)"
        )
        mean_time = sum(result.generation_seconds for result in matching) / len(
            matching
        )
        median_time = median(result.generation_seconds for result in matching)
        mean_attempts = sum(result.wfc_attempts for result in matching) / len(
            matching
        )
        total_recoveries = sum(result.recovery_attempts for result in matching)
        total_routes_retried = sum(result.routes_retried for result in matching)
        total_building_retries = sum(result.building_retry_count for result in matching)
        successful = [result for result in matching if result.success]
        logical_errors = sum(result.logical_errors for result in successful)
        geometry_warnings = sum(result.geometry_warnings for result in successful)
        print(
            f"resolution {resolution} / {strategy} / curvature radius "
            f"{radius:g} / tile-aware={tile_aware} / frontier={frontier} / "
            f"lookahead={lookahead}: {first_route_successes}/"
            f"{sum(result.first_route_success is not None for result in graph_valid)} "
            f"first routes; {successes}/{len(matching)} successful, "
            f"{len(matching) - successes} failed; "
            f"conditional on graph generation: {conditional_rate}; "
            f"runtime mean/median={mean_time:.2f}/{median_time:.2f}s; "
            f"mean WFC attempts={mean_attempts:.1f}; "
            f"recovery operations={total_recoveries}; "
            f"routes regenerated={total_routes_retried}; "
            f"whole-building retries={total_building_retries}; "
            f"successful-building logical errors={logical_errors}; "
            f"geometry warnings={geometry_warnings}"
        )


def run_controlled_comparison(
    seeds: Sequence[int],
    grid_resolutions: Sequence[int] = (2,),
    *,
    floors: int = 5,
    width: int = 5,
    length: int = 5,
    difficulty: int = 2,
    max_graph_attempts: int = 100,
    max_building_retries: int = 0,
    max_route_attempts: int = 30,
    max_route_backtrack_depth: int = 2,
    max_route_recovery_attempts: int = 1,
    minimum_curvature_radius: float = 0.02,
) -> tuple[ExperimentResult, ...]:
    """Compare recovery modes without curvature, then isolate curvature impact."""
    results = []
    for strategy in ("none", "recency", "conflict-aware"):
        results.extend(
            run_experiments(
                seeds,
                grid_resolutions,
                floors=floors,
                width=width,
                length=length,
                difficulty=difficulty,
                max_graph_attempts=max_graph_attempts,
                max_building_retries=max_building_retries,
                max_route_attempts=max_route_attempts,
                max_route_backtrack_depth=max_route_backtrack_depth,
                max_route_recovery_attempts=max_route_recovery_attempts,
                recovery_strategy=strategy,
                minimum_curvature_radius=0.0,
            )
        )
    results.extend(
        run_experiments(
            seeds,
            grid_resolutions,
            floors=floors,
            width=width,
            length=length,
            difficulty=difficulty,
            max_graph_attempts=max_graph_attempts,
            max_building_retries=max_building_retries,
            max_route_attempts=max_route_attempts,
            max_route_backtrack_depth=max_route_backtrack_depth,
            max_route_recovery_attempts=max_route_recovery_attempts,
            recovery_strategy="conflict-aware",
            minimum_curvature_radius=minimum_curvature_radius,
        )
    )
    return tuple(results)


def run_solver_heuristic_ablation(
    seeds: Sequence[int],
    grid_resolutions: Sequence[int] = (2,),
    *,
    floors: int = 5,
    width: int = 5,
    length: int = 5,
    difficulty: int = 2,
    max_graph_attempts: int = 100,
    max_building_retries: int = 0,
    max_route_attempts: int = 30,
    max_route_backtrack_depth: int = 2,
    max_route_recovery_attempts: int = 1,
    minimum_curvature_radius: float = 0.02,
) -> tuple[ExperimentResult, ...]:
    """Compare solver heuristics independently and together on matched seeds."""
    configurations = (
        ("baseline", False, False, False),
        ("tile-aware", True, False, False),
        ("endpoint-frontier", False, True, False),
        ("one-step-lookahead", False, False, True),
        ("default-combination", True, True, False),
        ("combined", True, True, True),
    )
    results = []
    for label, tile_aware, frontier, lookahead in configurations:
        print(f"Running solver configuration: {label}")
        results.extend(
            run_experiments(
                seeds,
                grid_resolutions,
                floors=floors,
                width=width,
                length=length,
                difficulty=difficulty,
                max_graph_attempts=max_graph_attempts,
                max_building_retries=max_building_retries,
                max_route_attempts=max_route_attempts,
                max_route_backtrack_depth=max_route_backtrack_depth,
                max_route_recovery_attempts=max_route_recovery_attempts,
                recovery_strategy="none",
                minimum_curvature_radius=minimum_curvature_radius,
                tile_aware_connectivity=tile_aware,
                endpoint_frontier=frontier,
                candidate_lookahead=lookahead,
            )
        )
    return tuple(results)


def _plot_results(results: Sequence[ExperimentResult], path: Path) -> None:
    """Save strategy success rates and per-seed runtimes, including graph failures."""
    resolutions = tuple(dict.fromkeys(result.grid_resolution for result in results))
    configurations = tuple(
        dict.fromkeys(
            (
                result.recovery_strategy,
                result.minimum_curvature_radius,
                result.tile_aware_connectivity,
                result.endpoint_frontier,
                result.candidate_lookahead,
            )
            for result in results
        )
    )
    figure, (rate_axes, time_axes) = plt.subplots(
        1, 2, figsize=(12, 5), constrained_layout=True
    )
    positions = list(range(len(resolutions)))
    width = 0.8 / max(1, len(configurations))
    strategy_colors = {
        "none": "#64748b",
        "recency": "#3977a8",
        "conflict-aware": "#58a878",
    }
    legend_labels: set[str] = set()
    for config_index, (
        strategy,
        radius,
        tile_aware,
        frontier,
        lookahead,
    ) in enumerate(configurations):
        config_label = (
            f"{strategy}, Rmin={radius:g}, tile={tile_aware}, "
            f"frontier={frontier}, lookahead={lookahead}"
        )
        for index, resolution in enumerate(resolutions):
            matching = [
                r for r in results
                if r.grid_resolution == resolution
                and r.recovery_strategy == strategy
                and r.minimum_curvature_radius == radius
                and r.tile_aware_connectivity == tile_aware
                and r.endpoint_frontier == frontier
                and r.candidate_lookahead == lookahead
            ]
            if not matching:
                continue
            graph_valid = [
                r for r in matching if r.failure_stage != "graph generation"
            ]
            success_rate = sum(r.success for r in matching) / len(matching)
            conditional_rate = (
                sum(r.success for r in graph_valid) / len(graph_valid)
                if graph_valid
                else 0.0
            )
            x_position = (
                positions[index]
                - 0.4
                + width * (config_index + 0.5)
            )
            color = strategy_colors[strategy]
            rate_axes.bar(
                x_position - width * 0.22,
                success_rate,
                width * 0.44,
                color=color,
                alpha=0.55,
                label=f"{config_label}: overall" if index == 0 else None,
            )
            rate_axes.bar(
                x_position + width * 0.22,
                conditional_rate,
                width * 0.44,
                color=color,
                alpha=1.0,
                label=f"{config_label}: graph-valid" if index == 0 else None,
            )
            rate_axes.text(
                x_position - width * 0.22,
                success_rate + 0.015,
                f"{sum(r.success for r in matching)}/{len(matching)}",
                ha="center",
                fontsize=7,
                rotation=90,
            )
            rate_axes.text(
                x_position + width * 0.22,
                conditional_rate + 0.015,
                f"{sum(r.success for r in graph_valid)}/{len(graph_valid)}"
                if graph_valid
                else "n/a",
                ha="center",
                fontsize=7,
                rotation=90,
            )
            for result in matching:
                category = (
                    "success"
                    if result.success
                    else "graph failure"
                    if result.failure_stage == "graph generation"
                    else "WFC failure"
                )
                marker = {
                    "success": "o",
                    "WFC failure": "x",
                    "graph failure": "s",
                }[category]
                offset_seed = result.seed + (config_index - (len(configurations)-1)/2) * 0.08
                time_axes.scatter(
                    offset_seed,
                    result.generation_seconds,
                    color=color,
                    marker=marker,
                    s=32 + 10 * result.recovery_attempts,
                    alpha=0.8,
                    label=(
                        f"{config_label}: {category}"
                        if f"{config_label}: {category}" not in legend_labels
                        else None
                    ),
                )
                legend_labels.add(f"{config_label}: {category}")

    rate_axes.set_xticks(positions, [str(value) for value in resolutions])
    rate_axes.set_ylim(0, 1.15)
    rate_axes.set_xlabel("Horizontal grid resolution")
    rate_axes.set_ylabel("Generation success rate")
    rate_axes.set_title("Overall vs graph-conditional success")
    rate_axes.legend(fontsize=7)
    time_axes.set_xlabel("Random seed")
    time_axes.set_ylabel("Generation time (seconds)")
    time_axes.set_title("Per-seed runtime (color=strategy, marker=outcome)")
    time_axes.set_yscale("log")
    handles, labels = time_axes.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    time_axes.legend(unique.values(), unique.keys(), fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> tuple[ExperimentResult, ...]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(range(20)),
        help="deterministic graph/WFC seeds (default: 0 through 19)",
    )
    parser.add_argument(
        "--grid-resolutions",
        nargs="+",
        type=int,
        default=[2],
        help="horizontal route-grid resolutions (default: 2)",
    )
    parser.add_argument("--floors", type=int, default=5)
    parser.add_argument("--width", type=int, default=5)
    parser.add_argument("--length", type=int, default=5)
    parser.add_argument("--difficulty", type=int, default=2)
    parser.add_argument("--max-graph-attempts", type=int, default=100)
    parser.add_argument("--max-building-retries", type=int, default=0)
    parser.add_argument("--max-route-attempts", type=int, default=30)
    parser.add_argument("--max-route-backtrack-depth", type=int, default=2)
    parser.add_argument("--max-route-recovery-attempts", type=int, default=1)
    parser.add_argument(
        "--recovery-strategy",
        choices=("none", "recency", "conflict-aware"),
        default="conflict-aware",
    )
    parser.add_argument("--minimum-curvature-radius", type=float, default=0.02)
    comparison = parser.add_mutually_exclusive_group()
    comparison.add_argument(
        "--compare-strategies",
        action="store_true",
        help="run no-recovery, recency, conflict-aware, then curvature comparison",
    )
    comparison.add_argument(
        "--compare-solver-heuristics",
        action="store_true",
        help="ablate tile-aware pruning, endpoint-frontier ordering, and lookahead",
    )
    parser.add_argument(
        "--tile-aware-connectivity",
        action="store_true",
        help="use tile-compatible route-state pruning",
    )
    parser.add_argument(
        "--no-tile-aware-connectivity",
        dest="tile_aware_connectivity",
        action="store_false",
        help="disable tile-compatible route-state pruning",
    )
    parser.add_argument(
        "--endpoint-frontier",
        action="store_true",
        help="collapse the next unresolved cell from the source along a supported route",
    )
    parser.add_argument(
        "--no-endpoint-frontier",
        dest="endpoint_frontier",
        action="store_false",
        help="disable source-frontier collapse ordering",
    )
    parser.add_argument(
        "--candidate-lookahead",
        action="store_true",
        help="reject route-tile choices that immediately contradict propagation",
    )
    parser.set_defaults(
        tile_aware_connectivity=False,
        endpoint_frontier=True,
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="optional CSV output path, including per-route metrics",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        help="optional PNG path for success-rate and per-seed runtime charts",
    )
    arguments = parser.parse_args(argv)
    experiment_parameters = {
        "floors": arguments.floors,
        "width": arguments.width,
        "length": arguments.length,
        "difficulty": arguments.difficulty,
        "max_graph_attempts": arguments.max_graph_attempts,
        "max_building_retries": arguments.max_building_retries,
        "max_route_attempts": arguments.max_route_attempts,
        "max_route_backtrack_depth": arguments.max_route_backtrack_depth,
        "max_route_recovery_attempts": arguments.max_route_recovery_attempts,
    }
    if arguments.compare_strategies:
        results = run_controlled_comparison(
            arguments.seeds,
            arguments.grid_resolutions,
            minimum_curvature_radius=arguments.minimum_curvature_radius,
            **experiment_parameters,
        )
    elif arguments.compare_solver_heuristics:
        results = run_solver_heuristic_ablation(
            arguments.seeds,
            arguments.grid_resolutions,
            minimum_curvature_radius=arguments.minimum_curvature_radius,
            **experiment_parameters,
        )
    else:
        results = run_experiments(
            arguments.seeds,
            arguments.grid_resolutions,
            recovery_strategy=arguments.recovery_strategy,
            minimum_curvature_radius=arguments.minimum_curvature_radius,
            tile_aware_connectivity=arguments.tile_aware_connectivity,
            endpoint_frontier=arguments.endpoint_frontier,
            candidate_lookahead=arguments.candidate_lookahead,
            **experiment_parameters,
        )
    _print_results(results)
    if arguments.csv is not None:
        _write_csv(arguments.csv, results)
        print(f"Wrote experiment results to {arguments.csv}")
    if arguments.plot is not None:
        _plot_results(results, arguments.plot)
        print(f"Wrote experiment plot to {arguments.plot}")
    return results


if __name__ == "__main__":
    main()
