"""Generate, trace, and plot graph-constrained 3D stair and slide routes."""

import argparse
from collections.abc import Mapping
from enum import Enum
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
from src.building_generation import generate_building
from src.route_diagnostics import diagnose_routes
from src.wfc_building import WFCBuilding
from src.wfc_tiles import PortKind


def _jsonable(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_jsonable(item) for item in value]
    return value


def _print_trace(
    event: Mapping[str, object],
    *,
    detailed_trace: bool,
) -> None:
    event_type = event.get("event")
    route_id = event.get("route_id")
    prefix = f"[{route_id}] " if route_id else ""
    if event_type == "route_summary":
        exclusions = event["excluded_cells_by_reason"]
        counts = ", ".join(
            f"{reason}={len(cells)}"
            for reason, cells in exclusions.items()
        ) or "none"
        print(
            f"{prefix}{event['route_kind']} {event['source']} -> {event['target']}; "
            f"WFC endpoints {event['start_cell']} -> {event['end_cell']}; "
            f"grid={event['grid_dimensions']}; "
            f"available={event['available_cell_count']}; excluded: {counts}; "
            f"initial domains={event['initial_domain_size_distribution']}; "
            f"initial fixed={event['initial_fixed_cell_count']}, "
            f"unresolved={len(event['initial_unresolved_cells'])}"
        )
    elif event_type in (
        "route_attempt_start",
        "route_attempt_failure",
        "route_attempt_geometry_rejection",
        "route_attempt_success",
        "route_attempt_no_route",
        "global_route_recovery_attempt",
    ):
        print(f"{prefix}{event_type}: {dict(event)}")
    elif event_type == "building_attempt_start":
        print(
            f"Building candidate {event['building_attempt']} "
            f"(seed {event['candidate_seed']})"
        )
    elif event_type in ("building_attempt_failure", "building_attempt_accepted"):
        print(f"{event_type}: {dict(event)}")
    elif event_type == "route_extracted":
        comparisons = event["path_comparisons"]
        print(
            f"{prefix}extracted route length={event['final_discrete_route_length']}; "
            f"DFS path comparisons={comparisons}"
        )
    elif event_type == "solver_trace":
        solver_event = event["solver_event"]
        solver_type = solver_event["event"]
        if not detailed_trace and not (
            solver_type == "solver_attempt_start"
            or (
                solver_type == "domain_reduction"
                and solver_event.get("phase") in {
                    "initial_propagation",
                    "global_reachability_pruning",
                }
            )
        ):
            return
        if solver_type == "solver_attempt_start":
            print(
                f"{prefix}attempt {solver_event['attempt']} starts; "
                f"domain sizes={solver_event['initial_domain_size_distribution']}; "
                f"pre-collapsed={len(solver_event['initial_pre_collapsed_cells'])}; "
                f"unresolved={len(solver_event['initial_unresolved_cells'])}"
            )
        elif solver_type == "domain_reduction":
            print(
                f"{prefix}{solver_event['phase']}: removed "
                f"{solver_event['possibilities_removed']} tile possibilities; "
                f"newly resolved={len(solver_event['newly_resolved_cells'])}"
            )
        elif solver_type == "route_candidate":
            print(
                f"{prefix}DFS candidate length={solver_event['candidate_path_length']}, "
                f"horizontal steps={solver_event['candidate_horizontal_steps']}, "
                f"backtracks={solver_event['dfs_backtracks']}, "
                f"budget exhausted={solver_event['dfs_search_budget_exhausted']}; "
                f"path={solver_event['candidate_path']}"
            )
        elif solver_type == "route_guided_selection":
            print(
                f"{prefix}DFS-filtered candidates "
                f"{solver_event['candidate_count_before_dfs_filter']} -> "
                f"{solver_event['candidate_count_after_dfs_filter']}"
            )
        elif solver_type == "candidate_lookahead_result":
            print(
                f"{prefix}lookahead checks={solver_event['checks']}, "
                f"rejections={solver_event['rejections']}, "
                f"candidates={solver_event['candidate_count_before']} -> "
                f"{solver_event['candidate_count_after']}"
            )
        elif solver_type == "collapse_selected":
            print(
                f"{prefix}collapse {solver_event['iteration']}: "
                f"mode={solver_event['selection_mode']}, "
                f"cell={solver_event['selected_cell']}, "
                f"domain={solver_event['original_domain_size']}, "
                f"candidates={solver_event['candidate_tile_names']}, "
                f"selected={solver_event['selected_tile_name']} "
                f"{solver_event['selected_tile_ports']}"
            )
        elif solver_type == "collapse_complete":
            print(
                f"{prefix}collapse {solver_event['iteration']} complete; "
                f"unresolved={solver_event['unresolved_cell_count']}; "
                f"required route feasible={solver_event['required_route_feasible']}"
            )
        elif solver_type == "solver_contradiction":
            print(
                f"{prefix}contradiction on attempt {solver_event['attempt']} "
                f"during {solver_event['failure_phase']}: "
                f"{solver_event['failure_message']}"
            )


def _save_progress_frame(
    event: Mapping[str, object],
    route_summary: Mapping[str, object],
    candidate_path: tuple[tuple[int, int, int], ...],
    destination: Path,
) -> None:
    solver_event = event["solver_event"]
    domains = {
        tuple(cell): tuple(indices)
        for cell, indices in solver_event["domains"]
    }
    empty_index = route_summary["empty_tile_index"]
    figure = plt.figure(figsize=(9, 7))
    axes = figure.add_subplot(111, projection="3d")
    unresolved = [cell for cell, domain in domains.items() if len(domain) > 1]
    if unresolved:
        unresolved_plot = axes.scatter(
            [cell[1] for cell in unresolved],
            [cell[2] for cell in unresolved],
            [cell[0] for cell in unresolved],
            c=[len(domains[cell]) for cell in unresolved],
            cmap="viridis",
            marker="o",
            label="Unresolved (colour = domain size)",
        )
        figure.colorbar(unresolved_plot, ax=axes, label="Remaining tile choices")
    empty_cells = [
        cell for cell, domain in domains.items()
        if len(domain) == 1 and domain[0] == empty_index
    ]
    route_cells = [
        cell for cell, domain in domains.items()
        if len(domain) == 1 and domain[0] != empty_index
    ]
    if empty_cells:
        axes.scatter(
            [cell[1] for cell in empty_cells],
            [cell[2] for cell in empty_cells],
            [cell[0] for cell in empty_cells],
            color="lightgray",
            marker="s",
            label="Collapsed empty",
        )
    if route_cells:
        axes.scatter(
            [cell[1] for cell in route_cells],
            [cell[2] for cell in route_cells],
            [cell[0] for cell in route_cells],
            color="steelblue",
            marker="o",
            label="Collapsed route tile",
        )
    excluded = route_summary["excluded_cells_by_reason"]
    reason_styles = {
        "forbidden": ("black", "x"),
        "occupied_by_existing_route": ("orange", "x"),
        "graph_floor_cell": ("gray", "x"),
        "reserved_connector": ("purple", "x"),
        "outside_refined_routing_grid": ("silver", "."),
    }
    for reason, cells in excluded.items():
        if not cells:
            continue
        color, marker = reason_styles.get(reason, ("dimgray", "x"))
        axes.scatter(
            [cell[1] for cell in cells],
            [cell[2] for cell in cells],
            [cell[0] for cell in cells],
            color=color,
            marker=marker,
            alpha=0.35,
            label=f"Excluded: {reason}",
        )
    start = tuple(route_summary["start_cell"])
    end = tuple(route_summary["end_cell"])
    axes.scatter(*((start[1], start[2], start[0])), color="green",
                 marker="^", s=100, label="Head")
    axes.scatter(*((end[1], end[2], end[0])), color="red",
                 marker="v", s=100, label="Tail")
    if candidate_path:
        axes.plot(
            [cell[1] for cell in candidate_path],
            [cell[2] for cell in candidate_path],
            [cell[0] for cell in candidate_path],
            color="gold",
            linewidth=2,
            label="Current DFS candidate",
        )
    selected_cell = solver_event.get("selected_cell")
    if selected_cell is not None:
        cell = tuple(selected_cell)
        axes.scatter(cell[1], cell[2], cell[0], color="magenta",
                     marker="*", s=180, label="Explicitly selected cell")
    axes.set(
        xlabel="x",
        ylabel="y",
        zlabel="floor-grid step",
        title=(
            f"{event['route_id']} — {solver_event['event']} — "
            f"attempt {solver_event.get('attempt', 1)}, "
            f"iteration {solver_event.get('iteration', 0)}"
        ),
    )
    axes.legend(loc="upper left", fontsize="x-small")
    figure.tight_layout()
    figure.savefig(destination, dpi=120)
    plt.close(figure)


def main(
    show_diagnostics: bool = False,
    seed: int = 0,
    grid_resolution: int = 2,
    save_plot: Path | None = None,
    verbose: bool = False,
    trace_wfc: bool = False,
    route_id: str | None = None,
    diagnostics_json: Path | None = None,
    progress_dir: Path | None = None,
    gif_path: Path | None = None,
    wfc_seed: int | None = None,
    tile_aware_connectivity: bool = False,
    endpoint_frontier: bool = True,
    candidate_lookahead: bool = False,
    max_building_retries: int = 0,
    routing_strategy: str = "dfs_guided",
    only_route_id: str | None = None,
) -> WFCBuilding:
    if route_id is not None and not (
        verbose or trace_wfc or diagnostics_json is not None
        or progress_dir is not None or gif_path is not None
    ):
        raise ValueError("--route-id requires verbose, tracing, or progress output")
    if (progress_dir is not None or gif_path is not None) and route_id is None:
        raise ValueError("Progress visualization requires --route-id to limit frames")
    if gif_path is not None and progress_dir is None:
        progress_dir = gif_path.with_name(f"{gif_path.stem}_frames")

    events: list[dict[str, object]] = []
    route_summaries: dict[str, Mapping[str, object]] = {}
    candidate_paths: dict[str, tuple[tuple[int, int, int], ...]] = {}
    frame_paths: list[Path] = []
    trace_enabled = (
        verbose
        or trace_wfc
        or diagnostics_json is not None
        or progress_dir is not None
    )

    def trace_callback(event: Mapping[str, object]) -> None:
        stored = dict(event)
        if diagnostics_json is not None:
            events.append(stored)
        if event.get("event") == "route_summary":
            route_summaries[str(event["route_id"])] = event
        if verbose or trace_wfc:
            _print_trace(event, detailed_trace=trace_wfc)
        if not trace_wfc and progress_dir is None:
            return
        if event.get("event") != "solver_trace":
            return
        solver_event = event["solver_event"]
        if solver_event.get("event") == "route_candidate":
            candidate_paths[str(event["route_id"])] = tuple(
                tuple(cell) for cell in solver_event["candidate_path"]
            )
        if progress_dir is None:
            return
        should_render = (
            solver_event.get("event") == "solver_attempt_start"
            or (
                solver_event.get("event") == "domain_reduction"
                and solver_event.get("phase") == "initial_propagation"
            )
            or solver_event.get("event") == "collapse_complete"
        )
        if not should_render:
            return
        route_key = str(event["route_id"])
        summary = route_summaries.get(route_key)
        if summary is None:
            return
        progress_dir.mkdir(parents=True, exist_ok=True)
        safe_route_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", route_key)
        sequence = len(frame_paths)
        frame_path = progress_dir / f"{safe_route_id}_{sequence:04d}.png"
        _save_progress_frame(
            event,
            summary,
            candidate_paths.get(route_key, ()),
            frame_path,
        )
        frame_paths.append(frame_path)

    building = generate_building(
        seed=seed,
        max_building_retries=max_building_retries,
        trace_callback=trace_callback if trace_enabled else None,
        floors=4,
        width=10,
        length=10,
        difficulty=2,
        grid_resolution=grid_resolution,
        routing_seed=wfc_seed,
        only_route_id=only_route_id,
        trace_route_id=route_id,
        trace_wfc=(
            trace_wfc
            or verbose
            or diagnostics_json is not None
            or progress_dir is not None
        ),
        trace_states=progress_dir is not None,
        tile_aware_connectivity=tile_aware_connectivity,
        endpoint_frontier=endpoint_frontier,
        candidate_lookahead=candidate_lookahead,
        routing_strategy=routing_strategy,
    ).building
    if route_id is not None and all(
        route.route_id != route_id for route in building.routes
    ):
        raise ValueError(f"No generated route has ID {route_id!r}")

    print(
        f"Generated {len(building.routes)} inter-floor routes from "
        f"{building.graph.number_of_edges()} graph edges; "
        f"{building.recovery_attempts} recovery windows, "
        f"{building.routes_retried} previous routes regenerated; "
        f"building retries={building.building_retry_count}."
    )
    diagnostics = diagnose_routes(building) if show_diagnostics else None
    for route in building.routes:
        smooth_points = building.smooth_path(route)
        direction = (
            "strictly descending"
            if route.kind is PortKind.SLIDE
            else "ascending"
        )
        print(
            f"{route.kind.value}: {route.source} -> {route.target}; "
            f"{len(route.cells)} WFC cells; {len(smooth_points)} geometry "
            f"points; {direction}; {route.attempts} WFC attempts."
        )
    if diagnostics is not None:
        for route_report in diagnostics.routes:
            slope = (
                "n/a"
                if route_report.minimum_slope is None
                else f"{route_report.minimum_slope:.3g}.."
                f"{route_report.maximum_slope:.3g}"
            )
            maximum_turn = max(route_report.turn_angles_degrees, default=None)
            curvature = (
                "n/a"
                if route_report.maximum_curvature is None
                else f"{route_report.maximum_curvature:.3g}"
            )
            minimum_radius = (
                "n/a"
                if route_report.minimum_curvature_radius is None
                else f"{route_report.minimum_curvature_radius:.3g}"
            )
            print(
                f"DIAGNOSTICS {route_report.route_id}: "
                f"endpoint errors={route_report.endpoint_errors}; "
                f"continuous={route_report.continuity_valid}; "
                f"direction={route_report.direction_valid}; "
                f"slide slope range={slope}; "
                f"max turn={maximum_turn if maximum_turn is not None else 'n/a'}; "
                f"max curvature={curvature}; "
                f"minimum curvature radius={minimum_radius}"
            )
        for pair in diagnostics.route_pairs:
            clearance = (
                "n/a"
                if pair.minimum_clearance is None
                else f"{pair.minimum_clearance:.3g}"
            )
            print(
                f"ROUTE PAIR {pair.route_ids}: "
                f"overlap={pair.overlapping_cells}; "
                f"minimum clearance={clearance}"
            )
        for error in diagnostics.logical_errors:
            print(f"LOGICAL ERROR: {error}")
        for warning in diagnostics.warnings:
            print(f"GEOMETRY WARNING: {warning}")
        if not diagnostics.logical_errors and not diagnostics.warnings:
            print("No logical errors or configured geometry warnings.")

    if diagnostics_json is not None:
        diagnostics_json.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_json.write_text(
            json.dumps(_jsonable(events), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(f"Saved structured WFC events to {diagnostics_json}")
    if progress_dir is not None and not frame_paths:
        raise RuntimeError(
            f"No WFC progress frames were generated for route {route_id!r}"
        )
    if gif_path is not None:
        from PIL import Image

        images = [Image.open(path) for path in frame_paths]
        try:
            first = images[0]
            first.save(
                gif_path,
                save_all=True,
                append_images=images[1:],
                duration=250,
                loop=0,
            )
        finally:
            for image in images:
                image.close()
        print(f"Saved WFC progress animation to {gif_path}")

    figure, _ = building.plot(show_diagnostics=show_diagnostics)
    if save_plot is not None:
        figure.savefig(save_plot, dpi=160)
        print(f"Saved route visualization to {save_plot}")
    plt.show()
    return building


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--seed", type=int, default=0, help="building graph seed")
    parser.add_argument("--wfc-seed", type=int, help="route-solver seed; defaults to --seed")
    parser.add_argument("--grid-resolution", type=int, default=2)
    parser.add_argument("--verbose", action="store_true", help="print per-route summaries")
    parser.add_argument(
        "--trace-wfc",
        action="store_true",
        help="print detailed WFC collapse, propagation, and DFS events",
    )
    parser.add_argument("--route-id", help="limit trace/progress to an exact route ID")
    parser.add_argument(
        "--only-route-id",
        help="generate direct stairs and only this slide route for isolated comparisons",
    )
    parser.add_argument(
        "--diagnostics-json",
        type=Path,
        help="export structured route and solver-attempt events",
    )
    parser.add_argument(
        "--progress-dir",
        type=Path,
        help="save actual solver-state PNG frames for --route-id",
    )
    parser.add_argument(
        "--gif",
        type=Path,
        help="assemble solver-state frames into an animated GIF for --route-id",
    )
    parser.add_argument(
        "--tile-aware-connectivity",
        action="store_true",
        help="enable tile-aware global route pruning",
    )
    parser.add_argument(
        "--no-endpoint-frontier",
        action="store_true",
        help="use minimum-domain selection along the candidate route",
    )
    parser.add_argument(
        "--candidate-lookahead",
        action="store_true",
        help="test each DFS-filtered tile candidate before committing",
    )
    parser.add_argument(
        "--routing-strategy",
        choices=("dfs_guided",),
        default="dfs_guided",
        help="experimental route-generation strategy",
    )
    parser.add_argument("--max-building-retries", type=int, default=0)
    parser.add_argument("--save-plot", type=Path)
    arguments = parser.parse_args()
    main(
        show_diagnostics=arguments.diagnostics,
        seed=arguments.seed,
        wfc_seed=arguments.wfc_seed,
        grid_resolution=arguments.grid_resolution,
        save_plot=arguments.save_plot,
        verbose=arguments.verbose,
        trace_wfc=arguments.trace_wfc,
        route_id=arguments.route_id,
        diagnostics_json=arguments.diagnostics_json,
        progress_dir=arguments.progress_dir,
        gif_path=arguments.gif,
        tile_aware_connectivity=arguments.tile_aware_connectivity,
        endpoint_frontier=not arguments.no_endpoint_frontier,
        candidate_lookahead=arguments.candidate_lookahead,
        max_building_retries=arguments.max_building_retries,
        routing_strategy=arguments.routing_strategy,
        only_route_id=arguments.only_route_id,
    )
