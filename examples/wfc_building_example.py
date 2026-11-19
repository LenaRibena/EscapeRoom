"""Generate and plot graph-constrained 3D stair and slide routes."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from src.route_diagnostics import diagnose_routes
from src.wfc_building import WFCBuilding
from src.wfc_tiles import PortKind


def main(
    show_diagnostics: bool = False,
    seed: int = 0,
    grid_resolution: int = 2,
    save_plot: Path | None = None,
) -> WFCBuilding:
    building = WFCBuilding(
        floors=5,
        width=5,
        length=5,
        difficulty=2,
        grid_resolution=grid_resolution,
        seed=seed,
    )

    print(
        f"Generated {len(building.routes)} inter-floor routes from "
        f"{building.graph.number_of_edges()} graph edges; "
        f"{building.recovery_attempts} recovery windows, "
        f"{building.routes_retried} previous routes regenerated."
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

    figure, _ = building.plot(show_diagnostics=show_diagnostics)
    if save_plot is not None:
        figure.savefig(save_plot, dpi=160)
        print(f"Saved route visualization to {save_plot}")
    plt.show()
    return building


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="highlight route geometry warnings and close route approaches",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grid-resolution", type=int, default=2)
    parser.add_argument(
        "--save-plot",
        type=Path,
        help="optional image path for saving the graph and routed 3D view",
    )
    arguments = parser.parse_args()
    main(
        show_diagnostics=arguments.diagnostics,
        seed=arguments.seed,
        grid_resolution=arguments.grid_resolution,
        save_plot=arguments.save_plot,
    )
