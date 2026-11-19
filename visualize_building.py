"""Generate and interactively visualize one graph-constrained WFC building."""

from __future__ import annotations

import argparse
import sys

import matplotlib


def _use_interactive_backend() -> None:
    backend = matplotlib.get_backend().lower()
    if backend not in {"tkagg", "qtagg", "qt5agg", "wxagg", "macosx"}:
        try:
            matplotlib.use("TkAgg", force=True)
        except ImportError as error:
            raise RuntimeError(
                "An interactive Matplotlib backend is required. Install a "
                "Python distribution with Tk support, or install PyQt and use "
                "the QtAgg backend."
            ) from error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--grid-resolution", type=int, default=2)
    parser.add_argument("--max-building-retries", type=int, default=0)
    arguments = parser.parse_args()

    _use_interactive_backend()

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from src.route_diagnostics import diagnose_routes
    from src.building_generation import generate_building
    from src.wfc_building import WFCBuilding
    from src.wfc_tiles import PortKind

    options = dict(
        floors=5,
        width=5,
        length=5,
        difficulty=2,
        grid_resolution=arguments.grid_resolution,
        seed=arguments.seed,
        max_route_attempts=30,
        max_route_backtrack_depth=2,
        max_route_recovery_attempts=1,
        recovery_strategy="none",
        minimum_curvature_radius=0.02,
        endpoint_frontier=True,
    )
    if arguments.max_building_retries:
        result = generate_building(
            max_building_retries=arguments.max_building_retries, **options
        )
        building = result.building
        print(
            f"Requested seed {building.requested_seed}; accepted seed "
            f"{building.accepted_seed} after {building.building_retry_count} retries."
        )
    else:
        building = WFCBuilding(**options)
    diagnostics = diagnose_routes(building)
    if diagnostics.logical_errors:
        raise RuntimeError(
            "Generated building failed logical validation:\n"
            + "\n".join(diagnostics.logical_errors)
        )

    figure, (graph_axes, route_axes) = building.plot()
    figure.suptitle(
        f"Generated escape-room building — seed {arguments.seed}, "
        f"resolution {arguments.grid_resolution}"
    )
    graph_axes.set_title("Inherited graph (S = stair start, T = stair target)")

    graph_axes.legend(
        handles=(
            Line2D([], [], color="#64748b", linewidth=1.5, label="Floor connection"),
            Line2D([], [], color="#e58a22", linewidth=2.5, label="Stair connection"),
            Line2D([], [], color="#277da1", linewidth=2.5, label="Slide connection"),
            Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                color="#334155",
                markersize=6,
                label="Graph node",
            ),
            Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                color="#2a9d55",
                markersize=8,
                label="Start",
            ),
            Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                color="#d1495b",
                markersize=8,
                label="Exit",
            ),
        ),
        loc="upper left",
        fontsize=8,
    )
    route_axes.legend(
        handles=(
            Line2D([], [], color="#e58a22", linewidth=2.5, label="Accepted stairs"),
            Line2D([], [], color="#277da1", linewidth=2.5, label="Accepted slides"),
            Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                color="#2a9d55",
                markersize=8,
                label="Graph start",
            ),
            Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                color="#d1495b",
                markersize=8,
                label="Graph exit",
            ),
        ),
        loc="upper left",
        fontsize=8,
    )
    for node, label, color in (
        (building.start_pos, "START", "#2a9d55"),
        (building.exit_pos, "EXIT", "#d1495b"),
    ):
        route_axes.scatter(
            [node[0]],
            [node[1]],
            [node[2]],
            color=color,
            s=110,
            edgecolor="white",
            depthshade=False,
            zorder=10,
        )
        route_axes.text(
            node[0],
            node[1],
            node[2] + 0.1,
            label,
            color=color,
            weight="bold",
        )

    for axes in (graph_axes, route_axes):
        axes.set_xlim(-0.25, building.length - 0.75)
        axes.set_ylim(-0.25, building.width - 0.75)
        axes.set_zlim(-0.25, building.floors - 0.75)
        axes.set_zticks(range(building.floors))
        axes.set_box_aspect(
            (
                max(building.length - 1, 1),
                max(building.width - 1, 1),
                max(building.floors - 1, 1),
            )
        )

    for axes in (graph_axes, route_axes):
        for floor in range(building.floors):
            axes.plot(
                [0, building.length - 1, building.length - 1, 0, 0],
                [0, 0, building.width - 1, building.width - 1, 0],
                [floor] * 5,
                color="#94a3b8",
                linewidth=0.7,
                alpha=0.45,
            )

    print(
        f"Generated {len(building.routes)} validated routes "
        f"({sum(route.kind is PortKind.STAIR for route in building.routes)} "
        f"stairs, {sum(route.kind is PortKind.SLIDE for route in building.routes)} "
        f"slides); {building.total_wfc_attempts} WFC attempts."
    )
    if building.embedding_repair is not None:
        print(f"Graph embedding repair: {building.embedding_repair}")
    for warning in diagnostics.warnings:
        print(f"Geometry diagnostic warning: {warning}")
    print(f"Matplotlib backend: {matplotlib.get_backend()}")
    if not sys.stdout.isatty():
        print("Close the Matplotlib window to exit.")

    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
