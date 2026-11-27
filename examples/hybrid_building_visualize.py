"""Plot an exported building's inherited graph and accepted trajectories."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


COLORS = {"floor": "#a7adb7", "stair": "#df8b22", "slide": "#257da1"}


def create_figure(building):
    figure = plt.figure(figsize=(13, 6), constrained_layout=True)
    graph_axes = figure.add_subplot(121, projection="3d")
    route_axes = figure.add_subplot(122, projection="3d")
    nodes = np.asarray([node["position"] for node in building["graph"]["nodes"]])
    graph_axes.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], color="#a7adb7",
                       s=3, alpha=0.25, depthshade=False)
    for edge in building["graph"]["edges"]:
        start = np.asarray(edge["source"])
        end = np.asarray(edge["target"])
        kind = edge["attributes"]["kind"]
        delta = end - start
        graph_axes.quiver(*start, *delta, color=COLORS[kind],
                          linewidth=0.8 if kind == "floor" else 1.7,
                          alpha=0.65 if kind == "floor" else 0.9,
                          arrow_length_ratio=0.12)
    for route in building["routes"]:
        points = np.asarray(route["geometry"])
        route_axes.plot(points[:, 0], points[:, 1], points[:, 2],
                        color=COLORS[route["type"]], linewidth=2)
    dimensions = building["building"]
    for axes, title in ((graph_axes, "Inherited building graph"),
                        (route_axes, "Accepted Hybrid WFC trajectories")):
        for key, color, marker, label in (
            ("start", "#16a34a", "o", "Start"),
            ("exit", "#dc2626", "X", "Exit"),
        ):
            axes.scatter(*dimensions[key], color=color, marker=marker, s=55,
                         depthshade=False, label=label)
        axes.set(xlim=(-0.2, dimensions["length"] - 0.8),
                 ylim=(-0.2, dimensions["width"] - 0.8),
                 zlim=(-0.1, dimensions["floors"] - 0.9),
                 xlabel="x", ylabel="y", zlabel="floor", title=title)
        axes.set_box_aspect((dimensions["length"] - 1, dimensions["width"] - 1,
                             dimensions["floors"]))
        axes.view_init(elev=25, azim=-55)
    figure.legend(handles=[
        Line2D([0], [0], color=COLORS["floor"], label="Floor edge"),
        Line2D([0], [0], color=COLORS["stair"], label="Stair"),
        Line2D([0], [0], color=COLORS["slide"], label="Slide"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#16a34a",
               markeredgecolor="#16a34a", label="Start"),
        Line2D([0], [0], marker="X", color="none", markerfacecolor="#dc2626",
               markeredgecolor="#dc2626", label="Exit"),
    ], loc="lower center", ncol=5, frameon=False)
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("building", type=Path, help="exported building JSON")
    parser.add_argument("--output", type=Path, required=True, help="new PNG file")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    building = json.loads(args.building.read_text(encoding="utf-8"))
    figure = create_figure(building)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as output:
        figure.savefig(output, format="png", dpi=160)
    plt.close(figure)
    print(f"Saved graph and accepted trajectories to {args.output}")


if __name__ == "__main__":
    main()
