"""Generate a synchronized DFS-guided WFC decision trace and animation."""

import argparse
from collections import Counter
from io import BytesIO
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

from src.json_export import building_to_dict
from src.wfc_building import WFCBuilding, build_route_tiles
from src.wfc_tiles import Direction, PortKind, _DIRECTION_OFFSETS


ROUTE_ID = "0:(0, 4, 3)->(0, 0, 0):slide"
TILES = build_route_tiles(PortKind.SLIDE)


def _domains(event):
    return {tuple(cell): tuple(indices) for cell, indices in event["domains"]}


def _directions(path, cell):
    index = path.index(cell)
    neighbors = path[max(0, index - 1):index] + path[index + 1:index + 2]
    return tuple(
        direction.name
        for neighbor in neighbors
        for direction, offset in _DIRECTION_OFFSETS.items()
        if tuple(neighbor[i] - cell[i] for i in range(3)) == offset
    )


def collect_decisions(events):
    """Derive decisions and neighboring changes only from actual snapshots."""
    records = []
    summary = None
    route_attempt = 0
    candidate = None
    filtered = None
    previous = None
    current = None
    counts = Counter()
    for event in events:
        if event.get("route_id") != ROUTE_ID:
            continue
        event_type = event["event"]
        counts[event_type] += 1
        if event_type == "route_summary":
            summary = event
        elif event_type == "route_attempt_start":
            route_attempt = event["route_attempt"]
        elif event_type == "solver_trace":
            solver = event["solver_event"]
            kind = solver["event"]
            counts[kind] += 1
            if kind == "route_candidate":
                candidate = solver
            elif kind == "route_guided_selection":
                filtered = solver
            elif kind == "collapse_selected":
                cell = tuple(solver["selected_cell"])
                previous = _domains(solver)
                previous[cell] = (solver["selected_tile_index"],)
                current = {
                    "iteration": solver["iteration"], "attempt": route_attempt,
                    "cell": cell, "tile": solver["selected_tile_name"],
                    "tile_index": solver["selected_tile_index"],
                    "ports": tuple(name for name, value in solver["selected_tile_ports"]
                                   if value == "slide"),
                    "available": tuple(solver["candidate_tile_names"]),
                    "eligible": tuple(filtered["candidate_tile_names"]),
                    "dfs_directions": _directions(candidate["candidate_path"], cell),
                    "dfs_length": candidate["candidate_path_length"],
                    "neighbors": [], "global_cells": set(),
                    "local": 0, "global": 0, "global_resolved": 0,
                }
            elif kind == "domain_reduction" and current is not None:
                updated = _domains(solver)
                global_phase = "pruning" in solver["phase"]
                current["global" if global_phase else "local"] += solver["possibilities_removed"]
                if global_phase:
                    current["global_resolved"] += len(solver["newly_resolved_cells"])
                    current["global_cells"].update(
                        cell for cell in updated
                        if cell in previous and set(previous[cell]) - set(updated[cell])
                    )
                else:
                    cell = current["cell"]
                    for direction, offset in _DIRECTION_OFFSETS.items():
                        neighbor = tuple(cell[i] + offset[i] for i in range(3))
                        if neighbor not in previous or neighbor not in updated:
                            continue
                        removed = set(previous[neighbor]) - set(updated[neighbor])
                        if removed:
                            current["neighbors"].append({
                                "direction": direction.name, "cell": neighbor,
                                "before": len(previous[neighbor]), "after": len(updated[neighbor]),
                                "removed": tuple(TILES.tiles[i].name for i in sorted(removed)),
                            })
                previous = updated
            elif kind == "collapse_complete":
                current["unresolved"] = solver["unresolved_cell_count"]
                current["feasible"] = solver["required_route_feasible"]
                current["domains"] = _domains(solver)
                records.append(current)
                current = None
    if summary is None or not records or current is not None:
        raise RuntimeError("The route trace is incomplete")
    return summary, records, counts


def _caption(record):
    return (
        f"Iteration {record['iteration']} | cell {record['cell']} | "
        f"{record['tile']} ({'/'.join(record['ports'])})\n"
        f"DFS {len(record['available'])} -> {len(record['eligible'])} tiles | "
        f"local -{record['local']} | global -{record['global']} | "
        f"unresolved {record['unresolved']}"
    )


def _render_frame(record, summary):
    cell = record["cell"]
    grid = summary["grid_dimensions"]
    figure, axes = plt.subplots(1, 3, figsize=(12, 5))
    figure.subplots_adjust(left=.055, right=.985, bottom=.22, top=.79, wspace=.16)
    x_start = max(0, min(cell[1] - 4, grid[1] - 9))
    y_start = max(0, min(cell[2] - 4, grid[2] - 9))
    for axes_item, layer in zip(axes, (cell[0] + 1, cell[0], cell[0] - 1)):
        axes_item.set_aspect("equal")
        axes_item.set_title(f"vertical layer {layer}" if 0 <= layer < grid[0] else "outside grid")
        axes_item.set_xlim(x_start - .5, x_start + 8.5)
        axes_item.set_ylim(y_start - .5, y_start + 8.5)
        axes_item.set_xlabel("x grid cell")
        axes_item.set_ylabel("y grid cell")
        axes_item.grid(color="#e5e7eb", linewidth=0.5)
        for (z, x, y), domain in record["domains"].items():
            if z != layer or not (x_start <= x < x_start + 9 and y_start <= y < y_start + 9):
                continue
            if len(domain) != 1:
                color = "#fef3c7"
            elif domain[0] == TILES.empty_index:
                color = "#f3f4f6"
            else:
                color = "#bfdbfe"
            axes_item.add_patch(Rectangle((x - .45, y - .45), .9, .9,
                                          facecolor=color, edgecolor="#e5e7eb", linewidth=.3))
            if len(domain) == 1 and domain[0] != TILES.empty_index:
                tile = TILES.tiles[domain[0]]
                for direction, offset in _DIRECTION_OFFSETS.items():
                    if tile.ports[direction] is not PortKind.SLIDE:
                        continue
                    if direction in (Direction.U, Direction.D):
                        axes_item.text(x, y, direction.name, fontsize=6, ha="center",
                                       va="center", color="#1e40af")
                    else:
                        axes_item.plot((x, x + .42 * offset[1]),
                                       (y, y + .42 * offset[2]), color="#1d4ed8", linewidth=2)
            if (z, x, y) in record["global_cells"]:
                axes_item.add_patch(Rectangle((x - .45, y - .45), .9, .9,
                                              fill=False, edgecolor="#ea580c", linewidth=1))
        for change in record["neighbors"]:
            z, x, y = change["cell"]
            if z == layer and x_start <= x < x_start + 9 and y_start <= y < y_start + 9:
                axes_item.add_patch(Rectangle((x - .45, y - .45), .9, .9,
                                              fill=False, edgecolor="#0891b2", linewidth=1.5))
                axes_item.text(x, y + .24, f"{change['before']}>{change['after']}",
                               ha="center", fontsize=6, color="#0e7490")
        if layer == cell[0]:
            axes_item.add_patch(Rectangle((cell[1] - .48, cell[2] - .48), .96, .96,
                                          fill=False, edgecolor="#a21caf", linewidth=2.5))
        for name, endpoint, color in (
            ("H", summary["start_cell"], "#16a34a"),
            ("T", summary["end_cell"], "#dc2626"),
        ):
            if endpoint[0] == layer:
                axes_item.text(endpoint[1], endpoint[2], name, color=color,
                               ha="center", va="center", weight="bold", fontsize=10)
    figure.suptitle(_caption(record), fontsize=11, y=.96)
    figure.text(.5, .06, "Blue: fixed route ports   Yellow: unresolved   Grey: empty   "
                "Purple: selected   H/T: head/tail\n"
                "Cyan outline: locally reduced neighbor   Orange outline: globally reduced cell",
                ha="center", fontsize=8)
    stream = BytesIO()
    figure.savefig(stream, format="png", dpi=100)
    plt.close(figure)
    stream.seek(0)
    with Image.open(stream) as image:
        return image.convert("RGB")


def _text(records, counts):
    lines = [
        "DFS-guided WFC: one slide of the complete seed-10 building",
        f"Slide: {ROUTE_ID}",
        "Coordinates: (vertical grid layer, x grid cell, y grid cell).",
        "A DFS-supported candidate path determines required tile ports; "
        "local propagation and global reachability pruning then reduce domains.",
        "",
    ]
    for record in records:
        lines.extend([
            _caption(record).replace("\n", " | "),
            f"  Before DFS: {', '.join(record['available'])}.",
            f"  DFS candidate: {record['dfs_length']} cells; "
            f"connection directions {' and '.join(record['dfs_directions'])}.",
            f"  After DFS: {', '.join(record['eligible'])}. "
            f"Selected {record['tile']} with ports {' and '.join(record['ports'])}.",
        ])
        if record["neighbors"]:
            for change in record["neighbors"]:
                lines.append(
                    f"  Local neighbor {change['direction']} {change['cell']}: "
                    f"domain {change['before']} -> {change['after']}; removed "
                    f"{', '.join(change['removed'])}."
                )
        else:
            lines.append("  Local propagation made no change to an immediate neighbor.")
        lines.append(
            f"  Global pruning removed {record['global']} possibilities and resolved "
            f"{record['global_resolved']} cells; route feasibility: "
            f"{'valid' if record['feasible'] else 'invalid'}."
        )
        lines.append("")
    lines.extend([
        f"Summary: {len(records)} explicit collapses; "
        f"{counts['solver_contradiction']} solver contradictions; "
        f"{max(0, counts['route_attempt_start'] - 1)} route retries.",
        "The GIF frames show these same decisions after their propagation.",
    ])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/dfs_decisions_reproduced"))
    parser.add_argument("--baseline", type=Path, required=True,
                        help="Reproduced DFS-guided building JSON")
    args = parser.parse_args()
    text_path = args.output / "wfc_decisions.txt"
    gif_path = args.output / "wfc_decisions.gif"
    if text_path.exists() or gif_path.exists():
        raise FileExistsError("Decision output already exists; choose a new directory")
    events = []
    building = WFCBuilding(
        4, 10, 10, difficulty=2, seed=10, grid_resolution=3, routing_seed=10,
        max_route_attempts=30, max_route_backtrack_depth=2,
        max_route_recovery_attempts=1, recovery_strategy="none",
        minimum_curvature_radius=.02, endpoint_frontier=True,
        trace_callback=events.append, trace_route_id=ROUTE_ID,
        trace_wfc=True, trace_states=True,
    )
    reference = json.loads(args.baseline.read_text(encoding="utf-8"))
    generated = json.loads(json.dumps(building_to_dict(building), allow_nan=False))
    if generated != reference:
        raise RuntimeError("Traced building does not match the validated baseline JSON")
    summary, records, counts = collect_decisions(events)
    if counts["route_attempt_start"] != 1 or counts["solver_contradiction"]:
        raise RuntimeError("Unexpected retry or contradiction; inspect events before presentation")
    frames = [_render_frame(record, summary) for record in records]
    args.output.mkdir(parents=True, exist_ok=True)
    with gif_path.open("xb") as output:
        frames[0].save(output, format="GIF", save_all=True, append_images=frames[1:],
                       duration=1500, loop=0, optimize=False)
    for frame in frames:
        frame.close()
    with text_path.open("x", encoding="utf-8", newline="\n") as output:
        output.write(_text(records, counts))
    print(f"Saved {len(records)} synchronized decisions to {text_path} and {gif_path}")


if __name__ == "__main__":
    main()
