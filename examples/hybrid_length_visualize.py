"""Visualize recorded, accepted length-bounded hybrid decisions."""

import argparse
from io import BytesIO
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

from src.wfc_building import build_route_tiles
from src.wfc_tiles import PortKind, Direction, _DIRECTION_OFFSETS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--route-id")
    parser.add_argument("--reference-building", type=Path,
                        help="Require the recorded building to match this accepted JSON")
    args = parser.parse_args()
    targets = [args.output / name for name in ("decisions.txt", "decisions.gif")]
    if any(path.exists() for path in targets):
        raise FileExistsError("Output already exists")
    data = json.loads(args.record.read_text(encoding="utf-8"))
    if args.reference_building is not None:
        reference = json.loads(args.reference_building.read_text(encoding="utf-8"))
        if data.get("building") != reference:
            raise ValueError("Recorded building does not match the reference building")
    routes = [route for route in data["routes"]
              if args.route_id is None or route["id"] == args.route_id]
    if len(routes) != 1 or data["backtracks"]:
        raise ValueError("Select exactly one accepted slide with no traced backtracking")
    if len(data["routes"]) > 1 and data.get("trace_route_id") != routes[0]["id"]:
        raise ValueError("Full-building snapshots must be traced for the selected slide only")
    events = data["events"]
    choices = [event for event in events if event["event"] == "hybrid_length_choice"]
    propagated = {event["iteration"]: event for event in events
                  if event["event"] == "hybrid_length_propagation"}
    if not choices or any(event["iteration"] not in propagated for event in choices):
        raise ValueError("Incomplete recorded decisions")
    if any("domains" not in propagated[event["iteration"]] for event in choices):
        raise ValueError("Recorded domain snapshots are required")
    route = routes[0]
    args.output.mkdir(parents=True, exist_ok=True)
    tiles = build_route_tiles(PortKind.SLIDE)
    building = data["building"]
    resolution = building["generation"]["grid_resolution"]
    grid_x = (building["building"]["length"] - 1) * resolution + 1
    grid_y = (building["building"]["width"] - 1) * resolution + 1
    frames = []
    lines = [
        "Accepted slide, length-bounded hybrid.",
        "Coordinates: (vertical grid layer, x, y). The bound uses the committed",
        "prefix, horizontal-travel requirement, directed motion and local/global",
        "propagation. Passing is a necessary feasibility check, not a proof of",
        "a complete route. No DFS path determines the sampled tile.",
        f"Budget: {choices[0]['max_cells']} cells; accepted: {route['discrete_cells']} cells.",
        "Backtracks in this accepted execution: 0.",
        "",
    ]
    for event in choices:
        step = propagated[event["iteration"]]
        probability = next(item[3] for item in event["eligible"]
                           if item[0] == event["selected_tile_index"])
        caption = (f"Choice {event['iteration']} | cell {tuple(event['selected_cell'])} | "
                   f"{event['selected_tile_name']} | "
                   f"{len(event['eligible'])} eligible | sampled {probability:.1%}")
        lines.extend([caption, "  Candidate weights and probabilities:"] + [
            f"    {name}: weight {weight:.6f}; probability {probability:.2%}"
            for _, name, weight, probability in event["eligible"]
        ] + [
            "  Rejected: " + (", ".join(
                f"{tiles.tiles[index].name} ({reason})" for index, reason in event["rejected"]
            ) or "none"),
            f"  After selection: local -{step['local_removed']}; "
            f"global -{step['global_removed']} tile possibilities.",
            "",
        ])
        cell = event["selected_cell"]
        domains = {tuple(key): value for key, value in step["domains"]}
        fig, axes = plt.subplots(1, 3, figsize=(12, 5))
        fig.subplots_adjust(left=.055, right=.985, bottom=.24, top=.8, wspace=.16)
        x0 = max(0, min(cell[1] - 4, grid_x - 9))
        y0 = max(0, min(cell[2] - 4, grid_y - 9))
        for ax, layer in zip(axes, (cell[0] + 1, cell[0], cell[0] - 1)):
            ax.set_aspect("equal")
            ax.set(xlim=(x0 - .5, x0 + 8.5), ylim=(y0 - .5, y0 + 8.5),
                   xlabel="x", ylabel="y", title=f"vertical layer {layer}")
            ax.grid(color="#e5e7eb", linewidth=.5)
            for (z, x, y), domain in domains.items():
                if z != layer or not (x0 <= x < x0 + 9 and y0 <= y < y0 + 9):
                    continue
                color = "#fef3c7" if len(domain) > 1 else (
                    "#f3f4f6" if domain[0] == tiles.empty_index else "#bfdbfe"
                )
                ax.add_patch(Rectangle((x - .45, y - .45), .9, .9,
                                       facecolor=color, edgecolor="#e5e7eb", linewidth=.3))
                if len(domain) == 1 and domain[0] != tiles.empty_index:
                    tile = tiles.tiles[domain[0]]
                    for direction, offset in _DIRECTION_OFFSETS.items():
                        if tile.ports[direction] is not PortKind.SLIDE:
                            continue
                        if direction in (Direction.U, Direction.D):
                            ax.text(x, y, direction.name, fontsize=6, ha="center",
                                    va="center", color="#1e40af")
                        else:
                            ax.plot((x, x + .42 * offset[1]),
                                    (y, y + .42 * offset[2]), color="#1d4ed8", linewidth=2)
            if layer == cell[0]:
                ax.add_patch(Rectangle((cell[1] - .48, cell[2] - .48), .96, .96,
                                       fill=False, edgecolor="#a21caf", linewidth=2.5))
        fig.suptitle(caption + f"\nLocal -{step['local_removed']}; "
                     f"global -{step['global_removed']} possibilities", fontsize=11, y=.96)
        probability_text = "   |   ".join(
            f"{name}: {candidate_probability:.1%}"
            for _, name, _, candidate_probability in event["eligible"]
        )
        fig.text(.5, .12, probability_text, ha="center", fontsize=7)
        fig.text(.5, .04, "Blue: fixed slide tile   Yellow: unresolved   Grey: empty   "
                 "Purple: sampled cell",
                 ha="center", fontsize=8)
        stream = BytesIO()
        fig.savefig(stream, format="png", dpi=100)
        plt.close(fig)
        stream.seek(0)
        with Image.open(stream) as image:
            frames.append(image.convert("RGB"))
    with targets[0].open("x", encoding="utf-8", newline="\n") as output:
        output.write("\n".join(lines))
    with targets[1].open("xb") as output:
        frames[0].save(output, format="GIF", save_all=True, append_images=frames[1:],
                       duration=1500, loop=0, optimize=False)
    for frame in frames:
        frame.close()
    print(f"Rendered {len(frames)} recorded decisions")


if __name__ == "__main__":
    main()
