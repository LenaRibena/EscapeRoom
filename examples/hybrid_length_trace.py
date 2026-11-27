"""Write a text trace of recorded, accepted length-bounded hybrid decisions."""

import argparse
import json
from pathlib import Path

from src.wfc_building import build_route_tiles
from src.wfc_tiles import PortKind


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--route-id")
    parser.add_argument("--reference-building", type=Path,
                        help="Require the recorded building to match this accepted JSON")
    args = parser.parse_args()
    text_path = args.output / "decisions.txt"
    if text_path.exists():
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
    args.output.mkdir(parents=True, exist_ok=True)
    tiles = build_route_tiles(PortKind.SLIDE)
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
    with text_path.open("x", encoding="utf-8", newline="\n") as output:
        output.write("\n".join(lines))
    print(f"Wrote {len(choices)} recorded decisions to {text_path}")


if __name__ == "__main__":
    main()
