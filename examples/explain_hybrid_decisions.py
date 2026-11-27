"""Explain an accepted hybrid replay without changing the solver or its RNG."""

import argparse
from bisect import bisect
from collections import Counter
import hashlib
from itertools import accumulate
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile

from src.hybrid_wfc import _horizontal_turn_cost, _length_lower_bound, relative_cell_budget
from src.wfc_building import WFCBuilding, build_route_tiles
from src.wfc_tiles import Direction, PortKind, _DIRECTION_OFFSETS


ROUTE_ID = "0:(0, 4, 3)->(0, 0, 0):slide"
ACCEPTED_HASH = "dac79b23be631eb5611cfdf295107f8c2658cd77a523c11166743b804948ff0c"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(prefix, successor, end, horizontal_requirement, strength, geometry_strength):
    """Recalculate the implementation's inputs and weight for a replayed candidate."""
    completed = sum(a[1:] != b[1:] for a, b in zip(prefix, prefix[1:]))
    completed += int(prefix[-1][1:] != successor[1:])
    horizontal_distance = abs(successor[1] - end[1]) + abs(successor[2] - end[2])
    horizontal_needed = max(horizontal_distance, horizontal_requirement - completed)
    bound = _length_lower_bound(prefix, successor, end, horizontal_requirement)
    turn_cost = _horizontal_turn_cost(prefix, successor)
    compact_cost = strength * max(0, bound - len(prefix))
    geometry_cost = geometry_strength * turn_cost
    weight = max(1e-12, math.exp(-compact_cost - geometry_cost))
    return completed, horizontal_distance, horizontal_needed, bound, turn_cost, compact_cost, geometry_cost, weight


def turn_description(prefix, successor):
    directions = [(b[1] - a[1], b[2] - a[2])
                  for a, b in zip(prefix, prefix[1:]) if a[1:] != b[1:]]
    delta = (successor[1] - prefix[-1][1], successor[2] - prefix[-1][2])
    if not directions:
        return "no preceding horizontal direction"
    run = 1
    for direction in reversed(directions[:-1]):
        if direction != directions[-1]:
            break
        run += 1
    if delta == (0, 0):
        return f"vertical; last horizontal direction {directions[-1]}, run {run}"
    if delta == directions[-1]:
        return f"continues horizontal direction {directions[-1]}, run {run}"
    reversal = delta == (-directions[-1][0], -directions[-1][1])
    return (f"{'reversal' if reversal else 'turn'} from {directions[-1]} to {delta}; "
            f"preceding run {run}, cost {'3' if reversal else '1'} + 1/{run}")


def explain(record, reference, original_text, replay_text, route_seed,
            expected_hash=ACCEPTED_HASH, new_execution=False):
    if digest(reference) != expected_hash:
        raise ValueError("Reference JSON is not the accepted building")
    if original_text.read_bytes() != replay_text.read_bytes():
        raise ValueError("Replayed text differs from the accepted demonstration")
    building = json.loads(reference.read_text(encoding="utf-8"))
    if not record["success"] or record["building"] != building:
        raise ValueError("Replay did not reproduce the accepted building")
    expected = {
        "strategy": "hybrid", "weighting": "compact", "compactness_strength": 0.12,
        "routing_seed": 10, "relative_budget": True, "detour_fraction": 0.25,
        "detour_cap": 4, "minimum_clearance_required": 0.25,
        "geometry_preference": 1.5, "geometry_backtracking": True,
        "trace_route_id": ROUTE_ID, "backtracks": 0,
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("Replay settings differ from the accepted execution")
    route, = [item for item in record["routes"] if item["id"] == ROUTE_ID]
    cells = [tuple(cell) for cell in route["cells"]]
    if len(cells) != route["discrete_cells"]:
        raise ValueError("Recorded route length does not match its cells")
    budget, = route["budgets"]
    lower_bound = budget["lower_bound_cells"]
    max_cells = budget["max_cells"]
    if relative_cell_budget(lower_bound, 0.25, 4) != max_cells:
        raise ValueError("Unexpected recorded relative budget")
    resolution = building["generation"]["grid_resolution"]
    vertical = building["generation"]["vertical_grid_resolution"]
    if not new_execution and (resolution != 3 or len(cells) != 26
                              or lower_bound != 22 or max_cells != 26):
        raise ValueError("Replay is not the accepted resolution-3 slide")
    wfc = WFCBuilding(4, 10, 10, difficulty=2, seed=10, grid_resolution=resolution,
                      vertical_grid_resolution=vertical,
                      generate_routes=False)
    horizontal_requirement = wfc._required_slide_horizontal_steps(cells[0], cells[-1])
    tiles = build_route_tiles(PortKind.SLIDE).tiles
    choices = [event for event in record["events"] if event["event"] == "hybrid_length_choice"]
    steps = {event["iteration"]: event for event in record["events"]
             if event["event"] == "hybrid_length_propagation"}
    if not choices or (not new_execution and
                       (len(choices) != 21 or sum(len(c["eligible"]) > 1 for c in choices) != 18)):
        raise ValueError("Unexpected choice count")
    expected_iterations = list(range(1, len(choices) + 1))
    if [c["iteration"] for c in choices] != expected_iterations or set(steps) != set(expected_iterations):
        raise ValueError("Incomplete or out-of-order decision/propagation events")
    if route_seed != 10:
        raise ValueError("Unexpected slide-0 attempt seed")
    replay_rng = random.Random(route_seed)
    lines = [
        "ACCEPTED SLIDE 0: EXPLAINED HYBRID WFC DECISIONS",
        "Route (0,4,3) -> (0,0,0); cells use (vertical layer, x, y).",
        "Directions N/E/S/W are +y/+x/-y/-x; U/D are +layer/-layer.",
        f"The {len(choices)} numbered choices correspond to the recorded execution.",
        ("The building and TXT were produced from this same recorded execution."
         if new_execution else
         "The replay reproduced the accepted building and original TXT byte-for-byte."),
        "",
        "HARD CONSTRAINTS, PROPAGATION AND SOFT WEIGHTS",
        "Local compatibility requires adjacent faces to have matching port kinds, including",
        "empty faces. Initially invalid tiles are removed; each trial fixes one tile, then",
        "propagates neighbor-domain restrictions until stable. The domain listed per choice",
        "is the remaining domain before that choice's hard candidate filters, not all 22 tiles.",
        "Global BFS builds an over-approximate directed possible-port graph, reaches forward",
        "from the start and backward from the end, and removes slide tiles in cells outside",
        "their intersection. In this run tile-aware BFS is off. This envelope permits",
        "different possible tiles to support different edges; it is not proof of a valid route.",
        "BFS already uses horizontal/downward slide edges; the separate prefix/candidate checks",
        "also enforce no upward step for the particular chosen tile, preventing a possible",
        "BFS path from being mistaken for the committed tile's directed continuation.",
        "The committed prefix cannot revisit a cell: a single tile assignment cannot encode",
        "the repeated visits as a simple route, and a loop is not a valid slide. The relaxed",
        "initial shortest-walk bound can revisit cells; it is only a lower bound.",
        "A candidate can pass topology and still fail continuous interpolation, curvature,",
        "collision or 0.25 graph-unit centerline clearance after the route is completed.",
        "These checks are hard, while compactness and turn penalties only adjust sampling weights.",
        "After selecting a trial, local counts include propagation following global changes;",
        "global counts are possibilities removed by BFS pruning, excluding direct collapse.",
        "Thousands can disappear when a choice cuts off a large possible-path envelope; zero",
        "means BFS found nothing further to remove at that step, not that BFS was skipped.",
        "",
        "BUDGET AND WEIGHT EQUATIONS",
        f"Initial directed, port-compatible relaxed walk lower bound = {lower_bound} cells.",
        f"Maximum = {lower_bound} + min(4, ceil(0.25 * ({lower_bound} - 1))) = {max_cells} cells.",
        f"Required horizontal travel H = {horizontal_requirement} refined-grid steps.",
        "For prefix P including the selected frontier, and proposed successor s:",
        "h_done = horizontal steps already in P plus 1 if cell->s is horizontal;",
        "h_distance = Manhattan XY distance from s to end;",
        "h_needed = max(h_distance, H - h_done);",
        "B = len(P) + 1 + (s.layer - end.layer) + h_needed.",
        f"B is a lower bound on the entire route in cells; B > {max_cells} is rejected.",
        "compact = 0.12 * max(0, B - len(P)).",
        "turn = 0 for vertical, first horizontal or straight continuation; otherwise",
        "turn = (3 for reversal, 1 for a turn) + 1 / preceding horizontal run length.",
        "Vertical steps do not break that run. geometry = 1.5 * turn.",
        "weight = max(1e-12, exp(-compact - geometry)); probability = weight / sum(weights).",
        "The factors exp(-compact) and exp(-geometry) are shown for interpretation; the",
        "implementation uses a single exp of the summed costs (floating-point rounding may differ).",
        "Displayed full-precision values are Python binary floats, not values recovered from",
        "the rounded percentages in the original text. RNG sampling can choose any positive",
        "probability; a single eligible candidate is forced, not a substantive random choice.",
        "Sampling draws below are reconstructed after the run with a separate Random(10),",
        "not read from rounded percentages or taken from the solver's random generator.",
        "The reconstructed draw must reproduce every recorded selected candidate.",
        "",
    ]
    for choice in choices:
        number = choice["iteration"]
        prefix = [tuple(cell) for cell in choice["prefix"]]
        cell = tuple(choice["selected_cell"])
        if prefix != cells[:len(prefix)] or cell != prefix[-1] or choice["max_cells"] != max_cells:
            raise ValueError(f"Choice {number}: prefix differs from accepted route")
        previous = prefix[-2]
        incoming, = [direction for direction, offset in _DIRECTION_OFFSETS.items()
                     if tuple(cell[i] + offset[i] for i in range(3)) == previous]
        domains = {tuple(key): value for key, value in choice["domains"]}
        eligible = choice["eligible"]
        rejected = choice["rejected"]
        indices = [item[0] for item in eligible] + [item[0] for item in rejected]
        if Counter(indices) != Counter(domains[cell]):
            raise ValueError(f"Choice {number}: candidate domain is incomplete")
        total = sum(item[2] for item in eligible)
        if not math.isclose(sum(item[3] for item in eligible), 1.0, rel_tol=0, abs_tol=1e-14):
            raise ValueError(f"Choice {number}: probabilities do not sum to one")
        selected, = [item for item in eligible if item[0] == choice["selected_tile_index"]]
        draw = replay_rng.random()
        threshold = draw * total
        cumulative = list(accumulate(item[2] for item in eligible))
        chosen_index = bisect(cumulative, threshold, 0, len(cumulative) - 1)
        if eligible[chosen_index][0] != selected[0]:
            raise ValueError(f"Choice {number}: reconstructed random draw selects another tile")
        lines += [
            f"Choice {number} | cell {cell} | {selected[1]} | {len(eligible)} eligible | sampled {selected[3]:.1%}",
            f"  Position: cell {len(prefix)} of {len(cells)}, with {len(cells) - len(prefix)} cells after it in the accepted route.",
            f"  Committed prefix (including frontier): {prefix}",
            f"  Entering from {previous} through {incoming.name}; end = {cells[-1]}.",
            "  Domain before candidate filters: " + ", ".join(
                f"{tiles[i].name} ({'+'.join(d.name for d, p in tiles[i].ports.items() if p is PortKind.SLIDE) or 'no slide ports'})"
                for i in domains[cell]),
        ]
        if rejected:
            lines.append("  Rejected by hard filters:")
            for index, reason in rejected:
                tile = tiles[index]
                ports = [direction for direction, port in tile.ports.items() if port is PortKind.SLIDE]
                explanation = {
                    "local": "incoming/tile ports or successor support are locally incompatible",
                    "direction_or_loop": "successor ascends, leaves the grid, or revisits the prefix",
                    "propagation_or_global": "trial propagation, BFS or propagated prefix contradicted a constraint",
                    "length_lower_bound": f"the optimistic full-route cell bound exceeds {max_cells}",
                }[reason]
                if reason == "length_lower_bound":
                    outgoing, = [direction for direction in ports if direction != incoming]
                    offset = _DIRECTION_OFFSETS[outgoing]
                    successor = tuple(cell[i] + offset[i] for i in range(3))
                    bound = _length_lower_bound(prefix, successor, cells[-1], horizontal_requirement)
                    if bound <= max_cells:
                        raise ValueError(f"Choice {number}: rejected bound does not exceed budget")
                    explanation += f" (successor {successor}, B={bound})"
                lines.append(f"    {tile.name}: {reason} - {explanation}.")
        else:
            lines.append("  Rejected by these hard filters: none.")
        lines.append("  Eligible candidates (all costs and weights from the actual replay):")
        for index, name, weight, probability in eligible:
            tile = tiles[index]
            ports = [direction for direction, port in tile.ports.items() if port is PortKind.SLIDE]
            outgoing, = [direction for direction in ports if direction != incoming]
            offset = _DIRECTION_OFFSETS[outgoing]
            successor = tuple(cell[i] + offset[i] for i in range(3))
            done, distance, needed, bound, turn, compact, geom, recomputed = score(
                prefix, successor, cells[-1], horizontal_requirement, 0.12, 1.5)
            if recomputed != weight or probability != weight / total or bound > max_cells:
                raise ValueError(f"Choice {number}, {name}: reconstructed weight/probability mismatch")
            lines += [
                f"    {name} [{incoming.name}+{outgoing.name}] -> {successor}:",
                f"      h_done={done}, h_distance={distance}, h_needed=max({distance}, {horizontal_requirement}-{done})={needed};",
                f"      B={len(prefix)}+1+({successor[0]}-{cells[-1][0]})+{needed}={bound}; compact=0.12*({bound}-{len(prefix)})={compact!r}.",
                f"      {turn_description(prefix, successor)}; turn cost={turn!r}; geometry=1.5*{turn!r}={geom!r}.",
                f"      exp(-compact)={math.exp(-compact)!r}; exp(-geometry)={math.exp(-geom)!r};",
                f"      weight=max(1e-12, exp(-{compact!r}-{geom!r}))={weight!r}.",
                f"      probability={weight!r}/{total!r}={probability!r} ({probability:.2%}).",
            ]
        step = steps[number]
        selected_ports = [direction for direction, port in tiles[selected[0]].ports.items()
                          if port is PortKind.SLIDE]
        outgoing, = [direction for direction in selected_ports if direction != incoming]
        offset = _DIRECTION_OFFSETS[outgoing]
        successor = tuple(cell[i] + offset[i] for i in range(3))
        if cells[len(prefix)] != successor:
            raise ValueError(f"Choice {number}: selected tile does not extend accepted route")
        lines += [
            f"  Normalization: sum weights = {' + '.join(repr(item[2]) for item in eligible)} = {total!r};",
            f"    sum probabilities = {sum(item[3] for item in eligible)!r} (100% within floating-point rounding).",
            f"  Independent RNG replay: u={draw!r}, u*sum={threshold!r}; cumulative weights = {cumulative!r}.",
            f"  Sampled {selected[1]}: enters via {incoming.name}, exits via {outgoing.name} to {successor}.",
            ("  Stochastic: multiple eligible tiles had positive weights; a lower-probability tile can be sampled."
             if len(eligible) > 1 else "  Forced: this is the only eligible tile (100%)."),
            f"  Post-selection propagation: local removed {step['local_removed']}; global BFS removed {step['global_removed']} tile possibilities.",
            "",
        ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    parser.add_argument("--events", type=Path, required=True, help="replay .events.jsonl")
    parser.add_argument("--replay-visualization", type=Path, required=True)
    parser.add_argument("--final", type=Path, default=Path("results/final"))
    parser.add_argument("--new-building", type=Path,
                        help="building from a new verified execution, instead of the accepted res-3 reference")
    parser.add_argument("--building-sha256",
                        help="expected SHA-256 of --new-building")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if bool(args.new_building) != bool(args.building_sha256):
        parser.error("--new-building and --building-sha256 must be supplied together")
    record = json.loads(args.record.read_text(encoding="utf-8"))
    with args.events.open(encoding="utf-8") as events:
        starts = [event for line in events if (event := json.loads(line)).get("event")
                  == "route_attempt_start" and event.get("route_id") == ROUTE_ID]
    if len(starts) != 1 or starts[0]["route_attempt"] != 1:
        raise ValueError("Expected one accepted slide-0 attempt")
    new_execution = args.new_building is not None
    reference = args.new_building if new_execution else args.final / "building" / "building_seed10_hybrid.json"
    original_text = (args.replay_visualization / "decisions.txt" if new_execution
                     else args.final / "algorithm" / "wfc_decisions.txt")
    if new_execution:
        with tempfile.TemporaryDirectory() as directory:
            rendered = Path(directory) / "decisions"
            subprocess.run(
                [sys.executable, "-m", "examples.hybrid_length_trace", str(args.record),
                 "--reference-building", str(reference), "--route-id", ROUTE_ID,
                 "--output", str(rendered)],
                check=True,
            )
            text = explain(record, reference, original_text, rendered / "decisions.txt",
                           starts[0]["routing_seed"], args.building_sha256,
                           new_execution=True)
    else:
        text = explain(record, reference, original_text,
                       args.replay_visualization / "decisions.txt",
                       starts[0]["routing_seed"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as output:
        output.write(text)
    print(f"Explained {sum(event['event'] == 'hybrid_length_choice' for event in record['events'])} verified choices in {args.output}")


if __name__ == "__main__":
    main()
