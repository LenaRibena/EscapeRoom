"""Run one bounded seed-10 hybrid experiment; never writes production outputs."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import time

from src.json_export import building_to_dict
from src.route_diagnostics import diagnose_routes
from src.wfc_building import RouteGenerationError, WFCBuilding, _edge_name
from src.wfc_tiles import PortKind


ROUTE = "0:(0, 4, 3)->(0, 0, 0):slide"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=("hybrid",), required=True)
    parser.add_argument("--weighting", choices=("uniform", "compact"), default="uniform")
    parser.add_argument("--compactness-strength", type=float, default=0.12)
    parser.add_argument("--isolated", action="store_true")
    parser.add_argument("--routing-seed", type=int, default=10)
    parser.add_argument("--grid-resolution", type=int, default=3)
    parser.add_argument("--max-cells", type=int)
    parser.add_argument("--relative-budget", action="store_true")
    parser.add_argument("--detour-fraction", type=float, default=0.25)
    parser.add_argument("--detour-cap", type=int, default=4)
    parser.add_argument("--minimum-clearance", type=float)
    parser.add_argument("--max-backtracks", type=int, default=20)
    parser.add_argument("--max-seconds", type=float, default=30)
    parser.add_argument("--route-attempts", type=int, default=30)
    parser.add_argument("--trace-states", action="store_true")
    parser.add_argument("--trace-route-id")
    parser.add_argument("--geometry-preference", type=float, default=0.0)
    parser.add_argument("--geometry-backtracking", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--building-output", type=Path,
                        help="also export the accepted building as a standalone JSON")
    args = parser.parse_args()
    if args.relative_budget and args.max_cells is not None:
        parser.error("--relative-budget and --max-cells are mutually exclusive")
    if not (args.relative_budget or args.max_cells is not None):
        parser.error("hybrid requires --relative-budget or --max-cells")
    if (not math.isfinite(args.detour_fraction) or args.detour_fraction < 0
            or args.detour_cap < 0):
        parser.error("Detour allowance must be finite and non-negative")
    if args.minimum_clearance is not None and (
        not math.isfinite(args.minimum_clearance) or args.minimum_clearance < 0
    ):
        parser.error("Minimum clearance must be finite and non-negative")
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.building_output is not None:
        if args.isolated:
            parser.error("--building-output requires a full building")
        if args.building_output == args.output:
            parser.error("--building-output must differ from --output")
        if args.building_output.exists():
            raise FileExistsError(args.building_output)
    log_path = args.output.with_suffix(".events.jsonl")
    if args.building_output == log_path:
        parser.error("--building-output must differ from the event log")
    if log_path.exists():
        raise FileExistsError(log_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    events = []
    log = log_path.open("x", encoding="utf-8", newline="\n")

    def trace(event):
        events.append(event)
        if (
            event["event"] in (
                "route_attempt_start", "route_attempt_failure",
                "route_attempt_geometry_rejection", "route_attempt_success", "hybrid_route_budget",
            )
            or (event["event"] == "solver_trace" and event["solver_event"]["event"] in (
                "collapse_selected", "collapse_complete",
                "domain_reduction", "solver_contradiction", "hybrid_length_choice",
                "hybrid_length_propagation", "hybrid_backtrack", "hybrid_geometry_rejection",
            ))
        ):
            log.write(json.dumps(event, allow_nan=False) + "\n")
            log.flush()

    start = time.perf_counter()
    try:
        building = WFCBuilding(
            4, 10, 10, difficulty=2, seed=10, routing_seed=args.routing_seed,
            grid_resolution=args.grid_resolution,
            max_route_attempts=args.route_attempts, max_route_backtrack_depth=2,
            max_route_recovery_attempts=1, recovery_strategy="none",
            minimum_curvature_radius=.02, endpoint_frontier=True,
            routing_strategy=args.strategy, hybrid_weighting=args.weighting,
            hybrid_compactness_strength=args.compactness_strength,
            only_route_id=ROUTE if args.isolated else None,
            trace_callback=trace, trace_route_id=args.trace_route_id,
            trace_wfc=True, trace_states=args.trace_states,
            generate_routes=False,
        )
        building.hybrid_max_cells = args.max_cells
        building.hybrid_relative_budget = args.relative_budget
        building.hybrid_detour_fraction = args.detour_fraction
        building.hybrid_detour_cap = args.detour_cap
        building.hybrid_minimum_clearance = args.minimum_clearance
        building.hybrid_max_backtracks = args.max_backtracks
        building.hybrid_max_seconds = args.max_seconds
        building.hybrid_geometry_preference = args.geometry_preference
        building.hybrid_geometry_backtracking = args.geometry_backtracking
        frozen_graph = {
            "nodes": [{"position": node, "attributes": dict(attributes)}
                      for node, attributes in building.graph.nodes(data=True)],
            "edges": [{"source": source, "target": target,
                       "attributes": {**attributes, "kind": _edge_name(attributes)}}
                      for source, target, attributes in building.graph.edges(data=True)],
            "solution_path": building.path,
        }
        reference = json.loads(Path("results/final/building/building_seed10_hybrid.json").read_text(
            encoding="utf-8"
        ))
        if json.loads(json.dumps(frozen_graph)) != reference["graph"]:
            raise RuntimeError("Graph differs from the accepted hybrid building before routing")
        log.write(json.dumps({"event": "frozen_graph_verified"}) + "\n")
        log.flush()
        failure = None
        try:
            building.generate_routes()
        except RouteGenerationError as error:
            failure = {"message": str(error), "classification": error.failure_classification,
                       "details": error.failure_details}
    finally:
        log.close()
    elapsed = time.perf_counter() - start
    counts = Counter(event["event"] for event in events)
    details = [event["solver_event"] for event in events if event["event"] == "solver_trace"]
    bounded_selections = [event for event in details if event["event"] == "hybrid_length_choice"]
    slides = []
    for route in building.routes:
        if route.kind is not PortKind.SLIDE:
            continue
        horizontal = [(b[1] - a[1], b[2] - a[2])
                      for a, b in zip(route.cells, route.cells[1:]) if a[1:] != b[1:]]
        slides.append({
            "id": route.route_id, "discrete_cells": len(route.cells),
            "cells": route.cells, "geometry": route.accepted_geometry,
            "horizontal_turns": sum(a != b for a, b in zip(horizontal, horizontal[1:])),
            "curve_length": sum(math.dist(a, b) for a, b in
                                zip(route.accepted_geometry, route.accepted_geometry[1:])),
            "attempts": route.attempts,
            "budgets": [event for event in events if event["event"] == "hybrid_route_budget"
                        and event["route_id"] == route.route_id],
            "explicit_choices": sum(
                event["event"] == "solver_trace" and event["route_id"] == route.route_id
                and event["solver_event"]["event"] == "hybrid_length_choice"
                for event in events
            ),
            "non_forced_choices": sum(
                event["event"] == "solver_trace" and event["route_id"] == route.route_id
                and event["solver_event"]["event"] == "hybrid_length_choice"
                and len(event["solver_event"]["eligible"]) > 1
                for event in events
            ),
        })
    diagnostics = diagnose_routes(building)
    clearances = [pair.minimum_clearance for pair in diagnostics.route_pairs
                  if pair.minimum_clearance is not None]
    data = {
        "strategy": args.strategy, "weighting": args.weighting, "isolated": args.isolated,
        "compactness_strength": args.compactness_strength,
        "max_cells": args.max_cells, "routing_seed": args.routing_seed,
        "relative_budget": args.relative_budget, "detour_fraction": args.detour_fraction,
        "detour_cap": args.detour_cap, "minimum_clearance_required": args.minimum_clearance,
        "success": failure is None, "failure": failure,
        "geometry_preference": args.geometry_preference,
        "geometry_backtracking": args.geometry_backtracking,
        "trace_route_id": args.trace_route_id,
        "seconds": elapsed, "routes": slides,
        "graph": {
            "nodes": [(node, dict(attributes))
                      for node, attributes in sorted(building.graph.nodes(data=True))],
            "edges": [(source, target, dict(attributes)) for source, target, attributes
                      in sorted(building.graph.edges(data=True))],
            "solution_path": building.path,
        },
        "building": None if args.isolated or failure is not None else building_to_dict(building),
        "attempts": building.total_wfc_attempts, "recoveries": building.recovery_attempts,
        "failures": counts["route_attempt_failure"],
        "geometry_rejections": counts["route_attempt_geometry_rejection"],
        "solver_contradictions": sum(event["event"] == "solver_contradiction" for event in details),
        "explicit_collapses": sum(event["event"] in ("collapse_selected", "hybrid_length_choice")
                                  for event in details),
        "hybrid_selections": len(bounded_selections),
        "non_forced_collapses": sum(len(event["eligible"]) > 1
                                     for event in bounded_selections),
        "backtracks": sum(event["event"] == "hybrid_backtrack" for event in details),
        "backtracked_geometry_rejections": sum(
            event["event"] == "hybrid_geometry_rejection" for event in details
        ),
        "warnings": diagnostics.warnings, "logical_errors": diagnostics.logical_errors,
        "minimum_pair_clearance": min(clearances) if clearances else None,
        "pair_clearances": [
            {"route_ids": pair.route_ids, "minimum_clearance": pair.minimum_clearance,
             "warnings": pair.warnings, "closest_segments": pair.closest_segments}
            for pair in diagnostics.route_pairs if pair.warnings
        ],
        "events": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(data, output, indent=2, allow_nan=False)
        output.write("\n")
    if failure is None and args.building_output is not None:
        args.building_output.parent.mkdir(parents=True, exist_ok=True)
        with args.building_output.open("x", encoding="utf-8", newline="\n") as output:
            json.dump(data["building"], output, indent=2, allow_nan=False)
            output.write("\n")
    print(f"{args.strategy} {args.weighting}: {len(slides)} slides, "
          f"{data['non_forced_collapses']}/{data['explicit_collapses']} non-forced, "
          f"{elapsed:.2f}s", flush=True)
    if failure is not None:
        print(f"FAILED: {failure['message']}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
