"""Export an already generated building without reconstructing its routes."""

from collections import Counter
import json
from pathlib import Path

from src.route_diagnostics import diagnose_routes
from src.wfc_building import WFCBuilding, _edge_name


def building_to_dict(building: WFCBuilding) -> dict:
    """Return a versioned JSON-compatible representation of a valid building."""
    graph_edges = []
    for source, target, attributes in building.graph.edges(data=True):
        kind = _edge_name(attributes)
        if kind not in ("floor", "stair", "slide"):
            raise ValueError(f"Invalid graph edge label {kind!r} for {source} -> {target}")
        graph_edges.append((source, target, attributes, kind))
    expected = Counter(
        (source, target, kind)
        for source, target, _, kind in graph_edges
        if kind in ("stair", "slide")
    )
    actual = Counter(
        (route.source, route.target, route.kind.value) for route in building.routes
    )
    if expected != actual:
        raise ValueError(
            "Accepted routes do not match graph stair/slide edges: "
            f"missing={list((expected - actual).elements())}, "
            f"unexpected={list((actual - expected).elements())}"
        )
    route_ids = [route.route_id for route in building.routes]
    if len(set(route_ids)) != len(route_ids):
        raise ValueError("Accepted route IDs must be unique")

    diagnostics = diagnose_routes(building)
    if diagnostics.logical_errors:
        raise ValueError(
            "Building has logical route errors:\n" + "\n".join(diagnostics.logical_errors)
        )

    return {
        "schema_version": 1,
        "coordinate_system": {
            "axis_order": "x,y,z",
            "up_axis": "z",
            "unit": "graph_unit",
            "blender_units_per_graph_unit": 1.0,
            "origin": [0, 0, 0],
        },
        "generation": {
            "seed": building.requested_seed,
            "requested_seed": building.requested_seed,
            "accepted_seed": building.accepted_seed,
            "building_retry_count": building.building_retry_count,
            "grid_resolution": building.grid_resolution,
            "vertical_grid_resolution": building.vertical_grid_resolution,
        },
        "building": {
            "floors": building.floors,
            "length": building.length,
            "width": building.width,
            "floor_elevations": list(range(building.floors)),
            "start": building.start_pos,
            "exit": building.exit_pos,
        },
        "graph": {
            "nodes": [
                {"position": node, "attributes": dict(attributes)}
                for node, attributes in building.graph.nodes(data=True)
            ],
            "edges": [
                {
                    "source": source,
                    "target": target,
                    "attributes": {**attributes, "kind": kind},
                }
                for source, target, attributes, kind in graph_edges
            ],
            "solution_path": building.path,
        },
        "routes": [
            {
                "id": route.route_id,
                "type": route.kind.value,
                "source": route.source,
                "target": route.target,
                "geometry": route.accepted_geometry,
            }
            for route in building.routes
        ],
        "diagnostics": {"warnings": diagnostics.warnings},
    }


def export_building_json(building: WFCBuilding, destination: str | Path) -> None:
    """Write a new UTF-8 JSON file, refusing to overwrite an existing file."""
    content = json.dumps(building_to_dict(building), indent=2, allow_nan=False) + "\n"
    with Path(destination).open("x", encoding="utf-8", newline="\n") as output:
        output.write(content)
