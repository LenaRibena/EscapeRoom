"""Import exported EscapeRoom centerlines and floor outlines into Blender."""

import json
from pathlib import Path
import sys

import bpy


def _curve(name, points, collection, material, radius, closed=False):
    curve = bpy.data.curves.new(name, "CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = radius
    curve.bevel_resolution = 3
    spline = curve.splines.new("POLY")
    spline.points.add(len(points) - 1)
    for point, position in zip(spline.points, points):
        point.co = (*position, 1.0)
    spline.use_cyclic_u = closed
    curve.materials.append(material)
    obj = bpy.data.objects.new(name, curve)
    collection.objects.link(obj)
    return obj


def _material(name, color):
    material = bpy.data.materials.new(name)
    material.diffuse_color = color
    return material


def import_building(path):
    with Path(path).open(encoding="utf-8") as source:
        data = json.load(source)
    if data["schema_version"] != 1:
        raise ValueError(f"Unsupported schema version: {data['schema_version']}")
    coordinates = data["coordinate_system"]
    if (coordinates["axis_order"], coordinates["up_axis"],
            coordinates["blender_units_per_graph_unit"]) != ("x,y,z", "z", 1.0):
        raise ValueError("Unsupported coordinate system or scale")

    building = data["building"]
    routes = data["routes"]
    for route in routes:
        points = route["geometry"]
        if route["type"] not in ("stair", "slide") or len(points) < 2:
            raise ValueError(f"Invalid route: {route['id']}")
        if route["type"] == "stair" and len(points) != 2:
            raise ValueError(f"Stair must be a straight segment: {route['id']}")

    collection = bpy.data.collections.new("EscapeRoom")
    bpy.context.scene.collection.children.link(collection)
    materials = {
        "stair": _material("EscapeRoom Stair", (1.0, 0.35, 0.04, 1.0)),
        "slide": _material("EscapeRoom Slide", (0.03, 0.42, 0.8, 1.0)),
        "floor": _material("EscapeRoom Floor", (0.35, 0.4, 0.48, 1.0)),
    }
    for route in routes:
        obj = _curve(
            f"{route['type']}: {route['id']}", route["geometry"], collection,
            materials[route["type"]], 0.06,
        )
        obj["route_id"] = route["id"]
        obj["route_type"] = route["type"]

    x_max, y_max = building["length"] - 1, building["width"] - 1
    for index, z in enumerate(building["floor_elevations"]):
        _curve(
            f"Floor {index} outline",
            [(0, 0, z), (x_max, 0, z), (x_max, y_max, z), (0, y_max, z)],
            collection, materials["floor"], 0.015, closed=True,
        )
    return collection


if __name__ == "__main__":
    if "--" not in sys.argv or len(sys.argv[sys.argv.index("--") + 1:]) != 1:
        raise SystemExit("Usage: blender --python import_building.py -- building.json")
    json_path = sys.argv[sys.argv.index("--") + 1]
    imported = import_building(json_path)
    print(f"Imported {len(imported.objects)} objects into {imported.name}")
