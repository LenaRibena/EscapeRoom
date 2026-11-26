from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src.json_export import building_to_dict, export_building_json
from src.route_diagnostics import diagnose_routes
from src.wfc_building import WFCBuilding, _edge_name
from src.wfc_tiles import PortKind


class JsonExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.building = WFCBuilding(
            floors=4, width=10, length=10, difficulty=2, seed=10,
            grid_resolution=3, recovery_strategy="none",
        )

    def test_round_trip_preserves_graph_and_exact_accepted_geometry(self):
        building = self.building
        with patch.object(
            building, "smooth_path", side_effect=AssertionError("route regenerated")
        ), TemporaryDirectory() as directory:
            path = Path(directory) / "building.json"
            export_building_json(building, path)
            data = json.loads(path.read_text(encoding="utf-8"))
            with self.assertRaises(FileExistsError):
                export_building_json(building, path)

        self.assertEqual(data["building"]["floor_elevations"], list(range(4)))
        self.assertEqual(
            (data["building"]["floors"], data["building"]["length"],
             data["building"]["width"]),
            (4, 10, 10),
        )
        self.assertEqual(len(data["graph"]["nodes"]), 400)
        self.assertEqual(len(data["routes"]), 19)
        self.assertEqual(data["schema_version"], 1)
        self.assertEqual(data["coordinate_system"]["blender_units_per_graph_unit"], 1.0)
        self.assertEqual(data["generation"]["grid_resolution"], 3)
        self.assertEqual(data["generation"]["vertical_grid_resolution"], 3)
        self.assertEqual(data["building"]["start"], list(building.start_pos))
        self.assertEqual(data["building"]["exit"], list(building.exit_pos))
        self.assertEqual(
            {tuple(node["position"]): node["attributes"] for node in data["graph"]["nodes"]},
            {node: dict(attrs) for node, attrs in building.graph.nodes(data=True)},
        )
        self.assertEqual(data["graph"]["solution_path"], [list(p) for p in building.path])
        self.assertEqual(data["diagnostics"]["warnings"], list(diagnose_routes(building).warnings))
        self.assertEqual(
            {(tuple(edge["source"]), tuple(edge["target"]), edge["attributes"]["kind"])
             for edge in data["graph"]["edges"]},
            {(source, target, _edge_name(attrs))
             for source, target, attrs in building.graph.edges(data=True)},
        )
        self.assertEqual(len(data["routes"]), len(building.routes))
        self.assertTrue(any(route.kind is PortKind.STAIR for route in building.routes))
        self.assertTrue(any(route.kind is PortKind.SLIDE for route in building.routes))
        for exported, route in zip(data["routes"], building.routes):
            self.assertEqual(exported["id"], route.route_id)
            self.assertEqual(exported["type"], route.kind.value)
            self.assertEqual(exported["source"], list(route.source))
            self.assertEqual(exported["target"], list(route.target))
            self.assertEqual(exported["geometry"], [list(p) for p in route.accepted_geometry])
            if route.kind is PortKind.STAIR:
                self.assertEqual(exported["geometry"], [list(route.source), list(route.target)])
            else:
                self.assertGreater(len(exported["geometry"]), 2)

    def test_missing_and_invalid_geometry_are_rejected(self):
        building = self.building
        slide_index = next(i for i, route in enumerate(building.routes)
                           if route.kind is PortKind.SLIDE)
        missing = list(building.routes)
        missing[slide_index] = replace(missing[slide_index], accepted_geometry=None)
        with patch.object(building, "routes", missing), TemporaryDirectory() as directory:
            path = Path(directory) / "missing.json"
            with self.assertRaisesRegex(RuntimeError, "lacks stored geometry"):
                export_building_json(building, path)
            self.assertFalse(path.exists())

        stair_index = next(i for i, route in enumerate(building.routes)
                           if route.kind is PortKind.STAIR)
        invalid = list(building.routes)
        stair = invalid[stair_index]
        invalid[stair_index] = replace(stair, accepted_geometry=(stair.source, stair.source))
        with patch.object(building, "routes", invalid):
            with self.assertRaisesRegex(ValueError, "logical route errors"):
                building_to_dict(building)

    def test_incomplete_or_duplicate_routes_are_rejected(self):
        building = self.building
        with patch.object(building, "routes", building.routes[:-1]):
            with self.assertRaisesRegex(ValueError, "do not match graph"):
                building_to_dict(building)
        duplicated = list(building.routes)
        duplicated[1] = replace(duplicated[1], route_id=duplicated[0].route_id)
        with patch.object(building, "routes", duplicated):
            with self.assertRaisesRegex(ValueError, "IDs must be unique"):
                building_to_dict(building)

    def test_graph_edge_label_compatibility_and_validation(self):
        building = self.building
        graph = building.graph.copy()
        source, target, attributes = next(iter(graph.edges(data=True)))
        edge_type = attributes["type"]
        with patch.object(building, "graph", graph):
            exported = building_to_dict(building)
            edge = next(edge for edge in exported["graph"]["edges"]
                        if edge["source"] == source and edge["target"] == target)
            self.assertEqual(edge["attributes"]["type"], edge_type)
            self.assertEqual(edge["attributes"]["kind"], edge_type)
            attributes["kind"] = edge_type
            del attributes["type"]
            building_to_dict(building)
            attributes["type"] = "invalid"
            with self.assertRaisesRegex(ValueError, "Conflicting graph edge labels"):
                building_to_dict(building)
            del attributes["kind"]
            with self.assertRaisesRegex(ValueError, "Invalid graph edge label"):
                building_to_dict(building)
            del attributes["type"]
            with self.assertRaisesRegex(ValueError, "Invalid graph edge label"):
                building_to_dict(building)


if __name__ == "__main__":
    unittest.main()
