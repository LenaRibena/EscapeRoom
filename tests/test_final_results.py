import hashlib
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from examples.hybrid_building_visualize import create_figure
from examples.hybrid_length_visualize import main as visualize_decisions
from examples.explain_hybrid_decisions import score, turn_description


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "results" / "final"


class FinalResultsTests(unittest.TestCase):
    def test_manifest_hashes_and_building_routes(self):
        manifest = json.loads((FINAL / "manifest.json").read_text(encoding="utf-8"))
        for name, expected in manifest["sha256"].items():
            with self.subTest(file=name):
                self.assertEqual(hashlib.sha256((FINAL / name).read_bytes()).hexdigest(),
                                 expected)
        building_path = FINAL / "building" / "building_seed10_hybrid.json"
        self.assertEqual(
            hashlib.sha256(building_path.read_bytes()).hexdigest(),
            "dac79b23be631eb5611cfdf295107f8c2658cd77a523c11166743b804948ff0c",
        )
        building = json.loads(building_path.read_text(encoding="utf-8"))
        self.assertEqual(sum(route["type"] == "slide" for route in building["routes"]), 10)
        self.assertEqual(sum(route["type"] == "stair" for route in building["routes"]), 9)

    def test_decision_presentation_counts(self):
        text = (FINAL / "algorithm" / "wfc_decisions.txt").read_text(encoding="utf-8")
        counts = [int(value) for value in re.findall(
            r"^Choice \d+ \| .* \| (\d+) eligible \| sampled ", text, re.MULTILINE
        )]
        self.assertEqual(len(counts), 21)
        self.assertEqual(sum(count > 1 for count in counts), 18)
        self.assertIn("Backtracks in this accepted execution: 0.", text)
        with Image.open(FINAL / "algorithm" / "wfc_decisions.gif") as image:
            self.assertEqual(image.n_frames, len(counts))

    def test_explained_trace_covers_each_choice(self):
        explained = (FINAL / "algorithm" / "wfc_decisions_explained.txt").read_text(
            encoding="utf-8"
        )
        choices = re.findall(r"^Choice (\d+) \|", explained, re.MULTILINE)
        self.assertEqual([int(number) for number in choices], list(range(1, 22)))
        self.assertEqual(explained.count("  Independent RNG replay:"), 21)
        self.assertEqual(explained.count("  Post-selection propagation:"), 21)

    def test_choice_14_cost_calculation(self):
        prefix = [(9, 0, 12), (8, 0, 12), (8, 0, 11), (8, 0, 10),
                  (7, 0, 10), (7, 0, 9), (7, 0, 8), (6, 0, 8),
                  (5, 0, 8), (4, 0, 8), (4, 0, 7), (4, 0, 6),
                  (4, 0, 5), (4, 0, 4), (3, 0, 4), (2, 0, 4), (1, 0, 4)]
        north = score(prefix, (1, 0, 5), (0, 0, 0), 6, 0.12, 1.5)
        east = score(prefix, (1, 1, 4), (0, 0, 0), 6, 0.12, 1.5)
        south = score(prefix, (1, 0, 3), (0, 0, 0), 6, 0.12, 1.5)
        self.assertEqual((north[3], north[4], north[-1]), (24, 3.125, 0.003975916465864252))
        self.assertEqual((east[3], east[4], east[-1]), (24, 1.125, 0.07985841697862626))
        self.assertEqual((south[3], south[4], south[-1]), (22, 0.0, 0.5488116360940265))
        self.assertIn("preceding run 8", turn_description(prefix, (1, 1, 4)))

    def test_building_figure_uses_graph_and_exact_route_geometry(self):
        building = json.loads(
            (FINAL / "building" / "building_seed10_hybrid.json").read_text(encoding="utf-8")
        )
        figure = create_figure(building)
        try:
            graph, trajectories = figure.axes
            self.assertEqual(graph.get_title(), "Inherited building graph")
            self.assertEqual(trajectories.get_title(), "Accepted Hybrid WFC trajectories")
            self.assertEqual(len(graph.collections), len(building["graph"]["edges"]) + 3)
            self.assertEqual(len(graph.collections[0]._offsets3d[0]),
                             len(building["graph"]["nodes"]))
            self.assertEqual(len(trajectories.lines), len(building["routes"]))
            for line, route in zip(trajectories.lines, building["routes"]):
                np.testing.assert_array_equal(np.asarray(line.get_data_3d()).T,
                                              route["geometry"])
            self.assertEqual(graph.get_xlim(), trajectories.get_xlim())
            self.assertEqual(graph.get_ylim(), trajectories.get_ylim())
            self.assertEqual(graph.get_zlim(), trajectories.get_zlim())
            self.assertEqual((graph.elev, graph.azim), (trajectories.elev, trajectories.azim))
            np.testing.assert_array_equal(graph.get_box_aspect(), trajectories.get_box_aspect())
        finally:
            plt.close(figure)

    def test_visualizer_rejects_different_reference_building(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = root / "record.json"
            reference = root / "reference.json"
            record.write_text(json.dumps({"building": {"id": 1}}), encoding="utf-8")
            reference.write_text(json.dumps({"id": 2}), encoding="utf-8")
            argv = ["visualize", str(record), "--output", str(root / "output"),
                    "--reference-building", str(reference)]
            with patch("sys.argv", argv):
                with self.assertRaisesRegex(ValueError, "does not match"):
                    visualize_decisions()
            self.assertFalse((root / "output").exists())


if __name__ == "__main__":
    unittest.main()
