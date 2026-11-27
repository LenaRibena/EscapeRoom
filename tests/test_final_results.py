import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageChops

from examples.hybrid_building_visualize import create_figure
from examples.hybrid_length_trace import main as write_decision_trace
from examples.hybrid_slide_animation import load_choices
from examples.explain_hybrid_decisions import score, turn_description


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "results" / "final"


class FinalResultsTests(unittest.TestCase):
    def test_hybrid_runner_rejects_dfs_and_missing_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            output = str(Path(directory) / "record.json")
            for arguments, expected in (
                (["--strategy", "dfs_guided", "--relative-budget"], "invalid choice"),
                (["--strategy", "hybrid"], "requires --relative-budget or --max-cells"),
            ):
                with self.subTest(arguments=arguments):
                    result = subprocess.run(
                        [sys.executable, "-m", "examples.hybrid_experiment",
                         *arguments, "--output", output],
                        cwd=ROOT, capture_output=True, text=True,
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(expected, result.stderr)
                    self.assertFalse(Path(output).exists())

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

    def test_slide_animation_matches_recorded_choices(self):
        choices, path = load_choices(
            FINAL / "algorithm" / "wfc_decisions_explained.txt",
            FINAL / "algorithm" / "wfc_decisions.txt",
            FINAL / "building" / "building_seed10_hybrid.json",
        )
        self.assertEqual(len(choices), 21)
        self.assertEqual(len(path), 26)
        self.assertEqual(path[0], (9, 0, 12))
        self.assertEqual(path[-1], (0, 0, 0))
        self.assertEqual(choices[13].successor, (1, 1, 4))
        self.assertEqual(choices[13].candidates[1][1], 12.62)
        with Image.open(FINAL / "algorithm" / "slide_generation.gif") as image:
            self.assertEqual(image.n_frames, len(choices))
            self.assertEqual(image.size, (1200, 600))
            first = image.convert("RGB")
            image.seek(image.n_frames - 1)
            last = image.convert("RGB")
            self.assertIsNotNone(ImageChops.difference(first, last).getbbox())
        with Image.open(FINAL / "algorithm" / "slide_generation.png") as image:
            self.assertEqual(image.size, (1200, 600))

    def test_slide_animation_rejects_mismatched_traces(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = Path(directory) / "decisions.txt"
            original = (FINAL / "algorithm" / "wfc_decisions.txt").read_text(
                encoding="utf-8"
            )
            summary.write_text(original.replace("Choice 1 |", "Choice 2 |", 1),
                               encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differs between traces"):
                load_choices(FINAL / "algorithm" / "wfc_decisions_explained.txt",
                             summary, FINAL / "building" / "building_seed10_hybrid.json")

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

    def test_trace_writer_rejects_different_reference_building(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = root / "record.json"
            reference = root / "reference.json"
            record.write_text(json.dumps({"building": {"id": 1}}), encoding="utf-8")
            reference.write_text(json.dumps({"id": 2}), encoding="utf-8")
            argv = ["trace", str(record), "--output", str(root / "output"),
                    "--reference-building", str(reference)]
            with patch("sys.argv", argv):
                with self.assertRaisesRegex(ValueError, "does not match"):
                    write_decision_trace()
            self.assertFalse((root / "output").exists())

    def test_trace_writer_outputs_recorded_decision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = root / "record.json"
            record.write_text(json.dumps({
                "routes": [{"id": "slide", "discrete_cells": 2}],
                "backtracks": 0,
                "events": [
                    {"event": "hybrid_length_choice", "iteration": 1,
                     "max_cells": 2, "selected_cell": [0, 0, 0],
                     "selected_tile_index": 0, "selected_tile_name": "empty",
                     "eligible": [[0, "empty", 1.0, 1.0]], "rejected": []},
                    {"event": "hybrid_length_propagation", "iteration": 1,
                     "domains": [], "local_removed": 0, "global_removed": 0},
                ],
            }), encoding="utf-8")
            with patch("sys.argv", ["trace", str(record), "--output", str(root / "output")]):
                write_decision_trace()
            text = (root / "output" / "decisions.txt").read_text(encoding="utf-8")
            self.assertIn("Budget: 2 cells; accepted: 2 cells.", text)
            self.assertIn("Choice 1 | cell (0, 0, 0)", text)
            self.assertFalse((root / "output" / "decisions.gif").exists())


if __name__ == "__main__":
    unittest.main()
