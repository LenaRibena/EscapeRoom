import os
from pathlib import Path
import subprocess
import sys
import unittest


class HumanUpstreamTests(unittest.TestCase):
    def test_human_can_follow_wfc_building_solution_path(self):
        root = Path(__file__).resolve().parents[1]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(
            (str(root / "src"), str(root), environment.get("PYTHONPATH", ""))
        )
        environment["MPLBACKEND"] = "Agg"
        environment["PYTHONHASHSEED"] = "0"
        code = """
from human import Human
from src.wfc_building import WFCBuilding

building = WFCBuilding(4, 10, 10, difficulty=2, seed=10,
                       grid_resolution=3, generate_routes=False)
path, types = Human(exploration=0.0).get_path(building)
assert path == building.path
assert path[0] == building.start_pos and path[-1] == building.exit_pos
assert types == [building.graph.edges[a, b]["type"]
                 for a, b in zip(path, path[1:])]
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
