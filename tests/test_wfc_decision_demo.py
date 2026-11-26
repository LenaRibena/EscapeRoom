import unittest

from examples.wfc_decision_demo import ROUTE_ID, TILES, _directions, _text, collect_decisions


class DecisionDemoTests(unittest.TestCase):
    def test_decisions_use_snapshots_and_exclude_direct_collapse_from_propagation(self):
        names = [tile.name for tile in TILES.tiles]
        selected = next(i for i, name in enumerate(names) if name == "slide_straight_N_S")
        empty = TILES.empty_index
        alternative = next(i for i in range(len(names)) if i not in (selected, empty))
        cell, north = (0, 0, 1), (0, 0, 2)

        def wrap(kind, **fields):
            return {"route_id": ROUTE_ID, "event": kind, **fields}

        def solver(kind, **fields):
            return wrap("solver_trace", solver_event={"event": kind, **fields})

        events = [
            wrap("route_summary", start_cell=(0, 0, 0), end_cell=north),
            wrap("route_attempt_start", route_attempt=1),
            solver("route_candidate", candidate_path=[(0, 0, 0), cell, north],
                   candidate_path_length=3),
            solver("route_guided_selection", candidate_tile_names=[names[selected]]),
            solver("collapse_selected", iteration=1, selected_cell=cell,
                   selected_tile_index=selected, selected_tile_name=names[selected],
                   selected_tile_ports=[("N", "slide"), ("S", "slide")],
                   candidate_tile_names=[names[selected], names[empty]],
                   domains=[(cell, (selected, empty)),
                            (north, (selected, empty, alternative))]),
            solver("domain_reduction", phase="cell_collapse_propagation",
                   possibilities_removed=1, newly_resolved_cells=[],
                   domains=[(cell, (selected,)), (north, (selected, empty))]),
            solver("domain_reduction", phase="global_reachability_pruning_after_collapse",
                   possibilities_removed=1, newly_resolved_cells=[north],
                   domains=[(cell, (selected,)), (north, (selected,))]),
            solver("collapse_complete", unresolved_cell_count=0, required_route_feasible=True,
                   domains=[(cell, (selected,)), (north, (selected,))]),
        ]
        _, records, counts = collect_decisions(events)
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(set(record["dfs_directions"]), {"N", "S"})
        self.assertEqual(record["neighbors"], [{
            "direction": "N", "cell": north, "before": 3, "after": 2,
            "removed": (names[alternative],),
        }])
        self.assertEqual(record["global_cells"], {north})
        self.assertEqual((record["local"], record["global"], record["global_resolved"]),
                         (1, 1, 1))
        self.assertIn("DFS 2 -> 1 tiles", _text(records, counts))
        self.assertIn("0 solver contradictions; 0 route retries", _text(records, counts))

    def test_direction_coordinates(self):
        self.assertEqual(_directions([(1, 2, 3), (0, 2, 3), (0, 3, 3)], (0, 2, 3)),
                         ("U", "E"))

    def test_incomplete_trace_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            collect_decisions([])


if __name__ == "__main__":
    unittest.main()
