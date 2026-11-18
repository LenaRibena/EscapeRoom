import unittest

from examples.graph_wfc_example import format_floor_maps
from src.wfc_global import validate_global_constraints
from src.wfc_graph import graph_edge_connections, graph_path_connections
from src.wfc_solver import solve_wfc
from src.wfc_tiles import Direction, PortKind, Tile


class TestGraph:
    def __init__(self, edges):
        self.edges = edges

    def get_edge_data(self, source, target):
        return self.edges.get((source, target))


def tile(name, **ports):
    all_ports = {direction: PortKind.NONE for direction in Direction}
    all_ports.update(
        {Direction[direction]: PortKind[kind] for direction, kind in ports.items()}
    )
    return Tile(name, all_ports)


class GraphWfcIntegrationTests(unittest.TestCase):
    def test_floor_map_visualizes_start_to_exit_ports(self):
        corridor = tile("corridor", E="WALK", W="WALK")
        layout = {(0, 0, 0): 0, (0, 1, 0): 0}

        rendered = format_floor_maps(
            layout,
            [corridor],
            start=(0, 0, 0),
            exit=(0, 1, 0),
        )

        self.assertIn("Floor 0", rendered)
        self.assertIn("S--E", rendered)

    def test_graph_path_maps_to_wfc_coordinates_and_typed_ports(self):
        path = ((0, 0, 0), (1, 0, 0), (1, 0, 1))
        graph = TestGraph(
            {
                (path[0], path[1]): {"kind": "floor"},
                (path[1], path[2]): {"kind": "stair"},
            }
        )
        self.assertEqual(
            graph_path_connections(graph, path),
            (
                ((0, 0, 0), Direction.E, PortKind.WALK),
                ((0, 1, 0), Direction.U, PortKind.STAIR),
            ),
        )

    def test_graph_commit_type_edge_attribute_is_supported(self):
        path = ((0, 0, 0), (1, 0, 0))
        graph = TestGraph({(path[0], path[1]): {"type": "floor"}})

        self.assertEqual(
            graph_path_connections(graph, path),
            (((0, 0, 0), Direction.E, PortKind.WALK),),
        )

    def test_conflicting_graph_edge_labels_are_rejected(self):
        path = ((0, 0, 0), (1, 0, 0))
        graph = TestGraph(
            {(path[0], path[1]): {"kind": "floor", "type": "stair"}}
        )

        with self.assertRaisesRegex(
            ValueError,
            "Conflicting graph edge labels 'floor' and 'stair'",
        ):
            graph_path_connections(graph, path)

    def test_graph_path_connections_are_enforced_by_solver(self):
        path = ((0, 0, 0), (1, 0, 0))
        graph = TestGraph({(path[0], path[1]): {"kind": "floor"}})
        connections = graph_path_connections(graph, path)
        closed = tile("closed")
        corridor = tile("corridor", E="WALK", W="WALK")

        layout = solve_wfc(
            [(0, 0, 0), (0, 1, 0)],
            [closed, corridor],
            required_connections=connections,
            seed=1,
        )

        self.assertEqual(layout, {(0, 0, 0): 1, (0, 1, 0): 1})

    def test_graph_route_with_stair_is_realized_by_solver(self):
        path = ((0, 0, 0), (1, 0, 0), (1, 0, 1))
        graph = TestGraph(
            {
                (path[0], path[1]): {"kind": "floor"},
                (path[1], path[2]): {"kind": "stair"},
            }
        )
        cells = [(0, 0, 0), (0, 1, 0), (1, 1, 0)]
        tiles = [
            tile("start", E="WALK"),
            tile("middle", W="WALK", U="STAIR"),
            tile("upper_exit", D="STAIR"),
        ]

        layout = solve_wfc(
            cells,
            tiles,
            fixed_tiles={cell: index for index, cell in enumerate(cells)},
            required_connections=graph_path_connections(graph, path),
            seed=1,
        )

        self.assertEqual(layout, {cell: index for index, cell in enumerate(cells)})

    def test_required_connection_checks_expected_kind(self):
        corridor = tile("stair", E="STAIR", W="STAIR")

        with self.assertRaisesRegex(ValueError, "Required WALK connection"):
            validate_global_constraints(
                {(0, 0, 0): 0, (0, 1, 0): 0},
                [corridor],
                required_connections=[
                    ((0, 0, 0), Direction.E, PortKind.WALK)
                ],
            )

    def test_stair_and_slide_links_follow_graph_travel_direction(self):
        stair_tiles = [
            tile("stair_bottom", U="STAIR"),
            tile("stair_top", D="STAIR"),
        ]
        slide_tiles = [
            tile("slide_bottom", U="SLIDE"),
            tile("slide_top", D="SLIDE"),
        ]
        lower = (0, 0, 0)
        upper = (1, 0, 0)
        stair_layout = {lower: 0, upper: 1}
        slide_layout = {lower: 0, upper: 1}

        validate_global_constraints(
            stair_layout,
            stair_tiles,
            required_pairs=[(lower, upper)],
        )
        with self.assertRaisesRegex(ValueError, "not connected"):
            validate_global_constraints(
                stair_layout,
                stair_tiles,
                required_pairs=[(upper, lower)],
            )

        validate_global_constraints(
            slide_layout,
            slide_tiles,
            required_pairs=[(upper, lower)],
        )
        with self.assertRaisesRegex(ValueError, "not connected"):
            validate_global_constraints(
                slide_layout,
                slide_tiles,
                required_pairs=[(lower, upper)],
            )

    def test_graph_path_rejects_non_adjacent_nodes(self):
        path = ((0, 0, 0), (3, 0, 0))
        graph = TestGraph({(path[0], path[1]): {"kind": "floor"}})

        with self.assertRaisesRegex(ValueError, "cardinally adjacent"):
            graph_path_connections(graph, path)

    def test_graph_stair_and_slide_edges_expand_across_floors(self):
        stair_path = ((0, 0, 0), (0, 0, 2))
        slide_path = ((0, 0, 2), (0, 0, 0))
        graph = TestGraph(
            {
                (stair_path[0], stair_path[1]): {"kind": "stair"},
                (slide_path[0], slide_path[1]): {"kind": "slide"},
            }
        )

        self.assertEqual(
            graph_path_connections(graph, stair_path),
            (
                ((0, 0, 0), Direction.U, PortKind.STAIR),
                ((1, 0, 0), Direction.U, PortKind.STAIR),
            ),
        )
        self.assertEqual(
            graph_path_connections(graph, slide_path),
            (
                ((2, 0, 0), Direction.D, PortKind.SLIDE),
                ((1, 0, 0), Direction.D, PortKind.SLIDE),
            ),
        )

    def test_graph_transition_with_offset_position_adds_corridor_to_stairs(self):
        source = (0, 0, 0)
        target = (2, 1, 1)
        graph = TestGraph({(source, target): {"kind": "stair"}})

        self.assertEqual(
            graph_path_connections(graph, (source, target)),
            (
                ((0, 0, 0), Direction.U, PortKind.STAIR),
                ((1, 0, 0), Direction.E, PortKind.WALK),
                ((1, 1, 0), Direction.E, PortKind.WALK),
                ((1, 2, 0), Direction.N, PortKind.WALK),
            ),
        )

    def test_graph_edges_include_branches_and_deduplicate_floor_links(self):
        first = (0, 0, 0)
        second = (1, 0, 0)
        upper = (1, 0, 1)

        connections = graph_edge_connections(
            [
                (first, second, {"kind": "floor"}),
                (second, first, {"kind": "floor"}),
                (second, upper, {"kind": "stair"}),
            ]
        )
        self.assertEqual(
            connections,
            (
                ((0, 0, 0), Direction.E, PortKind.WALK),
                ((0, 1, 0), Direction.U, PortKind.STAIR),
            ),
        )

    def test_graph_edges_choose_other_vertical_anchor_on_port_conflict(self):
        stair_source = (0, 0, 0)
        stair_target = (0, 0, 1)
        slide_source = (0, 0, 1)
        slide_target = (1, 0, 0)

        connections = graph_edge_connections(
            [
                (stair_source, stair_target, {"type": "stair"}),
                (slide_source, slide_target, {"type": "slide"}),
            ]
        )

        self.assertIn(
            ((1, 1, 0), Direction.D, PortKind.SLIDE),
            connections,
        )
        self.assertIn(
            ((1, 0, 0), Direction.E, PortKind.WALK),
            connections,
        )

    def test_graph_path_requires_each_edge_to_exist(self):
        path = ((0, 0, 0), (1, 0, 0))

        with self.assertRaisesRegex(ValueError, "missing edge"):
            graph_path_connections(TestGraph({}), path)


if __name__ == "__main__":
    unittest.main()
