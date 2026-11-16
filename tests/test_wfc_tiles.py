import unittest

from src.wfc_tiles import (
    ContradictionError,
    Direction,
    PortKind,
    Tile,
    build_compatibility,
    faces_match,
    propagate,
)


def tile(name, **ports):
    all_ports = {direction: PortKind.NONE for direction in Direction}
    all_ports.update(
        {Direction[direction]: PortKind[kind] for direction, kind in ports.items()}
    )
    return Tile(
        name=name,
        ports=all_ports,
    )


class FaceMatchingTests(unittest.TestCase):
    def test_matching_connection_types_are_compatible_on_all_six_faces(self):
        opposite = {
            Direction.N: Direction.S,
            Direction.E: Direction.W,
            Direction.S: Direction.N,
            Direction.W: Direction.E,
            Direction.U: Direction.D,
            Direction.D: Direction.U,
        }

        for direction, neighbor_direction in opposite.items():
            with self.subTest(direction=direction):
                first_ports = {face: PortKind.NONE for face in Direction}
                neighbor_ports = {face: PortKind.NONE for face in Direction}
                first_ports[direction] = PortKind.WALK
                neighbor_ports[neighbor_direction] = PortKind.WALK
                first = Tile("first", first_ports)
                neighbor = Tile("neighbor", neighbor_ports)

                self.assertTrue(faces_match(first, direction, neighbor))

    def test_different_connection_types_are_incompatible(self):
        corridor = tile("corridor", E="WALK")
        stair = tile("stair", W="STAIR")

        self.assertFalse(faces_match(corridor, Direction.E, stair))

    def test_empty_faces_match_empty_faces(self):
        first = tile("first", E="NONE")
        second = tile("second", W="NONE")

        self.assertTrue(faces_match(first, Direction.E, second))

    def test_tile_requires_a_port_for_every_direction(self):
        with self.assertRaises(ValueError):
            Tile("incomplete", {Direction.E: PortKind.WALK})


class CompatibilityTableTests(unittest.TestCase):
    def test_table_contains_neighbor_indices_with_matching_faces(self):
        tiles = [
            tile("empty"),
            tile("east_walk", E="WALK"),
            tile("west_walk", W="WALK"),
        ]

        compatibility = build_compatibility(tiles)

        self.assertEqual(compatibility[Direction.E][1], frozenset({2}))
        self.assertEqual(compatibility[Direction.W][2], frozenset({1}))
        self.assertEqual(compatibility[Direction.E][0], frozenset({0, 1}))

    def test_empty_tile_list_has_empty_rows_for_each_direction(self):
        compatibility = build_compatibility([])

        self.assertEqual(set(compatibility), set(Direction))
        self.assertTrue(all(not rows for rows in compatibility.values()))


class PropagationTests(unittest.TestCase):
    def test_domain_reductions_propagate_through_a_chain(self):
        tiles = [
            tile("anchor", E="WALK"),
            tile("first_link", W="WALK", E="STAIR"),
            tile("wrong_first_link", W="STAIR"),
            tile("second_link", W="STAIR", E="WALK"),
            tile("wrong_second_link"),
            tile("end", W="WALK"),
            tile("wrong_end"),
        ]
        domains = {
            (0, 0, 0): {0},
            (0, 1, 0): {1, 2},
            (0, 2, 0): {3, 4},
            (0, 3, 0): {5, 6},
        }

        propagate(domains, build_compatibility(tiles), [(0, 0, 0)])

        self.assertEqual(domains[(0, 1, 0)], {1})
        self.assertEqual(domains[(0, 2, 0)], {3})
        self.assertEqual(domains[(0, 3, 0)], {5})

    def test_propagation_uses_all_six_coordinate_directions(self):
        all_walk = Tile(
            "all_walk", {direction: PortKind.WALK for direction in Direction}
        )
        all_empty = tile("all_empty")
        neighbor_ports = {
            Direction.N: Direction.S,
            Direction.E: Direction.W,
            Direction.S: Direction.N,
            Direction.W: Direction.E,
            Direction.U: Direction.D,
            Direction.D: Direction.U,
        }
        tiles = [all_walk, all_empty]
        domains = {(1, 1, 1): {0}}

        for direction, port in neighbor_ports.items():
            offset = {
                Direction.N: (0, 0, 1),
                Direction.E: (0, 1, 0),
                Direction.S: (0, 0, -1),
                Direction.W: (0, -1, 0),
                Direction.U: (1, 0, 0),
                Direction.D: (-1, 0, 0),
            }[direction]
            neighbor = tuple(
                coordinate + delta
                for coordinate, delta in zip((1, 1, 1), offset)
            )
            matching_tile = tile(f"{direction.name}_match", **{port.name: "WALK"})
            tiles.append(matching_tile)
            domains[neighbor] = {1, len(tiles) - 1}

        propagate(domains, build_compatibility(tiles), [(1, 1, 1)])

        for direction, offset in {
            Direction.N: (0, 0, 1),
            Direction.E: (0, 1, 0),
            Direction.S: (0, 0, -1),
            Direction.W: (0, -1, 0),
            Direction.U: (1, 0, 0),
            Direction.D: (-1, 0, 0),
        }.items():
            neighbor = tuple(
                coordinate + delta
                for coordinate, delta in zip((1, 1, 1), offset)
            )
            self.assertEqual(len(domains[neighbor]), 1)

    def test_empty_neighbor_domain_raises_contradiction(self):
        tiles = [tile("east_walk", E="WALK"), tile("empty")]
        domains = {(0, 0, 0): {0}, (0, 1, 0): {1}}

        with self.assertRaises(ContradictionError):
            propagate(domains, build_compatibility(tiles), [(0, 0, 0)])

    def test_changed_cell_must_exist(self):
        with self.assertRaises(ValueError):
            propagate({}, build_compatibility([]), [(0, 0, 0)])


if __name__ == "__main__":
    unittest.main()
