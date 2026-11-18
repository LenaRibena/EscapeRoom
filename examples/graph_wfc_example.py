"""Use the existing Building graph as a topology constraint for WFC."""

from collections.abc import Iterable

import numpy as np

from graph import Building
from src.wfc_graph import graph_edge_connections
from src.wfc_solver import solve_wfc
from src.wfc_tiles import Cell, Direction, PortKind, Tile


_OFFSETS = {
    Direction.N: (0, 0, 1),
    Direction.E: (0, 1, 0),
    Direction.S: (0, 0, -1),
    Direction.W: (0, -1, 0),
    Direction.U: (1, 0, 0),
    Direction.D: (-1, 0, 0),
}
_OPPOSITES = {
    Direction.N: Direction.S,
    Direction.E: Direction.W,
    Direction.S: Direction.N,
    Direction.W: Direction.E,
    Direction.U: Direction.D,
    Direction.D: Direction.U,
}


class HeadlessBuilding(Building):
    def display_graph(self, graph) -> None:
        """Skip the interactive plots so the example can run from a terminal."""
        del graph


def _wfc_tiles_for_graph(
    nodes: Iterable[tuple[int, int, int]],
    connections: Iterable[tuple[Cell, Direction, PortKind]],
) -> tuple[list[Cell], list[Tile]]:
    cells = [(floor, x, y) for x, y, floor in nodes]
    cell_set = set(cells)
    cell_ports = {
        cell: {direction: PortKind.NONE for direction in Direction}
        for cell in cells
    }

    def add_port(cell: Cell, direction: Direction, kind: PortKind) -> None:
        existing = cell_ports[cell][direction]
        if existing not in (PortKind.NONE, kind):
            raise ValueError(
                f"Graph requires conflicting ports at {cell} toward {direction.name}"
            )
        cell_ports[cell][direction] = kind

    for cell, direction, kind in connections:
        offset = _OFFSETS[direction]
        neighbor = (
            cell[0] + offset[0],
            cell[1] + offset[1],
            cell[2] + offset[2],
        )
        if cell not in cell_set or neighbor not in cell_set:
            raise ValueError(f"Graph connection {cell} -> {neighbor} is outside nodes")

        add_port(cell, direction, kind)
        add_port(neighbor, _OPPOSITES[direction], kind)

    unique_ports = {}
    for ports in cell_ports.values():
        signature = tuple(ports[direction] for direction in Direction)
        unique_ports.setdefault(signature, ports)

    tiles = [
        Tile(f"graph_pattern_{index}", ports)
        for index, ports in enumerate(unique_ports.values())
    ]
    return cells, tiles


def format_floor_maps(
    layout: dict[Cell, int],
    tiles: list[Tile],
    start: Cell,
    exit: Cell,
) -> str:
    """Render WFC walk ports as a compact ASCII map for each floor."""
    cells_by_floor: dict[int, list[Cell]] = {}
    for cell in layout:
        cells_by_floor.setdefault(cell[0], []).append(cell)

    maps = []
    for floor in sorted(cells_by_floor, reverse=True):
        floor_cells = cells_by_floor[floor]
        min_x = min(cell[1] for cell in floor_cells)
        max_x = max(cell[1] for cell in floor_cells)
        min_y = min(cell[2] for cell in floor_cells)
        max_y = max(cell[2] for cell in floor_cells)
        canvas = [
            [" " for _ in range((max_x - min_x + 1) * 3)]
            for _ in range((max_y - min_y + 1) * 3)
        ]

        for cell in floor_cells:
            _, x, y = cell
            tile = tiles[layout[cell]]
            column = (x - min_x) * 3
            row = (max_y - y) * 3
            horizontal_ports = [
                direction
                for direction in (Direction.N, Direction.E, Direction.S, Direction.W)
                if tile.ports[direction] is PortKind.WALK
            ]
            for direction, row_offset, column_offset, symbol in (
                (Direction.N, 0, 1, "|"),
                (Direction.E, 1, 2, "-"),
                (Direction.S, 2, 1, "|"),
                (Direction.W, 1, 0, "-"),
            ):
                if tile.ports[direction] is PortKind.WALK:
                    canvas[row + row_offset][column + column_offset] = symbol

            if cell == start:
                center = "S"
            elif cell == exit:
                center = "E"
            elif tile.ports[Direction.U] is PortKind.STAIR:
                center = "^"
            elif tile.ports[Direction.D] is PortKind.SLIDE:
                center = "v"
            elif len(horizontal_ports) >= 3:
                center = "+"
            elif horizontal_ports:
                center = "o"
            else:
                center = "."
            canvas[row + 1][column + 1] = center

        lines = [f"Floor {floor}"]
        lines.extend("".join(row).rstrip() for row in canvas)
        maps.append("\n".join(lines))

    return "\n\n".join(maps)


def main(seed: int = 0) -> dict[Cell, int]:
    np.random.seed(seed)
    building = HeadlessBuilding(
        floors=5,
        width=5,
        length=5,
        difficulty=2,
    )
    connections = graph_edge_connections(building.graph.edges(data=True))
    cells, tiles = _wfc_tiles_for_graph(building.graph.nodes, connections)

    start: Cell = (
        building.start_pos[2],
        building.start_pos[0],
        building.start_pos[1],
    )
    exit: Cell = (
        building.exit_pos[2],
        building.exit_pos[0],
        building.exit_pos[1],
    )
    layout = solve_wfc(
        cells,
        tiles,
        required_pairs=[(start, exit)],
        required_connections=connections,
        require_closed_boundaries=True,
        seed=seed,
    )

    print(
        f"Generated {len(layout)} WFC cells from "
        f"{len(building.graph.edges)} graph edges "
        f"({len(connections)} required connections, {len(tiles)} port patterns)."
    )
    print(f"START {start} is connected to EXIT {exit}.")
    print("\n" + format_floor_maps(layout, tiles, start, exit))
    return layout


if __name__ == "__main__":
    main()
