from collections import deque
from collections.abc import Iterable, Mapping, Sequence

from src.wfc_tiles import (
    Cell,
    Direction,
    PortKind,
    Tile,
    _DIRECTION_OFFSETS,
)


def validate_global_constraints(
    layout: Mapping[Cell, int],
    tiles: Sequence[Tile],
    required_pairs: Iterable[tuple[Cell, Cell]] = (),
    require_closed_boundaries: bool = False,
) -> None:
    """Validate port consistency and connectivity between required cell pairs.

    Connectivity is currently undirected: any matching non-empty port creates
    a traversable link. Directed stair/slide rules and reserved-volume
    collisions require additional tile metadata and are not checked here.
    """
    for cell, tile_index in layout.items():
        if not isinstance(tile_index, int) or not 0 <= tile_index < len(tiles):
            raise ValueError(f"Invalid tile index {tile_index!r} at cell {cell}")

    adjacency = {cell: set() for cell in layout}
    for cell, tile_index in layout.items():
        tile = tiles[tile_index]
        for direction, offset in _DIRECTION_OFFSETS.items():
            neighbor_cell: Cell = (
                cell[0] + offset[0],
                cell[1] + offset[1],
                cell[2] + offset[2],
            )
            port = tile.ports[direction]

            if neighbor_cell not in layout:
                if require_closed_boundaries and port is not PortKind.NONE:
                    raise ValueError(
                        f"Open {direction.name} port at boundary cell {cell}"
                    )
                continue

            neighbor = tiles[layout[neighbor_cell]]
            opposite_direction = {
                Direction.N: Direction.S,
                Direction.E: Direction.W,
                Direction.S: Direction.N,
                Direction.W: Direction.E,
                Direction.U: Direction.D,
                Direction.D: Direction.U,
            }[direction]
            neighbor_port = neighbor.ports[opposite_direction]
            if port != neighbor_port:
                raise ValueError(
                    f"Mismatched ports between cells {cell} and {neighbor_cell}"
                )
            if port is not PortKind.NONE:
                adjacency[cell].add(neighbor_cell)

    for start, end in required_pairs:
        if start not in layout or end not in layout:
            raise ValueError(
                f"Required connection cells must exist in layout: {start}, {end}"
            )

        pending = deque([start])
        visited = {start}
        while pending and end not in visited:
            cell = pending.popleft()
            for neighbor in adjacency[cell] - visited:
                visited.add(neighbor)
                pending.append(neighbor)

        if end not in visited:
            raise ValueError(
                f"Required cells {start} and {end} are not connected"
            )
