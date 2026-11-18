from collections import deque
from collections.abc import Iterable, Mapping, Sequence

from src.wfc_tiles import (
    Cell,
    Direction,
    PortKind,
    Tile,
    _DIRECTION_OFFSETS,
    _OPPOSITE_DIRECTION,
)


def validate_global_constraints(
    layout: Mapping[Cell, int],
    tiles: Sequence[Tile],
    required_pairs: Iterable[tuple[Cell, Cell]] = (),
    require_closed_boundaries: bool = False,
    required_connections: Iterable[tuple[Cell, Direction, PortKind]] = (),
) -> None:
    """Validate ports, required direct links, and required cell connectivity.

    Walk links are bidirectional. Stair links are traversable upward and slide
    links downward when their ports face the corresponding vertical direction.
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
            opposite_direction = _OPPOSITE_DIRECTION[direction]
            neighbor_port = neighbor.ports[opposite_direction]
            if port != neighbor_port:
                raise ValueError(
                    f"Mismatched ports between cells {cell} and {neighbor_cell}"
                )
            if (
                port is PortKind.WALK
                or (port is PortKind.STAIR and direction is Direction.U)
                or (port is PortKind.SLIDE and direction is Direction.D)
            ):
                adjacency[cell].add(neighbor_cell)

    for cell, direction, kind in required_connections:
        if cell not in layout:
            raise ValueError(f"Required connection cell {cell} is not in layout")
        if not isinstance(direction, Direction):
            raise ValueError(f"Invalid required connection direction {direction!r}")
        if not isinstance(kind, PortKind) or kind is PortKind.NONE:
            raise ValueError(
                f"Required connection kind must be a non-empty PortKind: {kind!r}"
            )

        offset = _DIRECTION_OFFSETS[direction]
        neighbor_cell: Cell = (
            cell[0] + offset[0],
            cell[1] + offset[1],
            cell[2] + offset[2],
        )
        if neighbor_cell not in layout:
            raise ValueError(
                f"Required connection from {cell} toward {direction.name} "
                f"leaves the layout"
            )

        opposite_direction = _OPPOSITE_DIRECTION[direction]
        source_port = tiles[layout[cell]].ports[direction]
        target_port = tiles[layout[neighbor_cell]].ports[opposite_direction]
        if source_port is not kind or target_port is not kind:
            raise ValueError(
                f"Required {kind.name} connection from {cell} toward "
                f"{direction.name} is missing"
            )

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
