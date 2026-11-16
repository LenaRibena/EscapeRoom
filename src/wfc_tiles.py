from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType


class Direction(Enum):
    N = "N"
    E = "E"
    S = "S"
    W = "W"
    U = "U"
    D = "D"


class PortKind(Enum):
    NONE = "none"
    WALK = "walk"
    STAIR = "stair"
    SLIDE = "slide"


Cell = tuple[int, int, int]


_DIRECTION_OFFSETS = {
    Direction.N: (0, 0, 1),
    Direction.E: (0, 1, 0),
    Direction.S: (0, 0, -1),
    Direction.W: (0, -1, 0),
    Direction.U: (1, 0, 0),
    Direction.D: (-1, 0, 0),
}

_OPPOSITE_DIRECTION = {
    Direction.N: Direction.S,
    Direction.E: Direction.W,
    Direction.S: Direction.N,
    Direction.W: Direction.E,
    Direction.U: Direction.D,
    Direction.D: Direction.U,
}


class ContradictionError(ValueError):
    """Raised when propagation leaves a cell with no possible tiles."""


@dataclass(frozen=True)
class Tile:
    name: str
    ports: Mapping[Direction, PortKind]

    def __post_init__(self):
        if set(self.ports) != set(Direction):
            raise ValueError("A tile must define a port for all six directions")

        if any(not isinstance(port, PortKind) for port in self.ports.values()):
            raise TypeError("Tile ports must be PortKind values")

        object.__setattr__(self, "ports", MappingProxyType(dict(self.ports)))


def faces_match(tile: Tile, direction: Direction, neighbor: Tile) -> bool:
    opposite = _OPPOSITE_DIRECTION[direction]
    return tile.ports[direction] == neighbor.ports[opposite]


def build_compatibility(
    tiles: Sequence[Tile],
) -> dict[Direction, tuple[frozenset[int], ...]]:
    return {
        direction: tuple(
            frozenset(
                neighbor_index
                for neighbor_index, neighbor in enumerate(tiles)
                if faces_match(tile, direction, neighbor)
            )
            for tile in tiles
        )
        for direction in Direction
    }


def propagate(
    domains: MutableMapping[Cell, set[int]],
    compatibility: Mapping[Direction, Sequence[frozenset[int]]],
    changed_cells: Iterable[Cell],
) -> None:
    """Propagate reduced tile domains through all neighboring cells.

    Cells use (floor, x, y) coordinates. Neighbors outside ``domains`` are
    treated as grid boundaries.
    """
    queue = list(changed_cells)
    if any(cell not in domains for cell in queue):
        raise ValueError("Every changed cell must exist in domains")

    while queue:
        cell = queue.pop()
        if not domains[cell]:
            raise ContradictionError(f"Cell {cell} has no possible tiles")

        for direction, offset in _DIRECTION_OFFSETS.items():
            neighbor_cell: Cell = (
                cell[0] + offset[0],
                cell[1] + offset[1],
                cell[2] + offset[2],
            )
            if neighbor_cell not in domains:
                continue

            neighbor_domain = domains[neighbor_cell]
            supported_tiles = set().union(
                *(compatibility[direction][tile_index] for tile_index in domains[cell])
            )
            reduced_domain = neighbor_domain & supported_tiles

            if not reduced_domain:
                raise ContradictionError(
                    f"Cell {neighbor_cell} has no compatible tiles"
                )

            if reduced_domain != neighbor_domain:
                domains[neighbor_cell] = reduced_domain
                queue.append(neighbor_cell)
