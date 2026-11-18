from collections.abc import Iterable, Mapping, Sequence
from typing import Literal, Protocol

from src.wfc_tiles import (
    Cell,
    Direction,
    PortKind,
    _DIRECTION_OFFSETS,
)


GraphNode = tuple[int, int, int]
GraphConnection = tuple[Cell, Direction, PortKind]
GraphEdge = tuple[GraphNode, GraphNode, Mapping[str, object]]

_GRAPH_DIRECTIONS = {
    (1, 0, 0): Direction.E,
    (-1, 0, 0): Direction.W,
    (0, 1, 0): Direction.N,
    (0, -1, 0): Direction.S,
}
_GRAPH_EDGE_KINDS = {
    "floor": PortKind.WALK,
    "stair": PortKind.STAIR,
    "slide": PortKind.SLIDE,
}


class DirectedGraph(Protocol):
    def get_edge_data(
        self,
        source: GraphNode,
        target: GraphNode,
    ) -> Mapping[str, object] | None: ...


def graph_path_connections(
    graph: DirectedGraph,
    path: Sequence[GraphNode],
) -> tuple[GraphConnection, ...]:
    """Translate a graph.py path into required WFC links.

    Graph nodes use ``(x, y, floor)`` coordinates; WFC cells use
    ``(floor, x, y)``. Floor edges must be cardinally adjacent. Vertical
    stair/slide edges expand to adjacent vertical links at the source position,
    followed by a horizontal walk on the destination floor.
    """
    if len(path) < 2:
        raise ValueError("A graph path must contain at least two nodes")

    connections: list[GraphConnection] = []
    for source, target in zip(path, path[1:]):
        if (
            not isinstance(source, tuple)
            or len(source) != 3
            or not isinstance(target, tuple)
            or len(target) != 3
        ):
            raise ValueError("Graph path nodes must be (x, y, floor) tuples")
        if any(
            not isinstance(coordinate, int)
            for coordinate in (*source, *target)
        ):
            raise ValueError("Graph path nodes must contain integer coordinates")

        edge = graph.get_edge_data(source, target)
        if edge is None:
            raise ValueError(f"Graph path is missing edge {source} -> {target}")
        connections.extend(_connections_for_edge(source, target, edge))

    return tuple(connections)


def graph_edge_connections(
    edges: Iterable[GraphEdge],
) -> tuple[GraphConnection, ...]:
    """Translate a graph's edges, including its optional branches, to WFC links."""
    connections: list[GraphConnection] = []
    seen = set()
    ports: dict[tuple[Cell, Direction], PortKind] = {}
    graph_edges = list(edges)
    graph_edges.sort(key=lambda edge: _edge_kind(edge[2]) != "floor")

    for source, target, attributes in graph_edges:
        edge_kind = _edge_kind(attributes)
        candidates = (
            (_connections_for_edge(source, target, attributes, "source"),)
            if edge_kind == "floor"
            else (
                _connections_for_edge(source, target, attributes, "source"),
                _connections_for_edge(source, target, attributes, "target"),
            )
        )
        candidate = next(
            (item for item in candidates if not _has_port_conflicts(item, ports)),
            None,
        )
        if candidate is None:
            raise ValueError(
                f"Graph edge {source} -> {target} cannot be embedded without "
                "conflicting WFC ports"
            )

        for connection in candidate:
            cell, direction, kind = connection
            neighbor_offset = _DIRECTION_OFFSETS[direction]
            neighbor: Cell = (
                cell[0] + neighbor_offset[0],
                cell[1] + neighbor_offset[1],
                cell[2] + neighbor_offset[2],
            )
            key = (
                (frozenset((cell, neighbor)), kind)
                if kind is PortKind.WALK
                else (cell, direction, kind)
            )
            if key not in seen:
                seen.add(key)
                connections.append(connection)
            ports[(cell, direction)] = kind
            ports[(neighbor, _opposite(direction))] = kind
    return tuple(connections)


def _connections_for_edge(
    source: GraphNode,
    target: GraphNode,
    edge: Mapping[str, object],
    vertical_anchor: Literal["source", "target"] = "source",
) -> tuple[GraphConnection, ...]:
    if (
        not isinstance(source, tuple)
        or len(source) != 3
        or not isinstance(target, tuple)
        or len(target) != 3
    ):
        raise ValueError("Graph path nodes must be (x, y, floor) tuples")
    if any(
        not isinstance(coordinate, int)
        for coordinate in (*source, *target)
    ):
        raise ValueError("Graph path nodes must contain integer coordinates")

    edge_kind = _edge_kind(edge)

    delta = tuple(target[index] - source[index] for index in range(3))
    if edge_kind == "floor":
        direction = _GRAPH_DIRECTIONS.get(delta)
        if direction is None:
            raise ValueError(
                f"Floor graph edge {source} -> {target} must be "
                "cardinally adjacent on the same floor"
            )
        source_cell: Cell = (source[2], source[0], source[1])
        return ((source_cell, direction, PortKind.WALK),)
    if edge_kind == "stair":
        if delta[2] <= 0:
            raise ValueError(
                f"Stair graph edge {source} -> {target} must ascend"
            )
        anchor = source if vertical_anchor == "source" else target
        vertical = tuple(
            (
                (floor, anchor[0], anchor[1]),
                Direction.U,
                PortKind.STAIR,
            )
            for floor in range(source[2], target[2])
        )
        if vertical_anchor == "source":
            destination_start = (source[0], source[1], target[2])
            return vertical + _walk_connections(destination_start, target)
        return _walk_connections(source, target) + vertical
    if delta[2] >= 0:
        raise ValueError(
            f"Slide graph edge {source} -> {target} must descend"
        )
    anchor = source if vertical_anchor == "source" else target
    vertical = tuple(
        (
            (floor, anchor[0], anchor[1]),
            Direction.D,
            PortKind.SLIDE,
        )
        for floor in range(source[2], target[2], -1)
    )
    if vertical_anchor == "source":
        destination_start = (source[0], source[1], target[2])
        return vertical + _walk_connections(destination_start, target)
    return _walk_connections(source, target) + vertical


def _edge_kind(edge: Mapping[str, object]) -> str:
    edge_kind = edge.get("kind")
    edge_type = edge.get("type")
    if edge_kind is not None and edge_type is not None and edge_kind != edge_type:
        raise ValueError(
            f"Conflicting graph edge labels {edge_kind!r} and {edge_type!r}"
        )
    if edge_kind is None:
        edge_kind = edge_type
    if not isinstance(edge_kind, str) or edge_kind not in _GRAPH_EDGE_KINDS:
        raise ValueError(f"Unsupported graph edge kind {edge_kind!r}")
    return edge_kind


def _has_port_conflicts(
    connections: Sequence[GraphConnection],
    existing: Mapping[tuple[Cell, Direction], PortKind],
) -> bool:
    requested = dict(existing)
    for cell, direction, kind in connections:
        neighbor_offset = _DIRECTION_OFFSETS[direction]
        neighbor: Cell = (
            cell[0] + neighbor_offset[0],
            cell[1] + neighbor_offset[1],
            cell[2] + neighbor_offset[2],
        )
        for port in (
            (cell, direction),
            (neighbor, _opposite(direction)),
        ):
            if port in requested and requested[port] is not kind:
                return True
            requested[port] = kind
    return False


def _opposite(direction: Direction) -> Direction:
    return {
        Direction.N: Direction.S,
        Direction.E: Direction.W,
        Direction.S: Direction.N,
        Direction.W: Direction.E,
        Direction.U: Direction.D,
        Direction.D: Direction.U,
    }[direction]


def _walk_connections(
    source: GraphNode,
    target: GraphNode,
) -> tuple[GraphConnection, ...]:
    connections = []
    current_x, current_y = source[0], source[1]
    target_x, target_y = target[0], target[1]

    while current_x != target_x:
        direction = Direction.E if target_x > current_x else Direction.W
        connections.append(
            ((source[2], current_x, current_y), direction, PortKind.WALK)
        )
        current_x += 1 if direction is Direction.E else -1

    while current_y != target_y:
        direction = Direction.N if target_y > current_y else Direction.S
        connections.append(
            ((source[2], current_x, current_y), direction, PortKind.WALK)
        )
        current_y += 1 if direction is Direction.N else -1

    return tuple(connections)
