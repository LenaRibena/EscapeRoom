from collections import deque
from collections.abc import Iterable, Mapping, Sequence

from src.wfc_tiles import (
    Cell,
    ContradictionError,
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
    required_typed_pairs: Iterable[tuple[Cell, Cell, PortKind]] = (),
) -> None:
    """Validate ports, required direct links, and required cell connectivity.

    Walk links are bidirectional. Stair links are traversable upward and slide
    links downward when their ports face the corresponding vertical direction.
    """
    for cell, tile_index in layout.items():
        if not isinstance(tile_index, int) or not 0 <= tile_index < len(tiles):
            raise ValueError(f"Invalid tile index {tile_index!r} at cell {cell}")

    adjacency = {cell: set() for cell in layout}
    typed_adjacency = {
        kind: {cell: set() for cell in layout}
        for kind in (PortKind.STAIR, PortKind.SLIDE)
    }
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
            horizontal = direction in (
                Direction.N,
                Direction.E,
                Direction.S,
                Direction.W,
            )
            if (
                port is PortKind.WALK
                or (port is PortKind.STAIR and (horizontal or direction is Direction.U))
                or (port is PortKind.SLIDE and (horizontal or direction is Direction.D))
            ):
                adjacency[cell].add(neighbor_cell)
            if (
                port in typed_adjacency
                and (
                    horizontal
                    or (port is PortKind.STAIR and direction is Direction.U)
                    or (port is PortKind.SLIDE and direction is Direction.D)
                )
            ):
                typed_adjacency[port][cell].add(neighbor_cell)

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

    for start, end, kind in required_typed_pairs:
        if kind not in typed_adjacency:
            raise ValueError(
                f"Required route kind must be STAIR or SLIDE: {kind!r}"
            )
        if start not in layout or end not in layout:
            raise ValueError(
                f"Required route cells must exist in layout: {start}, {end}"
            )

        pending = deque([start])
        visited = {start}
        while pending and end not in visited:
            cell = pending.popleft()
            for neighbor in typed_adjacency[kind][cell] - visited:
                visited.add(neighbor)
                pending.append(neighbor)
        if end not in visited:
            raise ValueError(
                f"Required {kind.name.lower()} route from {start} to {end} "
                "is not connected in the required direction"
            )


def prune_unreachable_route_domains(
    domains: Mapping[Cell, set[int]],
    tiles: Sequence[Tile],
    required_typed_pairs: Iterable[tuple[Cell, Cell, PortKind]],
    *,
    tile_aware: bool = False,
    required_horizontal_steps: int = 0,
) -> set[Cell]:
    """Force route tiles outside every possible endpoint path to be empty.

    The domain adjacency is an over-approximation: an edge is possible if
    some tile in each endpoint domain supports it. Every tile on a realizable
    directed route therefore remains inside the possible-path envelope.
    """
    changed = set()
    for start, end, kind in required_typed_pairs:
        if kind not in (PortKind.STAIR, PortKind.SLIDE):
            raise ValueError("Required route kind must be STAIR or SLIDE")
        if start not in domains or end not in domains:
            raise ValueError(
                f"Required route cells must exist in domains: {start}, {end}"
            )

        if tile_aware:
            viable_tiles, route_exists = _tile_supported_route_tiles(
                domains,
                tiles,
                start,
                end,
                kind,
                required_horizontal_steps,
            )
            if not route_exists:
                raise ContradictionError(
                    f"No tile-compatible {kind.name.lower()} route from "
                    f"{start} to {end}",
                    cell=start,
                    domain=domains[start],
                )
            for cell, domain in domains.items():
                reduced = {
                    tile_index
                    for tile_index in domain
                    if not any(
                        port is kind for port in tiles[tile_index].ports.values()
                    )
                    or tile_index in viable_tiles.get(cell, set())
                }
                if not reduced:
                    raise ContradictionError(
                        f"Route tiles at {cell} cannot belong to a "
                        "tile-compatible endpoint route",
                        cell=cell,
                        domain=domain,
                    )
                if reduced != domain:
                    domain.intersection_update(reduced)
                    changed.add(cell)
            continue

        forward, reverse = _possible_route_adjacency(domains, tiles, kind)
        reachable_from_start = _reachable(start, forward)
        can_reach_end = _reachable(end, reverse)
        viable_cells = reachable_from_start & can_reach_end
        if end not in reachable_from_start:
            raise ContradictionError(
                f"Required {kind.name.lower()} route from {start} to {end} "
                "cannot connect",
                cell=start,
                domain=domains[start],
            )

        for cell, domain in domains.items():
            if cell in viable_cells:
                continue
            reduced = {
                tile_index
                for tile_index in domain
                if all(
                    port is not kind
                    for port in tiles[tile_index].ports.values()
                )
            }
            if not reduced:
                raise ContradictionError(
                    f"Route tiles at {cell} cannot connect the required endpoints",
                    cell=cell,
                    domain=domain,
                )
            if reduced != domain:
                domain.intersection_update(reduced)
                changed.add(cell)
    return changed


def has_tile_supported_route(
    domains: Mapping[Cell, set[int]],
    tiles: Sequence[Tile],
    start: Cell,
    end: Cell,
    kind: PortKind,
    required_horizontal_steps: int = 0,
) -> bool:
    """Check directed route feasibility with one tile supporting both ports."""
    _, route_exists = _tile_supported_route_tiles(
        domains,
        tiles,
        start,
        end,
        kind,
        required_horizontal_steps,
    )
    return route_exists


def _tile_supported_route_tiles(
    domains: Mapping[Cell, set[int]],
    tiles: Sequence[Tile],
    start: Cell,
    end: Cell,
    kind: PortKind,
    required_horizontal_steps: int,
) -> tuple[dict[Cell, set[int]], bool]:
    """Return domain tiles participating in a mutually supported typed route."""
    horizontal = {
        Direction.N,
        Direction.E,
        Direction.S,
        Direction.W,
    }
    travel_directions = horizontal | (
        {Direction.U} if kind is PortKind.STAIR else {Direction.D}
    )
    back_directions = {_OPPOSITE_DIRECTION[direction] for direction in travel_directions}
    route_states: list[
        tuple[Cell, int, Direction | None, Direction | None]
    ] = []

    for cell, domain in domains.items():
        for tile_index in domain:
            tile = tiles[tile_index]
            if cell == start:
                route_states.extend(
                    (cell, tile_index, None, direction)
                    for direction in travel_directions
                    if tile.ports[direction] is kind
                )
            elif cell == end:
                route_states.extend(
                    (cell, tile_index, direction, None)
                    for direction in back_directions
                    if tile.ports[direction] is kind
                )
            else:
                route_states.extend(
                    (cell, tile_index, entry, exit_direction)
                    for entry in back_directions
                    for exit_direction in travel_directions
                    if entry is not exit_direction
                    and tile.ports[entry] is kind
                    and tile.ports[exit_direction] is kind
                )

    if start == end:
        return (
            {start: set(domains[start])},
            required_horizontal_steps == 0 and bool(domains[start]),
        )

    states_by_cell_entry: dict[
        tuple[Cell, Direction | None],
        list[tuple[Cell, int, Direction | None, Direction | None]],
    ] = {}
    for state in route_states:
        cell, _, entry, _ = state
        states_by_cell_entry.setdefault((cell, entry), []).append(state)

    start_states = [
        state for state in route_states if state[0] == start and state[2] is None
    ]
    if not start_states:
        return {}, False

    required_horizontal_steps = max(0, required_horizontal_steps)
    starts = [(state, 0) for state in start_states]
    pending = deque(starts)
    predecessors: dict[
        tuple[tuple[Cell, int, Direction | None, Direction | None], int],
        set[tuple[tuple[Cell, int, Direction | None, Direction | None], int]],
    ] = {node: set() for node in starts}
    discovered = set(starts)
    terminals = []
    while pending:
        state, horizontal_steps = pending.popleft()
        cell, _, _, exit_direction = state
        if cell == end and exit_direction is None:
            if horizontal_steps >= required_horizontal_steps:
                terminals.append((state, horizontal_steps))
            continue
        if exit_direction is None:
            continue
        offset = _DIRECTION_OFFSETS[exit_direction]
        neighbor: Cell = (
            cell[0] + offset[0],
            cell[1] + offset[1],
            cell[2] + offset[2],
        )
        entry = _OPPOSITE_DIRECTION[exit_direction]
        next_horizontal_steps = min(
            required_horizontal_steps,
            horizontal_steps + int(exit_direction in horizontal),
        )
        for next_state in states_by_cell_entry.get((neighbor, entry), ()):
            next_node = (next_state, next_horizontal_steps)
            predecessors.setdefault(next_node, set()).add(
                (state, horizontal_steps)
            )
            if next_node not in discovered:
                discovered.add(next_node)
                pending.append(next_node)

    if not terminals:
        return {}, False

    viable_nodes = set()
    pending = deque(terminals)
    while pending:
        node = pending.popleft()
        if node in viable_nodes:
            continue
        viable_nodes.add(node)
        pending.extend(predecessors[node])

    viable_tiles: dict[Cell, set[int]] = {}
    for state, _ in viable_nodes:
        cell, tile_index, _, _ = state
        viable_tiles.setdefault(cell, set()).add(tile_index)
    return viable_tiles, True


def _possible_route_adjacency(
    domains: Mapping[Cell, set[int]],
    tiles: Sequence[Tile],
    kind: PortKind,
) -> tuple[dict[Cell, set[Cell]], dict[Cell, set[Cell]]]:
    forward = {cell: set() for cell in domains}
    reverse = {cell: set() for cell in domains}
    horizontal = {
        Direction.N,
        Direction.E,
        Direction.S,
        Direction.W,
    }
    for cell, domain in domains.items():
        for direction, offset in _DIRECTION_OFFSETS.items():
            if not (
                direction in horizontal
                or (kind is PortKind.STAIR and direction is Direction.U)
                or (kind is PortKind.SLIDE and direction is Direction.D)
            ):
                continue
            neighbor: Cell = (
                cell[0] + offset[0],
                cell[1] + offset[1],
                cell[2] + offset[2],
            )
            if neighbor not in domains:
                continue
            if not any(
                tiles[tile_index].ports[direction] is kind
                for tile_index in domain
            ):
                continue
            opposite = _OPPOSITE_DIRECTION[direction]
            if not any(
                tiles[tile_index].ports[opposite] is kind
                for tile_index in domains[neighbor]
            ):
                continue
            forward[cell].add(neighbor)
            reverse[neighbor].add(cell)
    return forward, reverse


def _reachable(start: Cell, adjacency: Mapping[Cell, set[Cell]]) -> set[Cell]:
    pending = deque([start])
    visited = {start}
    while pending:
        cell = pending.popleft()
        for neighbor in adjacency[cell] - visited:
            visited.add(neighbor)
            pending.append(neighbor)
    return visited
