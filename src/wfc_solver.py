import random
from collections.abc import Callable, Iterable, Mapping, Sequence

from src.wfc_global import validate_global_constraints
from src.wfc_tiles import (
    Cell,
    ContradictionError,
    Direction,
    PortKind,
    Tile,
    _DIRECTION_OFFSETS,
    _OPPOSITE_DIRECTION,
    build_compatibility,
    propagate,
)


def solve_wfc(
    cells: Iterable[Cell],
    tiles: Sequence[Tile],
    fixed_tiles: Mapping[Cell, int] | None = None,
    seed: int | None = None,
    max_attempts: int = 20,
    required_pairs: Iterable[tuple[Cell, Cell]] = (),
    require_closed_boundaries: bool = False,
    on_step: Callable[[Mapping[Cell, frozenset[int]]], None] | None = None,
    required_connections: Iterable[tuple[Cell, Direction, PortKind]] = (),
) -> dict[Cell, int]:
    """Solve a local tile layout with seeded random restarts.

    ``fixed_tiles`` maps cells to tile indices and pre-collapses those cells.
    ``required_pairs`` are pairs of cells that must be connected by matching,
    non-empty ports. ``required_connections`` specifies direct links as
    ``(cell, direction, port_kind)`` triples that every solution must include.
    ``max_attempts`` includes the initial attempt. Cells missing from the
    supplied set are treated as boundaries by :func:`propagate`.
    When supplied, ``on_step`` is called synchronously after initial propagation
    and after each collapse. It also receives snapshots from attempts that
    later fail a constraint.
    """
    cell_list = list(cells)
    if any(
        not isinstance(cell, tuple)
        or len(cell) != 3
        or any(not isinstance(coordinate, int) for coordinate in cell)
        for cell in cell_list
    ):
        raise ValueError("Cells must be (floor, x, y) integer tuples")
    if len(set(cell_list)) != len(cell_list):
        raise ValueError("Cells must not contain duplicates")
    if not isinstance(max_attempts, int) or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer")

    tile_list = tuple(tiles)
    if cell_list and not tile_list:
        raise ValueError("At least one tile is required to solve a non-empty grid")

    fixed = dict(fixed_tiles or {})
    if any(cell not in cell_list for cell in fixed):
        raise ValueError("Every fixed tile cell must exist in cells")
    if any(
        not isinstance(tile_index, int) or not 0 <= tile_index < len(tile_list)
        for tile_index in fixed.values()
    ):
        raise ValueError("Fixed tile indices must refer to an available tile")
    pairs = tuple(required_pairs)
    connections = tuple(required_connections)
    if not cell_list:
        validate_global_constraints(
            {},
            tile_list,
            required_pairs=pairs,
            required_connections=connections,
            require_closed_boundaries=require_closed_boundaries,
        )
        return {}

    compatibility = build_compatibility(tile_list)
    rng = random.Random(seed)
    last_contradiction: ContradictionError | None = None
    cell_set = set(cell_list)
    for connection in connections:
        if not isinstance(connection, (tuple, list)) or len(connection) != 3:
            raise ValueError(
                "Required connections must be (cell, direction, port_kind) triples"
            )
        cell, direction, kind = connection
        if cell not in cell_set:
            raise ValueError(f"Required connection cell {cell} must exist in cells")
        if not isinstance(direction, Direction):
            raise ValueError(f"Invalid required connection direction {direction!r}")
        if not isinstance(kind, PortKind) or kind is PortKind.NONE:
            raise ValueError(
                f"Required connection kind must be a non-empty PortKind: {kind!r}"
            )
        offset = _DIRECTION_OFFSETS[direction]
        neighbor = (
            cell[0] + offset[0],
            cell[1] + offset[1],
            cell[2] + offset[2],
        )
        if neighbor not in cell_set:
            raise ValueError(
                f"Required connection from {cell} toward {direction.name} "
                "must have a neighboring cell"
            )
        opposite_direction = _OPPOSITE_DIRECTION[direction]
        for fixed_cell, required_direction in (
            (cell, direction),
            (neighbor, opposite_direction),
        ):
            fixed_tile = fixed.get(fixed_cell)
            if (
                fixed_tile is not None
                and tile_list[fixed_tile].ports[required_direction] is not kind
            ):
                raise ValueError(
                    f"Fixed tile at {fixed_cell} does not satisfy the required "
                    f"{kind.name} connection"
                )

    def initial_domain(cell: Cell) -> set[int]:
        if not require_closed_boundaries:
            return set(range(len(tile_list)))

        boundary_directions = [
            direction
            for direction, offset in _DIRECTION_OFFSETS.items()
            if (
                cell[0] + offset[0],
                cell[1] + offset[1],
                cell[2] + offset[2],
            )
            not in cell_set
        ]
        return {
            tile_index
            for tile_index, tile in enumerate(tile_list)
            if all(
                tile.ports[direction] is PortKind.NONE
                for direction in boundary_directions
            )
        }

    for _ in range(max_attempts):
        domains = {cell: initial_domain(cell) for cell in cell_list}
        for cell, tile_index in fixed.items():
            if tile_index not in domains[cell]:
                raise ValueError(
                    f"Fixed tile at {cell} has an open port outside the layout"
                )
            domains[cell] = {tile_index}
        for cell, direction, kind in connections:
            offset = _DIRECTION_OFFSETS[direction]
            neighbor = (
                cell[0] + offset[0],
                cell[1] + offset[1],
                cell[2] + offset[2],
            )
            opposite_direction = _OPPOSITE_DIRECTION[direction]
            domains[cell] = {
                tile_index
                for tile_index in domains[cell]
                if tile_list[tile_index].ports[direction] is kind
            }
            domains[neighbor] = {
                tile_index
                for tile_index in domains[neighbor]
                if tile_list[tile_index].ports[opposite_direction] is kind
            }

        try:
            propagate(domains, compatibility, cell_list)
            if on_step is not None:
                on_step(
                    {cell: frozenset(domain) for cell, domain in domains.items()}
                )

            while True:
                unresolved = [
                    cell for cell in cell_list if len(domains[cell]) > 1
                ]
                if not unresolved:
                    result = {
                        cell: next(iter(domains[cell]))
                        for cell in cell_list
                    }
                    try:
                        validate_global_constraints(
                            result,
                            tile_list,
                            required_pairs=pairs,
                            required_connections=connections,
                            require_closed_boundaries=require_closed_boundaries,
                        )
                    except ValueError as error:
                        raise ContradictionError(str(error)) from error
                    return result

                smallest_domain = min(len(domains[cell]) for cell in unresolved)
                candidates = [
                    cell
                    for cell in unresolved
                    if len(domains[cell]) == smallest_domain
                ]
                selected_cell = rng.choice(candidates)
                selected_tile = rng.choice(sorted(domains[selected_cell]))
                domains[selected_cell] = {selected_tile}
                propagate(domains, compatibility, [selected_cell])
                if on_step is not None:
                    on_step(
                        {
                            cell: frozenset(domain)
                            for cell, domain in domains.items()
                        }
                    )
        except ContradictionError as error:
            last_contradiction = error

    raise ContradictionError(
        f"No solution found after {max_attempts} attempts; "
        f"last contradiction: {last_contradiction}"
    ) from last_contradiction
