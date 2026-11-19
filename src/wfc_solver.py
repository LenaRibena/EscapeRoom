import random
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

from src.wfc_global import (
    _possible_route_adjacency,
    has_tile_supported_route,
    prune_unreachable_route_domains,
    validate_global_constraints,
)
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


@dataclass(frozen=True)
class SolverFailureDetail:
    phase: str
    attempt: int
    message: str
    cell: Cell | None
    domain: tuple[int, ...]


class SolverContradiction(ContradictionError):
    """Contradiction annotated with the solver stage and affected domain."""

    def __init__(
        self,
        message: str,
        failure: SolverFailureDetail,
        phase_counts: Mapping[str, int],
    ) -> None:
        super().__init__(message, cell=failure.cell, domain=failure.domain)
        self.failure = failure
        self.phase_counts = dict(phase_counts)


class CandidateLookaheadError(ContradictionError):
    """Raised when every route-tile candidate immediately contradicts constraints."""


def _failure_detail(
    error: ContradictionError,
    phase: str,
    attempt: int,
    domains: Mapping[Cell, set[int]],
    trigger_cell: Cell | None = None,
) -> SolverFailureDetail:
    cell = error.cell
    if cell is None:
        match = re.search(r"\((-?\d+),\s*(-?\d+),\s*(-?\d+)\)", str(error))
        if match is not None:
            cell = tuple(int(value) for value in match.groups())
    cell = cell if cell in domains else trigger_cell
    domain = (
        tuple(sorted(domains[cell]))
        if cell in domains
        else tuple(error.domain)
    )
    return SolverFailureDetail(phase, attempt, str(error), cell, domain)


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
    allowed_tiles: Mapping[Cell, Iterable[int]] | None = None,
    required_typed_pairs: Iterable[tuple[Cell, Cell, PortKind]] = (),
    required_horizontal_steps: int = 0,
    preferred_tile_index: int | None = None,
    tile_aware_connectivity: bool = False,
    endpoint_frontier: bool = False,
    candidate_lookahead: bool = False,
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
    if not isinstance(required_horizontal_steps, int) or required_horizontal_steps < 0:
        raise ValueError("required_horizontal_steps must be a non-negative integer")

    tile_list = tuple(tiles)
    if cell_list and not tile_list:
        raise ValueError("At least one tile is required to solve a non-empty grid")
    if preferred_tile_index is not None and (
        not isinstance(preferred_tile_index, int)
        or not 0 <= preferred_tile_index < len(tile_list)
    ):
        raise ValueError("preferred_tile_index must refer to an available tile")
    if not all(
        isinstance(option, bool)
        for option in (
            tile_aware_connectivity,
            endpoint_frontier,
            candidate_lookahead,
        )
    ):
        raise ValueError("WFC heuristic options must be booleans")

    fixed = dict(fixed_tiles or {})
    allowed = {
        cell: frozenset(indices)
        for cell, indices in (allowed_tiles or {}).items()
    }
    if any(cell not in cell_list for cell in fixed):
        raise ValueError("Every fixed tile cell must exist in cells")
    if any(
        not isinstance(tile_index, int) or not 0 <= tile_index < len(tile_list)
        for tile_index in fixed.values()
    ):
        raise ValueError("Fixed tile indices must refer to an available tile")
    if any(cell not in cell_list for cell in allowed):
        raise ValueError("Every allowed-tile cell must exist in cells")
    if any(
        not isinstance(tile_index, int)
        or not 0 <= tile_index < len(tile_list)
        for indices in allowed.values()
        for tile_index in indices
    ):
        raise ValueError("Allowed tile indices must refer to available tiles")
    pairs = tuple(required_pairs)
    connections = tuple(required_connections)
    typed_pairs = tuple(required_typed_pairs)
    if not cell_list:
        validate_global_constraints(
            {},
            tile_list,
            required_pairs=pairs,
            required_connections=connections,
            require_closed_boundaries=require_closed_boundaries,
            required_typed_pairs=typed_pairs,
        )
        return {}

    compatibility = build_compatibility(tile_list)
    rng = random.Random(seed)
    last_failure: SolverFailureDetail | None = None
    phase_counts: Counter[str] = Counter()
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
    for start, end, kind in typed_pairs:
        if start not in cell_set or end not in cell_set:
            raise ValueError(
                f"Required route cells must exist in cells: {start}, {end}"
            )
        if kind not in (PortKind.STAIR, PortKind.SLIDE):
            raise ValueError("Required route kind must be STAIR or SLIDE")

    def initial_domain(cell: Cell) -> set[int]:
        domain = set(allowed.get(cell, range(len(tile_list))))
        if require_closed_boundaries:
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
            domain = {
                tile_index
                for tile_index in domain
                if all(
                    tile_list[tile_index].ports[direction] is PortKind.NONE
                    for direction in boundary_directions
                )
            }
        return domain

    for attempt_number in range(1, max_attempts + 1):
        domains = {cell: initial_domain(cell) for cell in cell_list}
        for cell, tile_index in fixed.items():
            if tile_index not in domains[cell]:
                raise ValueError(
                    f"Fixed tile at {cell} has an open port outside the layout"
                )
            domains[cell] = {tile_index}
        for cell, tile_indices in allowed.items():
            domains[cell].intersection_update(tile_indices)
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

        empty_domain = next(
            (cell for cell, domain in domains.items() if not domain),
            None,
        )
        if empty_domain is not None:
            error = ContradictionError(
                f"Cell {empty_domain} has no possible tiles",
                cell=empty_domain,
                domain=domains[empty_domain],
            )
            last_failure = _failure_detail(
                error,
                "initial_domain_construction",
                attempt_number,
                domains,
            )
            phase_counts[last_failure.phase] += 1
            continue

        trigger_cell: Cell | None = None

        def propagate_domains(
            changed_cells: Iterable[Cell],
            propagation_phase: str,
            pruning_phase: str,
        ) -> None:
            changed_list = tuple(changed_cells)
            phase_counts[propagation_phase] += 1
            try:
                propagate(domains, compatibility, changed_list)
            except ContradictionError as error:
                raise SolverContradiction(
                    str(error),
                    _failure_detail(
                        error,
                        propagation_phase,
                        attempt_number,
                        domains,
                        trigger_cell,
                    ),
                    phase_counts,
                ) from error
            while typed_pairs:
                phase_counts[pruning_phase] += 1
                try:
                    changed = prune_unreachable_route_domains(
                        domains,
                        tile_list,
                        typed_pairs,
                        tile_aware=tile_aware_connectivity,
                        required_horizontal_steps=required_horizontal_steps,
                    )
                except ContradictionError as error:
                    raise SolverContradiction(
                        str(error),
                        _failure_detail(
                            error,
                            pruning_phase,
                            attempt_number,
                            domains,
                            trigger_cell,
                        ),
                        phase_counts,
                    ) from error
                if not changed:
                    break
                phase_counts[propagation_phase] += 1
                try:
                    propagate(domains, compatibility, changed)
                except ContradictionError as error:
                    raise SolverContradiction(
                        str(error),
                        _failure_detail(
                            error,
                            propagation_phase,
                            attempt_number,
                            domains,
                            trigger_cell,
                        ),
                        phase_counts,
                    ) from error

        try:
            propagate_domains(
                cell_list,
                "initial_propagation",
                "global_reachability_pruning",
            )
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
                        phase_counts["final_validation"] += 1
                        validate_global_constraints(
                            result,
                            tile_list,
                            required_pairs=pairs,
                            required_connections=connections,
                            require_closed_boundaries=require_closed_boundaries,
                            required_typed_pairs=typed_pairs,
                        )
                    except ValueError as error:
                        contradiction = ContradictionError(str(error))
                        failure = SolverContradiction(
                            str(error),
                            _failure_detail(
                                contradiction,
                                "final_validation",
                                attempt_number,
                                domains,
                            ),
                            phase_counts,
                        )
                        raise failure from error
                    return result

                phase_counts["cell_collapse_selection"] += 1
                try:
                    route_choice = _select_route_tile(
                        domains,
                        tile_list,
                        typed_pairs,
                        rng,
                        required_horizontal_steps,
                        compatibility=compatibility,
                        endpoint_frontier=endpoint_frontier,
                        candidate_lookahead=candidate_lookahead,
                        tile_aware_connectivity=tile_aware_connectivity,
                        phase_counts=phase_counts,
                    )
                except ContradictionError as error:
                    failure_phase = (
                        "candidate_lookahead"
                        if isinstance(error, CandidateLookaheadError)
                        else "cell_collapse_selection"
                    )
                    failure = SolverContradiction(
                        str(error),
                        _failure_detail(
                            error,
                            failure_phase,
                            attempt_number,
                            domains,
                            typed_pairs[0][0] if typed_pairs else None,
                        ),
                        phase_counts,
                    )
                    raise failure from error
                if route_choice is not None:
                    selected_cell, selected_tile = route_choice
                else:
                    preferred_cells = (
                        [
                            cell
                            for cell in unresolved
                            if preferred_tile_index in domains[cell]
                        ]
                        if preferred_tile_index is not None
                        else []
                    )
                    if preferred_tile_index is not None and preferred_cells:
                        smallest_domain = min(
                            len(domains[cell]) for cell in preferred_cells
                        )
                        candidates = [
                            cell
                            for cell in preferred_cells
                            if len(domains[cell]) == smallest_domain
                        ]
                        selected_cell = rng.choice(candidates)
                        selected_tile = preferred_tile_index
                    else:
                        smallest_domain = min(
                            len(domains[cell]) for cell in unresolved
                        )
                        candidates = [
                            cell
                            for cell in unresolved
                            if len(domains[cell]) == smallest_domain
                        ]
                        selected_cell = rng.choice(candidates)
                        selected_tile = rng.choice(sorted(domains[selected_cell]))
                trigger_cell = selected_cell
                domains[selected_cell] = {selected_tile}
                try:
                    propagate_domains(
                        [selected_cell],
                        "cell_collapse_propagation",
                        "global_reachability_pruning_after_collapse",
                    )
                except SolverContradiction as error:
                    raise error
                trigger_cell = None
                if on_step is not None:
                    on_step(
                        {
                            cell: frozenset(domain)
                            for cell, domain in domains.items()
                        }
                    )
        except SolverContradiction as error:
            last_failure = error.failure

    if last_failure is None:
        last_failure = SolverFailureDetail(
            "unknown",
            max_attempts,
            "No solver attempt produced a result",
            None,
            (),
        )
    raise SolverContradiction(
        f"No solution found after {max_attempts} attempts; "
        f"last contradiction during {last_failure.phase}: "
        f"{last_failure.message}",
        last_failure,
        phase_counts,
    )


def _select_route_tile(
    domains: Mapping[Cell, set[int]],
    tiles: Sequence[Tile],
    required_typed_pairs: Sequence[tuple[Cell, Cell, PortKind]],
    rng: random.Random,
    required_horizontal_steps: int = 0,
    *,
    compatibility: Mapping[Direction, Sequence[frozenset[int]]],
    endpoint_frontier: bool = False,
    candidate_lookahead: bool = False,
    tile_aware_connectivity: bool = False,
    phase_counts: Counter[str],
) -> tuple[Cell, int] | None:
    for start, end, kind in required_typed_pairs:
        path = _find_tile_supported_route(
            domains,
            tiles,
            start,
            end,
            kind,
            rng,
            required_horizontal_steps,
        )
        if path is None:
            raise ContradictionError(
                f"No tile-supported {kind.name.lower()} path from {start} to {end}",
                cell=start,
                domain=domains[start],
            )
        unresolved = [cell for cell in path if len(domains[cell]) > 1]
        if not unresolved:
            continue
        if endpoint_frontier:
            cell = unresolved[0]
        else:
            smallest_domain = min(len(domains[cell]) for cell in unresolved)
            candidates = [
                cell for cell in unresolved
                if len(domains[cell]) == smallest_domain
            ]
            cell = rng.choice(candidates)
        path_index = path.index(cell)
        required_directions = set()
        if path_index:
            required_directions.add(
                _direction_between(cell, path[path_index - 1])
            )
        if path_index + 1 < len(path):
            required_directions.add(
                _direction_between(cell, path[path_index + 1])
            )
        candidates_for_cell = [
            tile_index
            for tile_index in domains[cell]
            if {
                direction
                for direction, port in tiles[tile_index].ports.items()
                if port is kind
            }
            == required_directions
        ]
        if not candidates_for_cell:
            raise ContradictionError(
                f"No {kind.name.lower()} tile fits the selected route at {cell}",
                cell=cell,
                domain=domains[cell],
            )
        ordered_tile_candidates = sorted(candidates_for_cell)
        if not candidate_lookahead:
            return cell, rng.choice(ordered_tile_candidates)

        rng.shuffle(ordered_tile_candidates)
        for tile_index in ordered_tile_candidates:
            phase_counts["candidate_lookahead_checks"] += 1
            trial_domains = {
                candidate_cell: set(domain)
                for candidate_cell, domain in domains.items()
            }
            trial_domains[cell] = {tile_index}
            try:
                propagate(trial_domains, compatibility, [cell])
                while required_typed_pairs:
                    phase_counts["candidate_lookahead_global_checks"] += 1
                    changed = prune_unreachable_route_domains(
                        trial_domains,
                        tiles,
                        required_typed_pairs,
                        tile_aware=tile_aware_connectivity,
                        required_horizontal_steps=required_horizontal_steps,
                    )
                    if not changed:
                        break
                    propagate(trial_domains, compatibility, changed)
                if any(
                    not has_tile_supported_route(
                        trial_domains,
                        tiles,
                        start,
                        end,
                        kind,
                        required_horizontal_steps,
                    )
                    for start, end, kind in required_typed_pairs
                ):
                    phase_counts["candidate_lookahead_rejections"] += 1
                    continue
            except ContradictionError:
                phase_counts["candidate_lookahead_rejections"] += 1
                continue
            return cell, tile_index
        raise CandidateLookaheadError(
            f"One-step lookahead rejected every route tile at {cell}",
            cell=cell,
            domain=domains[cell],
        )
    return None


def _find_tile_supported_route(
    domains: Mapping[Cell, set[int]],
    tiles: Sequence[Tile],
    start: Cell,
    end: Cell,
    kind: PortKind,
    rng: random.Random,
    required_horizontal_steps: int = 0,
) -> tuple[Cell, ...] | None:
    forward, _ = _possible_route_adjacency(domains, tiles, kind)
    path = [start]
    visited = {start}
    search_budget = max(100, len(domains) * 24)

    def search(cell: Cell, horizontal_steps: int) -> bool:
        nonlocal search_budget
        if cell == end:
            return horizontal_steps >= required_horizontal_steps
        search_budget -= 1
        if search_budget < 0:
            return False

        candidates = list(forward[cell] - visited)
        rng.shuffle(candidates)
        def candidate_key(candidate: Cell) -> tuple[int, float]:
            direction = _direction_between(cell, candidate)
            is_horizontal = direction in (
                Direction.N,
                Direction.E,
                Direction.S,
                Direction.W,
            )
            needs_horizontal = horizontal_steps < required_horizontal_steps
            horizontal_priority = (
                0 if is_horizontal else 1
            ) if needs_horizontal else 0
            return (
                horizontal_priority,
                sum(abs(candidate[index] - end[index]) for index in range(3))
                + rng.random() * 0.35,
            )

        candidates.sort(key=candidate_key)
        previous = path[-2] if len(path) > 1 else None
        for neighbor in candidates:
            direction = _direction_between(cell, neighbor)
            next_horizontal_steps = horizontal_steps + int(
                direction
                in (Direction.N, Direction.E, Direction.S, Direction.W)
            )
            if (
                neighbor == end
                and next_horizontal_steps < required_horizontal_steps
            ):
                continue
            directions = {_direction_between(cell, neighbor)}
            if previous is not None:
                directions.add(_direction_between(cell, previous))
            if not _has_tile_with_route_ports(
                domains[cell], tiles, kind, directions
            ):
                continue
            if neighbor == end and not _has_tile_with_route_ports(
                domains[end],
                tiles,
                kind,
                {_direction_between(end, cell)},
            ):
                continue
            path.append(neighbor)
            visited.add(neighbor)
            if search(neighbor, next_horizontal_steps):
                return True
            path.pop()
            visited.remove(neighbor)
        return False

    return tuple(path) if search(start, 0) else None


def _has_tile_with_route_ports(
    domain: set[int],
    tiles: Sequence[Tile],
    kind: PortKind,
    directions: set[Direction],
) -> bool:
    return any(
        {
            direction
            for direction, port in tiles[tile_index].ports.items()
            if port is kind
        }
        == directions
        for tile_index in domain
    )


def _direction_between(source: Cell, target: Cell) -> Direction:
    delta = tuple(target[index] - source[index] for index in range(3))
    for direction, offset in _DIRECTION_OFFSETS.items():
        if delta == offset:
            return direction
    raise ValueError(f"Route cells {source} and {target} are not adjacent")
