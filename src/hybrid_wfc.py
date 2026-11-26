"""Length-bounded stochastic hybrid WFC routing."""

import math
import random
import time
from collections import deque

from src.wfc_global import prune_unreachable_route_domains
from src.wfc_tiles import (
    ContradictionError, Direction, PortKind, _DIRECTION_OFFSETS,
    _OPPOSITE_DIRECTION, build_compatibility, propagate,
)


class HybridBudgetExhausted(RuntimeError):
    """The bounded hybrid search stopped without proving infeasibility."""


class HybridGeometryRejected(Exception):
    """A completed route failed the existing continuous geometry validation."""


def route_length_lower_bound(cells, tiles, allowed_tiles, start, end,
                             required_horizontal_steps):
    """Shortest directed port-compatible walk; revisits are allowed.

    The walk relaxes simultaneous tile assignments and off-route compatibility,
    so its cell count is a lower bound, not a shortest feasible route.
    """
    available = set(cells)
    if start not in available or end not in available:
        raise ValueError("Route endpoints must be in the available cells")
    pending = deque([(start, None, 0, 1)])
    seen = {(start, None, 0)}
    while pending:
        cell, incoming, horizontal, length = pending.popleft()
        for index in allowed_tiles[cell]:
            ports = [direction for direction, kind in tiles[index].ports.items()
                     if kind is PortKind.SLIDE]
            if cell == end:
                if incoming in ports and len(ports) == 1 and horizontal >= required_horizontal_steps:
                    return length
                continue
            if cell == start and incoming is None:
                if len(ports) != 1:
                    continue
                outgoing = ports[0]
            else:
                if len(ports) != 2 or incoming not in ports:
                    continue
                outgoing = next(direction for direction in ports if direction != incoming)
            if outgoing is Direction.U:
                continue
            offset = _DIRECTION_OFFSETS[outgoing]
            neighbor = tuple(cell[i] + offset[i] for i in range(3))
            if neighbor not in available or neighbor == start:
                continue
            next_incoming = _OPPOSITE_DIRECTION[outgoing]
            next_horizontal = min(required_horizontal_steps, horizontal + int(outgoing is not Direction.D))
            state = (neighbor, next_incoming, next_horizontal)
            if state not in seen:
                seen.add(state)
                pending.append((*state, length + 1))
    raise ContradictionError("No directed port-compatible walk reaches the destination")


def relative_cell_budget(lower_bound_cells, detour_fraction, detour_cap):
    if (lower_bound_cells < 2 or not math.isfinite(detour_fraction)
            or detour_fraction < 0 or detour_cap < 0):
        raise ValueError("Invalid relative hybrid route budget")
    return lower_bound_cells + min(
        detour_cap, math.ceil(detour_fraction * (lower_bound_cells - 1))
    )


def validate_hybrid_clearance(candidate, existing_routes, minimum_clearance):
    """Compare stored smoothed centerlines with the diagnostic endpoint exclusion."""
    from src.route_diagnostics import DiagnosticThresholds, _polyline_clearance

    if not math.isfinite(minimum_clearance) or minimum_clearance < 0:
        raise ValueError("Hybrid minimum clearance must be finite and non-negative")
    if candidate.accepted_geometry is None:
        raise RuntimeError("Hybrid candidate lacks accepted geometry")
    for existing in existing_routes:
        if existing.accepted_geometry is None:
            raise RuntimeError(f"Accepted route {existing.route_id} lacks stored geometry")
        shared = {candidate.source, candidate.target} & {existing.source, existing.target}
        distance, _ = _polyline_clearance(
            candidate.accepted_geometry, existing.accepted_geometry, tuple(shared),
            DiagnosticThresholds().shared_endpoint_exclusion_radius,
        )
        if distance is not None and distance < minimum_clearance:
            raise HybridGeometryRejected(
                f"clearance {distance:.9g} below {minimum_clearance:.9g} "
                f"against {existing.route_id}"
            )


def _horizontal_turn_cost(prefix, successor):
    directions = [
        (b[1] - a[1], b[2] - a[2])
        for a, b in zip(prefix, prefix[1:])
        if a[1:] != b[1:]
    ]
    delta = (successor[1] - prefix[-1][1], successor[2] - prefix[-1][2])
    if delta == (0, 0) or not directions or directions[-1] == delta:
        return 0.0
    run_length = 1
    for direction in reversed(directions[:-1]):
        if direction != directions[-1]:
            break
        run_length += 1
    reversal = delta == (-directions[-1][0], -directions[-1][1])
    return (3.0 if reversal else 1.0) + 1.0 / run_length


def _prefix(domains, tiles, start, end):
    path = [start]
    while path[-1] != end and len(domains[path[-1]]) == 1:
        cell = path[-1]
        tile = tiles[next(iter(domains[cell]))]
        next_cells = []
        for direction, port in tile.ports.items():
            if port is not PortKind.SLIDE:
                continue
            offset = _DIRECTION_OFFSETS[direction]
            neighbor = tuple(cell[i] + offset[i] for i in range(3))
            if len(path) > 1 and neighbor == path[-2]:
                continue
            if direction is Direction.U or neighbor not in domains:
                raise ContradictionError("Fixed slide exits or ascends", cell=cell)
            next_cells.append(neighbor)
        if len(next_cells) != 1 or next_cells[0] in path:
            raise ContradictionError("Fixed slide has a loop or invalid ports", cell=cell)
        path.append(next_cells[0])
    return path


def _length_lower_bound(prefix, successor, end, required_horizontal_steps):
    """Lower bound on cells, including the fixed prefix and successor."""
    completed_horizontal = sum(a[1:] != b[1:] for a, b in zip(prefix, prefix[1:]))
    completed_horizontal += int(prefix[-1][1:] != successor[1:])
    horizontal_distance = abs(successor[1] - end[1]) + abs(successor[2] - end[2])
    horizontal_needed = max(horizontal_distance, required_horizontal_steps - completed_horizontal)
    if successor[0] < end[0]:
        return math.inf
    return len(prefix) + 1 + (successor[0] - end[0]) + horizontal_needed


def solve_length_bounded_hybrid(
    cells, tiles, *, start, end, allowed_tiles, seed, max_cells,
    required_horizontal_steps, empty_index, tile_aware_connectivity=False,
    weighting="compact", compactness_strength=0.12, max_backtracks=20,
    max_seconds=30.0, trace_callback=None, trace_states=False,
    geometry_preference=0.0, validate_complete=None,
):
    """Solve one slide with a hard cell limit and bounded stochastic backtracking."""
    if (max_cells < 2 or max_backtracks < 0 or max_seconds <= 0
            or not math.isfinite(geometry_preference) or geometry_preference < 0):
        raise ValueError("Invalid hybrid search budget")
    cell_set = set(cells)
    compatibility = build_compatibility(tiles)
    rng = random.Random(seed)
    started = time.monotonic()
    backtracks = 0
    iteration = 0
    domains = {}
    for cell in cells:
        boundary = [
            direction for direction, offset in _DIRECTION_OFFSETS.items()
            if tuple(cell[i] + offset[i] for i in range(3)) not in cell_set
        ]
        domains[cell] = {
            index for index in allowed_tiles[cell]
            if all(tiles[index].ports[direction] is PortKind.NONE for direction in boundary)
        }

    def emit(event, state=None, **details):
        if trace_callback is not None:
            data = {"event": event, **details}
            if trace_states and state is not None:
                data["domains"] = tuple((cell, tuple(sorted(domain)))
                                        for cell, domain in state.items())
            trace_callback(data)

    def propagate_global(state, changed):
        before = sum(map(len, state.values()))
        propagate(state, compatibility, changed)
        local = before - sum(map(len, state.values()))
        global_removed = 0
        while True:
            before = sum(map(len, state.values()))
            changed_cells = prune_unreachable_route_domains(
                state, tiles, [(start, end, PortKind.SLIDE)],
                tile_aware=tile_aware_connectivity,
                required_horizontal_steps=required_horizontal_steps,
            )
            global_removed += before - sum(map(len, state.values()))
            if not changed_cells:
                break
            before = sum(map(len, state.values()))
            propagate(state, compatibility, changed_cells)
            local += before - sum(map(len, state.values()))
        return local, global_removed

    propagate_global(domains, cells)

    def search(state):
        nonlocal backtracks, iteration
        if time.monotonic() - started > max_seconds or backtracks > max_backtracks:
            raise HybridBudgetExhausted(
                "UNKNOWN: hybrid time or backtracking budget exhausted"
            )
        prefix = _prefix(state, tiles, start, end)
        if len(prefix) > max_cells:
            raise ContradictionError("Fixed prefix exceeds slide length limit")
        if prefix[-1] == end:
            horizontal = sum(a[1:] != b[1:] for a, b in zip(prefix, prefix[1:]))
            if horizontal < required_horizontal_steps:
                raise ContradictionError("Slide has insufficient horizontal travel")
            finished = {cell: set(domain) for cell, domain in state.items()}
            for cell in finished:
                if cell not in prefix:
                    if empty_index not in finished[cell]:
                        raise ContradictionError("Disconnected forced route component", cell=cell)
                    finished[cell] = {empty_index}
            from src.wfc_global import validate_global_constraints
            layout = {cell: next(iter(domain)) for cell, domain in finished.items()}
            try:
                validate_global_constraints(
                    layout, tiles, required_pairs=[(start, end)],
                    required_typed_pairs=[(start, end, PortKind.SLIDE)],
                    require_closed_boundaries=True,
                )
            except ValueError as error:
                raise ContradictionError(str(error)) from error
            if validate_complete is not None:
                try:
                    validate_complete(layout, tuple(prefix))
                except HybridGeometryRejected as error:
                    emit("hybrid_geometry_rejection", state, reason=str(error),
                         path=tuple(prefix))
                    raise
            return layout
        cell, previous = prefix[-1], prefix[-2]
        incoming = next(direction for direction, offset in _DIRECTION_OFFSETS.items()
                        if tuple(cell[i] + offset[i] for i in range(3)) == previous)
        eligible = []
        rejected = []
        for index in sorted(state[cell]):
            tile = tiles[index]
            ports = [direction for direction, port in tile.ports.items()
                     if port is PortKind.SLIDE]
            if len(ports) != 2 or incoming not in ports:
                rejected.append((index, "local"))
                continue
            outgoing = next(direction for direction in ports if direction != incoming)
            offset = _DIRECTION_OFFSETS[outgoing]
            successor = tuple(cell[i] + offset[i] for i in range(3))
            if outgoing is Direction.U or successor not in state or successor in prefix:
                rejected.append((index, "direction_or_loop"))
                continue
            bound = _length_lower_bound(prefix, successor, end, required_horizontal_steps)
            if bound > max_cells:
                rejected.append((index, "length_lower_bound"))
                continue
            if not any(tiles[j].ports[_OPPOSITE_DIRECTION[outgoing]] is PortKind.SLIDE
                       for j in state[successor]):
                rejected.append((index, "local"))
                continue
            trial = {key: set(domain) for key, domain in state.items()}
            trial[cell] = {index}
            try:
                local, global_removed = propagate_global(trial, [cell])
                trial_prefix = _prefix(trial, tiles, start, end)
                if len(trial_prefix) > max_cells:
                    raise ContradictionError("Propagated prefix exceeds length limit")
            except ContradictionError:
                rejected.append((index, "propagation_or_global"))
                continue
            cost = _horizontal_turn_cost(prefix, successor)
            compact_cost = 0.0 if weighting == "uniform" else (
                compactness_strength * max(0, bound - len(prefix))
            )
            weight = max(1e-12, math.exp(-compact_cost - geometry_preference * cost))
            eligible.append((index, weight, trial, local, global_removed, bound))
        if not eligible:
            raise ContradictionError("No feasible length-bounded frontier candidate", cell=cell)
        choices = eligible[:]
        while choices:
            if time.monotonic() - started > max_seconds:
                raise HybridBudgetExhausted("UNKNOWN: hybrid time budget exhausted")
            selected = rng.choices(choices, weights=[item[1] for item in choices], k=1)[0]
            total = sum(item[1] for item in choices)
            iteration += 1
            emit("hybrid_length_choice", state, iteration=iteration, selected_cell=cell,
                 prefix=tuple(prefix), max_cells=max_cells,
                 eligible=tuple((item[0], tiles[item[0]].name, item[1], item[1] / total)
                                for item in choices),
                 rejected=tuple(rejected), selected_tile_index=selected[0],
                 selected_tile_name=tiles[selected[0]].name)
            emit("hybrid_length_propagation", selected[2], iteration=iteration,
                 local_removed=selected[3], global_removed=selected[4])
            try:
                return search(selected[2])
            except HybridBudgetExhausted:
                raise
            except (ContradictionError, HybridGeometryRejected) as error:
                choices.remove(selected)
                if choices:
                    backtracks += 1
                emit("hybrid_backtrack", state, iteration=iteration, cell=cell,
                     tile_index=selected[0], reason=str(error), backtracks=backtracks,
                     alternatives_remaining=len(choices))
                if backtracks > max_backtracks:
                    raise HybridBudgetExhausted(
                        "UNKNOWN: hybrid backtrack budget exhausted"
                    ) from error
        raise ContradictionError("All length-bounded choices failed", cell=cell)

    return search(domains)
