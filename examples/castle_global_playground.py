"""Form a Castle tileset floorplan in a live window using local and global WFC."""

import random
import time
import tkinter as tk
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from src.wfc_tiles import Cell, Direction


_TILESET_PATH = (
    Path(__file__).resolve().parents[1]
    / "wfc_exercise_playground"
    / "tilesets"
    / "Castle"
)
_TILE_NAMES = (
    "bridge",
    "ground",
    "river",
    "riverturn",
    "road",
    "roadturn",
    "t",
    "tower",
    "wall",
    "wallriver",
    "wallroad",
)
_DIRECTIONS = (Direction.N, Direction.E, Direction.S, Direction.W)
_OFFSETS = {
    Direction.N: (0, 0, 1),
    Direction.E: (0, 1, 0),
    Direction.S: (0, 0, -1),
    Direction.W: (0, -1, 0),
}
_OPPOSITES = {
    Direction.N: Direction.S,
    Direction.E: Direction.W,
    Direction.S: Direction.N,
    Direction.W: Direction.E,
}
_BASE_ROAD_PORTS = {
    "bridge": frozenset((Direction.E, Direction.W)),
    "road": frozenset((Direction.N, Direction.S)),
    "roadturn": frozenset((Direction.N, Direction.E)),
    "t": frozenset((Direction.N, Direction.E, Direction.W)),
    "wallroad": frozenset((Direction.E, Direction.W)),
}
_GRID_SIZE = 25
_ZOOM = 4
_CELL_PIXELS = 7 * _ZOOM
_START = (0, 10, 12)
_GOAL = (0, 14, 12)


@dataclass(frozen=True)
class CastleTile:
    name: str
    pixels: tuple[tuple[str, ...], ...]
    edges: Mapping[Direction, tuple[str, ...]]
    road_ports: frozenset[Direction]
    image: tk.PhotoImage | None = None


def _rotate_pixels(
    pixels: tuple[tuple[str, ...], ...],
) -> tuple[tuple[str, ...], ...]:
    size = len(pixels)
    return tuple(
        tuple(pixels[size - 1 - x][y] for x in range(size))
        for y in range(size)
    )


def _rotate_road_ports(ports: frozenset[Direction]) -> frozenset[Direction]:
    clockwise = {
        Direction.N: Direction.E,
        Direction.E: Direction.S,
        Direction.S: Direction.W,
        Direction.W: Direction.N,
    }
    return frozenset(clockwise[direction] for direction in ports)


def _tile_edges(
    pixels: tuple[tuple[str, ...], ...],
) -> dict[Direction, tuple[str, ...]]:
    return {
        Direction.N: pixels[0],
        Direction.E: tuple(row[-1] for row in pixels),
        Direction.S: pixels[-1],
        Direction.W: tuple(row[0] for row in pixels),
    }


def _photo_from_pixels(
    master: tk.Misc,
    pixels: tuple[tuple[str, ...], ...],
) -> tk.PhotoImage:
    image = tk.PhotoImage(master=master, width=len(pixels), height=len(pixels))
    for y, row in enumerate(pixels):
        image.put("{" + " ".join(row) + "}", to=(0, y))
    return image.zoom(_ZOOM, _ZOOM)


def load_castle_tiles(
    master: tk.Misc,
) -> tuple[CastleTile, ...]:
    """Load and rotate the course Castle tiles, matching exact image edges."""
    tiles = []
    for tile_name in _TILE_NAMES:
        image = tk.PhotoImage(
            master=master,
            file=str(_TILESET_PATH / f"{tile_name}.png"),
        )
        base_pixels = tuple(
            tuple(
                "#%02x%02x%02x" % image.get(x, y)
                for x in range(image.width())
            )
            for y in range(image.height())
        )
        pixels = base_pixels
        ports = _BASE_ROAD_PORTS.get(tile_name, frozenset())
        for rotation in range(4):
            rotated_name = (
                tile_name if rotation == 0 else f"{tile_name}_rot{rotation}"
            )
            tiles.append(
                CastleTile(
                    name=rotated_name,
                    pixels=pixels,
                    edges=_tile_edges(pixels),
                    road_ports=ports,
                    image=_photo_from_pixels(master, pixels),
                )
            )
            pixels = _rotate_pixels(pixels)
            ports = _rotate_road_ports(ports)
    return tuple(tiles)


def _build_compatibility(
    tiles: Sequence[CastleTile],
) -> dict[Direction, tuple[frozenset[int], ...]]:
    return {
        direction: tuple(
            frozenset(
                neighbor_index
                for neighbor_index, neighbor in enumerate(tiles)
                if tile.edges[direction]
                == neighbor.edges[_OPPOSITES[direction]]
            )
            for tile in tiles
        )
        for direction in _DIRECTIONS
    }


def _propagate(
    domains: dict[Cell, set[int]],
    compatibility: Mapping[Direction, Sequence[frozenset[int]]],
    changed: Iterable[Cell],
) -> None:
    pending = deque(changed)
    while pending:
        cell = pending.popleft()
        if not domains[cell]:
            raise ValueError(f"WFC contradiction at {cell}")
        for direction, offset in _OFFSETS.items():
            neighbor = (
                cell[0] + offset[0],
                cell[1] + offset[1],
                cell[2] + offset[2],
            )
            if neighbor not in domains:
                continue
            supported = set().union(
                *(compatibility[direction][index] for index in domains[cell])
            )
            reduced = domains[neighbor] & supported
            if not reduced:
                raise ValueError(f"WFC contradiction at {neighbor}")
            if reduced != domains[neighbor]:
                domains[neighbor] = reduced
                pending.append(neighbor)


def _road_is_connected(
    layout: Mapping[Cell, int],
    tiles: Sequence[CastleTile],
    start: Cell,
    goal: Cell,
) -> bool:
    pending = deque([start])
    visited = {start}
    while pending:
        cell = pending.popleft()
        if cell == goal:
            return True
        tile = tiles[layout[cell]]
        for direction, offset in _OFFSETS.items():
            if direction not in tile.road_ports:
                continue
            neighbor = (
                cell[0] + offset[0],
                cell[1] + offset[1],
                cell[2] + offset[2],
            )
            if neighbor not in layout or neighbor in visited:
                continue
            neighbor_tile = tiles[layout[neighbor]]
            if _OPPOSITES[direction] in neighbor_tile.road_ports:
                visited.add(neighbor)
                pending.append(neighbor)
    return False


def solve_castle_layout(
    tiles: Sequence[CastleTile],
    seed: int,
    on_step: Callable[[Mapping[Cell, frozenset[int]]], None],
    max_attempts: int = 100,
) -> tuple[dict[Cell, int], Cell, Cell]:
    """Collapse a Castle grid and retry layouts until the road route connects."""
    cells = [
        (0, x, y)
        for y in range(_GRID_SIZE)
        for x in range(_GRID_SIZE)
    ]
    start = _START
    goal = _GOAL
    tile_by_name = {tile.name: index for index, tile in enumerate(tiles)}
    fixed_tiles = {
        start: tile_by_name["road_rot1"],
        goal: tile_by_name["road_rot1"],
    }
    compatibility = _build_compatibility(tiles)
    rng = random.Random(seed)
    last_error = "no connected road between START and EXIT"

    for _ in range(max_attempts):
        domains = {cell: set(range(len(tiles))) for cell in cells}
        for cell, tile_index in fixed_tiles.items():
            domains[cell] = {tile_index}
        try:
            _propagate(domains, compatibility, fixed_tiles)
            on_step({cell: frozenset(domain) for cell, domain in domains.items()})

            while True:
                unresolved = [cell for cell in cells if len(domains[cell]) > 1]
                if not unresolved:
                    layout = {cell: next(iter(domains[cell])) for cell in cells}
                    if _road_is_connected(layout, tiles, start, goal):
                        on_step(
                            {
                                cell: frozenset((tile_index,))
                                for cell, tile_index in layout.items()
                            }
                        )
                        return layout, start, goal
                    last_error = "no connected road between START and EXIT"
                    break

                min_options = min(len(domains[cell]) for cell in unresolved)
                candidates = [
                    cell for cell in unresolved
                    if len(domains[cell]) == min_options
                ]
                cell = rng.choice(candidates)
                domains[cell] = {rng.choice(sorted(domains[cell]))}
                _propagate(domains, compatibility, [cell])
                on_step(
                    {cell: frozenset(domain) for cell, domain in domains.items()}
                )
        except ValueError as error:
            last_error = str(error)

    raise RuntimeError(
        f"Could not generate a connected Castle layout in "
        f"{max_attempts} attempts: {last_error}"
    )


class LiveCastleView:
    def __init__(
        self,
        root: tk.Tk,
        tiles: Sequence[CastleTile],
        start: Cell,
        goal: Cell,
    ) -> None:
        self.root = root
        self.tiles = tiles
        self.start = start
        self.goal = goal
        self.image_items: dict[Cell, int] = {}
        self.other_items: dict[Cell, tuple[int, int]] = {}
        self.previous_collapsed = 0
        self.status = tk.StringVar(value="Starting Castle WFC…")
        root.title("Castle Floorplan — Wave Function Collapse")
        root.resizable(False, False)
        tk.Label(root, textvariable=self.status, padx=8, pady=8).pack()
        self.canvas = tk.Canvas(
            root,
            width=_GRID_SIZE * _CELL_PIXELS,
            height=_GRID_SIZE * _CELL_PIXELS,
            background="#98c415",
            highlightthickness=0,
        )
        self.canvas.pack()
        self.marker_items = {}
        for cell, label in ((start, "START"), (goal, "EXIT")):
            x, y = cell[1], cell[2]
            self.marker_items[label] = self.canvas.create_text(
                x * _CELL_PIXELS + _CELL_PIXELS / 2,
                (_GRID_SIZE - 1 - y) * _CELL_PIXELS + _CELL_PIXELS / 2,
                text=label,
                fill="#111111",
                font=("Segoe UI", 7, "bold"),
                tags=(f"marker-{label}",),
            )

    def show_step(self, domains: Mapping[Cell, frozenset[int]]) -> None:
        collapsed = sum(len(domain) == 1 for domain in domains.values())
        if collapsed < self.previous_collapsed:
            self.status.set("Retrying: checking for a global road connection…")
        else:
            self.status.set(
                f"Castle floorplan forming — {collapsed}/{len(domains)} tiles collapsed"
            )
        self.previous_collapsed = collapsed

        for cell, domain in domains.items():
            x, y = cell[1], cell[2]
            canvas_x = x * _CELL_PIXELS
            canvas_y = (_GRID_SIZE - 1 - y) * _CELL_PIXELS
            if len(domain) == 1:
                tile = self.tiles[next(iter(domain))]
                if tile.image is None:
                    raise RuntimeError(f"Tile image missing for {tile.name}")
                if cell not in self.image_items:
                    self.image_items[cell] = self.canvas.create_image(
                        canvas_x,
                        canvas_y,
                        image=tile.image,
                        anchor="nw",
                    )
                else:
                    self.canvas.itemconfigure(
                        self.image_items[cell],
                        image=tile.image,
                    )
                if cell in self.other_items:
                    rectangle, label = self.other_items.pop(cell)
                    self.canvas.delete(rectangle)
                    self.canvas.delete(label)
            else:
                if cell in self.image_items:
                    self.canvas.delete(self.image_items.pop(cell))
                if cell not in self.other_items:
                    rectangle = self.canvas.create_rectangle(
                        canvas_x,
                        canvas_y,
                        canvas_x + _CELL_PIXELS,
                        canvas_y + _CELL_PIXELS,
                        fill="#d9e7bc",
                        outline="",
                    )
                    label = self.canvas.create_text(
                        canvas_x + _CELL_PIXELS / 2,
                        canvas_y + _CELL_PIXELS / 2,
                        text=str(len(domain)),
                        fill="#52613d",
                        font=("Segoe UI", 7),
                    )
                    self.other_items[cell] = (rectangle, label)
                else:
                    _, label = self.other_items[cell]
                    self.canvas.itemconfigure(label, text=str(len(domain)))

        for marker in self.marker_items.values():
            self.canvas.tag_raise(marker)
        self.root.update()
        time.sleep(0.025)

    def finish(self, message: str) -> None:
        self.status.set(message)
        self.root.update()
        self.root.mainloop()


def run_live_castle(seed: int = 7) -> dict[Cell, int]:
    root = tk.Tk()
    tiles = load_castle_tiles(root)
    view = LiveCastleView(root, tiles, _START, _GOAL)
    layout, _, _ = solve_castle_layout(
        tiles,
        seed=seed,
        on_step=view.show_step,
    )
    view.show_step(
        {cell: frozenset((tile_index,)) for cell, tile_index in layout.items()}
    )
    view.finish("Complete — START and EXIT are connected through the Castle roads.")
    return layout


if __name__ == "__main__":
    run_live_castle()
