"""Render the saved hybrid slide decision trace as an animation and summary."""

import argparse
import ast
from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "results" / "final"
ROUTE_ID = "0:(0, 4, 3)->(0, 0, 0):slide"
Cell = tuple[int, int, int]


@dataclass(frozen=True)
class Choice:
    number: int
    cell: Cell
    tile: str
    prefix: tuple[Cell, ...]
    successor: Cell
    candidates: tuple[tuple[str, float], ...]


def _cell(value):
    cell = tuple(value)
    if len(cell) != 3 or any(type(coordinate) is not int for coordinate in cell):
        raise ValueError(f"Invalid grid cell: {value!r}")
    return cell


def load_choices(explained_path, decisions_path, building_path):
    """Reconstruct recorded prefixes and probabilities; reject inconsistent traces."""
    explained = explained_path.read_text(encoding="utf-8")
    decisions = decisions_path.read_text(encoding="utf-8")
    header = r"^Choice (\d+) \| cell (\([^)]+\)) \| (\S+) \| (\d+) eligible \| sampled ([\d.]+)%$"
    explained_blocks = re.split(r"(?=^Choice \d+ \|)", explained, flags=re.MULTILINE)[1:]
    decision_blocks = re.split(r"(?=^Choice \d+ \|)", decisions, flags=re.MULTILINE)[1:]
    if not explained_blocks or len(explained_blocks) != len(decision_blocks):
        raise ValueError("Decision traces have different numbers of choices")
    choices = []
    end = None
    for number, (detail, summary) in enumerate(zip(explained_blocks, decision_blocks), 1):
        detail_header = re.search(header, detail, re.MULTILINE)
        summary_header = re.search(header, summary, re.MULTILINE)
        prefix_match = re.search(r"^  Committed prefix \(including frontier\): (.+)$",
                                 detail, re.MULTILINE)
        successor_match = re.search(r"^  Sampled (\S+): .* exits via \w+ to (\([^)]+\))\.$",
                                    detail, re.MULTILINE)
        end_match = re.search(r"^  Entering from .*; end = (\([^)]+\))\.$",
                              detail, re.MULTILINE)
        if not (detail_header and summary_header and prefix_match and successor_match and end_match):
            raise ValueError(f"Choice {number} is incomplete")
        if detail_header.group(0) != summary_header.group(0) or int(detail_header[1]) != number:
            raise ValueError(f"Choice {number} differs between traces")
        prefix = tuple(_cell(item) for item in ast.literal_eval(prefix_match[1]))
        cell = _cell(ast.literal_eval(detail_header[2]))
        successor = _cell(ast.literal_eval(successor_match[2]))
        current_end = _cell(ast.literal_eval(end_match[1]))
        candidates = tuple(
            (name, float(probability))
            for name, probability in re.findall(
                r"^    (\S+): weight [\d.eE+-]+; probability ([\d.]+)%$",
                summary, re.MULTILINE)
        )
        if (not prefix or prefix[-1] != cell or len(set(prefix)) != len(prefix)
                or successor_match[1] != detail_header[3]
                or len(candidates) != int(detail_header[4])
                or sum(name == detail_header[3] for name, _ in candidates) != 1
                or any(not 0 <= probability <= 100 for _, probability in candidates)
                or abs(sum(probability for _, probability in candidates) - 100) > 0.1
                or abs(next(probability for name, probability in candidates
                            if name == detail_header[3]) - float(detail_header[5])) > 0.051
                or any(sum(abs(a - b) for a, b in zip(left, right)) != 1
                       for left, right in zip(prefix, prefix[1:] + (successor,)))):
            raise ValueError(f"Choice {number} contains inconsistent path or candidates")
        if choices and prefix[:len(choices[-1].prefix) + 1] != (
                choices[-1].prefix + (choices[-1].successor,)):
            raise ValueError(f"Choice {number} does not extend the previous choice")
        if end is not None and end != current_end:
            raise ValueError("The route endpoint changed between choices")
        end = current_end
        choices.append(Choice(number, cell, detail_header[3], prefix, successor, candidates))

    building = json.loads(building_path.read_text(encoding="utf-8"))
    route, = (route for route in building["routes"] if route["id"] == ROUTE_ID)
    horizontal = building["generation"]["grid_resolution"]
    vertical = building["generation"]["vertical_grid_resolution"]

    def grid_cell(point):
        return (point[2] * vertical, point[0] * horizontal, point[1] * horizontal)

    if choices[0].prefix[0] != grid_cell(route["source"]) or end != grid_cell(route["target"]):
        raise ValueError("Trace endpoints differ from the accepted building")
    full_path = choices[-1].prefix + (choices[-1].successor, end)
    budget = re.search(r"^Budget: (\d+) cells; accepted: (\d+) cells\.$",
                       decisions, re.MULTILINE)
    if (not budget or len(full_path) != int(budget[2])
            or len(full_path) > int(budget[1]) or len(set(full_path)) != len(full_path)
            or sum(abs(a - b) for a, b in zip(full_path[-2], end)) != 1):
        raise ValueError("Completed path does not match the recorded route")
    return choices, full_path


def render(choices, full_path, index):
    """Show the committed prefix and the recorded sampled successor."""
    choice = choices[index]
    fig = plt.figure(figsize=(12, 6), facecolor="#f4f7fb")
    grid = fig.add_gridspec(2, 2, width_ratios=(1.15, 1), height_ratios=(1, 1),
                            left=.07, right=.97, top=.80, bottom=.13, wspace=.34, hspace=.45)
    plan = fig.add_subplot(grid[:, 0])
    elevation = fig.add_subplot(grid[0, 1])
    bars = fig.add_subplot(grid[1, 1])
    completed = choice.prefix + (choice.successor,)
    for ax, a, b, title, labels in (
        (plan, 1, 2, "Top view", ("x grid cell", "y grid cell")),
        (elevation, 2, 0, "Elevation", ("y grid cell", "vertical layer")),
    ):
        ax.set_facecolor("white")
        ax.plot([c[a] for c in full_path], [c[b] for c in full_path],
                color="#cbd5e1", lw=3, label="Complete route reference", zorder=1)
        ax.plot([c[a] for c in choice.prefix], [c[b] for c in choice.prefix],
                color="#087e8b", lw=3.5, marker="o", markersize=4,
                label="Committed prefix", zorder=2)
        ax.plot([c[a] for c in completed[-2:]], [c[b] for c in completed[-2:]],
                color="#e67e22", lw=4, marker="o", markersize=7,
                label="Sampled step", zorder=3)
        ax.scatter([full_path[0][a]], [full_path[0][b]], s=95, marker="^",
                   color="#15945f", edgecolor="white", zorder=4)
        ax.scatter([full_path[-1][a]], [full_path[-1][b]], s=95, marker="*",
                   color="#c43b42", edgecolor="white", zorder=4)
        ax.set_title(title, loc="left", fontsize=12, weight="bold", color="#163047")
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.grid(color="#e4ebf2", linewidth=.7)
        ax.margins(.13)
    plan.set_title("Top view (axes scaled independently)", loc="left",
                   fontsize=12, weight="bold", color="#163047")
    labels = [name.removeprefix("slide_").replace("_", " ") for name, _ in choice.candidates]
    values = [value for _, value in choice.candidates]
    colors = ["#e67e22" if name == choice.tile else "#9bb2c6"
              for name, _ in choice.candidates]
    bars.barh(labels, values, color=colors, height=.65)
    bars.invert_yaxis()
    bars.set_xlim(0, 112)
    bars.set_xlabel("Sampling probability (%)")
    bars.set_title("Eligible tiles", loc="left", fontsize=12, weight="bold", color="#163047")
    bars.tick_params(axis="y", labelsize=8)
    for row, value in enumerate(values):
        bars.text(value + 1, row, f"{value:.2f}%", va="center", fontsize=8)
    fig.text(.07, .94, "HYBRID WFC  /  SLIDE GENERATION",
             fontsize=18, weight="bold", color="#163047")
    fig.text(.07, .88, f"Choice {choice.number:02d} / {len(choices)}  |  "
             f"cell {choice.cell}  ->  {choice.successor}",
             fontsize=12, color="#405970")
    fig.text(.07, .035, "Teal: committed prefix    Orange: sampled step    "
             "Grey: completed route reference    Green: start    Red: end",
             fontsize=9, color="#405970")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final", type=Path, default=FINAL)
    parser.add_argument("--output", type=Path, required=True,
                        help="Directory for slide_generation.gif and slide_generation.png")
    args = parser.parse_args()
    gif_path = args.output / "slide_generation.gif"
    png_path = args.output / "slide_generation.png"
    if gif_path.exists() or png_path.exists():
        raise FileExistsError("Output already exists; choose a new directory")
    choices, full_path = load_choices(
        args.final / "algorithm" / "wfc_decisions_explained.txt",
        args.final / "algorithm" / "wfc_decisions.txt",
        args.final / "building" / "building_seed10_hybrid.json",
    )
    args.output.mkdir(parents=True, exist_ok=True)
    frames = []
    try:
        for index in range(len(choices)):
            fig = render(choices, full_path, index)
            try:
                stream = BytesIO()
                fig.savefig(stream, format="png", dpi=100, facecolor=fig.get_facecolor())
                stream.seek(0)
                with Image.open(stream) as image:
                    frames.append(image.convert("RGB"))
            finally:
                plt.close(fig)
        with png_path.open("xb") as output:
            frames[-1].save(output, format="PNG")
        with gif_path.open("xb") as output:
            frames[0].save(output, format="GIF", save_all=True, append_images=frames[1:],
                           duration=[1200] * (len(frames) - 1) + [3000], loop=0,
                           optimize=False)
    finally:
        for frame in frames:
            frame.close()
    print(f"Saved {len(choices)} recorded decisions to {gif_path} and {png_path}")


if __name__ == "__main__":
    main()
