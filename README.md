# EscapeRoom
Exam project for the course 02563 Generative Methods for Computer Graphics

## Constraining WFC with a building route

`graph.py` represents building nodes as `(x, y, floor)` and records path and
branch edge kinds as `floor`, `stair`, or `slide`. Convert the generated graph
edges into required WFC port connections, then pass them to the solver:

```python
from src.wfc_graph import graph_edge_connections
from src.wfc_solver import solve_wfc

required_connections = graph_edge_connections(building.graph.edges(data=True))
layout = solve_wfc(cells, tiles, required_connections=required_connections)
```

WFC cells use `(floor, x, y)` coordinates. Floor edges must be cardinally
adjacent and translate to `WALK` ports. Stair and slide edges can span floors
and change `(x, y)` in the current graph model; the adapter normally places
the vertical transition at the source position and adds a horizontal walk on
the destination floor. If that would collide with another required port, it
anchors the vertical transition at the destination and adds the horizontal
walk on the source floor instead. Bidirectional floor edges are deduplicated.
Solver domains are filtered to those ports before collapse,
preserving the graph's start-to-finish route and its represented branches in
the generated layout.

Run `python -m examples.graph_wfc_example` for a headless end-to-end example
using the existing `Building` graph class. It derives a port-pattern tile set
from the graph itself to demonstrate topology preservation; it does not yet
generate semantic rooms, walls, or doors. The example prints an ASCII
floor-by-floor map of the generated walk connections and marks start, exit,
and vertical transitions.

For 3D WFC stair/slide generation, JSON export, route visualization, and the
Blender handoff, see [BLENDER_INTEGRATION.md](BLENDER_INTEGRATION.md).
