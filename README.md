# EscapeRoom
Exam project for the course 02563 Generative Methods for Computer Graphics.

This project generates a directed, multistory escape-room graph with a start
and an exit, then realizes its inter-floor connections as validated 3D
centerlines. Straight graph-defined stairs ascend; WFC-generated slides
descend. The current output is a Python data model and Matplotlib
visualization, not a finished playable building or Blender scene.

## Get started (Windows PowerShell)

From the `EscapeRoom` directory, create a virtual environment and install the
runtime dependencies:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install numpy scipy networkx matplotlib
```

SciPy is used by the archived previous graph implementation and its regression
fixtures; the active graph generator does not require it.
There is no dependency lockfile or declared supported version range. The
audited environment used Python 3.13.12, NumPy 2.5.5, SciPy 1.18.1, NetworkX
3.7 and Matplotlib 3.11.2. Installing unpinned versions may change outputs.
Interactive plotting also requires a working Matplotlib GUI backend such as
TkAgg (with Tk support).

```powershell
.\.venv\Scripts\python.exe visualize_building.py --seed 10 --grid-resolution 2
.\.venv\Scripts\python.exe visualize_building.py --seed 10 --grid-resolution 3
.\.venv\Scripts\python.exe visualize_building.py --seed 13 --grid-resolution 2 --max-building-retries 3
.\.venv\Scripts\python.exe -m unittest discover -s tests
.\.venv\Scripts\python.exe -m unittest discover -s archive
```

The visualization opens the inherited graph on the left and accepted route
geometry on the right. VS Code tasks in a parent workspace are outside this
repository and may not be available in a standalone checkout. The tracked
Castle playground and its tests are separate from the active 3D workflow.
Other commands below use `python`; substitute
`.\.venv\Scripts\python.exe` if the environment is not activated.

## Data flow and important files

1. [`src/graph.py`](src/graph.py) creates a directed graph, a start-to-exit
   solution path and additional edges. It validates reachability and repairs
   dead ends, regenerating invalid candidates up to `max_graph_attempts`
   (default 100). Exhaustion raises `GraphGenerationError`. Same-floor
   connections are generally bidirectional, with exceptions near directional
   route heads and the exit.
2. [`src/wfc_building.py`](src/wfc_building.py) checks and, where possible,
   repairs the graph embedding, reserves direct stairs and slide connectors,
   and generates slides with WFC. An embedding repair may change coordinates
   on an interior floor while preserving topology and start/exit.
3. [`src/wfc_tiles.py`](src/wfc_tiles.py),
   [`src/wfc_solver.py`](src/wfc_solver.py) and
   [`src/wfc_global.py`](src/wfc_global.py) implement typed ports, local
   propagation, collapse, directed reachability and final global checks.
4. The validated `WFCPath.accepted_geometry` is the source for route plotting
   and JSON export. [`src/route_diagnostics.py`](src/route_diagnostics.py)
   separates logical errors from advisory shape/clearance warnings.

Other entry points and resources:

- [`visualize_building.py`](visualize_building.py): interactive graph and
  accepted routes, with explicit seed and horizontal resolution.
- [`examples/wfc_building_example.py`](examples/wfc_building_example.py):
  demonstration with diagnostics and optional image saving; its recovery
  defaults differ from the interactive script.
- [`examples/route_robustness_experiment.py`](examples/route_robustness_experiment.py):
  seed sweeps, recovery comparisons, diagnostics and CSV/plot output.
- [`src/json_export.py`](src/json_export.py) and
  [`examples/export_building_json.py`](examples/export_building_json.py):
  export a validated building; [`blender/import_building.py`](blender/import_building.py)
  creates Blender curve placeholders.
- [`src/building_generation.py`](src/building_generation.py): optional bounded
  whole-building regeneration after geometric routing failures.
- [`tests/`](tests/) and [`archive/legacy_graph_wfc/`](archive/legacy_graph_wfc/):
  active tests and an older prototype. Experiment outputs may be generated
  locally but are not included in the repository.
- [`BLENDER_INTEGRATION.md`](BLENDER_INTEGRATION.md): current data contract
  and JSON schema; [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md):
  algorithm, dependency map and development guidance.
- [`escape_room_todo.md`](escape_room_todo.md): project planning and proposals,
  not a guarantee that every listed feature exists.

The graph uses `(x, y, floor)` coordinates. Route-grid cells use
`(floor_step, x_step, y_step)`; their world coordinates divide horizontal and
vertical indices by their respective resolutions. `floor_grid` contains all
graph grid nodes, not a list of constructed rooms. No real-world units,
building meshes, wall volumes or physical unit scale are defined.
The same seed and configuration are reproducible in a matching software
environment; changes in dependencies, iteration order, heuristics or
resolution can change results. A bounded search failure is not proof of
geometric impossibility; bounded graph generation can also fail. Its dead-end
policy checks reachable dead ends for an exit but does not require every
reachable cycle to have a path to the exit.

Whole-building regeneration is opt-in. The visualization and JSON-export
commands accept `--max-building-retries 3` for at most four complete
candidates. Candidate zero uses the requested seed unchanged; subsequent
candidate seeds are `(seed + index * 0x9E3779B9) % 2**32`. Both graph
generation and WFC use the candidate seed. `max_graph_attempts` remains a
separate per-candidate bound; graph-search exhaustion is reported immediately.
Unexpected errors and non-geometric failures are not retried. A retry can
produce a different graph and topology; the accepted graph and all of its
routes remain together. Exhaustion reports the candidate failures rather
than accepting invalid geometry. The limit bounds candidate count, not the
runtime of an individual candidate.
The robustness experiment also accepts `--max-building-retries`; its CSV
includes the accepted seed and per-candidate effort.

```python
from src.building_generation import generate_building

result = generate_building(
    floors=5, width=5, length=5, difficulty=2,
    seed=13, grid_resolution=2, recovery_strategy="none",
    max_building_retries=3, max_graph_attempts=100,
)
building = result.building
print(building.requested_seed, building.accepted_seed, result.attempts)
```

## Active 3D route generator

The supported WFC workflow uses `src/graph.py`'s `Building` as the source of
the floor graph. `src/wfc_building.py` defines `WFCBuilding(Building)`,
inheriting the graph without changing its topology. Graph nodes use
`(x, y, floor)` coordinates, and stair/slide edges define fixed head and tail
endpoints.

`WFCBuilding` represents each stair as the straight 3D segment between its
original graph endpoints. It validates and reserves these fixed segments
before routing slides; it does not generate stair WFC routes. An obstructed
direct stair fails explicitly with the conflicting geometry. Stair records
have no WFC cells or attempts, and `accepted_geometry` holds their two endpoints.
Before routing, graph-generated stair segments are checked against unrelated
stair and slide endpoints using the existing collision tolerance. If an
interior floor contains a conflict, the generator tries a bounded set of
whole-floor rotations/reflections that preserve graph topology, same-floor
adjacency, start/exit and the solution path. `embedding_repair` records the
change. If no proposed embedding clears all stair/endpoint conflicts, the
original embedding remains and routing rejects the collision explicitly.

For slides, `WFCBuilding` maps endpoints into a separate, configurable 3D
inter-floor grid using `(floor_step, x_step, y_step)` cells. `grid_resolution`
controls horizontal sub-cells; `vertical_grid_resolution` independently
controls vertical subdivisions. By default the vertical resolution is at
least three cells per floor interval, leaving enough intermediate levels for
aligned endpoints to route horizontally while descending. Graph endpoints
remain fixed. WFC chooses discrete slide route tiles between them; it does not
regenerate the building graph or replace routing with a pathfinding algorithm.

`src/wfc_tiles.py` defines directional, typed ports and tile compatibility.
`src/wfc_solver.py` handles cell domains, minimum-remaining-values collapse,
propagation, constraints, and retries. `src/wfc_global.py` enforces typed
endpoint reachability and route direction, and prunes domains that cannot
belong to a required route. Direct stairs ascend in their graph direction;
slides descend. Slide routes share occupancy, respect forbidden cells and
reserved stair segments, and reject disconnected or invalid results. Slide
paths can be smoothed while preserving their endpoints and descending height.
The default WFC configuration uses source-frontier collapse ordering.
Slide connector reservations are graph-node-aware: incident slides may share a
landing connector only when they reserve the same graph node and vertical
direction. Direct stairs do not reserve artificial vertical connectors;
their actual straight geometry is checked against slide connectors, other
stairs, slide centerlines, forbidden cells and graph route endpoints. Shared
graph endpoints are permitted, but overlapping segments are not.
Slide smoothing keeps endpoint positions and terminal approach directions
fixed. If full smoothing fails validation, the same discrete route is retried
at two bounded lower smoothing strengths; every variant must still pass the
unchanged slope, curvature, clearance, and collision checks. Other graph route
endpoints are treated as geometric obstacles, and actual smoothed centerlines
are checked for intersections. Contact at a common graph endpoint is allowed
only at that exact point; overlapping segments remain disallowed unless they
are an explicitly compatible shared connector.
Each candidate is also checked against the previously accepted smoothed
geometry in the reverse direction, so a later route cannot place occupied cells
too close to an earlier centerline. The exact accepted sampled geometry,
including any local slide smoothing strength and height anchor, is retained
for diagnostics and plotting. Direct stair-to-cell tests cover the entire
segment, not only its endpoints. WFC excludes grid cells within the existing
`0.2 / grid_resolution` clearance of a direct stair at candidate validation,
without changing WFC domain propagation. Missing accepted geometry is an error.
The JSON exporter uses `accepted_geometry`; the Blender importer creates
visual placeholders, not physical meshes.
Tile-aware directed-route pruning and one-step candidate lookahead are
independently configurable but off by default: the tested tile-aware option
added substantial runtime without improving success, and lookahead did not
improve success either. Tile-aware pruning tracks a specific tile and both
route ports at each cell, rather than combining an incoming port from one
domain value with an outgoing port from another. These checks guide or prune
WFC domains; they do not construct routes. A failure within the configured
attempt budget means no solution was found, not that infeasibility was proved.
When a slide candidate collides with a fixed stair, subsequent attempts may
try up to four route-specific, one-cell empty-tile constraints ranked by the
collision location. Constraints are temporary and do not alter global
occupancy or connector reservations. Previously rejected discrete signatures
are not smoothed repeatedly. All candidate geometry remains subject to the
same checks, and retries stay within `max_route_attempts`.
Domain-level reachability pruning uses an over-approximation of compatible
edges, so it preserves tiles on every realizable directed endpoint route.
Solver failures record their phase, attempt, cell, and domain; route failures
also include endpoint connector constraints and grid availability. Once a
required route is fixed, unrelated unresolved cells prefer the empty tile to
avoid creating disconnected route fragments.

Slide elevation is parameterized by horizontal arc length, not sample index.
The configurable `WFCBuilding` settings `minimum_slide_horizontal_travel`
(default `0.5` graph units), `maximum_slide_grade` (default `1.5`), and
`minimum_turning_radius` (default `0.02` graph units) are enforced when WFC
candidates are accepted. A vertically aligned slide must find a horizontal
detour sufficient for both the minimum travel and its height drop; otherwise
generation fails explicitly. Longer and larger floor-span routes are attempted
first, and each route gets an independent deterministic retry-seed range.
Slide recovery is bounded by `max_route_backtrack_depth` and
`max_route_recovery_attempts`, with `recovery_strategy` set to `"none"`,
`"recency"`, or `"conflict-aware"`. Recency recovery reopens the most recently
accepted routes. Conflict-aware recovery counterfactually restores owned route
cells and connector reservations, then checks six-neighbor free-space
reachability between the failed endpoints. It ranks the smallest route sets
that restore spatial reachability and asks WFC to regenerate those routes with
the failed route. This is a blocker heuristic, not proof of typed-tile or
geometric feasibility; the original WFC constraints still determine acceptance.
If the heuristic finds no workable candidate, bounded recency recovery remains
the fallback. Fixed stair segments are never reopened. Unaffected routes are
preserved, and occupancy is rebuilt after
each attempt. Recovery seed offsets are deterministic.

The configurable `minimum_curvature_radius` (default `0.02` graph units) is
enforced on each smoothed slide before route acceptance. Curvature uses a
two-sided, multi-sample discrete tangent estimate, including endpoint-adjacent
sections; it is less sensitive to sample spacing than a three-point
circumcircle, but remains an approximation of the sampled centerline rather
than a Blender-scale physical radius. A failing candidate may use a
less-smoothed variant of the same discrete path, but no variant that violates
the configured geometric constraints is accepted.
Set `vertical_grid_resolution` below three only when a coarser vertical route
grid is intentional; vertically aligned slides may then be geometrically
impossible and will be rejected rather than made artificially steep.

Run the 3D Matplotlib demonstration:

```powershell
python -m examples.wfc_building_example
```

For an optional overlay of problematic turns, slopes, curvature violations,
and close route approaches, run:

```powershell
python -m examples.wfc_building_example --diagnostics
```

`src/route_diagnostics.py` reports logical checks (endpoint accuracy, slide cell
continuity or direct-stair segment shape, and direction) separately from geometric warnings (slide
slope, turning angles and curvature, route overlap, and minimum clearance).
An accepted route that fails geometric revalidation is a logical error rather
than a warning; the diagnostic reports its retained geometry.
Default diagnostic thresholds are heuristic graph-coordinate values, not
Blender dimensions or building-code limits; warnings do not alter the graph or
routes.

To compare deterministic seeds and route-grid resolutions, run:

```powershell
New-Item -ItemType Directory -Force results | Out-Null
python -m examples.route_robustness_experiment --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 --grid-resolutions 2 --compare-strategies --csv results\route-results.csv --plot results\route-results.png
```

The experiment reports success/failure, graph-generation versus WFC failures,
route types and discrete lengths, WFC attempts, bounded recovery activity, and
generation time. It also labels graph, discrete WFC, and smoothed-geometry
feasibility separately. `--compare-strategies` runs four matched configurations:
no recovery, recency recovery, conflict-aware recovery, then conflict-aware
recovery with the curvature threshold enabled. The first three disable only the
new minimum-radius rejection so the effect of recovery strategy is isolated.
All configurations use the same seeds and remaining parameters. The summary
includes overall and graph-conditional success, mean and median runtime, WFC
attempts, recovery operations, regenerated routes, and logical/geometric
diagnostic counts for successful buildings. CSV columns preserve detailed
failed-route and recovery records as JSON values. The plot compares
graph-conditional success and per-seed runtime by strategy; marker shape
distinguishes success, WFC failure, and inherited graph failure.

To compare the route-feasibility and collapse heuristics independently on
identical seeds, use:

```powershell
New-Item -ItemType Directory -Force results | Out-Null
python -m examples.route_robustness_experiment --seeds 0 1 2 3 4 --grid-resolutions 2 --compare-solver-heuristics --minimum-curvature-radius 0 --csv results\solver-ablation.csv
```

This runs baseline, each heuristic individually, the default frontier
configuration, and combined options, with recovery disabled for a controlled
solver comparison. Endpoint-frontier ordering is enabled by default; use
`--no-endpoint-frontier` to disable it. Tile-aware pruning can be enabled with
`--tile-aware-connectivity`; one-step lookahead can be enabled with
`--candidate-lookahead`.

The archived graph-to-WFC plot documents an earlier development stage: it
translated abstract graph edges into tile constraints. It is not directly
equivalent to the active 3D route plot, which solves the separate spatial
routing problem between fixed graph endpoints.

To inspect a particular seeded 3D result, including routes reopened and
regenerated during recovery, run:

```powershell
New-Item -ItemType Directory -Force output | Out-Null
python -m examples.wfc_building_example --seed 18 --diagnostics --save-plot output\building-seed-18.png
```

Run the active test suite:

```powershell
python -m unittest discover -s tests
```

Run the active WFC tests individually:

```powershell
python -m unittest discover -s tests -p test_wfc_building.py
python -m unittest discover -s tests -p test_wfc_tiles.py
python -m unittest discover -s tests -p test_wfc_solver.py
python -m unittest discover -s tests -p test_wfc_global.py
python -m unittest discover -s tests -p test_route_diagnostics.py
python -m unittest discover -s tests -p test_route_robustness_experiment.py
```

## Repository layout

- Active implementation: `src/graph.py`, `src/wfc_building.py`,
  `src/wfc_tiles.py`, `src/wfc_solver.py`, `src/wfc_global.py`, and
  `src/route_diagnostics.py`.
- The tracked root `graph.py` is historical; the active building generator
  imports `src.graph`.
- Active 3D demonstration: `examples/wfc_building_example.py`.
- Seed/resolution experiment: `examples/route_robustness_experiment.py`.
- Active tests: `tests/test_wfc_building.py`, `tests/test_wfc_tiles.py`,
  `tests/test_wfc_solver.py`, `tests/test_wfc_global.py`, and
  `tests/test_route_diagnostics.py`,
  `tests/test_route_robustness_experiment.py`.
- The legacy Castle playground and `tests/test_wfc_exercise_playground.py`
  are separate from the active 3D workflow.
- Archived prototype: `archive/legacy_graph_wfc/` preserves the old approach
  that translated graph edges directly into required WFC tile connections.
  It does not generate spatial routes between fixed endpoints and is not part
  of active test discovery. To run its demonstration or tests explicitly:

```powershell
python -m archive.legacy_graph_wfc.examples.graph_wfc_example
python -m unittest archive.legacy_graph_wfc.tests.test_wfc_graph
```
