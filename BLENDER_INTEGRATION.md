# Blender integration handoff

## What exists

Python generates a directed NetworkX graph and validated inter-floor
centerlines. [`src/json_export.py`](src/json_export.py) exports them as JSON;
[`blender/import_building.py`](blender/import_building.py) creates visible
curve placeholders and floor outlines in Blender. It does not create floor
or wall meshes, a player, or physically dimensioned stairs or slides.

See [`src/wfc_building.py`](src/wfc_building.py) for the concrete data model,
[`src/graph.py`](src/graph.py) for the graph, and
[`src/route_diagnostics.py`](src/route_diagnostics.py) for diagnostics.

## Data to consume

After constructing `WFCBuilding` with routes enabled:

| Python attribute | Meaning |
| --- | --- |
| `building.floors`, `length`, `width` | Graph-grid dimensions: floor count, x count and y count. |
| `building.start_pos`, `exit_pos` | Start and exit graph nodes. |
| `building.graph.nodes(data=True)` | All graph-grid nodes and their attributes. Grid nodes are not necessarily occupied rooms. |
| `building.graph.edges(data=True)` | Directed connections. The current generator's edge attribute `type` is `floor`, `stair` or `slide`; older graphs use `kind`. Preserve edge direction; do not infer connectivity from spatial proximity. |
| `building.path` | Ordered graph-node solution path from start to exit; not a continuous player animation path. |
| `building.embedding_repair` | `None` or a record of an interior-floor coordinate transform applied after graph generation. Use the resulting graph and routes together. |
| `building.routes` | Accepted `WFCPath` records for inter-floor graph edges; there are no separate WFC routes for same-floor edges. |
| `building.total_wfc_attempts` | WFC attempts for the accepted candidate only. With whole-building regeneration, sum `attempt.wfc_attempts` across the factory result's `attempts` for the total effort. |
| `building.requested_seed`, `accepted_seed`, `building_retry_count` | Original seed, the seed used to generate the accepted graph and routes, and the number of rejected complete candidates. They are identical seeds and zero retries without whole-building regeneration. |

Each route has `route_id`, `kind` (`PortKind.STAIR` or `PortKind.SLIDE`),
`source`, `target`, `cells`, `accepted_geometry`, `attempts`,
`smoothing_strength`, and `height_profile_anchor`. IDs are generated from
sorted route order and graph endpoints: retain the exact ID rather than
assuming it is stable across configurations or graph changes. Source and
target are directed graph nodes. Stairs ascend and slides descend; a floor
edge is represented by the graph rather than a `WFCPath`. A direct stair has
`direct_stair=True`, no WFC cells, zero attempts, and two accepted points.
Slide geometry is a sampled polyline, not a stair-like sequence of tiles.

**Use `route.accepted_geometry` as the centerline.**
`building.validate_accepted_geometry(route)` revalidates and returns those
same stored points, raising if geometry is absent or invalid. Normal route
plotting reads the stored points. `route.world_points(...)` only converts
discrete WFC cells, and `building.smooth_path(...)` recomputes geometry; neither
is a substitute for exporting the exact accepted result. In particular,
local height anchors and accepted smoothing strength matter. Logical errors
from `diagnose_routes(building)` should block export; warnings are advisory
and must be reviewed for the intended Blender scale.

## Coordinates and scale

- Graph nodes and accepted points use `(x, y, z)`, where integer `z` is the
  floor index: ground floor `0`, higher floors positive. The origin is
  `(0, 0, 0)` at the exit; the default start is
  `(length - 1, width - 1, floors - 1)`.
- X increases with `length`, Y with `width`, Z upward with floor number.
  Graph units have **no specified meter conversion** or building-code
  dimensions. The importer maps X, Y and Z directly to Blender's X, Y and Z;
  one graph unit becomes one Blender unit, without an axis swap or scaling.
- A graph node `(x, y, floor)` maps through `building.graph_node_to_cell(node)`
  to `(floor * vertical_grid_resolution, x * grid_resolution,
  y * grid_resolution)`. A cell `(f, x, y)` maps back to
  `(x / grid_resolution, y / grid_resolution,
  f / vertical_grid_resolution)`. Horizontal and vertical resolutions may
  differ; by default the latter is `max(grid_resolution, 3)`.
- Accepted geometry is continuous in graph units and may lie between cells.
  A Blender adapter can multiply X and Y by a chosen cell width and Z by a
  chosen floor height, but this nonuniform scaling changes slopes, curvature
  and distances. Reassess physical mesh clearances after scaling; the Python
  centerline checks do not certify Blender mesh volumes.

## Minimal end-to-end Python example

Run from the project directory using `.\.venv\Scripts\python.exe`; this
configuration matches the interactive seed-10 example:

```python
from src.route_diagnostics import diagnose_routes
from src.wfc_building import WFCBuilding

building = WFCBuilding(
    floors=5, width=5, length=5, difficulty=2,
    seed=10, grid_resolution=2, recovery_strategy="none",
    max_route_attempts=30, max_route_backtrack_depth=2,
    max_route_recovery_attempts=1, minimum_curvature_radius=0.02,
    endpoint_frontier=True,
)
report = diagnose_routes(building)
if report.logical_errors:
    raise RuntimeError("\n".join(report.logical_errors))
for warning in report.warnings:
    print("Advisory:", warning)

for route in building.routes:
    points = building.validate_accepted_geometry(route)
    print(route.route_id, route.kind.value, route.source, route.target, points)
```

The graph can also be inspected with `list(building.graph.nodes(data=True))` and
`list(building.graph.edges(data=True))`.

## Export and import

From PowerShell in the project directory, create a **new** file using the
same generation settings as the interactive visualization:

```powershell
.\.venv\Scripts\python.exe -m examples.export_building_json output\building.json --seed 10 --grid-resolution 3
.\.venv\Scripts\python.exe -m examples.export_building_json output\retry-building.json --seed 13 --grid-resolution 2 --max-building-retries 3
```

Create `output` first if it does not exist. The exporter refuses to overwrite
an existing file. Generated JSON examples are not included in the repository.
To export a building already in memory:

```python
from src.json_export import export_building_json
export_building_json(building, "output/building.json")
```

In PowerShell, start Blender with the importer and the generated example
(replace the executable path if Blender is installed elsewhere):

```powershell
& "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" --python blender\import_building.py -- output\building.json
```

The script adds an `EscapeRoom` collection with orange straight stair curves,
blue slide polyline curves, and gray closed floor-outline curves. Their bevel
depths are display placeholders. Existing scene objects are not deleted; a
repeated import adds another collection. In the viewport use Material Preview
or Material color mode to see the colors. Save the Blender scene manually if
desired. To call it from Blender's Python console instead:

```python
import runpy
module = runpy.run_path(r"C:\absolute\path\to\EscapeRoom\blender\import_building.py")
module["import_building"](r"C:\absolute\path\to\EscapeRoom\output\building.json")
```

## JSON schema (version 1)

The UTF-8 JSON document has the following fields. Positions are three-number
arrays `[x, y, z]`, with Z as the floor coordinate; no WFC cells appear in
route geometry.

| Field | Meaning |
| --- | --- |
| `schema_version` | Integer `1`. |
| `coordinate_system` | `axis_order: "x,y,z"`, `up_axis: "z"`, `unit: "graph_unit"`, `origin: [0,0,0]`, and `blender_units_per_graph_unit: 1.0`. No physical size is implied. |
| `generation` | `seed` and `requested_seed` are the original requested seed; `accepted_seed` is the candidate seed used for the exported geometry; `building_retry_count` records the number of rejected complete candidates. Also contains horizontal `grid_resolution` and `vertical_grid_resolution`. This is not a complete record of every generation option. Historical version-1 files may lack the three newer seed/retry fields. |
| `building` | Grid-node counts `floors`, `length` (X), `width` (Y), integer `floor_elevations` from 0 to `floors - 1`, and `start`/`exit` positions. Outlines span X=0 to length−1 and Y=0 to width−1, not physical room boundaries. |
| `graph.nodes` | Every graph node: `position` and its `attributes` object. Positions identify nodes, including unused grid nodes. |
| `graph.edges` | Every directed connection: `source`, `target`, and `attributes` containing canonical `kind` (`floor`, `stair`, or `slide`). Original attributes are preserved, including `type` from the current generator. Opposite floor edges are separate entries when present. |
| `graph.solution_path` | Ordered graph-node positions on the generated solution path. |
| `routes` | One entry per stair/slide graph edge: `id`, `type`, `source`, `target`, and `geometry`, an ordered array of accepted continuous positions. Stairs have exactly two points; slides contain the sampled smoothed centerline. |
| `diagnostics.warnings` | Advisory diagnostic strings. Logical errors prevent export. |

`building_to_dict(building)` checks that every stair/slide edge has exactly
one accepted route and that route IDs are unique. It runs the existing
diagnostics, including accepted-geometry revalidation; missing or invalid
geometry and logical errors fail explicitly. Advisory warnings remain in
the JSON. `export_building_json` serializes with finite JSON numbers only and
does not reconstruct, resmooth or change routes. Graph attributes must be
JSON serializable; unsupported values cause an explicit serialization error.
JSON positions are exact stored numbers, while Blender stores coordinates at
its own floating-point precision.

## Suggested Blender work

1. Import all nodes and directed edges; use same-floor `floor` edges for
   corridor connectivity. Decide which nodes represent rooms before generating
   surfaces: the full grid is not a room occupancy map.
2. Place floor slabs and walls using chosen cell dimensions and openings
   derived from connectivity. This layout/mesh generation is not implemented.
3. Build stair meshes along each accepted straight stair segment and slide
   tubes or surfaces along sampled slide polylines. Design cross sections,
   landing transitions and local frames, and check mesh-to-mesh collisions.
4. Add start/exit markers and optionally construct a player path that follows
   directed graph edges, with inter-floor portions using matching accepted
   route centerlines. Blender animation and gameplay are future work.

Shared endpoints are legitimate; do not merge unrelated crossings merely
because mesh geometry overlaps. Advisory route warnings and Python
centerline validity alone do not guarantee that chosen mesh thicknesses,
railings or walls will fit.
