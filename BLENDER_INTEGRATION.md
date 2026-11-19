# Blender integration handoff

## WFC output

`src/wfc_building.py` extends the original `src/graph.py` without changing the
graph generator. It preserves graph-defined inter-floor routes on a shared WFC
grid. `src/route_diagnostics.py` checks logical and geometric properties;
`src/building_generation.py` can retry complete buildings after geometric
routing failures. Graph-generation errors propagate unchanged.

The graph is directed and its nodes are `(x, y, floor)` positions. `floor`
edges connect nodes on the same floor; `stair` and `slide` edges define
inter-floor routes. `graph.solution_path` is the directed start-to-exit path.
The JSON includes all graph nodes and edges, not only those on the solution
path.

Each route's accepted geometry is its authoritative centerline.
`src/json_export.py` validates accepted routes and exports the exact stored
points as `routes[*].geometry`; it does not re-smooth or reconstruct them.
Direct stairs have two points and slides are sampled polylines. Blender must
consume these stored `[x, y, z]` points rather than rebuilding geometry from
WFC cells or calling `world_points()` or `smooth_path()`.

## Reproduce the accepted seed-10 example

Run from the repository root in PowerShell. The exporter requires a new output
file and refuses to overwrite an existing one.

```powershell
New-Item -ItemType Directory -Force results | Out-Null
python -m examples.export_building_json results\building_seed10_grid3.json --seed 10 --grid-resolution 3 --max-building-retries 3
```

The command uses five floors, a 5-by-5 footprint, difficulty 2, 30 attempts
per route, and at most three additional whole-building retries. It writes a
new file and refuses to overwrite an existing one. The validated reference
export, `results/building_seed10_grid3.json`, records requested and accepted
seed `10`, zero building retries, resolution `3`, and 20 accepted routes
(12 slides and 8 stairs).

To reproduce the reference visualization and diagnostics:

```powershell
$env:MPLBACKEND = "Agg"
python -m examples.wfc_building_example --diagnostics --seed 10 --grid-resolution 3 --save-plot results\building_seed10_grid3.png
```

The corresponding visual reference is `results/building_seed10_grid3.png`.
These generated reference files are local validation outputs and are not
included in the WFC source commit; the commands above recreate them. The plot
highlights the closest segments of advisory close-route pairs in magenta, and
other route warnings in red.

## JSON schema and coordinates

The versioned JSON object contains:

- `schema_version`: currently `1`.
- `coordinate_system`: axis order `x,y,z`, `z` up, origin `[0,0,0]`,
  graph units, and `blender_units_per_graph_unit: 1.0`.
- `generation`: requested `seed`, `requested_seed`, `accepted_seed`,
  `building_retry_count`, horizontal `grid_resolution`, and
  `vertical_grid_resolution`.
- `building`: floor count, length, width, floor elevations, start, and exit.
- `graph`: node and directed edge positions/metadata plus `solution_path`.
- `routes`: route ID, type (`stair` or `slide`), graph endpoints, and
  `geometry`, an array of stored `[x,y,z]` centerline points.
- `diagnostics`: advisory warning strings.

Graph coordinates and route geometry are in `(x,y,z)` order; floor index is
`z`. The exporter stores accepted floating-point geometry directly. Blender
should use `routes[*].geometry`, not WFC cells, `world_points()`, or
`smooth_path()`, which recomputes geometry. Coordinates currently map directly
to Blender units (one graph unit per Blender unit); scaling or changing floor
height changes slopes, curvature, and clearance and requires new validation.

For the seed-10 JSON above, the schema is `1`, horizontal and vertical
resolutions are both `3`, and the graph contains 125 nodes, 168 directed
edges, and a 21-node solution path. All 20 consecutive directed edges in the
solution path are present from start to exit. The export contains all 20
accepted route geometries exactly as generated.

## Clearance warnings and Blender thickness

The diagnostic minimum route-clearance threshold is `0.25` graph units. It is
an advisory centerline-separation threshold, not the WFC acceptance rule or a
physical-width guarantee. At resolution `3`, the accepted-geometry checks use
a `0.2 / 3` (`about 0.0667`) graph-unit tolerance for blocked cells and
reserved connectors, and reject actual route-segment intersections. The two
seed-10 close approaches have no overlapping WFC cells or logical errors:

- Route `7:(4, 3, 1)->(0, 0, 0):slide` and route
  `8:(1, 4, 0)->(2, 0, 1):stair` are separated by about `0.116` graph units.
- Route `10:(4, 0, 0)->(2, 1, 1):stair` and route
  `18:(3, 0, 1)->(3, 0, 0):slide` are separated by about `0.202` graph units.

The plot marks the nearest segment on each route in magenta. Coordinates
currently map directly to Blender units (one graph unit per Blender unit).
For tube-like geometry with the same diameter on each route, the diameter
must be below about `0.116` units for the first pair and `0.202` units for the
second pair to avoid contact at those closest approaches. More generally, the
sum of the two route radii must be less than the measured centerline
separation. Wider stairs, slide surfaces, bevels, or changed scaling can
intersect even though the zero-thickness accepted centerlines do not. Check
the actual Blender meshes and their thickness/clearance before treating the
building as physically collision-free.

The remaining seed-10 warning is route
`5:(0, 4, 3)->(0, 4, 1):slide`, whose maximum sampled turn is about `107.593°`,
above the advisory `100°` turn threshold. This is not a route intersection.
All three warnings are retained in the JSON; they are not silently treated as
logical route failures.

## Visualize and diagnose routes

For interactive use, remove `$env:MPLBACKEND = "Agg"` and omit `--save-plot`
if no image file is needed. The plot shows the graph and accepted route
geometry side-by-side; it is a visualization, not a Blender mesh or
physical-clearance certification.

## Blender import

The Blender-side importer reads the JSON's building dimensions and
`routes[*].geometry`, creates an `EscapeRoom` collection with curve objects
for stored stair/slide centerlines, and creates closed floor outlines. The
route curves use the exported points; the importer does not regenerate or
smooth routes. This is an import of reference geometry, not a complete
building: it does not create building meshes, route thickness, or
player/gameplay assets. In Blender, run:

```powershell
blender --background --python blender\import_building.py -- results\building_seed10_grid3.json
```

`blender/import_building.py` is intentionally not part of this WFC candidate;
the Blender implementation remains unchanged and is to be brought into the
Blender development stage separately. Its current implementation was checked
against the reference JSON using a Blender API stub: all 20 route curves
received the exact stored points and all five floor outlines were created.
Blender itself is not installed in this environment, so an in-Blender scene
smoke test and physical mesh-clearance check remain for the Blender stage.

## Tests

Run all repository tests from the project root:

```powershell
python -m unittest discover -s tests
```
