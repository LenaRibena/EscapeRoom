# Blender integration handoff

The preferred hybrid seed-10 centerlines are in
[`results/final/building/building_seed10_hybrid.json`](results/final/building/building_seed10_hybrid.json).
This is a schema-version-1 JSON. Its 10 slides and 9 stairs match their graph edges
and endpoints; the saved accepted centerlines have not been resmoothed.
The [hybrid results README](results/final/README.md) records the
settings, hashes, diagnostics and reproduction commands.
The [Inherited Graph vs Hybrid WFC figure](results/final/building/building_seed10_hybrid.png)
shows the stored graph and centerlines, not a second generated building.

The hybrid routes the inherited directed graph on a refined 3D grid with a
22-tile vocabulary. It samples stochastic tiles with geometry-aware weights,
propagates local port compatibility and global BFS connectivity, and applies
per-slide relative length bounds. It interpolates accepted slide paths and
checks curvature, collisions and centerline clearance with bounded
backtracking and retries. The relaxed shortest-walk bound is not an optimal
feasible route.

Do not overwrite a teammate's existing Blender input without coordination.

## Export contract

The JSON uses schema version 1 and includes:

- Generation metadata: requested and accepted seed, building retry count,
  horizontal resolution and vertical resolution.
- Building dimensions, floor elevations, start and exit.
- All graph nodes and directed edges, their attributes and the solution path.
- Route IDs, types, endpoints and `routes[*].geometry`.
- Advisory diagnostics.

Geometry points are `[x, y, z]`, with `z` up and one graph unit per Blender
unit. **Use `routes[*].geometry` exactly as exported.** These are the accepted
centerlines; rebuilding from WFC cells or smoothing them again can change
previously validated geometry. Stairs have two endpoints, while slides use
sampled polylines.

For the seed-10 hybrid result, the refined grid shape is `(10, 28, 28)` in
`(vertical, x, y)` order. The result has no diagnostic warnings. The exporter validates route
completeness and logical geometry; advisory warnings are not ignored or treated
as acceptance failures.

Accepted centerlines are not a physical mesh-clearance guarantee. Slide and
stair thickness, scaling, and floor-height changes may cause intersections or
change slopes and must be checked in Blender.

## Reproduce the JSON

The hybrid runner refuses to overwrite existing output files:

```powershell
$env:PYTHONHASHSEED = "0"
python -m examples.hybrid_experiment --strategy hybrid --weighting compact --relative-budget --minimum-clearance 0.25 --routing-seed 10 --max-seconds 60 --max-backtracks 8 --route-attempts 3 --geometry-preference 1.5 --geometry-backtracking --trace-states --trace-route-id '0:(0, 4, 3)->(0, 0, 0):slide' --output results\hybrid_reproduced.json --building-output results\hybrid_building_reproduced.json
```

More reproduction details are documented in
[`results/final/README.md`](results/final/README.md). Only the
hybrid JSON is needed as input to the existing external importer; the PNGs,
animation, text traces and manifest are presentation and provenance files. No Blender
importer or scene is in this worktree, so an actual import and physical mesh
clearance remain to be checked in the teammate's Blender project. Do not
overwrite their existing input or scene without coordination.
