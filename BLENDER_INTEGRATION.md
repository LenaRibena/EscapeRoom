# Blender integration handoff

The preferred hybrid seed-10 centerlines are in
[`results/final/building/building_seed10_hybrid.json`](results/final/building/building_seed10_hybrid.json).
This is a separate schema-version-1 JSON, with the same graph and export
contract as the baseline. Its 10 slides and 9 stairs match their graph edges
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
feasible route. The separate DFS-guided baseline uses DFS to propose routes.

The unchanged DFS-guided presentation baseline can be reproduced with the
command below; its SHA-256 is
`67c4a5161ececea97e5b0214ab7df9a71ff07e640d4694390015561878b9f345`.
It uses the exact upstream graph generation behavior, requested and accepted
seed 10, four floors, a 10-by-10 footprint, and WFC grid resolution 3. It
contains 19 accepted inter-floor routes: 10 slides and 9 stairs.

The older five-floor, 5-by-5 DFS-guided export is no longer included.
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

For both seed-10 results, the refined grid shape is `(10, 28, 28)` in
`(vertical, x, y)` order. The hybrid result has no diagnostic warnings; the
DFS baseline has one sharp-turn advisory. The exporter validates route
completeness and logical geometry; advisory warnings are not ignored or treated
as acceptance failures.

Accepted centerlines are not a physical mesh-clearance guarantee. Slide and
stair thickness, scaling, and floor-height changes may cause intersections or
change slopes and must be checked in Blender.

## Reproduce the JSON

The exporter refuses to overwrite an existing file:

```powershell
$env:PYTHONHASHSEED = "0"
python -m examples.export_building_json results\building_reproduced.json --seed 10 --grid-resolution 3
```

The command above reproduces the DFS baseline; hybrid reproduction is documented
in [`results/final/README.md`](results/final/README.md). Only the
hybrid JSON is needed as input to the existing external importer; the PNGs,
animation, text traces and manifest are presentation and provenance files. No Blender
importer or scene is in this worktree, so an actual import and physical mesh
clearance remain to be checked in the teammate's Blender project. Do not
overwrite their existing input or scene without coordination.
