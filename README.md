# EscapeRoom

Exam project for the course 02563 Generative Methods for Computer Graphics.

## Graph-constrained WFC routing

The preferred hybrid configuration and the preserved DFS-guided baseline route
an inherited directed building graph on a refined 3D grid using a 22-tile
vocabulary. The graph source is the
exact upstream `src/graph.py` from commit `6daab25`. The upstream graph sets
a four-floor, 10-by-10 footprint. The adapter rejects inconsistent dimensions.

The relative-budget hybrid samples a tile from eligible WFC candidates using
seeded positive weights. Local port compatibility and global BFS reachability
pruning propagate constraints; a committed-prefix length lower bound filters
choices that cannot fit the route's budget. A directed port-compatible walk
provides a per-route lower bound, not an exact shortest feasible path. The
optional geometry-aware weight softly discourages sharp or closely spaced
horizontal turns; it does not guarantee curvature. Completed slide paths are
continuously interpolated and checked for curvature and collisions, with
optional centerline clearance checks, bounded backtracking and route retries.
Centerline clearance does not prove Blender mesh clearance.

The hybrid is explicitly configured by
[`examples/hybrid_experiment.py`](examples/hybrid_experiment.py); the existing
`WFCBuilding` default remains `dfs_guided`. The accepted seed-10 configuration,
results, limitations and reproduction commands are in the
[final hybrid results](results/final/README.md).

In the baseline, DFS proposes a supported route, and the solver collapses compatible
tiles and propagates constraints. The representative trace shows one candidate
after DFS filtering at every route-guided collapse: DFS is therefore an
important source of route choices, rather than evidence of independent random
tile choices in that trace.

## Upstream human simulation

`src/human.py` is unchanged from upstream. `Human` reads the building's graph,
start, exit and solution path; it does not consume accepted WFC or Blender
geometry. Its `from graph import Building` requires `src` on Python's import
path when importing it from the repository root. Its standalone example still
uses the old `Building` constructor and currently fails with a `TypeError`;
it must be updated upstream before it can be run directly.

## Final deliverables and reproduction

The [final results](results/final/README.md) include the accepted hybrid
building JSON, a full-building figure, a synchronized animation and decision
trace of the accepted hybrid slide solve, and a manifest. The route geometry
is the stored accepted geometry; it is not reconstructed or smoothed again.
See the [Inherited Graph vs Hybrid WFC figure](results/final/building/building_seed10_hybrid.png).
See the final results README for the hybrid settings and reproduction commands.

To reproduce the DFS-guided baseline from the repository root in PowerShell,
use a fresh output path:

```powershell
$env:PYTHONHASHSEED = "0"
python -m examples.export_building_json results\dfs_reproduced.json --seed 10 --grid-resolution 3
Get-FileHash results\dfs_reproduced.json -Algorithm SHA256
```

The baseline hash is
`67c4a5161ececea97e5b0214ab7df9a71ff07e640d4694390015561878b9f345`.
The old DFS-guided demonstration remains reproducible with
`python -m examples.wfc_decision_demo --baseline results\dfs_reproduced.json --output results\dfs_decisions_reproduced`;
it is not the final hybrid demonstration.

The exporter refuses to overwrite an existing JSON file. See
[BLENDER_INTEGRATION.md](BLENDER_INTEGRATION.md) for the handoff contract and
clearance limitations.

Run the repository tests with:

```powershell
python -m unittest discover -s tests
```
