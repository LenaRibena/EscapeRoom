# EscapeRoom

Exam project for the course 02563 Generative Methods for Computer Graphics.

## Setup

The final result was reproduced with Python 3.13.11 and the following
dependencies. From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install numpy==2.4.2 scipy==1.17.0 networkx==3.7 matplotlib==3.10.8 pillow==12.1.0
```

## Graph-constrained WFC routing

The final hybrid configuration routes an inherited directed building graph on
a refined 3D grid using a 22-tile vocabulary. The graph source is the
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

To reproduce the final hybrid building from the repository root in PowerShell,
use fresh output paths:

```powershell
$env:PYTHONHASHSEED = "0"
$env:MPLBACKEND = "Agg"
python -m examples.hybrid_experiment --strategy hybrid --weighting compact --relative-budget --minimum-clearance 0.25 --routing-seed 10 --max-seconds 60 --max-backtracks 8 --route-attempts 3 --geometry-preference 1.5 --geometry-backtracking --trace-states --trace-route-id '0:(0, 4, 3)->(0, 0, 0):slide' --output results\hybrid_reproduced.json --building-output results\hybrid_building_reproduced.json
Get-FileHash results\hybrid_building_reproduced.json -Algorithm SHA256
```

The expected building hash is
`dac79b23be631eb5611cfdf295107f8c2658cd77a523c11166743b804948ff0c`.
Timing may affect retries and the reproduced result; the saved JSON remains
authoritative. The runner refuses to overwrite existing output files. See
[BLENDER_INTEGRATION.md](BLENDER_INTEGRATION.md) for the handoff contract and
clearance limitations.

Run the repository tests with:

```powershell
python -m unittest discover -s tests
```
