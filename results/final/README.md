# Accepted hybrid WFC result

This package is the authoritative seed-10 relative-budget, geometry-aware
hybrid building. `building/building_seed10_hybrid.json` contains the unchanged
accepted schema-version-1 graph and centerlines, with SHA-256
`dac79b23be631eb5611cfdf295107f8c2658cd77a523c11166743b804948ff0c`.
The building has ten slides and nine stairs. The
[Inherited Graph vs Hybrid WFC figure](building/building_seed10_hybrid.png)
shows the inherited directed graph and the accepted trajectories using the
same axes and camera. It is rendered directly from the accepted JSON.

The [decision trace](algorithm/wfc_decisions.txt) records the same accepted
execution for slide `(0, 4, 3) -> (0, 0, 0)`: 21 choices, 18 non-forced,
with no backtracking. The
[explained decision trace](algorithm/wfc_decisions_explained.txt) gives the
full-precision costs, normalization and reconstructed random draw for each
choice. The [slide animation](algorithm/slide_generation.gif) shows each
recorded choice in top and elevation views, with the committed prefix in teal,
the sampled step in orange, the complete route as a grey reference, and the
eligible tile probabilities. The
[last-choice frame](algorithm/slide_generation.png) is available separately.
These are grid-cell views, not the smoothed Blender centerline or a visualization
of all WFC domains; intermediate forced cells appear in the next recorded
prefix. The traces were generated from a replay whose
building and original text matched the accepted artifacts; no solver
instrumentation or changes to the solver were needed.

## Settings and findings

Graph and routing seeds are 10, resolution is 3, compactness strength is
0.12, geometry preference is 1.5, geometry backtracking is enabled, and
there are at most eight alternative-choice backtracks and three route
attempts. For each slide the cell budget is
`lower_bound_cells + min(4, ceil(0.25 * (lower_bound_cells - 1)))`.
The directed port-compatible walk permits revisits and relaxes simultaneous
assignments; it is a lower bound, not a proven feasible shortest route.
Minimum centerline clearance and the shared-endpoint exclusion radius are
both 0.25 graph units; the minimum curvature radius is 0.02. Recorded
diagnostics found no logical errors or warnings and a minimum inter-route
centerline distance of 0.5679793828224292. This does not guarantee Blender
mesh clearance.

Early unconstrained hybrid full-building trials timed out; among six
isolated absolute-budget trials only two passed geometry validation.
Adding a soft geometry preference led all six matched isolated cases to
pass, but the first full-building 29-cell absolute-budget run had four
clearance advisories and long detours on short slides. In the later
seed-10 relative-budget run, total slide cells fell from 246 to 134 and
curve length from 69.756 to 32.922 graph units compared with that run.
Seeds 10, 11 and 12 completed without logical errors or clearance
advisories; seeds 11 and 12 still had sharp-turn advisories. These are
small exploratory comparisons, not general reliability or timing claims.

## Reproduction

Use fresh paths outside this package. The hybrid runner explicitly selects the
length-bounded hybrid, generates a recorded execution, and embeds the
Blender-compatible accepted building as the `building` object. The first
visualization command uses the frozen JSON directly. The animation command
reconstructs the saved trace without rerunning the solver; the remaining
commands produce and verify a newly recorded execution:

```powershell
$env:PYTHONHASHSEED = "0"
$env:MPLBACKEND = "Agg"
python -m examples.hybrid_building_visualize results\final\building\building_seed10_hybrid.json --output results\hybrid_figure_reproduced.png
python -m examples.hybrid_slide_animation --output results\slide_animation_reproduced
python -m examples.hybrid_experiment --strategy hybrid --weighting compact --relative-budget --minimum-clearance 0.25 --routing-seed 10 --max-seconds 60 --max-backtracks 8 --route-attempts 3 --geometry-preference 1.5 --geometry-backtracking --trace-states --trace-route-id '0:(0, 4, 3)->(0, 0, 0):slide' --output results\hybrid_reproduced.json --building-output results\hybrid_building_reproduced.json
python -m examples.hybrid_length_trace results\hybrid_reproduced.json --reference-building results\final\building\building_seed10_hybrid.json --route-id '0:(0, 4, 3)->(0, 0, 0):slide' --output results\hybrid_reproduced_slide
python -m examples.explain_hybrid_decisions results\hybrid_reproduced.json --events results\hybrid_reproduced.events.jsonl --replay-visualization results\hybrid_reproduced_slide --output results\hybrid_decisions_explained_reproduced.txt
Get-FileHash results\hybrid_building_reproduced.json -Algorithm SHA256
```

Reproduce the DFS baseline separately with
`python -m examples.export_building_json results\dfs_reproduced.json --seed 10 --grid-resolution 3`.
Its expected SHA-256 is `67c4a5161ececea97e5b0214ab7df9a71ff07e640d4694390015561878b9f345`.
The original untraced run used a 10-second per-solve limit; snapshot
recording required a 60-second limit. Timing can affect retries. The
`--reference-building` check prevents presenting a trace of different
geometry as the accepted execution. The explanation tool additionally requires
byte-identical regenerated decision text, verifies the route seed and
reconstructs every candidate's weight, probability and sampled choice.
Reproduction may not match the saved JSON or presentation bytes if
timing changes retries; the saved JSON remains authoritative.
