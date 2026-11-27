# Resolution-2 hybrid experiment

This is a separate seed-10 result, not a replacement for the authoritative
resolution-3 output. The inherited directed graph matches the resolution-3
JSON; accepted centerlines have been exported without resmoothing.

Horizontal resolution is 2. As in the original configuration, vertical
resolution defaults to at least 3, giving a `(10, 19, 19)` grid in
`(vertical, x, y)` order. Graph and routing seeds are 10; compact weighting
strength is 0.12, geometry preference 1.5, relative allowance 0.25 capped
at 4 cells, minimum centerline clearance 0.25, geometry backtracking enabled,
at most 8 alternative-choice backtracks per solve and 3 route attempts.
The untraced per-solve limit is 10 seconds; the traced limit is 60 seconds,
as in the resolution-3 experiment.

The untraced building completed in 36.54 seconds with 11 slide attempts.
Slide 2's first attempt exhausted its backtrack budget (an `UNKNOWN`
result); its second attempt succeeded. There were no building-level
recoveries. Ten slides and nine stairs were accepted; diagnostics reported
no logical errors or warnings and minimum pair centerline clearance
0.44611033447206316 graph units. This is not a mesh-clearance guarantee.
The traced full-building execution took 43.78 seconds and exported a
byte-identical building JSON. Slide 0's accepted execution used its first
attempt, has 18 cells (5.0 graph units of continuous length), 13 choices,
all non-forced, and no slide-0 backtracks. The GIF and text use those same
recorded states; the detailed explanation recalculates and verifies each
weight and independent random draw.

The building SHA-256 is
`363e6c46666e3b411148cd3df7687dfb2078d67f2c4280c69b0e39b9a9092830`.
Timing measurements are exploratory and not controlled benchmarks; retries
may differ on another machine.

## Reproduction

Use fresh paths. A traced run is needed for the decision visualizations:

```powershell
$env:PYTHONHASHSEED = "0"
$env:MPLBACKEND = "Agg"
python -m examples.hybrid_experiment --strategy hybrid --weighting compact --grid-resolution 2 --relative-budget --minimum-clearance 0.25 --routing-seed 10 --max-seconds 60 --max-backtracks 8 --route-attempts 3 --geometry-preference 1.5 --geometry-backtracking --trace-states --trace-route-id '0:(0, 4, 3)->(0, 0, 0):slide' --output results\res2_reproduced.json --building-output results\res2_building_reproduced.json
Get-FileHash results\res2_building_reproduced.json -Algorithm SHA256
python -m examples.hybrid_building_visualize results\res2_building_reproduced.json --output results\res2_figure_reproduced.png
python -m examples.hybrid_length_visualize results\res2_reproduced.json --reference-building results\final\res2\building_seed10_hybrid_res2.json --route-id '0:(0, 4, 3)->(0, 0, 0):slide' --output results\res2_decisions_reproduced
python -m examples.explain_hybrid_decisions results\res2_reproduced.json --events results\res2_reproduced.events.jsonl --replay-visualization results\res2_decisions_reproduced --new-building results\final\res2\building_seed10_hybrid_res2.json --building-sha256 363e6c46666e3b411148cd3df7687dfb2078d67f2c4280c69b0e39b9a9092830 --output results\res2_explained_reproduced.txt
```

The visualizer checks the recorded building against the saved JSON and
rejects slide traces with backtracking. The explanation tool verifies each
recorded candidate weight, normalization and reconstructed sampled choice,
and independently rerenders the GIF and text to check they match the trace.
