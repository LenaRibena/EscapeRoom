# EscapeRoom

Exam project for the course 02563 Generative Methods for Computer Graphics.

## Final 3D WFC building generation

The current WFC implementation generates graph-constrained stair and slide
routes on a refined 3D grid. It starts with the tile and port rules in
`src/wfc_tiles.py`, solves tile domains with `src/wfc_solver.py` and
`src/wfc_global.py`, and applies the solver to building routes in
`src/wfc_building.py`. The graph-to-WFC adapter experiment is not part of this
workflow.

Run the building visualization from the repository root:

```powershell
python -m examples.wfc_building_example --diagnostics --seed 10 --grid-resolution 3
```

To save the visualization, add `--save-plot results\building.png`. To export
the accepted route geometry as JSON for the Blender handoff:

```powershell
python -m examples.export_building_json results\building.json --seed 10 --grid-resolution 3
```

See [BLENDER_INTEGRATION.md](BLENDER_INTEGRATION.md) for generation settings,
the JSON schema, coordinate conventions, and Blender integration details.

Run the tests from the repository root:

```powershell
python -m unittest discover -s tests
```
