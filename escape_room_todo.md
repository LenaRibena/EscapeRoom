# Automatic Escape Room — Fire Building with Stairs and Slides

## Project goal

Procedurally generate a multi-floor 3D escape-room building in Blender. The player starts at the **upper-right corner of the top floor** and must reach an exit (initially fixed on the ground floor) in as few steps as possible. Players traverse rooms/corridors, **descend via slides**, and **ascend via stairs**. The system must generate solvable levels, build the scene, and animate a player following a route.

**Core tools:** Python (generation, validation, pathfinding, WFC), Blender Python API (scene orchestration and animation), Geometry Nodes (procedural/instanced visual geometry). The logical graph is generated in Python and passed to Geometry Nodes; Geometry Nodes should not be responsible for complex graph-search logic.

**Initial MVP dimensions:** 3 floors × 4 × 4 grid cells; one start; one exit; a few stair/slide connections; simple capsule player; fixed isometric camera. Expand only after a complete end-to-end example works.

## Game rules and assumptions to finalize

- [ ] Define one **step** for scoring: movement between adjacent grid cells = 1; stair/slide traversal = 1 (recommended baseline). If movement time matters, use weighted edges and Dijkstra instead of BFS.
- [ ] Define directional movement: walking normally bidirectional; slides **downward only**; stairs **upward only** under the proposed game rules (or bidirectional if desired—make it explicit).
- [ ] Define start coordinate, e.g. `(floor=n-1, x=width-1, y=height-1)`, and exit coordinate, e.g. `(0, 0, 0)`.
- [ ] Decide whether crossings are allowed at different elevations (allowed if clearances are sufficient), versus within the same occupied 3D volume (disallowed).
- [ ] Define whether cycles are allowed. **Cycles are not automatically invalid**; reject only disallowed movement/connectivity patterns and unreachable exits.
- [ ] Decide whether the game is player-controlled, a shortest-path demo, or both. MVP: automatic route playback.

## Architecture and shared data

The project has **three connected layers**:

1. **Abstract level graph (Helena):** logical nodes, directed movement links, entrances/exits, and valid vertical-connection endpoints.
2. **3D WFC layout (Alexandra / WFC workstream):** maps the graph's intended connections to a non-intersecting arrangement of modular corridor, slide, and stair pieces using pre-collapsed endpoints.
3. **Blender player / scene (Dimitra):** renders modules, animates traversals, controls floor visibility, and optionally plays back the shortest route.

> Important: the graph and physical layout must agree. A graph can be reachable but impossible to embed without collisions. Perform graph validation **and** post-WFC physical validation.

### Shared coordinate and data conventions

- Grid cell: `(f, x, y)` with `f=0` as ground floor; world position maps to `(x * cell_size, y * cell_size, f * floor_height)`.
- Directions: `N, E, S, W, U, D`.
- Each WFC module has a set of **connection ports**: type (`walk`, `stair`, `slide`, or `none`), direction, and (where necessary) entry/exit polarity and height.
- Each slide/stair occupies a geometric **volume** or reserved cells; two modules cannot overlap unless explicitly compatible.
- Store deterministic `seed` in each generation output.

Suggested repository layout:

```text
escape-room/
  README.md
  docs/
    rules.md
    tile_catalog.md
    data_contract.md
  src/
    graph_generator.py
    graph_validator.py
    wfc_tiles.py
    wfc_solver.py
    global_constraints.py
    route_solver.py
    export_level.py
  blender/
    build_scene.py
    geometry_nodes_setup.py
    animate_player.py
    visibility.py
    assets/
  data/
    example_level.json
  tests/
    test_graph.py
    test_wfc.py
    test_integration.py
```

### Proposed level interchange format

Agree on an explicit JSON contract **before implementing independently**. Minimum fields:

```json
{
  "seed": 42,
  "dimensions": {"floors": 3, "width": 4, "height": 4},
  "cell_size": 2.0,
  "floor_height": 3.0,
  "start": [2, 3, 3],
  "exit": [0, 0, 0],
  "nodes": [{"id": "2,3,3", "position": [2, 3, 3]}],
  "edges": [{"from": "2,3,3", "to": "1,3,2", "kind": "slide", "cost": 1}],
  "modules": [{"cell": [2, 3, 3], "tile": "slide_entry", "rotation": 0}],
  "route": ["2,3,3", "1,3,2"],
  "route_cost": 1
}
```

This is a **schema sketch**, not a valid sample complete level. In the real exporter, `nodes`, `edges`, `modules`, and `route` must describe the *same* complete level. Separate planned graph connections from physically realized ones if helpful (e.g. `requested_edges` and `realized_edges`).

---

## Workstream A — Global-constrained WFC (Alexandra / you)

**Research goal:** Use WFC to connect fixed endpoints into a coherent 3D network, with constraints that make complete stairs/slides physically valid.

### A1. Research and tile representation

- [ ] Read up on **simple tiled WFC**, constraint propagation, entropy-based selection, backtracking/restarts, and fixed-cell pre-collapse.
- [ ] Distinguish **local** adjacency constraints (standard WFC) from **global** constraints (endpoint pairing, path connectivity, no overlap, complete segments).
- [ ] Design an initial small tile catalog: empty, straight corridor, turn, T-junction, stair entry, stair middle, stair exit, slide entry, slide middle, slide exit.
- [ ] Define tile orientations and six-face ports; ensure neighboring faces only match compatible ports.
- [ ] Specify explicit entry/exit *polarity*: no two exits/ends can mate, and no two starts/entrances can mate when direction requires start-to-end matching.
- [ ] Decide whether multi-cell stairs/slides are (a) atomic multi-cell prefabs with reserved footprint, or (b) segment tiles with continuity constraints. **For MVP, atomic prefabs are simpler**; segmented WFC is a stretch goal.

#### Starting literature

Read these in order; the first two establish the WFC/CSP baseline, and the later papers are extensions:

1. Maxim Gumin, [WaveFunctionCollapse](https://github.com/mxgmn/WaveFunctionCollapse) (2016). Reference description of the observation/propagation loop and the simple tiled model.
2. Isaac Karth and Adam M. Smith, [“WaveFunctionCollapse is constraint solving in the wild”](https://doi.org/10.1145/3102071.3110566) (2017). Useful framing of WFC as a constraint-satisfaction approach, including its limitations.
3. Arunpreet Sandhu, Zeyuan Chen, and Joshua McCoy, [“Enhancing wave function collapse with design-level constraints”](https://doi.org/10.1145/3337722.3337752) (2019). Directly relevant to adding constraints beyond local tile adjacency.
4. Bahar Bateni, Isaac Karth, and Adam Smith, [“Better Resemblance without Bigger Patterns: Making Context-sensitive Decisions in WFC”](https://doi.org/10.1145/3582437.3582441) (FDG 2023), with [code](https://github.com/iambb5445/Context-sensitive-WFC). This changes decision heuristics for better resemblance; it is not, by itself, a solution to global endpoint connectivity.

For our first version, use a **simple tiled WFC solver with global constraints around it**: propagate six-direction local compatibility, then check endpoint pairing, directed connectivity, and reserved-volume collisions on a complete layout. Retry invalid layouts with a bounded limit. Only add global constraint propagation inside the solver if these checks prove too inefficient. In this sense “global-constrained WFC” describes our extension; it is not a separate standard WFC algorithm we need to find and adopt.

### A2. Implement basic WFC

- [x] Implement tile compatibility table for N/E/S/W/U/D.
- [x] Implement possible tile sets for each cell and local constraint propagation.
- [ ] Choose next cell by minimum remaining possibilities (entropy heuristic; randomized tie breaks).
- [ ] Implement contradiction handling: backtracking or reset with a bounded retry count.
- [ ] Support seeded randomness and reproducible outputs.
- [ ] Support **pre-collapsed cells** for mandatory slide/stair endpoints provided by the graph workstream.
- [ ] Write unit tests for tile rotation, port matching, contradictions, and pre-collapsed endpoints.

### A3. Global constraints and geometric validation

- [ ] Enforce required endpoint pairs: every required connection has one legal start and one legal end.
- [ ] Reject dangling ports, incompatible connection types, and endpoint polarity errors.
- [ ] Track reserved volumes to reject intersecting stairs/slides; consider clearance above walking surfaces.
- [ ] Ensure each mandated pair is connected by a continuous, directed route; connectivity cannot be guaranteed by local port matching alone.
- [ ] Ensure realized physical traversals match allowed directed graph edges.
- [ ] Verify start-to-exit reachability **after** WFC; regenerate or backtrack if invalid.
- [ ] Export module placement and a realized movement graph as JSON.

### A4. Experiments

- [ ] Generate at least 20 fixed-seed test layouts and record solver success, contradictions/restarts, generation time, and invalid intersections.
- [ ] Compare basic local-only WFC against WFC plus global constraints (validity rate and generation time).
- [ ] Prepare a debug visualization with colored ports/connection endpoints.

**Deliverable:** Standalone Python WFC solver, tile catalog, validation tests, and JSON output importable into Blender.

---

## Workstream B — Graph generation and solvability (Helena)

### B1. Logical model

- [ ] Create a graph with grid-addressable nodes `(f,x,y)` and directed edges labeled `walk`, `stair`, or `slide`.
- [ ] Define allowed horizontal walk edges and vertical movement rules.
- [ ] Generate random but spatially plausible candidate connections and endpoint pairs.
- [ ] Avoid conflicting assignments of the same endpoint/cell and enforce bounds.
- [ ] Decide whether graph generation precedes WFC entirely or whether WFC can request graph changes after a failed embedding.

### B2. Graph validation and routing

- [ ] Check start/end existence and reachability using BFS or DFS.
- [ ] Compute shortest escape route with BFS (unit costs) or Dijkstra (weighted costs).
- [ ] Identify dead ends and disconnected components; set policy for whether they are allowed.
- [ ] Check forbidden directional edges, duplicate edges, and unintended slide/stair cycles.
- [ ] **Do not reject all cycles automatically**: loops can be valid puzzle structure.
- [ ] Produce a route annotated with mode (`walk`, `stair`, `slide`) and cost.

### B3. Graph-to-WFC interface

- [ ] Export mandatory fixed endpoints and their required pairing/type/direction.
- [ ] Export forbidden cells/edges and any global constraints the WFC solver must obey.
- [ ] Accept WFC's realized graph and re-check solvability and shortest path.
- [ ] Provide a small deterministic fixture graph for integration testing.

### B4. Difficulty extension

- [ ] Define **difficulty** separately from geometric map size. Shortest-path length alone measures route length, not necessarily player difficulty.
- [ ] Start with target shortest-path steps (e.g. easy <= 8; medium 9–15; hard >= 16, to be calibrated for selected dimensions).
- [ ] Add optional measures: branching factor, misleading dead ends, number of floor changes, ratio between average random route and optimal route.
- [ ] Use generate–evaluate–reject or constraint-guided sampling to hit target difficulty bands.

**Deliverable:** Python graph generator + validator + shortest-path solver and JSON contract for fixed endpoints.

---

## Workstream C — Blender scene, player and visibility (Dimitra)

### C1. Procedural scene rendering

- [ ] Build a Blender importer for the common level JSON.
- [ ] Create a Geometry Nodes setup for floor plates/grid, glass walls, and instanced slide/stair modules.
- [ ] Organize geometry by floor and module type using collections or attributes.
- [ ] Add clearly distinguishable start marker, ground-floor exit, and connection endpoints.
- [ ] Use consistent cell size/floor height from JSON.
- [ ] Choose an isometric or orthographic camera with optional orbit control.

### C2. Player movement and animation

- [ ] Create a visible player placeholder (sphere/capsule) with a known location/origin.
- [ ] Read ordered route waypoints and edge types from the level JSON.
- [ ] Animate walking between adjacent cells.
- [ ] Animate slide traversal from slide entrance to exit (curve-following or keyframed path).
- [ ] Animate stair ascent along a stair path, preserving player orientation if possible.
- [ ] Define a movement timeline (frames per walk edge / slide / stair) and ensure no teleportation.
- [ ] Provide automatic shortest-route playback; optional controls to play/pause or step through route.

### C3. Occlusion and visibility (required)

- [ ] **Hide all floors above the player's current floor**, including floor slabs/walls; retain slides and stairs as requested.
- [ ] Group objects by floor so visibility changes are reliable and inexpensive.
- [ ] On a transition, update floor visibility at a defined animation frame; avoid exposing/hiding the player halfway through a stair/slide.
- [ ] For same-floor obstructions, identify camera-to-player occluders and reduce their opacity or hide them (stretch goal: ray casting; MVP: mark front-facing wall groups as transparent).
- [ ] Keep the player conspicuous using a contrasting material, outline, or marker.
- [ ] Test when player is on top/middle/ground floor, walking, climbing, and sliding.

### C4. Presentation

- [ ] Configure simple lighting and transparent/glass materials that remain readable in viewport/render.
- [ ] Show current floor, step count, and goal (text overlay or Blender text objects).
- [ ] Create a short rendered animation or realtime Blender demonstration.

**Deliverable:** Blender `.blend` template and Python scripts that render and animate any valid level JSON.

---

## Integration tasks — everyone

- [ ] **First meeting:** Agree on tile dimensions, graph/node conventions, movement direction, exact meaning of one step, and JSON fields.
- [ ] Create one hand-authored level with 3 floors, 4×4 cells, at least one slide and one stair, and a known correct route.
- [ ] Import the fixture into Blender **before** any generator is complete.
- [ ] Replace hand-authored graph with Helena's generated graph.
- [ ] Replace hand-authored module placement with WFC output.
- [ ] Rebuild the *realized* movement graph from placed modules and compare it to the intended graph.
- [ ] Run pathfinding on the realized layout and animate the computed path.
- [ ] Confirm no volume collisions, no dangling connectors, reachable exit, and correct floor hiding.
- [ ] Log failures and save failing seeds for debugging.

## Definition of done (MVP)

- [ ] One command/Blender script generates a seeded 3-floor level.
- [ ] Generated level passes logical and physical validity checks.
- [ ] Every slide/stair has correct connected endpoints and no forbidden intersections.
- [ ] Start can reach the exit through valid movement.
- [ ] Solver reports the minimum steps and actual route.
- [ ] Blender builds the geometry and animates player through the route.
- [ ] Floors above the current floor are hidden; stairs/slides stay visible; player is readable.
- [ ] At least 10 different seeds generate valid playable levels (allowing bounded solver retries).

## Optional features (only after MVP)

### Difficulty presets

- [ ] Introduce desired shortest-route distance and/or number of floor transitions.
- [ ] Generate candidate levels until one matches a chosen target range.
- [ ] Record shortest-path cost, graph branching, dead ends, generation time, and route alternatives.
- [ ] Evaluate whether the metric matches perceived difficulty in a small user test.

### Spreading fire and time pressure

- [ ] Choose fire source cell(s), ignition time, and propagation rule on the physical cell adjacency graph.
- [ ] Compute `fire_arrival_time[cell]` using a multi-source BFS for uniform spread or Dijkstra for weighted spread.
- [ ] Add timed transitions to player simulation. A route is safe only if the player reaches each traversed cell/connection before it becomes hazardous.
- [ ] Use a **time-expanded graph** or other temporal pathfinding when fire moves during traversal; ordinary BFS shortest steps is insufficient for safety.
- [ ] Show spreading fire and countdown in Blender; verify a safe escape is achievable.

### Visual enhancements

- [ ] Smooth camera orbit; automatic camera tracking.
- [ ] Ray-cast-based fading of walls between camera and player.
- [ ] Highlight optimal route, alternative routes, and invalid WFC connection attempts.
- [ ] Add polished slide curves, stair rails, materials, lighting, and sound.

## Suggested milestone sequence

| Milestone | Alexandra — WFC | Helena — Graph | Dimitra — Blender |
|---|---|---|---|
| **M1: Interface + baseline** | Tile/port catalog, basic WFC prototype | Graph schema, hand-authored valid fixture, BFS | Import fixture, scene blocks, player placeholder |
| **M2: Independent modules** | Pre-collapse, propagation, collision checks | Random graph, reachability, endpoints | Geometry Nodes modules, path animation, floor visibility |
| **M3: Integration** | Connect endpoint pairs and export realized layout | Validate realized graph, shortest route | Animate real generated levels, camera/occlusion tests |
| **M4: Evaluation + polish** | Validity and generation-time benchmarks | Difficulty metrics and route statistics | Demo video, UI, optional visual upgrades |

## Key risks and mitigation

| Risk | Mitigation |
|---|---|
| Graph is solvable but impossible to embed | Restrict endpoint placement; validate after WFC; retry bounded times |
| Standard WFC satisfies local ports but produces disconnected paths | Add explicit graph/connectivity checks and global constraints |
| Slides or stairs intersect other geometry | Reserve occupied 3D volumes and test clearance |
| Too many tile types make WFC expensive | Start with simple prefabs, small grids, and a constrained tile catalog |
| Player hidden by upper floors, walls, or slides | Per-floor visibility, occluder transparency, simple player marker |
| Team modules don't integrate | Shared JSON schema and hand-authored fixture established in M1 |

## Next meeting: concrete decisions

1. Are stairs **up-only** and slides **down-only**, as described, or can stairs be descended?
2. Is every graph node a **grid cell** or a **room** spanning multiple cells?
3. Should WFC generate **walkable corridors as well as stair/slide shapes**, or only the latter?
4. Are stairs/slides made of **complete prefabs** (recommended for MVP) or WFC-chained segment tiles?
5. What is the exact target for the first playable demonstration (floor count, dimensions, number of connections)?

**Immediate next action:** agree on the JSON contract and build the single hand-authored test level. This lets all three team members work in parallel from the same reference.
