# ProcTHOR → Scene Graph → PDDL → Fast Downward

This project turns a [ProcTHOR-10k](https://procthor.allenai.org/) house into a task plan in three steps:

1. **Scene graph.** Load a house in [AI2-THOR](https://ai2thor.allenai.org/) and build a scene graph from it: apartment → rooms → containers → objects. The loader also builds an occupancy grid from the simulator's reachable positions.
2. **Problem file.** Write a `problem.pddl` from the scene graph for a pick-and-place `domain.pddl`. Moves cost their real walking distance over the occupancy grid.
3. **Plan.** Solve the task with the [Fast Downward](https://www.fast-downward.org) planner, which is downloaded and built locally on first use.

Everything runs natively with the [uv](https://docs.astral.sh/uv/) package manager.

---

## 1. Requirements

| | |
|---|---|
| OS | macOS (tested on Apple Silicon) or Linux |
| Tools | [uv](https://docs.astral.sh/uv/), `git`, [`git-lfs`](https://git-lfs.com), `make`, CMake ≥ 3.16, a C++20 compiler |
| Disk | About 5 GB: Python environment 1.1 GB plus 1 GB uv cache, AI2-THOR simulator 1 GB (`~/.ai2thor`), data 750 MB plus 550 MB download caches (`~/.prior`, `~/.cache/huggingface`), Fast Downward 240 MB |
| Network | Needed on first run, for the downloads above |

Install the tools:

```bash
# macOS
xcode-select --install                # C++ compiler, git, make
brew install cmake uv git-lfs
softwareupdate --install-rosetta      # Apple Silicon only: the AI2-THOR simulator is an Intel build

# Linux (Debian/Ubuntu)
sudo apt-get install build-essential cmake git git-lfs
curl -LsSf https://astral.sh/uv/install.sh | sh
```

uv installs the right Python (3.12, pinned in `.python-version`) by itself; you don't need a system Python.

On Linux, AI2-THOR needs an X display. On a headless machine, prefix commands with `xvfb-run -a`. Only macOS has been tested.

---

## 2. Setup from scratch

Run every command from the repository root. The code finds `./data` and `./third_party` relative to the working directory.

```bash
git clone git@github.com:RAIL-group/symbolic-planning.git
cd symbolic-planning

# 1. Create .venv and install all Python dependencies (about 1 minute on first install)
uv sync

# 2. Download the ProcTHOR-10k houses and the sentence-BERT model into ./data (about 1 minute)
make download

# 3. Clone and build Fast Downward (release-24.06.1) into ./third_party (1–3 minutes)
make fast-downward
```

`make download` writes:
- `data/procthor-data/data.jsonl`: the 10,000 ProcTHOR-10k training houses, one JSON per line, fetched with [`prior`](https://github.com/allenai/prior).
- `data/sentence_transformers/`: the `bert-base-nli-stsb-mean-tokens` model, used to embed node names in the scene graph.

Step 3 is optional: the pipeline builds Fast Downward automatically the first time it needs it. To use an existing checkout instead, set `FAST_DOWNWARD_DIR=/path/to/downward`.

The first time a house is loaded, AI2-THOR downloads its simulator build (a 775 MB zip, 1 GB unpacked) to `~/.ai2thor`, so the first pipeline run takes about a minute longer.

---

## 3. Run one example

Put an apple in the fridge in house 3:

```bash
make pipeline SEED=3 GOALS="--goal apple fridge"
```

Expected output (noise trimmed):

```
Scene graph: 46 nodes, 13 containers, 28 objects
Goal: [('apple_44', 'fridge_11')]
Plan (5 steps, cost 341.0):
  move robot1 base countertop_17
  pick robot1 apple_44 countertop_17 hand1
  move robot1 countertop_17 fridge_11
  place robot1 apple_44 fridge_11 hand1
  move robot1 fridge_11 base
```

This takes about 15 seconds, most of it starting AI2-THOR and embedding node names. The cost is 10 (pick) + 10 (place) + 321 for the three moves, measured in 0.05 m grid cells.

Results are written to `data/pipeline/seed_3/`:

| File | Contents |
|---|---|
| `scene_graph.json` | Nodes (name, type, grid position), edges, container and object indices |
| `scene_graph.png` | Picture of the graph |
| `domain.pddl` | Copy of the domain used |
| `problem.pddl` | Generated problem |
| `plan.txt` | Fast Downward plan, ending with `; cost = ...` |
| `fast_downward.log` | Full planner output |

More ways to run it:

```bash
make pipeline SEED=0                                                 # one random goal
make pipeline SEED=7 GOALS="--goal apple fridge --goal mug sink"     # several goals
make pipeline SEED=7036 PLANNER=ff-wastar2                           # faster planner for big houses
uv run python -m procgraph.scripts.run_pipeline --seed 7036 --num-random-goals 3 --planner ff-wastar2
uv run python -m procgraph.scripts.run_pipeline --help
```

| Option | Make variable | Default | Meaning |
|---|---|---|---|
| `--seed N` | `SEED` | 0 | ProcTHOR-10k house index (0–9999) |
| `--goal OBJECT CONTAINER` | `GOALS` | random | Put `OBJECT` in `CONTAINER`; repeatable |
| `--num-random-goals N` | | 1 | Number of random goals if no `--goal` is given |
| `--planner P` | `PLANNER` | `ff-astar` | See [Planners](#planners) |
| `--output-dir DIR` | | `data/pipeline` | Where outputs go |

---

## 4. How it works

### Step 1: loading the house and building the scene graph

[`house.py`](./src/procgraph/house.py), `ProcTHORHouse(seed)`:

1. Read line `seed` of `data.jsonl` and load that house into an AI2-THOR `Controller`.
2. **Occupancy grid.** Ask the simulator for its reachable positions (`GetReachablePositions`, 0.05 m grid). Reachable cells are free (`0`); everything else is occupied (`1`). Each container, and the objects inside it, is moved to the nearest reachable cell; each room gets a position at the free cell nearest its centre. The robot's start cell is stored as `robot_position`.
3. **Preprocessing:**
   - drop containers named in `IGNORED_ASSETS` (lamps, paintings, TVs, …) and containers with no objects;
   - give repeated container names a number;
   - keep each object name only once across the house;
   - keep at most 7 objects per container.

The simulator is stopped once the grid is built (`house.close()` in `load_scene`). `house.path_cost(start, goal)` gives the shortest-path length between two grid cells.

[`scene_graph.py`](./src/procgraph/scene_graph.py) then builds the scene graph:
- `build_scene_graph(house)` makes nodes for the apartment, each room, each container and each object. Each node has an `id`, `name`, `type` and grid `position`. Edges run apartment–room, room–room (when a door connects them), room–container and container–object. The graph also lists the `containers` and `objects` node indices.
- `add_node_features(graph)` gives each node a feature vector (`node_features`): the 768-dimensional sentence-BERT embedding of its name plus a one-hot node type.
- `save_scene_graph(graph, dir)` writes `scene_graph.json` and a picture, `scene_graph.png`.

### Step 2: the problem file

[`pddl/problem.py`](./src/procgraph/pddl/problem.py):

- **PDDL names.** Every graph node gets a unique name made of its generic name and node index, e.g. `fridge_19` or `apple_63`.
- **Initial state:**
  - `(Robot robot1)`, two hands `hand1` and `hand2`, both empty, and the robot `(At robot1 base)`;
  - `(Object o)` and `(In o c)` for every object `o` in container `c`;
  - `(ContainerUnsafe base)`, so nothing can be put down at the start position;
  - `(Motion a b)` and `(= (MotionCost a b) d)` for every pair of locations (containers and `base`) with a path between them.
- **Move costs.** `compute_motion_costs(house, graph)` runs Dijkstra ([`grid.py`](./src/procgraph/grid.py)) over the occupancy grid once from each location. The cost `d` is the shortest-path length in 0.05 m cells, rounded to an integer because Fast Downward only accepts integer costs.
- **Goal:** `(In object container)` for each requested pair (`random_goals` picks random ones), plus the robot back `At base` with both hands empty.
- **Goal names** can be PDDL names (`apple_63`) or generic names (`apple`); a generic name picks the first match. Unknown names raise an error listing the valid ones.

The domain, [`pddl/domain.pddl`](./src/procgraph/pddl/domain.pddl), has three actions:

| Action | Precondition | Cost |
|---|---|---|
| `move ?r ?s1 ?s2` | robot at `?s1`, `(Motion ?s1 ?s2)` | `MotionCost ?s1 ?s2` |
| `pick ?r ?b ?s ?h` | robot at `?s`, `?b` in `?s`, hand `?h` empty | 10 |
| `place ?r ?b ?s ?h` | robot at `?s`, holding `?b` in `?h`, `?s` not `ContainerUnsafe` | 10 |

### Step 3: solving with Fast Downward

[`pddl/fast_downward.py`](./src/procgraph/pddl/fast_downward.py):
- `ensure_installed()` clones `aibasel/downward` at `release-24.06.1` into `./third_party/fast-downward` and runs its `build.py`, unless it's already built.
- `solve(domain, problem, planner=...)` runs `fast-downward.py` and parses the plan file into a list of action tuples plus the total cost. It returns `(None, None)` if no plan is found.

#### Planners

| `PLANNER` | Fast Downward search | When to use |
|---|---|---|
| `ff-astar` (default) | A* with the FF heuristic | Small houses; tends to find cheaper plans, but can take minutes on houses with many containers |
| `ff-wastar2` | Lazy weighted A* (w=2) with FF and preferred operators | Larger houses; plans in seconds, possibly more expensive |
| any other string | passed as `--search`, e.g. `"astar(lmcut())"` | Your own configuration. Optimal search is usually too slow at this size |

The names follow pddlstream's planner names.

---

## 5. Using it from Python

```python
from procgraph.pddl import fast_downward, problem
from procgraph.scripts.run_pipeline import load_scene

house, graph = load_scene(seed=3)                        # step 1
motion_costs = problem.compute_motion_costs(house, graph)
pddl = problem.build_problem(graph, [('apple', 'fridge')], motion_costs)
with open('problem.pddl', 'w') as f:                     # step 2
    f.write(pddl)
actions, cost = fast_downward.solve(problem.DOMAIN_PATH, 'problem.pddl',
                                    planner='ff-wastar2')  # step 3
```

`run_pipeline(seed, goals, ...)` in the same module does all three steps and writes the output files.

---

## 6. Tests and checks

```bash
make test     # pytest
make flake8   # style check
```

- `tests/test_grid.py`: shortest paths on small grids.
- `tests/test_problem.py`: problem generation and solving on a hand-made scene graph. The solve test is skipped until Fast Downward is built.
- `tests/test_pipeline.py`: the full pipeline on ProcTHOR house 0. Skipped until `make download` has been run.

---

## 7. Project layout

```
.
├── Makefile                  install / download / fast-downward / pipeline / test / flake8
├── pyproject.toml            package metadata and dependencies (Python 3.12)
├── uv.lock                   locked dependency versions
├── src/procgraph/
│   ├── config.py             data, model and Fast Downward paths
│   ├── house.py              ProcTHORHouse: load a house, build its occupancy grid
│   ├── grid.py               shortest paths over the occupancy grid
│   ├── scene_graph.py        build, featurise, draw and save the scene graph
│   ├── pddl/
│   │   ├── domain.pddl       pick/place/move domain
│   │   ├── problem.py        scene graph -> problem.pddl
│   │   └── fast_downward.py  install Fast Downward and solve
│   └── scripts/
│       ├── download_data.py  ProcTHOR-10k houses + sentence-BERT model
│       └── run_pipeline.py   the three steps for one house
├── tests/                    test_grid.py, test_problem.py, test_pipeline.py
├── data/                     (created) houses, sentence-BERT model, pipeline outputs
└── third_party/              (created) Fast Downward checkout and build
```

---

## 8. Troubleshooting

- **`FileNotFoundError: ./data/procthor-data/data.jsonl`**: run `make download`, and run commands from the repository root.
- **`make download` says `git-lfs is required`** (or Python fails with `No such file or directory: 'wget'`): the ProcTHOR-10k download uses Git LFS. Install it with `brew install git-lfs` or `sudo apt-get install git-lfs`.
- **AI2-THOR hangs or crashes on start (macOS, Apple Silicon)**: install Rosetta 2 with `softwareupdate --install-rosetta`.
- **AI2-THOR fails on Linux with a display error**: run under `xvfb-run -a`, or set `DISPLAY` to a running X server.
- **Fast Downward build fails**: check that `cmake --version` is ≥ 3.16 and that a C++20 compiler is installed. Delete `third_party/fast-downward` and rerun `make fast-downward`.
- **The planner runs for minutes**: use `PLANNER=ff-wastar2`.
- **`ValueError: No object named ...`**: the goal name isn't in this house after preprocessing. The error message lists the valid names, and `scene_graph.json` has them too.
- **The editor can't find imports**: point it at `.venv/bin/python`.

---

## Credits

The scene graph and ProcTHOR loading code come from the `procgraph` and `antplan` modules of [RAIL-core](https://github.com/RAIL-group) (Robotic Anticipatory Intelligence & Learning Group, George Mason University). Planning uses [Fast Downward](https://www.fast-downward.org) (GPL-3.0). See [LICENSE](./LICENSE).
