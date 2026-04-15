# Implementation Plan
## Drone Q-Learning Simulation
**Version:** 1.00  
**Author:** AVISHAGL  
**Course:** RL-L02 — Reinforcement Learning, Bar-Ilan University  
**Instructor:** Dr. Yoram Segal  
**Date:** 2026-04-14  
**PRD Reference:** [docs/PRD.md](PRD.md)

---

## Table of Contents
1. [Architecture Overview](#1-architecture-overview)
2. [Layer Diagram](#2-layer-diagram)
3. [Module Breakdown](#3-module-breakdown)
4. [Directory Structure](#4-directory-structure)
5. [Data Flow](#5-data-flow)
6. [Implementation Phases](#6-implementation-phases)
7. [Class Design](#7-class-design)
8. [Configuration Files](#8-configuration-files)
9. [Testing Strategy](#9-testing-strategy)
10. [Tooling Setup](#10-tooling-setup)
11. [Risk Register](#11-risk-register)

---

## 1. Architecture Overview

The project follows a strict **three-layer architecture** mandated by Dr. Segal's guidelines:

```
┌─────────────────────────────────────┐
│             GUI Layer               │  tkinter desktop app
│   (views, widgets, event handlers)  │  ← calls SDK only
├─────────────────────────────────────┤
│             SDK Layer               │  public façade API
│   (DroneSimSDK — single entry point)│  ← orchestrates core
├─────────────────────────────────────┤
│            Core Layer               │  pure business logic
│   (GridWorld  +  QLearningAgent)    │  ← no GUI imports
└─────────────────────────────────────┘
         ↑ config/*.json loaded at startup
         ↑ no hardcoded values anywhere
```

**Key rules enforced by the architecture:**
- GUI modules import **only** from `sdk/`.
- `sdk/` modules import **only** from `core/`.
- `core/` modules have **no** knowledge of GUI or SDK.
- All magic numbers live in `config/env.json` or `config/rl.json`.
- Every `.py` file stays under 150 lines; split when approaching the limit.

---

## 2. Layer Diagram

```
src/
├── core/
│   ├── cell_type.py          # CellType enum
│   ├── grid_world.py         # GridWorld environment
│   ├── q_agent.py            # QLearningAgent
│   ├── episode_stats.py      # EpisodeStats dataclass
│   └── config_loader.py      # load_env_config() / load_rl_config()
│
├── sdk/
│   ├── drone_sim_sdk.py      # DroneSimSDK façade
│   ├── training_loop.py      # background thread runner
│   └── persistence.py        # save/load Q-Table + grid
│
├── gui/
│   ├── app.py                # main window, layout
│   ├── grid_canvas.py        # tkinter Canvas — grid + arrows
│   ├── drone_sprite.py       # drone icon rendering + path trail
│   ├── dashboard.py          # stats panel widget
│   ├── graph_panel.py        # matplotlib convergence graph
│   ├── control_panel.py      # buttons + hyperparameter fields
│   └── grid_editor.py        # click/drag cell-type painting
│
└── main.py                   # entry point: uv run main.py
```

---

## 3. Module Breakdown

### 3.1 Core Layer

| Module | Class / Function | Responsibility | PRD Refs |
|---|---|---|---|
| `cell_type.py` | `CellType(Enum)` | Enum values: EMPTY, BUILDING, TRAP, WIND, TARGET, START | FR-ENV-02 |
| `grid_world.py` | `GridWorld` | 2-D grid, `reset()`, `step()`, reward lookup, boundary logic | FR-ENV-01 to FR-ENV-07 |
| `q_agent.py` | `QLearningAgent` | Q-Table (NumPy), `select_action()`, `update()`, stats recording | FR-RL-01 to FR-RL-09 |
| `episode_stats.py` | `EpisodeStats` | Dataclass: episode, steps, total_reward, max_delta_q, reached_target | FR-RL-05 |
| `config_loader.py` | `load_env_config()` `load_rl_config()` | Parse `config/*.json` into typed dicts | NFR-CODE-05 |

### 3.2 SDK Layer

| Module | Class / Function | Responsibility | PRD Refs |
|---|---|---|---|
| `drone_sim_sdk.py` | `DroneSimSDK` | Public API: `train()`, `pause()`, `resume()`, `stop()`, `reset()`, `evaluate()`, `get_policy()`, `get_q_table()`, `set_hyperparams()` | NFR-CODE-04 |
| `training_loop.py` | `TrainingLoop` | Runs episodes in a background `threading.Thread`; posts `EpisodeStats` to a `queue.Queue` | NFR-PERF-02 |
| `persistence.py` | `save_q_table()` `load_q_table()` `save_grid()` `load_grid()` | NumPy `.npy` / JSON serialisation | FR-RL-07, FR-RL-08 |

### 3.3 GUI Layer

| Module | Class / Function | Responsibility | PRD Refs |
|---|---|---|---|
| `app.py` | `App(tk.Tk)` | Root window, layout grid, SDK wiring, event loop, queue polling | All GUI |
| `grid_canvas.py` | `GridCanvas(tk.Canvas)` | Draw cells, policy arrows, drone position | FR-GUI-GRID-01 to FR-GUI-GRID-04 |
| `drone_sprite.py` | `DroneSprite` | Drone icon + path trail rendering mixin | FR-GUI-DRONE-01 to FR-GUI-DRONE-03 |
| `dashboard.py` | `Dashboard(tk.Frame)` | Live stats labels | FR-GUI-DASH-01 to FR-GUI-DASH-07 |
| `graph_panel.py` | `GraphPanel(tk.Frame)` | Embedded matplotlib figure, live update, PNG export | FR-GUI-GRAPH-01 to FR-GUI-GRAPH-05 |
| `control_panel.py` | `ControlPanel(tk.Frame)` | Buttons, speed slider, hyperparameter entries | FR-GUI-CTRL-01 to FR-GUI-CTRL-09 |
| `grid_editor.py` | `GridEditorMixin` | Click/drag paint logic, Start/Target constraint enforcement | FR-GUI-GRID-05, FR-GUI-GRID-06 |

---

## 4. Directory Structure

```
ex1/
├── config/
│   ├── env.json              # grid size, cell types, rewards, max steps
│   └── rl.json               # alpha, gamma, epsilon, decay, episodes
│
├── docs/
│   ├── PRD.md
│   ├── PLAN.md               # this file
│   └── TODO.md
│
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── cell_type.py
│   │   ├── config_loader.py
│   │   ├── episode_stats.py
│   │   ├── grid_world.py
│   │   └── q_agent.py
│   ├── sdk/
│   │   ├── __init__.py
│   │   ├── drone_sim_sdk.py
│   │   ├── persistence.py
│   │   └── training_loop.py
│   ├── gui/
│   │   ├── __init__.py
│   │   ├── app.py
│   │   ├── control_panel.py
│   │   ├── dashboard.py
│   │   ├── graph_panel.py
│   │   ├── grid_canvas.py
│   │   ├── grid_editor.py
│   │   └── drone_sprite.py
│   └── main.py
│
├── tests/
│   ├── core/
│   │   ├── test_cell_type.py
│   │   ├── test_grid_world.py
│   │   ├── test_q_agent.py
│   │   └── test_config_loader.py
│   ├── sdk/
│   │   ├── test_drone_sim_sdk.py
│   │   └── test_persistence.py
│   └── conftest.py
│
├── assets/
│   └── drone.png             # drone icon (or use Unicode fallback)
│
├── .env-example              # template for any env-level secrets
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## 5. Data Flow

### 5.1 Training Loop (background thread → GUI)

```
TrainingLoop (thread)
  │
  ├─ env.reset() → state
  │
  └─ for each step:
       action = agent.select_action(state)   # ε-greedy
       next_state, reward, done = env.step(action)
       agent.update(state, action, reward, next_state)
       state = next_state
       if done: break
  │
  └─ EpisodeStats → queue.put(stats)
       │
       └─ App.after() polls queue on main thread
            ├─ GridCanvas.refresh(policy, drone_pos, trail)
            ├─ Dashboard.update(stats)
            └─ GraphPanel.append(stats)
```

### 5.2 User Interaction → SDK → Core

```
ControlPanel button click
  └─ DroneSimSDK.start_training()
       └─ TrainingLoop.start()   (daemon thread)

Hyperparameter entry change
  └─ DroneSimSDK.set_hyperparams(alpha, gamma, epsilon, ...)
       └─ QLearningAgent.update_params(...)   (takes effect next episode)

GridEditor cell paint
  └─ App.on_cell_edit(row, col, cell_type)
       └─ DroneSimSDK.update_grid(row, col, cell_type)
            └─ GridWorld.set_cell(row, col, cell_type)
```

---

## 6. Implementation Phases

Phases correspond to PRD milestones (M1–M7). Each phase ends with passing tests before the next begins — **TDD throughout**.

### Phase 1 — Project Bootstrap
**Goal:** Skeleton compiles, runs, and lints clean.

- [ ] Initialise project with `uv init`
- [ ] Configure `pyproject.toml` (name, version=1.00, dependencies: numpy, matplotlib, pytest, pytest-cov, ruff)
- [ ] Add Ruff config to `pyproject.toml` (`line-length = 88`, all rules enabled)
- [ ] Create `config/env.json` and `config/rl.json` with defaults
- [ ] Create empty `__init__.py` files across all packages
- [ ] Verify `ruff check .` returns zero errors

**Completion gate:** `uv run ruff check .` → 0 errors; `uv run python src/main.py` → window opens.

---

### Phase 2 — Core: Grid World (M1)
**Goal:** `GridWorld` passes all unit tests.

**TDD order (Red → Green → Refactor):**

1. `test_cell_type.py` — enum values exist and map to correct reward/passable flags
2. `test_config_loader.py` — `load_env_config()` parses `config/env.json` correctly
3. `test_grid_world.py`:
   - `reset()` returns integer state for Start cell
   - `step(action)` returns correct `(next_state, reward, done)` for each cell type
   - Boundary / Building collision: drone stays, reward = −1
   - Target reached: `done=True`, reward = +50
   - Max steps exceeded: `done=True`

**Key design decisions:**
- `GridWorld` receives a config dict (injected), not file path — enables test isolation.
- Reward values come from config dict, never hardcoded.
- `step()` is a pure function given state; no mutable drone-position stored in env (state lives in the agent/loop).

**Completion gate:** `uv run pytest tests/core/ --cov=src/core` → all green, ≥ 85 % coverage.

---

### Phase 3 — Core: Q-Learning Agent (M2)
**Goal:** `QLearningAgent` passes all unit tests.

**TDD order:**

1. `test_q_agent.py`:
   - Q-Table initialised to zeros, correct shape `(num_states, 4)`
   - `update()` applies the Bellman equation correctly (numeric assertion)
   - `select_action()` with ε=1.0 → always random; ε=0.0 → always greedy
   - Epsilon decays correctly after `end_episode()`
   - Stats recorded: `EpisodeStats` fields populated
2. Save / load round-trip: `persistence.save_q_table()` + `load_q_table()` → Q-Table numerically identical

**Key design decisions:**
- `QLearningAgent` is constructed with a config dict and `(num_states, num_actions)` — no env coupling.
- `update_params()` method allows runtime hyperparameter changes (takes effect immediately on next call).
- `end_episode()` applies epsilon decay and returns an `EpisodeStats` snapshot.

**Completion gate:** `uv run pytest tests/core/ tests/sdk/test_persistence.py --cov` → all green.

---

### Phase 4 — SDK Façade (M3)
**Goal:** `DroneSimSDK` integrates core components; `TrainingLoop` runs episodes in background.

**Implementation steps:**

1. `DroneSimSDK.__init__()` — load configs, construct `GridWorld` and `QLearningAgent`
2. `DroneSimSDK.start_training()` — create `TrainingLoop`, start as daemon thread
3. `TrainingLoop.run()` — episode loop, posts `EpisodeStats` to `queue.Queue`
4. `DroneSimSDK.pause()` / `resume()` — threading.Event flag checked in loop
5. `DroneSimSDK.stop()` — stop flag; thread joins gracefully
6. `DroneSimSDK.reset()` — recreates Q-Table (zeros) and resets stats
7. `DroneSimSDK.get_policy()` → `dict[int, int]` mapping state → best action
8. `DroneSimSDK.evaluate()` → runs one greedy episode, returns path + stats

**Tests (`test_drone_sim_sdk.py`):**
- `start_training()` + short run → queue receives at least one `EpisodeStats`
- `pause()` / `resume()` → episode count stops / continues
- `reset()` → Q-Table back to zeros
- `get_policy()` → returns dict of correct shape

**Completion gate:** `uv run pytest tests/sdk/ --cov=src/sdk` → all green.

---

### Phase 5 — GUI Skeleton (M4)
**Goal:** Window opens with grid canvas and control panel; buttons call SDK.

**Implementation steps:**

1. `app.py` — `App(tk.Tk)`: create layout (grid canvas left, control + dashboard right, graph below)
2. `grid_canvas.py` — `GridCanvas(tk.Canvas)`: draw coloured rectangles for each cell type; policy arrow rendering using `canvas.create_text("↑↓←→")`
3. `control_panel.py` — buttons wired to `DroneSimSDK` methods; hyperparameter `ttk.Entry` widgets
4. `App._poll_queue()` — `self.after(50, self._poll_queue)` loop drains stats queue and calls `GridCanvas.refresh()` + `Dashboard.update()` + `GraphPanel.append()`
5. Speed slider → sets `TrainingLoop.vis_delay` (sleep between steps when visualising)

**GUI threading rule:** All `canvas.*` and widget calls happen **only** on the main thread via `after()`. The training thread **never** touches widgets directly.

**Completion gate:** `uv run python src/main.py` → window opens, Start Training runs episodes, grid refreshes.

---

### Phase 6 — Dashboard & Convergence Graph (M5)

1. `dashboard.py` — `ttk.Label` widgets for each `FR-GUI-DASH-*` stat; `update(stats: EpisodeStats)` method
2. `graph_panel.py` — embed `matplotlib.figure.Figure` in a `tk.Canvas` via `FigureCanvasTkAgg`; `append(stats)` adds data point; `export_png()` saves figure
3. Second graph line for max Δ Q-value (toggled by checkbox)

**Completion gate:** Dashboard and graph update live during training.

---

### Phase 7 — Drone Visualization & Grid Editor (M6)

1. `drone_sprite.py` (`DroneSpriteMixin`) — renders drone icon at current position; draws trail as low-opacity rectangles for cells visited in last episode
2. `grid_editor.py` (`GridEditorMixin`) — bind `<Button-1>` and `<B1-Motion>` on `GridCanvas`; enforce single Start + single Target constraint; call `DroneSimSDK.update_grid()` on each edit
3. Reset Grid button → `DroneSimSDK.load_default_grid()` → `GridCanvas.refresh()`

**Completion gate:** User can paint cells, press Start, watch drone navigate; trail visible after each episode.

---

### Phase 8 — Polish & QA (M7)

- [ ] Audit every file: ≤ 150 lines; split any exceeding the limit
- [ ] Run `uv run ruff check . --fix` then manually resolve remaining issues
- [ ] Run `uv run pytest --cov=src --cov-report=term-missing`; add tests until ≥ 85 %
- [ ] Add tooltips (`tk.ToolTip` or `ttk.ToolTip`) to all buttons and sliders
- [ ] Error handling: wrap file dialogs and Q-Table load with try/except; show `tk.messagebox.showerror`
- [ ] Version string from `pyproject.toml` displayed in `App.title()`
- [ ] Write `README.md` with install steps, quickstart, and config reference
- [ ] Write `docs/Q_LEARNING_ALGO.md` (per-algorithm PRD per NFR-DOC-05)
- [ ] Final review against PRD acceptance criteria

**Completion gate:** All acceptance criteria in PRD §4.2 satisfied.

---

## 7. Class Design

### 7.1 `GridWorld`

```python
class GridWorld:
    def __init__(self, config: dict) -> None: ...
    def reset(self) -> int: ...                          # returns start state
    def step(self, state: int, action: int
             ) -> tuple[int, float, bool]: ...           # next_state, reward, done
    def set_cell(self, row: int, col: int,
                 cell_type: CellType) -> None: ...
    def get_grid(self) -> list[list[CellType]]: ...
    def state_to_pos(self, state: int) -> tuple[int, int]: ...
    def pos_to_state(self, row: int, col: int) -> int: ...
    @property
    def num_states(self) -> int: ...
    @property
    def num_actions(self) -> int: ...                   # always 4
```

### 7.2 `QLearningAgent`

```python
class QLearningAgent:
    def __init__(self, num_states: int,
                 num_actions: int, config: dict) -> None: ...
    def select_action(self, state: int) -> int: ...     # ε-greedy
    def update(self, state: int, action: int,
               reward: float, next_state: int) -> float: ...  # returns Δq
    def end_episode(self) -> EpisodeStats: ...          # decay ε, snapshot stats
    def get_policy(self) -> dict[int, int]: ...
    def update_params(self, **kwargs) -> None: ...
    def get_q_table(self) -> np.ndarray: ...
    def set_q_table(self, q: np.ndarray) -> None: ...
    def reset(self) -> None: ...                        # zeros Q-table
```

### 7.3 `DroneSimSDK`

```python
class DroneSimSDK:
    def __init__(self) -> None: ...                     # loads configs
    def start_training(self) -> None: ...
    def pause(self) -> None: ...
    def resume(self) -> None: ...
    def stop(self) -> None: ...
    def reset(self) -> None: ...
    def evaluate(self) -> tuple[list[int], EpisodeStats]: ...
    def get_policy(self) -> dict[int, int]: ...
    def get_q_table(self) -> np.ndarray: ...
    def set_hyperparams(self, **kwargs) -> None: ...
    def update_grid(self, row: int, col: int,
                    cell_type: CellType) -> None: ...
    def get_stats_queue(self) -> queue.Queue: ...
    def save_q_table(self, path: str) -> None: ...
    def load_q_table(self, path: str) -> None: ...
```

---

## 8. Configuration Files

### `config/env.json`
```json
{
  "rows": 10,
  "cols": 10,
  "max_steps_per_episode": 200,
  "rewards": {
    "empty":    -1,
    "wind":     -3,
    "trap":     -10,
    "building": -1,
    "target":   50
  },
  "default_grid": []
}
```
`default_grid` is a 2-D array of cell-type strings (e.g., `"EMPTY"`, `"TRAP"`) representing the starting layout. If empty, a blank grid with default Start/Target positions is generated.

### `config/rl.json`
```json
{
  "alpha":           0.1,
  "gamma":           0.9,
  "epsilon":         1.0,
  "epsilon_decay":   0.995,
  "epsilon_min":     0.01,
  "episodes":        1000,
  "max_steps":       200,
  "vis_every_n":     10
}
```
`vis_every_n` — render the drone's path only every N episodes (reduces GUI overhead during fast training).

---

## 9. Testing Strategy

### 9.1 TDD Cycle per Feature
1. **Red** — write a failing test that describes the desired behaviour.
2. **Green** — write the minimum code to make the test pass.
3. **Refactor** — clean up without breaking tests.

### 9.2 Test Coverage Targets

| Module | Target | Key Scenarios |
|---|---|---|
| `core/grid_world.py` | 90 %+ | reset, step all cell types, boundary, max-steps |
| `core/q_agent.py` | 90 %+ | Bellman update (numeric), ε-greedy distribution, epsilon decay, reset |
| `core/config_loader.py` | 100 % | valid JSON, missing key, wrong type |
| `sdk/persistence.py` | 90 %+ | save/load round-trip `.npy`, save/load grid JSON |
| `sdk/drone_sim_sdk.py` | 80 %+ | start, pause, resume, stop, reset, evaluate |
| **Total** | ≥ 85 % | `pytest --cov=src --cov-report=term-missing` |

### 9.3 What We Do NOT Test
- GUI rendering (not unit-testable without display)
- `matplotlib` graph drawing internals
- `tkinter` widget layout

### 9.4 CI-Equivalent Local Check
```bash
uv run ruff check .
uv run pytest --cov=src --cov-fail-under=85
```
Both commands must pass with zero failures before any phase is considered complete.

---

## 10. Tooling Setup

### 10.1 `pyproject.toml` (key sections)

```toml
[project]
name = "drone-q-learning"
version = "1.00"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.26",
    "matplotlib>=3.8",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "ruff>=0.4",
]

[tool.ruff]
line-length = 88
select = ["E", "F", "W", "I", "N", "UP"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--cov=src --cov-report=term-missing"
```

### 10.2 Key `uv` Commands

| Purpose | Command |
|---|---|
| Install all deps | `uv sync --all-extras` |
| Run application | `uv run python src/main.py` |
| Run tests | `uv run pytest` |
| Lint | `uv run ruff check .` |
| Auto-fix lint | `uv run ruff check . --fix` |
| Add dependency | `uv add <package>` |

---

## 11. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| tkinter `after()` polling causes lag at high episode speed | Medium | Medium | Batch queue drain; skip GUI refresh for fast-mode episodes using `vis_every_n` |
| Q-Learning converges slowly on large/complex grids | Low | Low | Default grid is 10×10 with clear path; hyperparameters tunable at runtime |
| File exceeds 150-line limit during development | High | Low | Monitor line count during each phase; split proactively before hitting limit |
| matplotlib embedded in tkinter causes thread issues | Medium | Medium | All matplotlib calls on main thread only; use `FigureCanvasTkAgg.draw_idle()` |
| Test coverage falls below 85 % | Medium | Medium | Write tests before code (TDD); track `--cov-report=term-missing` continuously |
| Grid editor allows invalid state (no Start or Target) | Low | High | Constraint enforced in `GridEditorMixin`; validated by `DroneSimSDK` before training starts |

---

*This plan is the implementation blueprint for the Drone Q-Learning Simulation. Refer to [docs/PRD.md](PRD.md) for requirements and [docs/TODO.md](TODO.md) for the per-task checklist.*
