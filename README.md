# Drone Q-Learning Simulation

**Version:** 1.00  
**Author:** AVISHAGL  
**Course:** RL-L02 — Reinforcement Learning, Bar-Ilan University  

A tabular Q-Learning simulation where a drone learns to navigate a 2-D grid world from a start cell to a target cell while avoiding buildings and hazards.

---

## Features

- **Tabular Q-Learning** with ε-greedy exploration and Bellman updates
- **Interactive GUI** built with tkinter: live grid editor, drone animation, stats dashboard, reward/ΔQ graph
- **Config-driven**: all environment and RL hyperparameters in `config/env.json` and `config/rl.json`
- **Persistence**: save/load Q-Tables (`.npy`) and grid layouts (`.json`)
- **BFS reachability guard**: grid editor prevents unsolvable configurations
- **≥85% test coverage** (currently 96%) across 263 tests

---

## Project Structure

```
ex1/
├── config/
│   ├── env.json          # Grid dimensions, cell rewards, default layout
│   └── rl.json           # Alpha, gamma, epsilon, decay, episodes, vis params
├── src/
│   ├── core/             # Pure RL — no GUI/IO dependencies
│   │   ├── cell_type.py      # CellType enum (EMPTY, BUILDING, TRAP, WIND, TARGET, START)
│   │   ├── config_loader.py  # load_env_config(), load_rl_config()
│   │   ├── episode_stats.py  # EpisodeStats dataclass
│   │   ├── grid_world.py     # GridWorld environment (reset/step interface)
│   │   ├── q_agent.py        # QLearningAgent (select_action, update, end_episode)
│   │   └── version.py        # __version__ = "1.00"
│   ├── sdk/              # Façade layer — bridges core and GUI
│   │   ├── drone_sim_sdk.py  # DroneSimSDK: single entry point for GUI
│   │   ├── persistence.py    # save/load Q-Table and grid files
│   │   └── training_loop.py  # Daemon thread training loop with pause/resume/stop
│   ├── gui/              # tkinter GUI — calls SDK only, never core directly
│   │   ├── app.py            # App(tk.Tk): root window, queue polling, layout
│   │   ├── grid_canvas.py    # GridCanvas: grid drawing + drone animation
│   │   ├── grid_editor.py    # GridEditorMixin: click-to-paint + BFS check
│   │   ├── drone_sprite.py   # DroneSpriteMixin: drone and trail rendering
│   │   ├── control_panel.py  # ControlPanel: buttons, hyperparameter inputs, speed slider
│   │   ├── dashboard.py      # Dashboard: live episode/reward/epsilon stats
│   │   ├── graph_panel.py    # GraphPanel: matplotlib reward & ΔQ plots
│   │   └── tooltip.py        # add_tooltip() hover helper
│   └── main.py           # Entry point: python -m src.main
└── tests/
    ├── core/             # Unit tests for all core modules
    └── sdk/              # Unit tests for SDK modules
```

---

## Installation

Requires Python 3.11+ and [uv](https://github.com/astral-sh/uv).

```bash
# Create virtual environment and install dependencies
uv sync --dev

# Or with pip
pip install -e ".[dev]"
```

---

## Running the Simulation

```bash
# Activate the virtual environment first
source .venv/bin/activate

# Launch the GUI
python -m src.main
```

### GUI Controls

| Button | Action |
|--------|--------|
| Start Training | Begin Q-Learning episodes |
| Pause / Resume | Pause or resume training |
| Stop | Stop training (Q-Table preserved) |
| Reset | Clear Q-Table and restart |
| Evaluate | Run one greedy episode (ε=0), highlight path |
| Save Q-Table | Export Q-Table to `.npy` file |
| Load Q-Table | Import Q-Table from `.npy` file |
| Reset Grid | Restore default grid layout |

### Grid Editor

Click or drag on the grid to paint cells. Hold the mouse button and drag to paint continuously. A BFS check runs after each edit — if the target becomes unreachable, the change is automatically reverted.

**Cell types:**

| Color | Type | Effect |
|-------|------|--------|
| White | EMPTY | Reward: −1 |
| Yellow | START | Drone spawn point |
| Green | TARGET | Reward: +50, episode ends |
| Gray | BUILDING | Impassable wall |
| Red | TRAP | Reward: −10 |
| Blue | WIND | Reward: −3 |

### Hyperparameters

Edit values in the Hyperparameters panel and click **Apply** to update at runtime:

| Parameter | Default | Description |
|-----------|---------|-------------|
| α (alpha) | 0.1 | Learning rate |
| γ (gamma) | 0.9 | Discount factor |
| ε (epsilon) | 1.0 | Exploration rate (initial) |
| ε decay | 0.995 | Multiplier applied each episode |
| ε min | 0.01 | Minimum exploration rate |

### Speed Slider

Controls animation delay (0–500 ms per episode visualization). Set to 0 for maximum training speed.

---

## Configuration

### `config/env.json`

```json
{
  "version": "1.00",
  "rows": 10,
  "cols": 10,
  "max_steps_per_episode": 200,
  "rewards": {
    "empty": -1,
    "wind": -3,
    "trap": -10,
    "building": -1,
    "target": 50
  },
  "default_grid": []
}
```

Set `"default_grid"` to a 2-D array of CellType names to load a custom starting layout.

### `config/rl.json`

```json
{
  "version": "1.00",
  "alpha": 0.1,
  "gamma": 0.9,
  "epsilon": 1.0,
  "epsilon_decay": 0.995,
  "epsilon_min": 0.01,
  "episodes": 1000,
  "max_steps": 200,
  "vis_every_n": 10,
  "vis_delay_ms": 0
}
```

`vis_every_n`: refresh the canvas every N episodes (reduce for smoother animation, increase for faster training).

---

## Running Tests

```bash
# Run all tests with coverage report
.venv/bin/pytest tests/ -q

# Run a specific module
.venv/bin/pytest tests/core/test_q_agent.py -v
```

Coverage must remain ≥85%. Current: **96%**.

---

## Linting

```bash
.venv/bin/ruff check src/
```

Zero errors enforced. Rules: `E, F, W, I, N, UP, B, C4, SIM`.

---

## Architecture

```
GUI Layer  →  SDK Layer  →  Core Layer
(tkinter)     (DroneSimSDK)  (GridWorld, QLearningAgent)
```

- **Core** has zero GUI/IO imports — pure Python + NumPy
- **SDK** owns threading (TrainingLoop daemon thread) and persistence
- **GUI** calls SDK only, never core directly
- Stats flow: `TrainingLoop → queue.Queue → App._poll_queue() → Dashboard/GraphPanel`

---

## Q-Learning Algorithm

The Bellman update applied after each step:

```
Q(s, a) ← Q(s, a) + α · [r + γ · max_{a'} Q(s', a') − Q(s, a)]
```

Action selection uses ε-greedy: with probability ε choose a random action, otherwise choose `argmax Q(s, ·)`. ε decays multiplicatively each episode until it reaches `epsilon_min`.

---

## License

Academic project — Bar-Ilan University, 2026.
