# Product Requirements Document (PRD)
## Drone Q-Learning Simulation
**Version:** 1.00  
**Author:** AVISHAGL  
**Course:** RL-L02 — Reinforcement Learning, Bar-Ilan University  
**Instructor:** Dr. Yoram Segal  
**Date:** 2026-04-14  

---

## Table of Contents
1. [Overview](#1-overview)
2. [Problem Statement](#2-problem-statement)
3. [Target Audience](#3-target-audience)
4. [Goals and Success Metrics](#4-goals-and-success-metrics)
5. [Functional Requirements](#5-functional-requirements)
   - 5.1 [Grid World Environment](#51-grid-world-environment)
   - 5.2 [Reinforcement Learning Engine](#52-reinforcement-learning-engine)
   - 5.3 [GUI — Grid Visualization](#53-gui--grid-visualization)
   - 5.4 [GUI — Drone Visualization](#54-gui--drone-visualization)
   - 5.5 [GUI — Dashboard Panel](#55-gui--dashboard-panel)
   - 5.6 [GUI — Convergence Graph](#56-gui--convergence-graph)
   - 5.7 [GUI — Controls](#57-gui--controls)
6. [Non-Functional Requirements](#6-non-functional-requirements)
7. [Reward Specification](#7-reward-specification)
8. [User Stories](#8-user-stories)
9. [Architecture Constraints](#9-architecture-constraints)
10. [Assumptions](#10-assumptions)
11. [Out of Scope](#11-out-of-scope)
12. [Milestones](#12-milestones)

---

## 1. Overview

This project implements a **Drone Q-Learning Simulation** — a desktop application that demonstrates tabular Q-Learning in a 2-D grid-world environment. A drone agent learns, through trial-and-error, the optimal path from a start cell to a target cell while avoiding traps and navigating wind zones and buildings.

The application couples a visual, interactive grid editor with a real-time training dashboard, allowing users to observe how a Q-Table evolves episode by episode until the policy converges to an optimal strategy.

The project is built following Dr. Segal's professional software-writing guidelines (v3.00): SDK-based architecture, OOP with no code duplication, TDD (≥85 % test coverage with pytest-cov), Ruff linter with zero errors, `uv` as the sole package manager, and all configuration stored in JSON files.

---

## 2. Problem Statement

Classic path-finding algorithms (BFS, Dijkstra, A*) require full knowledge of the environment. Reinforcement Learning allows an agent to learn an optimal policy purely through interaction with the environment — without a pre-supplied map. This project illustrates that capability by training a drone agent to:

- Explore a configurable grid world.
- Learn, via Q-Learning, which actions maximise long-term cumulative reward.
- Visualise the learned policy as directional arrows overlaid on the grid.
- Converge to a stable Q-Table that the user can save and reload.

---

## 3. Target Audience

| Audience | Use Case |
|---|---|
| RL students (BIU RL-L02) | Understand Q-Table training dynamics through visual feedback |
| Course instructor | Evaluate project against submission checklist |
| Self-learners | Experiment with hyperparameters and grid layouts |

---

## 4. Goals and Success Metrics

### 4.1 Primary Goals
- Implement a correct, fully tabular Q-Learning algorithm.
- Provide a GUI that makes every training step visible and understandable.
- Deliver production-quality code per Dr. Segal's guidelines.

### 4.2 Acceptance Criteria / KPIs

| KPI | Target |
|---|---|
| Drone reaches target in evaluation episodes | ≥ 80 % success rate after training converges |
| Q-Table convergence visible in graph | Max Δ Q-value per episode drops below 0.01 |
| Training speed | 1,000 episodes complete in < 60 seconds on standard laptop |
| Test coverage | ≥ 85 % (pytest --cov) |
| Linter errors | 0 (Ruff) |
| Lines per source file | ≤ 150 |
| Hardcoded values | 0 — all configuration loaded from JSON |

---

## 5. Functional Requirements

### 5.1 Grid World Environment

**FR-ENV-01** The grid shall be a 2-D rectangular array of cells. Default size is configurable via `config/env.json` (e.g., 10 × 10).

**FR-ENV-02** Each cell shall belong to exactly one of the following types:

| Cell Type | Display Color | Description |
|---|---|---|
| Empty | Dark (heatmap-shaded) | Passable, normal step cost; shaded by V(s) during training |
| Building | Gray | Impassable; drone cannot enter |
| Trap | Red | **Terminal**; high negative reward; episode ends immediately on entry |
| Wind Zone | Blue | Passable, moderate negative reward |
| Target | Green | Terminal success state |
| Start | Yellow | Drone spawn position |

**FR-ENV-03** The environment shall expose a standard RL interface: `reset() → state`, `step(state, action) → (next_state, reward, done)`.

**FR-ENV-04** Episode termination conditions:
- Drone reaches the Target cell (success, `done=True`).
- Drone enters a Trap cell (failure, `done=True`).
- Drone exceeds the maximum steps per episode (failure, `done=True`). Default configurable in `config/env.json`.

**FR-ENV-05** State representation: a single integer index encoding the `(row, col)` position — `state = row * num_cols + col`.

**FR-ENV-06** Action space: four discrete actions — `UP=0`, `DOWN=1`, `LEFT=2`, `RIGHT=3`.

**FR-ENV-07** Boundary behaviour: attempting to move outside the grid or into a Building cell leaves the drone in its current position and applies the building step penalty.

### 5.2 Reinforcement Learning Engine

**FR-RL-01** Implement the **Q-Learning** (off-policy TD) update rule:

$$Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s,a) \right]$$

**FR-RL-02** Hyperparameters (all configurable via `config/rl.json`, no hardcoded defaults in source):

| Parameter | Symbol | Description |
|---|---|---|
| Learning rate | α (alpha) | Step size for Q-update |
| Discount factor | γ (gamma) | Default: 0.9 |
| Exploration rate | ε (epsilon) | Fraction of steps that choose a random action |
| Epsilon decay | — | Multiplicative decay applied after each episode |
| Epsilon minimum | — | Floor for ε after decay |
| Episodes | N | Total training episodes |
| Max steps/episode | T | Cap on steps before episode is truncated |
| Visual delay | — | Per-step animation delay in ms (0 = fast mode) |

**FR-RL-03** Action selection: **ε-Greedy** — with probability ε choose a uniformly random action; otherwise choose `argmax_a Q(s, a)`.

**FR-RL-04** The Q-Table shall be a 2-D NumPy array of shape `(num_states, num_actions)`, initialised to zeros.

**FR-RL-05** The RL engine shall record per-episode statistics: total reward, steps taken, and max Δ Q-value (convergence metric).

**FR-RL-06** The engine shall support **pause / resume** without resetting the Q-Table.

**FR-RL-07** Save Q-Table: export the current Q-Table to a NumPy `.npy` file selected by the user via file dialog.

**FR-RL-08** Load Q-Table: import a previously saved `.npy` Q-Table and resume training or run an evaluation episode.

**FR-RL-09** Run a **greedy evaluation episode** (ε = 0) automatically when training is paused, visualising the optimal path without modifying the Q-Table.

**FR-RL-10** Track visited states: the engine shall record which states the agent has taken an action from, resetting on hard reset.

### 5.3 GUI — Grid Visualization

**FR-GUI-GRID-01** Render the grid as a colour-coded tile map matching the cell-type colour scheme in §5.1.

**FR-GUI-GRID-02** Overlay the learned policy as **directional arrows** only on states the agent has visited (non-terminal, non-building cells). Each arrow indicates `argmax_a Q(s, a)`.

**FR-GUI-GRID-03** Highlight the **current drone position** distinctly during a live training episode.

**FR-GUI-GRID-04** The grid shall update in real time as episodes progress. Update frequency is configurable via `vis_every_n` in `config/rl.json`.

**FR-GUI-GRID-05** The user shall be able to **edit the grid** before or during training:
- Click a cell to select its type from the palette.
- Drag to paint multiple cells of the same type.
- There must be exactly one Start cell and one Target cell; the UI enforces this constraint and reverts illegal edits.
- A walkable path from Start to Target must exist; edits that would block all paths are reverted with a warning.

**FR-GUI-GRID-06** A **reset grid** button (or equivalent) shall restore the grid to the default layout loaded from `config/env.json`.

**FR-GUI-GRID-07** **Value heatmap**: EMPTY cells shall be shaded by their normalised state-value $V(s) = \max_a Q(s,a)$, transitioning from dark-burgundy (low) through neutral to deep-purple (high). The heatmap is toggled via the `[H]` hotkey.

### 5.4 GUI — Drone Visualization

**FR-GUI-DRONE-01** Display the drone as a distinct icon at its current grid position during live training.

**FR-GUI-DRONE-02** During training (step-by-step mode) the drone shall animate one cell at a time, with per-step delay controlled by the speed slider.

**FR-GUI-DRONE-03** When training is paused, a greedy evaluation episode runs automatically and its path is shown as a trail of markers from Start to the final cell; the drone icon is placed at the last reached position.

### 5.5 GUI — Dashboard Panel

The dashboard shall be a side panel displaying the following live statistics:

**FR-GUI-DASH-01** Current episode number.

**FR-GUI-DASH-02** Steps taken in the current episode.

**FR-GUI-DASH-03** Cumulative reward for the current episode.

**FR-GUI-DASH-04** Current ε (epsilon) value.

**FR-GUI-DASH-05** Total elapsed training time.

**FR-GUI-DASH-06** Best episode reward observed so far.

**FR-GUI-DASH-07** Whether the drone reached the target in the last episode (success / fail indicator).

### 5.6 GUI — Convergence Graph

**FR-GUI-GRAPH-01** Display a live line graph with **episode number on the X-axis** and **total reward per episode on the Y-axis**.

**FR-GUI-GRAPH-02** Optionally display a second line showing **max Δ Q-value per episode** (convergence indicator), toggled by a checkbox.

**FR-GUI-GRAPH-03** The graph shall update every N episodes (configurable via `vis_every_n`).

**FR-GUI-GRAPH-04** Provide an **Export PNG** button to save the graph as an image file.

### 5.7 GUI — Controls

All training controls are accessible via keyboard hotkeys displayed in a persistent **bottom status bar**. The status bar shows the current training mode (`Idle`, `Training`, `Paused`) at all times.

| Hotkey | Action |
|---|---|
| `[SPACE]` | Start training (Idle → Training); Pause (Training → Paused); Resume (Paused → Training) |
| `[F]` | Toggle fast mode — sets visual delay to 0; restoring restores prior delay |
| `[H]` | Toggle value heatmap on/off |
| `[A]` | Toggle policy arrows on/off |
| `[E]` | Toggle grid edit mode on/off |
| `[S]` | Open file dialog and save Q-Table to `.npy` |
| `[L]` | Open file dialog and load Q-Table from `.npy` |
| `[R]` | Hard reset — clears Q-Table, resets episode counter and epsilon |

Hotkeys are suppressed when a text Entry field has keyboard focus (hyperparameter editing).

**FR-GUI-CTRL-01** The status bar shall display the current training state at all times.

**FR-GUI-CTRL-02** The **Config tab** shall provide editable fields for α, γ, ε, epsilon decay, N episodes, and max steps. An **Apply** button commits changes to the running agent.

**FR-GUI-CTRL-03** **Speed slider** — controls the visualisation delay between steps (range: 0 → 500 ms/step). Configurable default in `config/rl.json`.

**FR-GUI-CTRL-04** **Save / Load Q-Table** buttons in the Train tab provide an alternative to the `[S]`/`[L]` hotkeys.

---

## 6. Non-Functional Requirements

### 6.1 Performance
**NFR-PERF-01** 1,000 training episodes on a 10 × 10 grid shall complete in ≤ 60 seconds on a standard laptop (no GPU required).  
**NFR-PERF-02** GUI shall remain responsive during training; the training loop runs in a background daemon thread and communicates with the main thread exclusively via a `queue.Queue`.

### 6.2 Usability
**NFR-USE-01** The application shall follow Nielsen's 10 Usability Heuristics (visibility of system status, user control, error prevention, recognition over recall, etc.).  
**NFR-USE-02** All interactive controls shall have tooltips or labels explaining their effect.  
**NFR-USE-03** Error states (e.g., invalid grid configuration, file-load failure) shall produce a user-visible error message, not a crash.

### 6.3 Code Quality
**NFR-CODE-01** All source files ≤ 150 lines.  
**NFR-CODE-02** Zero Ruff linter errors (`ruff check .`).  
**NFR-CODE-03** OOP design; no code duplication; use of mixins where appropriate.  
**NFR-CODE-04** SDK layer encapsulates all business logic; GUI layer calls SDK only — no direct RL/env logic in GUI code.  
**NFR-CODE-05** No hardcoded numeric or string constants in source files; all configuration loaded from `config/*.json`.  
**NFR-CODE-06** No secrets in source; use `.env` + `.env-example` for any sensitive values.

### 6.4 Testing
**NFR-TEST-01** TDD methodology: Red → Green → Refactor.  
**NFR-TEST-02** Test coverage ≥ 85 % (`pytest --cov`).  
**NFR-TEST-03** Tests cover: Q-update correctness, ε-greedy action selection, environment step/reset, reward delivery, trap termination, boundary behaviour, Q-Table save/load round-trip, visited-state tracking.

### 6.5 Dependency Management
**NFR-DEP-01** `uv` is the sole package manager; `pip` and standalone `venv` commands are prohibited.  
**NFR-DEP-02** Project metadata and dependencies declared in `pyproject.toml`; lockfile committed as `uv.lock`.

### 6.6 Documentation
**NFR-DOC-01** `README.md` — installation, quickstart, configuration reference.  
**NFR-DOC-02** `docs/PRD.md` (this document).  
**NFR-DOC-03** `docs/PLAN.md` — implementation plan and architecture diagram.  
**NFR-DOC-04** `docs/TODO.md` — task breakdown with status tracking.  
**NFR-DOC-05** Per-algorithm PRD document (Q-Learning) describing the algorithm, parameters, and design decisions.

### 6.7 Versioning
**NFR-VER-01** Version numbering starts at `1.00` and follows Dr. Segal's versioning convention.  
**NFR-VER-02** Version is stored in `pyproject.toml` and displayed in the application title bar.

---

## 7. Reward Specification

| Cell / Situation | Reward | Terminal? |
|---|---|---|
| Normal (empty) cell | −1 | No |
| Wind Zone cell | −3 | No |
| Trap cell | −10 | **Yes** — episode ends |
| Building cell | Move blocked; drone stays; reward −1 | No |
| Target cell | +50 | **Yes** — episode ends (success) |

All reward values are configurable via `config/env.json`.

---

## 8. User Stories

| ID | As a… | I want to… | So that… |
|---|---|---|---|
| US-01 | Student | Start training with `[SPACE]` on the default grid | I can see Q-Learning in action immediately |
| US-02 | Student | Edit the grid before training | I can experiment with different obstacle layouts |
| US-03 | Student | Pause training with `[SPACE]` | I can see the optimal greedy path at the current policy state |
| US-04 | Student | Watch the drone navigate step-by-step | I can intuitively understand the exploration process |
| US-05 | Student | View the convergence graph | I can confirm the algorithm is learning and stabilising |
| US-06 | Student | Save and reload a Q-Table with `[S]`/`[L]` | I can continue a training run across sessions |
| US-07 | Instructor | Run a greedy evaluation episode on pause | I can verify the final policy without further training |
| US-08 | Self-learner | Adjust hyperparameters at runtime | I can compare different α/γ/ε settings without restarting |
| US-09 | Student | Export the convergence graph as PNG | I can include it in my assignment report |
| US-10 | Student | Toggle `[H]` heatmap and `[A]` arrows | I can inspect Q-value distribution and policy independently |

---

## 9. Architecture Constraints

| Constraint | Details |
|---|---|
| GUI framework | **tkinter** (ttk widgets; dark theme via centralized `src/gui/theme.py`) |
| Language | Python 3.11+ |
| RL approach | Tabular Q-Learning only (no neural networks, no deep RL) |
| Package manager | `uv` exclusively |
| Linter | Ruff — zero errors required |
| Layer separation | `src/core/` (env + RL logic) → `src/sdk/` (DroneSimSDK facade) → `src/gui/` (display only) |
| Config | All tunable values in `config/*.json`; loaded at startup |
| File layout | Max 150 lines per `.py` file; split modules when limit approached |
| Thread model | Training loop in daemon thread; GUI/thread communication via `queue.Queue` only |

### Project Structure

```text
project-root/
├── src/
│   ├── core/          # GridWorld, QLearningAgent, CellType, episode stats
│   ├── sdk/           # DroneSimSDK facade, TrainingLoop, GridSDK, persistence
│   └── gui/           # App, GridCanvas, ControlPanel, StatusBar, Dashboard, GraphPanel
├── tests/             # Unit & Integration (≥85 % coverage)
├── docs/              # PRD, PLAN, TODO, CLAUDE.md
├── config/            # env.json, rl.json
├── assets/            # Static assets
├── pyproject.toml     # uv, Ruff, and coverage rules
└── README.md          # User manual
```

---

## 10. Assumptions

1. The grid is small enough (≤ 20 × 20) that a tabular Q-Table fits comfortably in memory.
2. The user runs the application on a machine with Python 3.11+ and `uv` installed.
3. A path from Start to Target always exists (the grid editor warns and reverts if it does not).
4. A single drone agent occupies one cell at a time; no multi-agent scenarios.
5. Training runs in a background thread; GUI updates happen on the main thread via a queue polled every 50 ms.
6. The assignment is individual work; no collaborative / networked features are required.

---

## 11. Out of Scope

| Item | Reason |
|---|---|
| Deep Reinforcement Learning (DQN, PPO, etc.) | Course focuses on tabular methods |
| Web / browser interface | Dr. Segal's guidelines prohibit web-based GUIs |
| Multi-agent simulation | Not specified in course requirements |
| 3-D environment | Complexity exceeds course scope |
| Continuous action / state spaces | Q-Table requires discrete state-action spaces |
| Database persistence | NumPy `.npy` file I/O is sufficient |
| Deployment / packaging (Docker, PyInstaller) | Not required by submission guidelines |
| Graph zoom/pan toolbar | Removed; Export PNG button retained |

---

## 12. Milestones

| Milestone | Deliverable | Notes |
|---|---|---|
| M1 — Environment | `GridWorld` class passing all unit tests | FR-ENV-01 to FR-ENV-07 |
| M2 — RL Engine | `QLearningAgent` with save/load and visited-state tracking | FR-RL-01 to FR-RL-10 |
| M3 — SDK Integration | `DroneSimSDK` facade + `TrainingLoop` with queue communication | NFR-CODE-04 |
| M4 — GUI Skeleton | Window, grid canvas, control panel wired to SDK | FR-GUI-GRID-01 to FR-GUI-CTRL-04 |
| M5 — Dashboard & Graph | Live stats panel and convergence graph integrated | FR-GUI-DASH, FR-GUI-GRAPH |
| M6 — Grid Editor | Click/drag cell editing with constraint enforcement | FR-GUI-GRID-05, FR-GUI-GRID-06 |
| M7 — Visualisation | Heatmap, visited-state arrows, step-by-step drone animation | FR-GUI-GRID-07, FR-GUI-DRONE-01 to 03 |
| M8 — Status Bar & Hotkeys | Bottom status bar with full keyboard control | FR-GUI-CTRL-01 to 04 |
| M9 — Polish & QA | ≥ 85 % test coverage, zero Ruff errors, README complete | All NFRs |

---

*This document is the authoritative specification for the Drone Q-Learning Simulation project. All implementation decisions should be traceable to a requirement listed here.*
