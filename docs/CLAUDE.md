# CLAUDE.MD

---

## 🔹 Project

**RL-L02 Drone Q-Table Simulation**

## 🔹 Role

**Senior AI Software Architect**

---

## 1. Vision & Professional Standards {#vision}

This project is developed as a high-excellence Reinforcement Learning simulation.
It strictly adheres to the professional software engineering lifecycle (SDLC) defined by Dr. Yoram Segal , ensuring that every line of code is backed by rigorous documentation and architectural planning.

---

## 2. Technical Framework & Compliance {#technical-framework}

To meet the "Highest Level of Excellence", the project is built upon the following quality pillars:

### 2.1 International Standards {#international-standards}

* ISO/IEC 25010: Full compliance with the product quality model, focusing on functional suitability, performance efficiency, and maintainability.
* Nielsen's 10 Heuristics: The Pygame GUI is designed to ensure visibility of system status, user control, and aesthetic minimalism.

---

### 2.2 Machine-Readable Grading (AI-Optimized) {#machine-readable}

All project artifacts are structured for automated evaluation:

* README.md: Contains a parseable mapping of all ISO and RL requirements.
* Automated Reports: pytest generates both coverage.xml and coverage.json to verify the mandatory 85%+ branch coverage.
* Structured Logs: All system events are emitted in JSON format via config/logging_config.json.

---

## 3. Mandatory Workflow (5-Phase Process) {#workflow}

As per the mandatory working process, development proceeds only after document approval:

### 3.1 Requirement Engineering {#requirement-engineering}

* Generation of docs/PRD.md, docs/PLAN.md (C4 Model), and docs/PRD_q_learning.md.
* Note: The API Gatekeeper is documented as "Mocked/Out-of-Scope" for this local simulation.

---

### 3.2 Granular Task Tracking {#task-tracking}

* Creation of docs/TODO.md featuring at least 800 micro-tasks to ensure extreme granularity.

---

### 3.3 Prompt Engineering {#prompt-engineering}

* Maintenance of docs/PROMPTS_AND_COSTS.md to log AI interactions and token costs.

---

### 3.4 Circular Validation {#circular-validation}

* Cross-referencing TODO.md against the PRD.md before execution.

---

### 3.5 TDD Execution {#tdd}

* Mandatory Red-Green-Refactor cycle.
* Tests must be written before implementation.

---

## 4. Architectural & Quality Constraints {#architecture}

* Modular SDK Design: All business logic is strictly encapsulated within the SDK layer. The GUI and CLI act only as consumers.

* File Size Constraint: Strictly MAX 150 lines of code per file.

* Environment Management: Exclusively managed via uv. No pip or requirements.txt allowed.

* Version Control: Every feature uses its own branch. Commits must be meaningful (e.g., feat(domain): ...).

* Security: Zero hardcoded secrets. Use .env-example for placeholders.

* LaTeX Formatting: All mathematical equations in notebooks and docs must use LaTeX.

---

## 5. RL Algorithm Specifications {#rl}

### 5.1 Algorithm {#algorithm}

* Algorithm: Tabular Q-Learning (Off-policy TD control).

---

### 5.2 State-Action Initialization {#initialization}

* State-Action Initialization: All pairs initialized to 0.0.

---

### 5.3 Bellman Equation {#bellman}

$$Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s,a) \right]$$

---

### 5.4 Hyperparameters {#hyperparameters}

* Hyperparameters: $\alpha = 0.1$, $\gamma = 0.9$, $\epsilon = 1.0$ (decaying at $0.995$ to a floor of $0.01$).

---

### 5.5 Reward Matrix {#reward}

* Reward Matrix:

  * Normal Move: $-1$
  * Goal: $+10$
  * Trap: $-100$

---

### 5.6 Stochasticity {#stochasticity}

* Stochasticity: Wind Zones apply a probability distribution over movement outcomes.

---

## 6. Project Structure (Building Block Design) {#structure}

The project follows the "Building Block" principle, where every component validates its Input, Output, and Setup data.

```text
project-root/
├── src/drone_rl/
│   ├── sdk/          # Single entry point for ALL logic [cite: 220]
│   ├── domain/       # Core Q-learning & environment logic
│   ├── shared/       # version.py (v1.00), constants, utils [cite: 122, 129]
│   └── gui/          # Pygame renderer (Logic-free)
├── tests/            # Unit & Integration (85%+ Coverage) [cite: 131, 319]
├── docs/             # PRD, PLAN, TODO, PROMPTS_AND_COSTS [cite: 138]
├── config/           # setup.json, logging_config.json (Versioned) [cite: 143, 377]
├── notebooks/        # Parameter research using LaTeX [cite: 163, 422]
├── pyproject.toml    # uv, Ruff, and Coverage rules [cite: 151, 344]
└── README.md         # Comprehensive User Manual [cite: 66, 164]
```

---

## 7. Quality Guardrails (Anti-Patterns) {#guardrails}

* ❌ No print() statements; use JSON structured logging.
* ❌ No absolute imports; use relative imports within src/.
* ❌ No code duplication; use Mixins and Base Classes for DRY compliance.
* ❌ Zero Ruff violations allowed

---
