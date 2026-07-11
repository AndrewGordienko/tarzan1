# Migration: v0.1 → v0.2

v0.1 was an architecture smoke test whose reported numbers **overstated** what
had been demonstrated. v0.2 keeps the architecture but makes the evaluation
truthful. This note lists every claim/metric that changed and why.

## Why the v0.1 numbers were inflated

| v0.1 claim | problem | v0.2 fix |
|---|---|---|
| "90% unseen-env success" | agent read **privileged `backend.state()`**; lighting/camera/perception noise never touched control | agent acts only on `BeliefState` from noisy, nameless percepts; `AgentEnv` forbids `state()`; architectural test enforces it |
| "transfers across objects" | object **roles/names were supplied** (`cube_a=manipuland`) | roles inferred by function in the demo; eval objects are nameless + shuffled + have distractors; correspondence recovers them by geometry |
| world-model planning | world model was **another `ToyTabletopSim`** (sim predicting itself) | separate analytic `PlanningModel` with different contact/actuator/friction equations; never imports the sim |
| "sparse router" | it was an if-statement over scripted controllers | still symbolic — now **labelled** as symbolic skill grounding, not a neural router |
| "RMA adaptation" | only actuator delay was estimated | `DynamicsContext` estimates delay/friction/grasp-stability and **reports estimation error vs ground truth**; mass is honestly flagged as poorly observable |
| task-level search | it was 4 copies of one plan at different heights | still trajectory-level; **labelled as such**; task-level alternatives are stubbed for the next milestone |

## Metric changes

| metric | v0.1 | v0.2 |
|---|---|---|
| success | believed success on privileged state | **ground-truth** success via `Scorer`, on the belief-driven run |
| "human interventions/hour" | autonomous replans, normalized to wall-clock | **autonomous replans** (own field) and **human interventions = 0** (only nonzero if a rescue API is called); rates use simulated time |
| p95 latency | percentile over **per-episode means** | percentiles (p50/p95/p99) over **individual planning calls**; sensor→action latency reported separately |
| completion time | wall-clock of the toy sim | control **steps** and **simulated seconds** at a documented control rate |
| cost per success | arbitrary `steps + plan_calls*5` | **removed** |
| recovery rate | any episode containing a replan | `recovered / recovery_opportunities`, where an opportunity means the disturbance actually perturbed a task-relevant object (ground truth) |
| first-attempt | "zero replans" | success **and** no detected failure event **and** no replan |
| collisions | not counted | counted; safety = force + irreversible; collisions reported too |
| — | — | **new:** wrong-belief rate, failure taxonomy, context-estimation error, bootstrap CIs, per-split breakdown |

## New guarantees / tests

- `test_agent_env_forbids_privileged_state`, `test_execution_does_not_touch_ground_truth`
  — the agent path cannot read ground truth (booby-trapped `state()`).
- `test_correspondence_ignores_names_and_order_with_distractors` — role binding is
  by geometry, not name/index, and survives distractors.
- `test_no_persistent_learning_across_episodes` + `test_no_optimizer_in_deployment_path`
  — replaces the old docstring-grep "no gradient" test: the compiled program is
  unmutated by evaluation, a repeated seed reproduces exactly (no hidden
  accumulation), and no training framework is imported at eval time.
- `test_relative_beats_absolute` — the relative-transform mechanism measurably matters.

## Headline delta

- v0.1: "90% success, 20-pt claim" (privileged, name-supplied, self-predicting).
- v0.2: **~41% success (CI 0.36–0.47)** with **~52% silent wrong-belief failures**,
  belief-state and ground-truth scored. Lower, but real — and now the failures are
  measured and attributable.

## Deliberately NOT done in v0.2

Per the review: **no ACT / Diffusion Policy / VLA baseline yet.** First make the
harness truthful and the benchmark meaningful; then wire baselines into the same
observation-only harness so a comparison actually means something.
