# One-Shot Task Compiler (OSC) — v0.2

**Research question:** can a robot watch *one* demonstration of an unseen task,
infer an executable task *program*, and complete it in changed environments
**without any task-time fine-tuning**?

> **What this repo is (read first).** This is a **symbolic / scripted
> architecture prototype and a truthful evaluation harness**, not a learned
> robot policy. The skills are hand-written closed-form controllers, the world
> model is an analytic parameter-ensemble, and perception runs in a toy tabletop
> simulator. The point of v0.2 is that **the evaluation is honest**: the agent
> acts only on an estimated belief state from noisy, nameless percepts, ground
> truth is read only by the scorer, and the metrics mean what they say. The
> numbers are deliberately *lower and harder-earned* than v0.1's inflated ones.
> See [`MIGRATION.md`](MIGRATION.md) for exactly what changed and why.

## Quickstart

```bash
pip install -e .
python -m osc.run_demo  --task stack        # one task, watch the loop
python -m osc.run_bench --seeds 20 --seed-groups 2 --out reports/v0_2
python -m osc.run_ablations --seeds 20      # component attribution
pytest -q                                    # 7 architectural + behavioural tests
```

## Headline (320 episodes, 2 seed groups, bootstrap CIs)

| metric | value |
|---|---|
| success (belief-state, ground-truth scored) | **41%** (CI95 0.36–0.47) |
| first-attempt success | 40% |
| **wrong-belief rate** (agent thinks it succeeded, ground truth says no) | **52%** |
| plan latency p50/p95/p99 | 0.2 / 0.3 / 0.4 ms |
| sensor→action p50/p95 | 0.14 / 0.19 ms |
| human interventions | **0** (no rescue API exists) |
| context est. error (mean abs) | delay **0.001**, friction 0.31, mass 1.59 |

| split | success | CI95 |
|---|---|---|
| seen_task_new_layout | 46% | 0.36–0.57 |
| unseen_instances | 33% | 0.23–0.42 |
| hidden_dynamics | 51% | 0.40–0.62 |
| disturbance_recovery | 35% | 0.25–0.46 |

The dominant failure is **perception / wrong-belief** (correspondence picks the
wrong object, or placement is imprecise under noise + distractors). That is the
honest state of the art for this prototype and the clearest next research target
— not something papered over.

## The wall between agent and ground truth

```
 backend (SimState, ground truth)         <- scorer only
        │  perceive()  (sensor model)
        ▼
 Percept: nameless, unordered detections + proprioception
        │  Corruptor: occlusion, drop, delay, false-contact, id-swap, noise
        ▼
 StateEstimator: association + coasting + grasp inference
        ▼
 BeliefState (anonymous track ids, per-object uncertainty)
        ▼
 correspondence → skills → world model → verifier → executor   (agent side)
```

`AgentEnv` exposes no `state()`; an architectural test (`tests/test_v0_2.py`)
boobytraps `backend.state()` and fails if the agent path ever reads it.

## Code → research-plan map

| stage | module | what it does |
|---|---|---|
| A task inference | `perception/`, `compiler/stage_a.py` | belief trajectory → role-based task graph (predicates + **relative transforms**), multi-grasp-episode capable |
| B skill grounding | `skills/correspondence.py`, `skills/grounding.py` | bind roles→tracks by geometry (name-free), instantiate only the needed skill experts |
| C imagined search | `worldmodel/planning_model.py` (analytic, **distinct from the sim**), `worldmodel/search.py` | score candidate plans on collision/uncertainty/force/irreversibility |
| D closed-loop exec | `execution/loop.py`, `execution/verifier.py` | act on belief, adapt `DynamicsContext` online (no weights), replan on events |
| scoring | `benchmark/scorer.py`, `metrics/metrics.py` | ground-truth success + safety, corrected metrics, bootstrap CIs |

## Ablations (component attribution)

`python -m osc.run_ablations` turns one thing off at a time. On the current
tasks the informative results are:

- **absolute vs relative transforms**: relative wins by ~25 points — the
  relative-frame task graph is what makes one-shot transfer work.
- **privileged vs belief state**: perfect state is only ~+5 points here, i.e.
  perception is costly but not the whole story.
- **world model / recovery / adaptation off**: ≈ no change on these tasks —
  an honest signal that the benchmark is **not yet hard enough** to exercise
  them. Making it harder (below) is the point of the next milestone.

## Implemented vs. future (learned) components

| implemented now (symbolic/scripted) | future (learned) |
|---|---|
| scripted skill controllers | small neural skill experts sharing an encoder |
| analytic parameter-ensemble planning model | learned latent world model (V-JEPA-2 / OSVI-WM) |
| geometry/size correspondence | learned object correspondence + open-vocab ID |
| RMA-style delay/friction estimation | learned dynamics-context encoder (payload/slip/compliance) |
| toy tabletop CPU sim | ManiSkill3 GPU / contact sim (behind the same `SimBackend`) |

## Honest limitations

- **~41% success** with **~52% silent (wrong-belief) failures** — perception and
  correspondence are the bottleneck.
- `double_stack` (a 2-object composition) **compiles** via multi-episode Stage A
  but its **execution-time multi-object tracking is not reliable**; it is
  excluded from the default benchmark and kept as a known-limitation task.
- No real physics (no liquids/deformables/contact forces); tasks are pick-place,
  side-place, and stacking. Pour/wipe/insert wait for a contact simulator.
- No learned components yet; **no baseline (ACT/DP/VLA) is wired** — deliberately.
  The next correct step is wiring ACT into *this* observation-only harness so the
  comparison is fair.

## Layout
```
src/osc/
  geometry.py            SE(2)+z relative transforms
  sim/                   backend interface, toy sim, randomization, disturbance
  perception/            nameless detections + corruptions; belief-based tracks/keyframes
  agent/                 BeliefState, StateEstimator, AgentEnv guard, DynamicsContext
  compiler/              Stage A: role-based task graph
  skills/                skill experts, correspondence (router), grounding
  worldmodel/            Stage C: analytic planning model + imagined search
  execution/             Stage D: verifier + event-driven closed loop
  benchmark/             scorer (ground truth), runner, splits, ablations, reports
  metrics/               corrected metric suite + bootstrap CIs
  tasks.py               scenes + oracles + ground-truth success predicates
```

## Roadmap
1. Attack wrong-belief: better correspondence + placement verification from belief.
2. Make the benchmark hard enough that world-model/recovery/adaptation matter.
3. Robust multi-object tracking → re-enable `double_stack` and add clear-obstruction.
4. Wire ACT / Diffusion Policy into the same observation-only harness (fair baseline).
5. Learned world model; then the ManiSkill3 GPU backend.
