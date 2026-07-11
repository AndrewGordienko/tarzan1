# One-Shot Task Compiler (OSC) — v0.3

**Research question:** can a robot watch *one* demonstration of an unseen task,
infer an executable task *program*, and complete it in changed environments
**without any task-time fine-tuning**?

> **What this repo is (read first).** A **symbolic / scripted architecture
> prototype with a truthful, self-diagnosing evaluation harness** — not a learned
> policy. The agent acts only on an estimated belief state from noisy, nameless
> percepts; ground truth is read only by the scorer. v0.3 adds a **paired oracle
> attribution ladder** that measures which component actually causes failure, a
> rebuilt correspondence module, active verification, and semantic retargeting.
> See [`MIGRATION.md`](MIGRATION.md) (v0.1→v0.2) and
> [`MIGRATION_V0_3.md`](MIGRATION_V0_3.md) (v0.2→v0.3).

## Quickstart

```bash
pip install -e .
python -m osc.run_attribution --seeds 25    # the error-budget ladder (headline)
python -m osc.run_bench --seeds 25 --out reports/v0_3
python -m osc.run_ablations --seeds 20      # component attribution
python -m osc.run_demo  --task stack        # watch one task
pytest -q                                    # 7 architectural + behavioural tests
```

## The result that matters: an evidence-based error budget

Perfect-component swaps (same task/split/seed) show what to fix — and refuted the
v0.2 guess that perception (tracking) was the bottleneck:

| oracle swap | success | Δ vs full |
|---|---|---|
| full (belief) | 74.5% | — |
| perfect state estimation | 82.0% | +7.5 |
| **perfect role correspondence** | 87.5% | **+13.0** |
| perfect perception + binding | 96.5% | +22.0 |
| perfect verifier | 72.5% | −2.0 |

**Correspondence, not tracking, is the lever.** Rebuilding it (greedy →
relational sticky `RoleBelief`) took full success **50.6% → 74.5%** and
wrong-belief **45% → 17.5%**, with **P(success | correct role) = 97.5%**.

## Headline (200 episodes, bootstrap CIs)

| metric | value |
|---|---|
| success (belief-state, ground-truth scored) | **74.5%** (CI95 0.69–0.80) |
| first-attempt success | 66% |
| silent false-completion (among **confident** claims) | **12.1%** (target <10%) |
| role-binding accuracy · P(success \| correct role) | 59% · **97.5%** |
| recovery (of genuine opportunities) | 78% |
| plan latency p50/p95/p99 | 0.7 / 1.7 / 3.6 ms |
| human interventions | **0** (no rescue API exists) |

| split | success | CI95 |
|---|---|---|
| seen_task_new_layout | 90% | 0.82–0.98 |
| hidden_dynamics | 90% | 0.82–0.98 |
| disturbance_recovery | 78% | 0.66–0.88 |
| unseen_instances | 40% | 0.26–0.54 |

The remaining failures are **role correspondence on identifiable scenes** and
**genuinely ambiguous scenes** (distractors that coincidentally match the demo —
labelled `ambiguous_or_unidentifiable`, not counted as bugs). `unseen_instances`
(heavy size jitter + 3 distractors) is where ambiguity concentrates.

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

`python -m osc.run_ablations` turns one thing off at a time (paired seeds):

- **retargeting**: absolute **54%** < raw-relative **72.5%** ≤ semantic **75%**.
  Semantic only edges relative because the toy sim's **settling forgives z-target
  error** — a stricter placement task is needed to show the full advantage.
- **event-driven vs every-step planning**: **75% vs 17.5%** — naive
  receding-horizon replanning thrashes; event-driven planning matters a lot.
- **recovery on/off**: +4.4 points (78% of genuine disturbance opportunities
  recovered).
- **world model / adaptation off**: ≈ no change — these tasks still don't give
  them decisions to make (the benchmark needs harder scenarios; see roadmap).
- **privileged vs belief state**: +7 points.

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
