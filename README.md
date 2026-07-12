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

> ### Gate status: **selective safety gate passed; ambiguity-resolution gate open.**
> The system now usually *refuses to confidently claim success* when it cannot
> identify the roles — silent false-completion is **6.7%**. But it does so by
> abstaining: **46.9% of completions are flagged uncertain**, and **~half the
> benchmark is visually unidentifiable from the provided evidence** (distractors
> overlap the demonstrated role sizes; appearance is independently randomized).
> **≈1/3 of the recent gain is better binding; ≈2/3 is honest ambiguity
> attribution.** 6.7% does **not** mean the correspondence problem is solved — it
> means the system is safely selective. The next objective is to *preserve* the
> low silent-error rate while converting ambiguous episodes into correctly
> resolved, autonomous executions (see Roadmap → resolution layer).

## Quickstart

```bash
pip install -e .
python -m osc.run_attribution --seeds 40    # the error-budget ladder (headline)
python -m osc.run_bench --seeds 40 --out reports/v0_3
python -m osc.run_ablations --seeds 20      # component attribution
python -m osc.run_demo  --task stack        # watch one task
pytest -q                                    # 18 architectural + behavioural tests
```

### Packing proof-of-concept

The `v0.5-packing-poc` branch demonstrates the broader one-shot task-acquisition
thesis with a deterministic geometric packing domain. A canonical demonstration
compiles into hard constraints, soft preferences, and a reusable packing program;
the planner then searches finite orientations/extreme-point placements and can
remove/repack an earlier item when a late arrival makes the current arrangement
infeasible.

```bash
osc-pack-demo --render artifacts/packing_demo.gif
osc-pack-bench --episodes 100 --perception oracle
osc-pack-bench --episodes 100 --perception belief
```

The controlled PoC reports lanes separately: 100% oracle feasible completion and
88% belief feasible completion in the regenerated 100-episode report (the earlier
91% belief run implies 93.25% overall correct decision on a 75% feasible / 25%
impossible mix). Both lanes have zero constraint violations and beat literal
replay and greedy next-fit; no gradient updates occur after the demonstration. The
packing implementation lives under `src/osc/packing/` and is intentionally a
finite geometric world model before learned residual dynamics are introduced.

The scientific packing artifact also runs a causal demo-dependence check: the same
inventory is evaluated after heavy-bottom, maximize-volume, minimize-rehandling,
shuffled, no-demo, conflicting, and oracle-program conditions. It records the
program posterior, constraint posterior, arrangement, and policy-behavior match.
An explicit `unknown_or_unexplained` hypothesis abstains on out-of-vocabulary
demonstrations. The matched late-item intervention forces both oracle and belief
lanes through the same initial placement prefix before revealing the large item.

Results are **cross-process deterministic** (identical JSON across
`PYTHONHASHSEED`, apart from wall-clock latency fields); CI enforces it.

### v0.6 embodied packing (MuJoCo adapter)

Intent, planning, and recovery remain in Tarzan while continuous arm control is
delegated to the separate [TinyVLA repository](https://github.com/AndrewGordienko/tinyvla).
The camera/contact-safe `SkillCommand` boundary is in `src/osc/embodied/`.

```bash
osc-pack-mujoco-demo --demo-policy heavy_bottom_fragile_top --controller scripted
osc-pack-mujoco-bench --episodes 20 --perception segdepth --controller scripted
```

Without MuJoCo/TinyVLA installed these commands return a structured `blocked`
report and never silently substitute simulator ground truth.

The physical rearrangement harness is separate:

```bash
osc-pack-mujoco-rearrange --out artifacts/embodied_rearrangement_dev_100_109.json
```

It runs the forced-removal execution and motor-isolation lanes on development
seeds 100--109, with no-removal/wrong-object controls and scorer-only checks.
Autonomous multi-object removal is intentionally reported as unimplemented until
the planner is connected; forced execution is not counted as planner success.

### Customer localhost demo

Run `./scripts/run_customer_demo.sh` and open `http://localhost:8787`. The
application calls the real packing compiler and executor for logical runs,
renders the recorded MuJoCo smoke trajectory, and labels forced and
unimplemented rearrangement evidence explicitly.

## The result that matters: an evidence-based error budget

Perfect-component swaps (same task/split/seed) show what to fix — and refuted the
v0.2 guess that perception (tracking) was the bottleneck:

| oracle swap | success | Δ vs full |
|---|---|---|
| full (belief) | 75.9% | — |
| perfect state estimation | 85.9% | +10.0 |
| **perfect role correspondence** | 83.8% | **+7.8** |
| perfect perception + binding | 97.8% | +21.9 |
| perfect verifier | 74.7% | −1.3 |

**Perception + correspondence together are the whole budget** (perfect binding on
correctly-perceived scenes → 97.8%). Rebuilding correspondence (greedy →
relational sticky `RoleBelief`, appearance-independent) with ground-truth
ambiguity attribution drove silent false-completion **18.6% → 6.7%** at
**P(success | correct role) = 97.9%**.

## Headline (320 episodes, bootstrap CIs, `PYTHONHASHSEED=0`)

| metric | value |
|---|---|
| success (belief-state, ground-truth scored) | **75.9%** (CI95 0.71–0.81) |
| first-attempt success | 67.8% |
| **silent false-completion** (fail among **confident, identifiable** claims) | **6.7%** |
| uncertain-completion (honestly flagged low-confidence/ambiguous) | 46.9% |
| role-binding accuracy · P(success \| correct role) | 58.4% · **97.9%** |
| recovery (of genuine, world-perturbing opportunities) | 73.0% |
| safety violations / ep (force + irreversible, both now **reachable**) | 0.034 |
| plan latency p50/p95/p99 | 0.5 / 0.7 / 0.8 ms |
| human interventions | **0** (no rescue API exists) |

| split | success | CI95 |
|---|---|---|
| hidden_dynamics | 91.2% | 0.85–0.98 |
| seen_task_new_layout | 88.7% | 0.81–0.95 |
| disturbance_recovery | 75.0% | 0.65–0.85 |
| unseen_instances | 48.7% | 0.38–0.60 |

Failure breakdown: `ambiguous_or_unidentifiable` **60**, `role_correspondence`
**13**, `control` 3, `verification_false_positive` 1. The dominant bucket is now
the honest one — scenes where a distractor matches the demonstrated size/shape
signature at least as well as the true object, so **no** agent could bind it from
vision alone. `unseen_instances` (heavy size jitter + 3 distractors) is where that
ambiguity concentrates.

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
| A task inference | `perception/`, `compiler/stage_a.py` | belief trajectory → role-based task graph (predicates + **semantic relations**), multi-grasp-episode capable |
| B skill grounding | `skills/correspondence.py`, `skills/grounding.py` | bind roles→tracks by **geometry only** (appearance-independent), instantiate only the needed skill experts |
| C imagined search | `worldmodel/planning_model.py` (analytic, **distinct from the sim**), `worldmodel/search.py` | score candidate plans on collision/uncertainty/force/irreversibility |
| D closed-loop exec | `execution/loop.py`, `execution/verifier.py` | act on belief, adapt `DynamicsContext` online (no weights), replan on events |
| scoring | `benchmark/scorer.py`, `metrics/metrics.py` | ground-truth success + safety + **identifiability**, corrected metrics, bootstrap CIs |

## Ablations (component attribution, paired seeds)

`python -m osc.run_ablations` turns one thing off at a time:

- **retargeting**: absolute **53.8%** < raw-relative **75.0%** < semantic **78.8%**.
- **event-driven vs every-step planning**: **78.8% vs 17.5%** — naive
  receding-horizon replanning thrashes; event-driven planning matters a lot.
- **recovery on/off**: +4.4 points (73% of genuine disturbance opportunities
  recovered).
- **world model / adaptation off**: ≈ no change — these tasks still don't give
  them decisions to make (the benchmark needs harder scenarios; see roadmap).
- **privileged vs belief state**: +5 points.

## Implemented vs. future (learned) components

| implemented now (symbolic/scripted) | future (learned) |
|---|---|
| scripted skill controllers | small neural skill experts sharing an encoder |
| analytic parameter-ensemble planning model | learned latent world model (V-JEPA-2 / OSVI-WM) |
| geometry/size correspondence | learned object correspondence + open-vocab ID |
| RMA-style delay/friction estimation | learned dynamics-context encoder (payload/slip/compliance) |
| toy tabletop CPU sim | ManiSkill3 GPU / contact sim (behind the same `SimBackend`) |

## Honest limitations

- **75.9% success**, but with **46.9% of completions flagged uncertain**: the
  system buys its low 6.7% silent-error rate largely by *abstaining*. That is a
  safety property, **not** autonomy — the ambiguity-resolution gate is open.
- **~half the benchmark is visually unidentifiable** from the demonstration alone
  (distractors overlap role sizes; appearance is independently randomized). Those
  episodes cannot be resolved by better vision — they need active inspection,
  metadata, or a user question. This is the argument for the resolution layer,
  not a bug to tune away.
- Even on *identifiable* scenes, role accuracy is only ~0.65 — a real remaining
  binding problem (stickiness / relational-ratio knobs don't move it).
- `double_stack` (a 2-object composition) **compiles** via multi-episode Stage A
  but its **execution-time multi-object tracking is not reliable**; excluded from
  the default benchmark and kept as a known-limitation task.
- No real physics (no liquids/deformables); tasks are pick-place, side-place, and
  stacking. Pour/wipe/insert wait for a contact simulator.
- No learned components yet; **no baseline (ACT/DP/VLA) is wired** — deliberately.

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
  benchmark/             scorer (ground truth + identifiability), runner, splits, ablations, reports
  metrics/               corrected metric suite + bootstrap CIs
  tasks.py               scenes + oracles + ground-truth success predicates
```

## Roadmap
1. **Resolution layer (next branch):** `TaskContext` + `RoleBelief` +
   `ResolutionPolicy` — separate **active inspection** (resolve poor-sensing
   ambiguity: more frames, viewpoint change, probe/lift, read a label) from
   **clarification** (introduce information not in the video: language, SKU
   metadata, a user selection). Report the **autonomous-coverage vs
   silent-error** curve, not just success. Attach answered clarifications to the
   compiled workflow so the customer isn't asked again per box.
2. Make the benchmark hard enough that world-model/recovery/adaptation matter.
3. Robust multi-object tracking → re-enable `double_stack` and add clear-obstruction.
4. Wire ACT / Diffusion Policy into the same observation-only harness (fair baseline).
5. Learned world model; then the ManiSkill3 GPU backend.
