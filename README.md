# One-Shot Task Compiler (OSC)

**Research question:** can a robot watch *one* demonstration of a previously
unseen task, infer an executable task *program*, and complete it in changed
environments **without any task-time fine-tuning**?

This repo is the **Phase-1 / Phase-2 vertical slice**: a single manipulation task
(`STACK`: place cube A on cube B) driven end-to-end through the full pipeline —
demo → task graph → skill grounding → imagined search → closed-loop execution
with event-driven recovery — evaluated across randomized environments with an
injected disturbance and **zero gradient updates after the demonstration**.

It runs on a laptop CPU today. The GPU/photoreal backend (ManiSkill3) plugs in
behind the same interface later.

---

## Quickstart

```bash
pip install -e .                 # numpy only
python -m osc.run_demo --episodes 30 --seed 0
pytest -q                        # 5 behavioural tests, ~0.3 s
```

Example output (seed 0, 30 episodes, disturbance on):

```
unseen-environment success   :  90.0%
first-attempt success        :  63.3%     <- before recovery
eventual success (w/ recovery):  90.0%     <- recovery closes the gap
recovery rate                : 100.0%
mean planning latency        :   16.4 ms   <- event-driven, not per-frame
p95  planning latency        :   21.0 ms
safety violations / episode  :   0.00
```

The gap between **first-attempt** and **eventual** success is the recovery story;
the low planning latency is the event-driven-replanning story.

---

## How the code maps to the research plan

| Plan stage | Module | What it does |
|---|---|---|
| **A** task inference | `perception/tracks.py`, `compiler/stage_a.py` | one demo → object tracks, contact states, keyframes → **task graph** of predicates + *relative* transforms |
| **B** skill grounding | `skills/library.py`, `skills/grounding.py` | retrieve reusable skill experts for each transition and **retarget** them to current objects (the sparse *router*) |
| **C** imagined search | `worldmodel/model.py`, `worldmodel/search.py` | roll candidate plans through an action-conditioned **world model**; penalize collision / uncertainty / force / irreversible states |
| **D** closed-loop exec | `execution/loop.py`, `execution/verifier.py` | execute a short horizon, adapt dynamics online (RMA-style, no weights), **replan only on events** |
| **E** autonomous improvement | *(hook)* `EpisodeResult.events` | every episode logs failures/recoveries for offline consolidation — the store is here; the consolidation trainer is future work |

The **metric suite** (`metrics/metrics.py`) is deliberately *not* average action
error: unseen-env success, first-attempt vs eventual success, recovery rate,
interventions/op-hour, planning latency (p50/p95), completion time, safety
violations, demos required (=1), cost per success.

### Why one demo transfers
The task graph stores **relative transforms** (`geometry.relative`), not world
poses or joint trajectories. A subgoal recorded once ("cube A ends 0 offset above
cube B") is invariant to where the objects sit, how they're turned, what they
look like, and the dynamics — which is exactly what changes between the demo and
each randomized evaluation episode.

### The modular / latency-first architecture
This slice already instantiates the "nervous-system" decomposition:
skill grounding **is** the sparse router (only the handful of experts a phase
needs are activated), Stage C **is** the predictive forward model, and Stage D's
replan-on-event **is** the event-driven planner that wakes expensive reasoning
only on novelty / failure — keeping the reactive loop cheap.

---

## Simulator backends

- **`ToyTabletopSim`** (`sim/toy.py`) — NumPy CPU rigid-body tabletop. Not a
  physics engine; it models exactly the phenomena the benchmark stresses (grasp,
  stack, lateral collision, force limits, irreversible off-table loss, actuator
  delay, friction/mass). Runs on this Mac now.
- **`ManiSkillBackend`** — same `SimBackend` interface, GPU/SAPIEN, added when a
  CUDA machine is available. Nothing above `sim/base.py` changes.

## Layout
```
src/osc/
  geometry.py          SE(2)+z transforms (relative/apply invariance)
  sim/                 backend interface, toy sim, randomization, disturbance
  perception/          Stage A front-end: tracks, contacts, keyframes
  compiler/            Stage A: task graph (predicates + relative transforms)
  skills/              reusable skill experts + grounding/router (Stage B)
  worldmodel/          Stage C: ensemble world model + imagined search
  execution/           Stage D: verifier + event-driven closed loop
  metrics/             the deployment-oriented metric suite
  tasks.py             STACK scene + scripted oracle that records the 1 demo
  run_demo.py          end-to-end entry point
```

## Status & honest limitations
- One task (STACK). Widening to the 15–25 task benchmark = add a scene + oracle
  per task; the pipeline is untouched.
- The world model is a **parameter-ensemble** analytic model, not yet a learned
  latent forward model (V-JEPA-2 / OSVI-WM). The `rollout` interface is the seam
  to swap it.
- Baselines (ACT, Diffusion Policy, OpenVLA/π0, pure TAMP) are **not yet wired**;
  the metric harness is built to receive them.
- Stage E stores experience but does not yet consolidate it into new experts.

## Roadmap (next)
1. Widen to ~6 tasks (drawer, insert, pour, wipe, push, handover).
2. Wire an ACT / Diffusion-Policy baseline into the same harness for the
   20-point head-to-head on held-out tasks.
3. Replace the ensemble world model with a learned latent model.
4. Implement Stage-E offline consolidation (replay → distilled skills).
5. Stand up the ManiSkill3 backend on a GPU box.
