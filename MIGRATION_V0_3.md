# Migration: v0.2 → v0.3

v0.2 made evaluation honest. v0.3 uses that honesty to **find** the bottleneck by
measurement, then fixes it — instead of guessing "perception."

## 1. The error-attribution ladder came first (as instructed)

`python -m osc.run_attribution` replays every episode (same task/split/seed) with
a perfect version of one component swapped in. The v0.2 claim "perception is the
bottleneck" was **wrong**:

| oracle swap | success | Δ vs full |
|---|---|---|
| full (belief) | 74.5% | — |
| + oracle tracks (perfect state est.) | 82.0% | **+7.5** |
| + oracle correspondence | 87.5% | **+13.0** |
| + oracle tracks + correspondence | 96.5% | +22.0 |
| + oracle verifier | 72.5% | −2.0 |
| full oracle (upper bound) | 96.5% | +22.0 |

**Role correspondence, not tracking, was the dominant failure.** Before the fix,
correspondence alone was worth **+33 points** and misbound in **62% of episodes**.
Tracking was worth ~2. The scorer no longer defaults silent failures to
"perception" — categories are derived from counterfactual role-binding checks:
`role_correspondence`, `ambiguous_or_unidentifiable`, `control`,
`verification_false_positive`, `irreversible`, `timeout`.

## 2. Correspondence rebuilt (the fix the budget pointed to)

Greedy per-role feature matching → **`RoleBelief`**: global one-to-one assignment,
a **relational cost** (the demonstrated size *ratio* between roles disambiguates
similarly-sized distractors), a null option, softmax **confidence + entropy**, an
**ambiguous** flag, and **stickiness** across replans. Deterministic throughout
(sorted iteration; `hash(color)` replaced with a fixed-palette code, since
Python's builtin hash varies between processes).

Result: full-system success **50.6% → 74.5%**; wrong-belief **45% → 17.5%**;
**P(success | correct role) = 97.5%**.

## 3. Active verification + honest uncertainty

- Completion is confirmed over several re-observations (re-observe after release),
  catching placements that look done for one noisy frame but slipped/fell.
- Genuinely near-tied scenes are labelled **ambiguous** rather than counted as
  correspondence bugs. New conditional metrics: **silent false-completion among
  CONFIDENT claims** (12.1%; the honest target is <10%), uncertain-completion
  rate, `P(success | correct role)`, role-binding accuracy.

## 4. Semantic retargeting (the one-shot-learning thesis)

Placements are stored as **relations**, not copied metric offsets. At deploy the
target is solved on current geometry:
`target_z = support_center + support_half + manip_half + clearance`; "beside" is a
size-normalized direction+distance. Ablation (`retarget_mode`):
absolute **53.8%** < relative **72.5%** ≤ semantic **75.0%**. Semantic edges
relative; the gap is small **because the toy sim's settling forgives z-target
error** (an object released too high just drops onto the support). Showing a large
semantic advantage needs a stricter placement task (insertion / tight fit) — a
documented next step, not a solved claim.

## 5. Ablation bug fixed

`every_step_planning` now lets the freshly-planned skill drive the next action
(v0.2 measured only the extra planning call). With the fix it drops to **17.5%**
vs event-driven **75%** — i.e. naive receding-horizon replanning thrashes here;
event-driven planning is materially better. Recovery now helps (+4.4 pts;
78% of genuine opportunities recovered).

## 6. Infra

Held-out vs dev seed generators (`benchmark/seeds.py`); GitHub Actions CI runs
tests + a deterministic benchmark/attribution smoke test.

## Gate status (per the review's v0.3 completion bar)

| gate | status |
|---|---|
| evidence-based failure attribution | ✅ done (ladder + counterfactual categories) |
| oracle correspondence distinct from oracle estimation | ✅ done |
| semantic beats raw relative on changed sizes | ⚠️ marginal (+2.5pt); sim settling masks it — needs a stricter task |
| silent false-completion < 10% | ⚠️ close (12.1%), not yet under 10% |
| role assignments stable + calibrated | ✅ sticky + confidence/entropy; ⚠️ full posterior/multi-hypothesis partial |
| a hard split gains from planning | ✅ event-driven vs every-step (75 vs 17.5) |
| a disturbance split gains from recovery | ✅ +4.4 pts, 78% recovered |
| multi-object composition reliable | ❌ `double_stack` still excluded |
| CI green | ✅ |

**v0.3 is not "complete" against every gate** — semantic-advantage demonstration,
<10% silent false-completion, and reliable multi-object composition remain open,
and ACT stays deferred until they close. This is reported honestly rather than
declared done.
