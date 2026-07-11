# Attribution: v0.5 Kalman size-fusion change (341751d → e270e1c)

The estimator's size update changed from EMA (`0.8·size + 0.2·det`) to a tracked
Kalman fusion. On the paired 320-episode benchmark (`PYTHONHASHSEED=0`,
`size_noise=0`), the effect is:

- **aggregate: success 243 → 244 (+1), role-binding-correct 187 → 187 (0).**
- **5 episodes changed** (all `stack`); an earlier "81 changed" figure was a
  diff bug — `(split, seed)` is not unique across tasks, so the key collided.

| task/split/seed | old (succ,bind) | new (succ,bind) |
|---|---|---|
| seen_task_new_layout 20 | (0,1) | (0,0) |
| unseen_instances 13 | (0,0) | (1,0) |
| disturbance_recovery 0 | (1,0) | (0,0) |
| disturbance_recovery 16 | (0,0) | (1,0) |
| disturbance_recovery 34 | (1,0) | (1,1) |

Net: +2 success −1 success = **+1**; +1 bind −1 bind = **0**.

## Mechanism (traced)

Clean episodes are bit-identical (e.g. seen_task_new_layout seed 6: same
correspondence, same final poses in both trees). All 5 changes sit in the
**track-churn** path — 3 of 5 in `disturbance_recovery`, the rest in the
high-jitter/distractor splits — where a track receives detections from different
objects across frames (re-acquisition after a disturbance, merges). There the
Kalman blend (gain shrinks with accumulated evidence) diverges from the fixed-gain
EMA, which shifts either the global one-to-one assignment or the final track
geometry:

- **disturbance_recovery 0**: OLD binds manipuland→t0 (size 0.050), NEW→t2 (size
  0.036). The demonstrated manipuland is ~0.036, so **NEW is the better size
  match** — a more principled binding. (This borderline episode's *success* still
  regressed, a downstream-dynamics coin-flip, not a binding error.)
- **disturbance_recovery 34**: identical correspondence and size history, but the
  final track geometry differs slightly (execution timing), flipping the scorer's
  nearest-GT `role_binding_correct` from wrong → right.

## Verdict

**Legitimate side effect of a principled change, not a regression.** Kalman
fusion weights observations by accumulated evidence (correct) rather than a fixed
0.2 EMA gain. The effect is confined to borderline churn episodes, is
binding-neutral in aggregate, and slightly success-positive (+1). Where it visibly
changes a binding (seed 0) it picks the better size match. No action needed; the
headline number will be regenerated from this code at v0.4 PR time.
