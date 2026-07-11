"""Ground-truth scoring. This is the ONLY place ground-truth SimState is read.

Builds one EpisodeRecord per episode by combining:
  * the final ground-truth SimState (real success + object fates),
  * AgentEnv.step_info_log (real collisions / force / irreversibility),
  * the AgentTrace (agent-side latencies, replans, believed success),
  * the disturbance (whether it created a genuine recovery opportunity),
  * the DynamicsContext vs the episode's true hidden params (context error).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..geometry import dist_xy
from ..sim.base import SimState

CONTROL_HZ = 20.0
NEAR_XY = 0.06
IDENT_EPS = 0.005   # a distractor within this weighted-feature gap of the true
                    # object == the role is not observably identifiable (small:
                    # only flag genuine ties, not merely "hard" scenes)


# -- ground-truth predicates ---------------------------------------------
def _alive(s: SimState, a: str) -> bool:
    return a in s.objects and a not in s.fallen

def gt_on_table(s: SimState, a: str) -> bool:
    if not _alive(s, a):
        return False
    o = s.objects[a]
    return abs(o.pose[2] - (s.table_z + o.size[2] / 2)) < 0.02 and s.grasped != a

def gt_near(s: SimState, a: str, b: str) -> bool:
    return _alive(s, a) and _alive(s, b) and dist_xy(s.objects[a].pose, s.objects[b].pose) < NEAR_XY

def gt_on_top(s: SimState, a: str, b: str) -> bool:
    if not (_alive(s, a) and _alive(s, b)):
        return False
    oa, ob = s.objects[a], s.objects[b]
    # a genuine stack: a is horizontally over b AND a's base rests on b's top
    # surface (b centre + half b's height + half a's height). Loose "somewhat
    # above" checks let a floating or geometrically invalid stack pass.
    rest_z = ob.pose[2] + ob.size[2] / 2 + oa.size[2] / 2
    return (dist_xy(oa.pose, ob.pose) < NEAR_XY and abs(oa.pose[2] - rest_z) < 0.02
            and s.grasped != a)

def gt_in_region(s: SimState, a: str, center, radius: float) -> bool:
    return _alive(s, a) and dist_xy(s.objects[a].pose, [center[0], center[1], 0, 0]) < radius


@dataclass
class EpisodeRecord:
    task: str
    split: str
    seed: int
    success: bool
    believed_success: bool
    wrong_belief: bool                 # agent believed success but ground truth says no
    first_attempt_success: bool        # success, no failure event, no replan
    steps: int
    sim_seconds: float
    autonomous_replans: int
    role_binding_correct: bool = True  # did the agent act on the right GT objects
    ambiguous: bool = False            # agent flagged OR objectively unidentifiable
    identifiable: bool = True          # OBJECTIVE: true role is the best observable match
    inspections: int = 0               # active-perception observe frames used
    role_confidence: float = 1.0
    role_entropy: float = 0.0          # agent-visible discrimination features (for the audit)
    role_margin: float = 1.0
    assignment_margin: float = 1.0
    top_cost: float = 0.0
    second_cost: float = 0.0
    n_candidates: int = 0
    track_uncertainty: float = 0.0
    staleness: float = 0.0
    human_interventions: int = 0       # only nonzero if a human-rescue API is called (none exists)
    # -- resolution layer --
    committed: bool = True             # robot committed to executing (vs abstained)
    clarifications: int = 0            # user questions asked
    resolution_inspection_frames: int = 0
    initially_ambiguous: bool = False  # first binding was not committable
    viewpoints: list = field(default_factory=list)
    viewpoint_frames: int = 0
    viewpoint_actions: int = 0
    association_contested: bool = False
    viewpoint_diagnostics: list = field(default_factory=list)
    assignment_diagnostics: dict = field(default_factory=dict)
    assignment_failure_class: str = ""
    recovery_opportunity: bool = False
    recovered: bool = False
    collisions: int = 0
    force_violations: int = 0
    irreversible_failures: int = 0
    timeout: bool = False
    failure_category: str = ""         # "" if success
    disturbance_to_correction_steps: int | None = None   # -> first corrective MOTOR action
    disturbance_to_replan_steps: int | None = None       # -> first replan (planner reaction)
    plan_latencies_ms: list = field(default_factory=list)
    step_latencies_ms: list = field(default_factory=list)
    context_error: dict = field(default_factory=dict)

    @property
    def safety_violations(self) -> int:
        return self.force_violations + self.irreversible_failures


class Scorer:
    def __init__(self, task, roles: dict, graph=None):
        self.task = task
        self.roles = roles
        self.graph = graph

    def identifiable(self, final_state: SimState) -> bool:
        """Ground-truth identifiability. A demonstration binds roles by observable
        geometry (size + shape; NOT the independently-randomized colour). If, for
        some role, a distractor matches the demonstrated signature at least as well
        as the TRUE object -- because size jitter drifted the true target and a
        distractor sits at the demonstrated size -- then no agent could reliably
        pick the right object. Such an episode is genuinely ambiguous, not a
        silent perception failure, and is excluded from the confident-completion
        denominator. Adjudicated with GT here; uses only agent-observable features.
        """
        r2g = getattr(self.graph, "role_to_gt", {}) if self.graph is not None else {}
        sigs = getattr(self.graph, "role_signatures", {}) if self.graph is not None else {}
        if not r2g or not sigs:
            return True
        W = (3.0, 3.0, 1.0)                               # size_x, size_z, shape (no colour)
        def feat(o):
            return (float(o.size[0]), float(o.size[2]), 0.0 if o.shape == "box" else 1.0)
        def wdist(a, b):
            return math.sqrt(sum((w * (ai - bi)) ** 2 for w, ai, bi in zip(W, a, b)))
        alive = [n for n in final_state.objects if n not in final_state.fallen]
        for role, gt_name in r2g.items():
            sig = sigs.get(role)
            if sig is None or gt_name not in final_state.objects:
                continue
            sig = tuple(float(x) for x in sig[:3])
            d = {n: wdist(sig, feat(final_state.objects[n])) for n in alive}
            if gt_name not in d:
                continue
            d_true = d[gt_name]
            d_other = min((v for n, v in d.items() if n != gt_name), default=d_true + 10.0)
            if d_true >= d_other - IDENT_EPS:            # a distractor is as good a match
                return False
        return True

    def role_binding_correct(self, trace, final_state: SimState) -> bool:
        """Evidence-based: did the agent's correspondence bind each role to the
        track nearest the CORRECT ground-truth object? Uses role_to_gt + the
        agent's final track poses (privileged, scoring-side only)."""
        r2g = getattr(self.graph, "role_to_gt", {}) if self.graph is not None else {}
        if not r2g or not trace.final_track_poses:
            return True
        return self.mapping_binding_correct(trace.correspondence, trace.final_track_poses, final_state)

    def mapping_binding_correct(self, mapping, track_poses, final_state: SimState) -> bool:
        r2g = getattr(self.graph, "role_to_gt", {}) if self.graph is not None else {}
        if not r2g or not track_poses:
            return True
        for role, gt_name in r2g.items():
            tid = mapping.get(role)
            if tid is None or tid not in track_poses or gt_name not in final_state.objects:
                return False
            p = track_poses[tid]
            nearest = min(final_state.objects, key=lambda n: dist_xy(final_state.objects[n].pose, p))
            if nearest != gt_name:
                return False
        return True

    def assignment_audit(self, trace, initial_state: SimState | None) -> tuple[dict, str]:
        """Adjudicate the exact posterior against the initial-scene GT tracks."""
        d = dict(getattr(trace, "assignment_diagnostics", {}) or {})
        if not d or initial_state is None:
            return d, ""
        role_to_gt = getattr(self.graph, "role_to_gt", {}) if self.graph is not None else {}
        poses = d.get("track_poses", {})
        gt_track = {}
        for role, name in role_to_gt.items():
            if name not in initial_state.objects or not poses:
                continue
            gp = initial_state.objects[name].pose
            gt_track[role] = min(poses, key=lambda tid: dist_xy(poses[tid], gp))
        target = {r: gt_track.get(r) for r in role_to_gt}
        posterior = d.get("posterior", [])
        present = []
        for item in posterior:
            mapping = item.get("mapping", {})
            present.append(mapping == target)
        rank = (present.index(True) + 1) if True in present else None
        top_correct = bool(present and present[0])
        d.update(dict(gt_assignment=target, gt_assignment_rank=rank,
                      gt_assignment_present=rank is not None,
                      top_assignment_correct=top_correct,
                      gt_assignment_posterior=(posterior[rank - 1]["prob"] if rank else 0.0)))
        if rank is None:
            category = "correct_assignment_excluded"
        elif rank > 1:
            category = "correct_assignment_present_but_scored_lower"
        elif not trace.committed and top_correct:
            category = "correct_assignment_ranked_first_policy_abstains"
        elif not trace.committed:
            category = "wrong_assignment_ranked_first_policy_abstains"
        elif not top_correct:
            category = "wrong_assignment_ranked_first_policy_commits"
        else:
            category = "correct_assignment_ranked_first_policy_commits"
        return d, category

    def score(self, split, seed, final_state: SimState, step_info_log, trace,
              disturbance, context, true_params, initial_state: SimState | None = None) -> EpisodeRecord:
        success = bool(self.task.success(final_state, self.roles))
        collisions = sum(int(i.collision) for i in step_info_log)
        forces = sum(int(i.force_violation) for i in step_info_log)
        irrev = sum(int(i.irreversible) for i in step_info_log)
        timeout = (not success) and trace.steps >= 0 and \
            trace.steps >= (self.task_max_steps() - 1)

        # a genuine recovery opportunity = the disturbance actually PERTURBED the
        # world (e.g. a "drop" that found nothing held is a no-op and does not
        # count) for a task-relevant object -- judged from ground truth, not from
        # whether the episode happened to succeed.
        perturbed = bool(disturbance and getattr(disturbance, "perturbed", disturbance.fired))
        opp = bool(perturbed and disturbance.target in self.task_relevant(final_state))
        recovered = opp and success

        # disturbance -> first corrective MOTOR ACTION (steps), distinct from the
        # planner's first replan reaction. Both measured only over reactions that
        # occur after the disturbance actually fired.
        d2c = d2r = None
        if perturbed:
            acts = [s for s in trace.corrective_action_steps if s >= disturbance.at_step]
            if acts:
                d2c = acts[0] - disturbance.at_step
            reps = [s for s in trace.replan_steps if s >= disturbance.at_step]
            if reps:
                d2r = reps[0] - disturbance.at_step

        rbc = self.role_binding_correct(trace, final_state)
        assignment_diagnostics, assignment_failure_class = self.assignment_audit(trace, initial_state)
        viewpoint_diagnostics = []
        for d in getattr(trace, "viewpoint_diagnostics", []):
            d = dict(d)
            d["binding_before_correct"] = self.mapping_binding_correct(
                d["assignment_before"], d["track_poses_before"], final_state)
            d["binding_after_correct"] = self.mapping_binding_correct(
                d["assignment_after"], d["track_poses_after"], final_state)
            viewpoint_diagnostics.append(d)
        # ambiguous = the agent flagged it OR the scene is objectively unidentifiable.
        # Identifiability is a property of the evidence available when choosing
        # roles, not of objects after the robot may have moved/dropped them.  Using
        # final_state here made labels configuration-dependent and could invert a
        # pooled AUROC without any change to the score itself.
        identifiable = self.identifiable(initial_state if initial_state is not None else final_state)
        ambiguous = bool(getattr(trace, "ambiguous", False)) or not identifiable
        cat = "" if success else self._categorize(final_state, trace, irrev, timeout, rbc, ambiguous)
        return EpisodeRecord(
            task=self.task.name, split=split, seed=seed, success=success,
            believed_success=trace.believed_success,
            wrong_belief=trace.believed_success and not success,
            role_binding_correct=rbc,
            ambiguous=ambiguous, identifiable=identifiable,
            committed=getattr(trace, "committed", True),
            clarifications=getattr(trace, "clarifications", 0),
            resolution_inspection_frames=getattr(trace, "resolution_inspection_frames", 0),
            initially_ambiguous=bool(getattr(trace, "initially_ambiguous", False)),
            viewpoints=list(getattr(trace, "viewpoints", [])),
            viewpoint_frames=getattr(trace, "viewpoint_frames", 0),
            viewpoint_actions=getattr(trace, "viewpoint_actions", 0),
            association_contested=bool(getattr(trace, "association_contested", False)),
            viewpoint_diagnostics=viewpoint_diagnostics,
            assignment_diagnostics=assignment_diagnostics,
            assignment_failure_class=assignment_failure_class,
            inspections=getattr(trace, "inspections", 0),
            role_confidence=getattr(trace, "role_confidence", 1.0),
            role_entropy=getattr(trace, "role_entropy", 0.0),
            role_margin=getattr(trace, "role_margin", 1.0),
            assignment_margin=getattr(trace, "assignment_margin", 1.0),
            top_cost=getattr(trace, "top_cost", 0.0),
            second_cost=getattr(trace, "second_cost", 0.0),
            n_candidates=getattr(trace, "n_candidates", 0),
            track_uncertainty=getattr(trace, "track_uncertainty", 0.0),
            staleness=getattr(trace, "staleness", 0.0),
            first_attempt_success=success and trace.autonomous_replans == 0
                and trace.first_failure_step is None,
            steps=trace.steps, sim_seconds=trace.steps / CONTROL_HZ,
            autonomous_replans=trace.autonomous_replans,
            recovery_opportunity=opp, recovered=recovered,
            collisions=collisions, force_violations=forces, irreversible_failures=irrev,
            timeout=timeout, failure_category=cat, disturbance_to_correction_steps=d2c,
            disturbance_to_replan_steps=d2r,
            plan_latencies_ms=list(trace.plan_latencies_ms),
            step_latencies_ms=list(trace.step_latencies_ms),
            context_error=context.error_vs(*true_params) if context else {})

    def task_max_steps(self) -> int:
        return getattr(self.task, "max_total_steps", 600)

    def task_relevant(self, s: SimState):
        return [n for n, r in self.roles.items() if r in ("manipuland", "target")]

    def _categorize(self, s: SimState, trace, irrev, timeout, role_binding_correct,
                    ambiguous) -> str:
        """Evidence-based, not a default. Role-binding correctness comes from the
        counterfactual check, so wrong-object failures are labelled as role
        correspondence rather than lumped into 'perception'."""
        if not role_binding_correct:
            # if the scene was genuinely ambiguous, this is not a correspondence
            # bug -- the observation did not determine the answer.
            return "ambiguous_or_unidentifiable" if ambiguous else "role_correspondence"
        if irrev > 0:
            return "irreversible"          # a task object was knocked off the table
        if trace.believed_success:
            # bound the right objects but still declared success wrongly:
            # the verifier accepted a placement that ground truth rejects.
            return "verification_false_positive"
        if timeout:
            return "timeout"
        if trace.autonomous_replans > self._max_replans():
            return "recovery_failed"
        return "control"                   # right objects, right stop, missed placement

    def _max_replans(self) -> int:
        return 4
