"""Ground-truth scoring. This is the ONLY place ground-truth SimState is read.

Builds one EpisodeRecord per episode by combining:
  * the final ground-truth SimState (real success + object fates),
  * AgentEnv.step_info_log (real collisions / force / irreversibility),
  * the AgentTrace (agent-side latencies, replans, believed success),
  * the disturbance (whether it created a genuine recovery opportunity),
  * the DynamicsContext vs the episode's true hidden params (context error).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..geometry import dist_xy
from ..sim.base import SimState

CONTROL_HZ = 20.0
NEAR_XY = 0.06


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
    return (dist_xy(oa.pose, ob.pose) < NEAR_XY and oa.pose[2] > ob.pose[2] + ob.size[2] / 4
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
    ambiguous: bool = False            # correspondence was genuinely ambiguous
    inspections: int = 0               # active-perception observe frames used
    role_confidence: float = 1.0
    human_interventions: int = 0       # only nonzero if a human-rescue API is called (none exists)
    recovery_opportunity: bool = False
    recovered: bool = False
    collisions: int = 0
    force_violations: int = 0
    irreversible_failures: int = 0
    timeout: bool = False
    failure_category: str = ""         # "" if success
    disturbance_to_correction_steps: int | None = None
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

    def role_binding_correct(self, trace, final_state: SimState) -> bool:
        """Evidence-based: did the agent's correspondence bind each role to the
        track nearest the CORRECT ground-truth object? Uses role_to_gt + the
        agent's final track poses (privileged, scoring-side only)."""
        r2g = getattr(self.graph, "role_to_gt", {}) if self.graph is not None else {}
        if not r2g or not trace.final_track_poses:
            return True
        for role, gt_name in r2g.items():
            tid = trace.correspondence.get(role)
            if tid is None or tid not in trace.final_track_poses or gt_name not in final_state.objects:
                return False
            p = trace.final_track_poses[tid]
            nearest = min(final_state.objects, key=lambda n: dist_xy(final_state.objects[n].pose, p))
            if nearest != gt_name:
                return False
        return True

    def score(self, split, seed, final_state: SimState, step_info_log, trace,
              disturbance, context, true_params) -> EpisodeRecord:
        success = bool(self.task.success(final_state, self.roles))
        collisions = sum(int(i.collision) for i in step_info_log)
        forces = sum(int(i.force_violation) for i in step_info_log)
        irrev = sum(int(i.irreversible) for i in step_info_log)
        timeout = (not success) and trace.steps >= 0 and \
            trace.steps >= (self.task_max_steps() - 1)

        # a genuine recovery opportunity = the disturbance actually perturbed a
        # task-relevant object (fired and moved/knocked it), judged from ground truth.
        opp = bool(disturbance and disturbance.fired
                   and disturbance.target in self.task_relevant(final_state))
        recovered = opp and success

        # disturbance -> first corrective action latency (in control steps)
        d2c = None
        if disturbance and disturbance.fired:
            after = [s for s in trace.replan_steps if s >= disturbance.at_step]
            if after:
                d2c = after[0] - disturbance.at_step

        rbc = self.role_binding_correct(trace, final_state)
        cat = "" if success else self._categorize(final_state, trace, irrev, timeout, rbc,
                                                  getattr(trace, "ambiguous", False))
        return EpisodeRecord(
            task=self.task.name, split=split, seed=seed, success=success,
            believed_success=trace.believed_success,
            wrong_belief=trace.believed_success and not success,
            role_binding_correct=rbc,
            ambiguous=getattr(trace, "ambiguous", False),
            inspections=getattr(trace, "inspections", 0),
            role_confidence=getattr(trace, "role_confidence", 1.0),
            first_attempt_success=success and trace.autonomous_replans == 0
                and trace.first_failure_step is None,
            steps=trace.steps, sim_seconds=trace.steps / CONTROL_HZ,
            autonomous_replans=trace.autonomous_replans,
            recovery_opportunity=opp, recovered=recovered,
            collisions=collisions, force_violations=forces, irreversible_failures=irrev,
            timeout=timeout, failure_category=cat, disturbance_to_correction_steps=d2c,
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
