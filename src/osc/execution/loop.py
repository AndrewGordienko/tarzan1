"""Stage D: closed-loop execution on BELIEF, with event-driven replanning.

The executor never touches ground truth. Each control step it: gets a percept
from AgentEnv, updates the StateEstimator -> BeliefState, updates the
DynamicsContext (online adaptation), selects an action from the current skill,
and steps. Expensive replanning (Stage C imagined search) fires only on events
(precondition break, believed drop, subgoal boundary) unless configured
otherwise. It produces an AgentTrace; the benchmark Scorer decides real success
and safety from ground truth afterward.

Ablation switches (all default to the full system):
  use_world_model, event_driven, adapt, allow_recovery.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..agent.env import AgentEnv
from ..agent.estimator import StateEstimator
from ..agent.dynamics_context import DynamicsContext
from ..compiler.task_graph import TaskGraph
from ..skills.correspondence import RoleBelief
from .resolution import ResolutionConfig, ResolutionPolicy, TaskContext
from ..skills.grounding import ground_goal, ground_goal_rel, ground_plan
from ..worldmodel.planning_model import PlanningModel
from ..worldmodel.search import ImaginedSearch
from .verifier import Verifier


def _safe_done(si, belief) -> bool:
    try:
        return si.done(belief)
    except KeyError:
        return False


class _OracleGoalVerifier:
    """Wraps a real Verifier but answers goal_satisfied from an oracle (ground
    truth). Used only by the attribution ladder's oracle_verifier mode."""
    def __init__(self, base, oracle_goal):
        self._base = base
        self._oracle = oracle_goal
        self.goal = base.goal
        self.rel_map = base.rel_map

    def goal_satisfied(self, belief):
        return bool(self._oracle(belief))

    def progress(self, belief):
        return self._base.progress(belief)

    def detect_failure(self, prev, cur, expected):
        return self._base.detect_failure(prev, cur, expected)


@dataclass
class AgentTrace:
    believed_success: bool = False
    steps: int = 0
    autonomous_replans: int = 0
    replan_steps: list = field(default_factory=list)   # control-step index of each replan
    corrective_action_steps: list = field(default_factory=list)  # step of first motor
                                                       # action AFTER a reactive replan
    plan_latencies_ms: list = field(default_factory=list)   # per planning CALL
    step_latencies_ms: list = field(default_factory=list)   # observation-ready -> action-dispatch
    events: list = field(default_factory=list)
    correspondence: dict = field(default_factory=dict)
    corr_history: list = field(default_factory=list)   # correspondence at each plan call
    first_failure_step: int | None = None
    believed_progress: float = 0.0
    n_belief_tracks: int = 0
    inspections: int = 0
    final_track_poses: dict = field(default_factory=dict)   # track_id -> pose (for scoring)
    role_confidence: float = 1.0        # confidence of the final role assignment
    min_role_confidence: float = 1.0    # lowest across planning calls
    role_entropy: float = 0.0           # posterior entropy at the FIRST binding
    role_margin: float = 1.0            # weakest-role top1-top2 margin at first binding
    assignment_margin: float = 1.0      # joint top1-top2 cost gap at first binding
    top_cost: float = 0.0               # best joint-assignment cost at first binding
    second_cost: float = 0.0            # runner-up joint-assignment cost
    n_candidates: int = 0               # tracks considered at first binding
    track_uncertainty: float = 0.0      # mean per-track position std at first binding
    staleness: float = 0.0              # mean frames-since-seen at first binding
    ambiguous: bool = False             # correspondence was ambiguous at some plan
    # -- resolution layer --
    committed: bool = True              # did the robot commit to executing (vs abstain)
    clarifications: int = 0             # user questions asked this episode
    resolution_inspection_frames: int = 0   # extra frames spent on active inspection
    initially_ambiguous: bool | None = None  # was the FIRST binding not committable


@dataclass
class ExecConfig:
    max_steps_per_skill: int = 60
    max_replans: int = 4
    max_total_steps: int = 600
    use_world_model: bool = True
    event_driven: bool = True
    adapt: bool = True
    allow_recovery: bool = True
    active_verify: bool = True       # confirm success over several re-observations
    confirm_frames: int = 4          # consecutive frames the goal must hold
    retarget_mode: str = "semantic"  # semantic | relative | absolute (ablation)
    # -- ambiguity-resolution layer (off => v0.3 behaviour: guess & commit) --
    resolution: bool = False         # enable the ResolutionPolicy
    allow_inspection: bool = True    # active inspection (more frames) when enabled
    allow_clarification: bool = True # ask-the-user clarification when enabled
    commit_threshold: float = 0.60   # weakest-role marginal required to commit


class ClosedLoopExecutor:
    def __init__(self, env: AgentEnv, graph: TaskGraph, estimator: StateEstimator,
                 context: DynamicsContext, planning_model: PlanningModel,
                 config: ExecConfig | None = None,
                 oracle_corr=None, oracle_goal=None, clarify_fn=None, task_context=None):
        self.env = env
        self.graph = graph
        self.est = estimator
        self.ctx = context
        self.search = ImaginedSearch(planning_model)
        self.cfg = config or ExecConfig()
        # ambiguity-resolution layer (benchmark-side clarify_fn is the "user":
        # clarify_fn(target_roles, belief) -> {role: track_id}, GT-derived).
        self.clarify_fn = clarify_fn
        if self.cfg.resolution:
            self.resolver = ResolutionPolicy(ResolutionConfig(
                allow_inspection=self.cfg.allow_inspection,
                allow_clarification=self.cfg.allow_clarification,
                commit_threshold=self.cfg.commit_threshold))
        else:
            self.resolver = None
        # persists across a WORKFLOW when supplied: a clarification answered once
        # during setup carries to every subsequent production box.
        self.task_context = task_context if task_context is not None else TaskContext()
        # ATTRIBUTION-LADDER hooks (benchmark-side, not the deployed agent):
        #   oracle_corr(belief) -> {role: track_id}  perfect role binding
        #   oracle_goal(belief) -> bool               ground-truth stop decision
        self.oracle_corr = oracle_corr
        self.oracle_goal = oracle_goal
        self.role_belief = RoleBelief(graph.role_signatures)

    def _apply_clarifications(self, corr, belief):
        """Re-apply roles the user already disambiguated (persisted on TaskContext).
        Re-derived each call because track ids churn -- the customer answered about
        an OBJECT once; we keep tracking it without asking again."""
        if self.clarify_fn is None or not self.task_context.user_selections:
            return corr
        corr = dict(corr)
        fixed = self.clarify_fn(tuple(sorted(self.task_context.user_selections)), belief)
        corr.update({r: t for r, t in fixed.items() if t is not None})
        return corr

    def _resolve_initial(self, belief, tr: AgentTrace):
        """Run the ResolutionPolicy once at episode start: inspect (gather frames),
        clarify (ask the user), or abstain -- rather than guess a low-confidence
        binding. Clarifications persist on self.task_context for later plan calls."""
        from ..sim.base import Action
        if self.resolver is None or self.oracle_corr:
            return belief                            # feature off / oracle binding
        ra = self.role_belief.update(belief)
        tr.initially_ambiguous = not self.resolver.committable(ra, self.task_context)
        inspections_used, last_gain = 0, None
        while True:
            action = self.resolver.decide(ra, self.task_context, inspections_used,
                                          tr.clarifications, last_gain)
            if action.kind == "commit":
                break
            if action.kind == "observe":
                prev_weak = min(ra.per_role_conf.values(), default=1.0)
                for _ in range(self.resolver.cfg.inspect_frames):
                    hold = belief.gripper.copy(); hold[2] += 0.02
                    belief = self.est.update(self.env.step(Action(target=hold, gripper_close=0.0)))
                    tr.steps += 1; tr.resolution_inspection_frames += 1
                ra = self.role_belief.update(belief)
                last_gain = min(ra.per_role_conf.values(), default=1.0) - prev_weak
                inspections_used += 1
            elif action.kind == "ask_user":
                if self.clarify_fn is None:
                    tr.committed = False; break      # nobody to ask -> abstain
                self.task_context.user_selections.update(action.target_roles)
                tr.clarifications += 1
                ra = self.role_belief.update(belief)  # clarified roles now excluded
            else:                                    # abstain
                tr.committed = False; break
        return belief

    def _plan(self, belief, tr: AgentTrace):
        """Re-resolve roles->tracks against the CURRENT belief (track ids churn
        under occlusion/merge), rebuild the goal+verifier, then select a plan."""
        t0 = time.perf_counter()
        if self.oracle_corr:
            corr = self.oracle_corr(belief)
        else:
            ra = self.role_belief.update(belief)
            corr = self._apply_clarifications(ra.mapping, belief)
            tr.role_confidence = ra.confidence
            tr.min_role_confidence = min(tr.min_role_confidence, ra.confidence)
            if not tr.corr_history:            # capture features at the FIRST binding
                tr.role_entropy = ra.entropy
                tr.role_margin = ra.margin
                tr.assignment_margin = ra.assignment_margin
                tr.top_cost = ra.top_cost
                tr.second_cost = ra.second_cost
                tr.n_candidates = ra.n_candidates
                objs = list(belief.objects.values())
                if objs:
                    tr.track_uncertainty = sum(getattr(o, "pos_std", 0.0) for o in objs) / len(objs)
                    tr.staleness = sum(belief.t - getattr(o, "last_seen", belief.t)
                                       for o in objs) / len(objs)
            tr.ambiguous = tr.ambiguous or ra.ambiguous
        tr.correspondence = corr
        tr.corr_history.append(dict(corr))
        verifier = Verifier(ground_goal(self.graph, corr), ground_goal_rel(self.graph, corr))
        if self.oracle_goal is not None:
            verifier = _OracleGoalVerifier(verifier, self.oracle_goal)
        base = ground_plan(self.graph, corr, self.cfg.retarget_mode)
        if self.cfg.use_world_model:
            plan = self.search.select(belief, base, verifier.goal_satisfied).plan
        else:
            plan = base
        tr.plan_latencies_ms.append(1000 * (time.perf_counter() - t0))
        return list(plan), verifier

    def _confirm(self, belief, verifier, tr):
        """Active completion verification: hold and re-observe for a few frames;
        require the goal to remain satisfied on every frame. Catches placements
        that look done for one noisy frame but have actually slipped/fallen."""
        if not self.cfg.active_verify:
            return True, belief
        from ..sim.base import Action
        hold = belief.gripper.copy(); hold[2] += 0.02
        for _ in range(self.cfg.confirm_frames):
            belief = self.est.update(self.env.step(Action(target=hold, gripper_close=0.0)))
            tr.steps += 1; tr.inspections += 1
            try:
                if not verifier.goal_satisfied(belief):
                    return False, belief
            except KeyError:
                return False, belief
        return True, belief

    def _replan(self, belief, tr, at_event: str) -> tuple:
        tr.autonomous_replans += 1
        tr.replan_steps.append(tr.steps)
        tr.events.append(f"replan({at_event}) @step{tr.steps}")
        return self._plan(belief, tr)

    def run(self) -> AgentTrace:
        tr = AgentTrace()
        belief = self.est.update(self.env.reset_percept())
        belief = self._resolve_initial(belief, tr)
        if not tr.committed:                        # abstained -- decline, do not guess
            tr.n_belief_tracks = len(belief.objects)
            tr.final_track_poses = {tid: o.pose.copy() for tid, o in belief.objects.items()}
            return tr
        plan, verifier = self._plan(belief, tr)

        idx = 0
        pending_corrective = False   # a reactive replan is awaiting its first action
        while idx < len(plan):
            si = plan[idx]
            try:
                ok = si.skill.precondition(belief, si.params)
            except KeyError:
                ok = False                          # a referenced track vanished
            if not ok:
                if self.cfg.allow_recovery and tr.autonomous_replans < self.cfg.max_replans:
                    plan, verifier = self._replan(belief, tr, "precond"); idx = 0
                    continue
                idx += 1; continue

            expected = si.params.get("object") if si.skill.name in ("move", "place") else None
            sk_steps, restart = 0, False
            while sk_steps < self.cfg.max_steps_per_skill:
                t0 = time.perf_counter()
                # receding-horizon (every-step) ablation: replan FIRST, then act
                # from the freshly planned current skill -- so the new plan
                # actually drives this step's action (v0.2 measured only the extra
                # planning call, not true receding horizon).
                if self.cfg.use_world_model and not self.cfg.event_driven:
                    plan, verifier = self._plan(belief, tr)
                    si = next((s for s in plan if not _safe_done(s, belief)), si)
                try:
                    if si.done(belief):
                        break
                    action = si.act(belief)
                    # sensor->action latency: observation-ready (t0) -> action
                    # dispatched. Simulator stepping and state estimation below are
                    # the NEXT observation's acquisition, not part of this latency.
                    tr.step_latencies_ms.append(1000 * (time.perf_counter() - t0))
                except KeyError:                    # track referenced by skill is gone
                    if self.cfg.allow_recovery and tr.autonomous_replans < self.cfg.max_replans:
                        plan, verifier = self._replan(belief, tr, "lost-track"); idx = 0
                        pending_corrective = True; restart = True
                    break
                prev = belief
                belief = self.est.update(self.env.step(action))
                if self.cfg.adapt:
                    self.ctx.update(prev, action.target, belief)
                tr.steps += 1; sk_steps += 1
                # first motor action dispatched after a reactive replan is the
                # corrective action -- distinct from the replan itself.
                if pending_corrective:
                    tr.corrective_action_steps.append(tr.steps)
                    pending_corrective = False

                event = verifier.detect_failure(prev, belief, expected)
                if event:
                    tr.events.append(f"event:{event} @step{tr.steps}")
                    if tr.first_failure_step is None:
                        tr.first_failure_step = tr.steps
                    if self.cfg.allow_recovery and tr.autonomous_replans < self.cfg.max_replans:
                        plan, verifier = self._replan(belief, tr, event.split(":")[0])
                        idx = 0; pending_corrective = True; restart = True
                    break
                if tr.steps >= self.cfg.max_total_steps:
                    idx = len(plan); break

            if restart:
                continue
            idx += 1
            try:
                if verifier.goal_satisfied(belief):
                    confirmed, belief = self._confirm(belief, verifier, tr)
                    if confirmed:
                        break
                    # placement slipped/fell on re-observation -> retry if allowed
                    if self.cfg.allow_recovery and tr.autonomous_replans < self.cfg.max_replans:
                        plan, verifier = self._replan(belief, tr, "unconfirmed")
                        idx = 0
                        continue
            except KeyError:
                pass

        try:
            ok = verifier.goal_satisfied(belief)
            if ok:
                ok, belief = self._confirm(belief, verifier, tr)
            tr.believed_success = ok
            tr.believed_progress = verifier.progress(belief)
        except KeyError:
            pass
        tr.n_belief_tracks = len(belief.objects)
        tr.final_track_poses = {tid: o.pose.copy() for tid, o in belief.objects.items()}
        return tr
