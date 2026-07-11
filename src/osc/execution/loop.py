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
from ..skills.correspondence import correspond
from ..skills.grounding import ground_goal, ground_goal_rel, ground_plan
from ..worldmodel.planning_model import PlanningModel
from ..worldmodel.search import ImaginedSearch
from .verifier import Verifier


@dataclass
class AgentTrace:
    believed_success: bool = False
    steps: int = 0
    autonomous_replans: int = 0
    replan_steps: list = field(default_factory=list)   # control-step index of each replan
    plan_latencies_ms: list = field(default_factory=list)   # per planning CALL
    step_latencies_ms: list = field(default_factory=list)   # per control step (sensor->action)
    events: list = field(default_factory=list)
    correspondence: dict = field(default_factory=dict)
    first_failure_step: int | None = None
    believed_progress: float = 0.0


@dataclass
class ExecConfig:
    max_steps_per_skill: int = 60
    max_replans: int = 4
    max_total_steps: int = 600
    use_world_model: bool = True
    event_driven: bool = True
    adapt: bool = True
    allow_recovery: bool = True


class ClosedLoopExecutor:
    def __init__(self, env: AgentEnv, graph: TaskGraph, estimator: StateEstimator,
                 context: DynamicsContext, planning_model: PlanningModel,
                 config: ExecConfig | None = None):
        self.env = env
        self.graph = graph
        self.est = estimator
        self.ctx = context
        self.search = ImaginedSearch(planning_model)
        self.cfg = config or ExecConfig()

    def _plan(self, belief, tr: AgentTrace):
        """Re-resolve roles->tracks against the CURRENT belief (track ids churn
        under occlusion/merge), rebuild the goal+verifier, then select a plan."""
        t0 = time.perf_counter()
        corr = correspond(belief, self.graph.role_signatures)
        tr.correspondence = corr
        verifier = Verifier(ground_goal(self.graph, corr), ground_goal_rel(self.graph, corr))
        base = ground_plan(self.graph, corr)
        if self.cfg.use_world_model:
            plan = self.search.select(belief, base, verifier.goal_satisfied).plan
        else:
            plan = base
        tr.plan_latencies_ms.append(1000 * (time.perf_counter() - t0))
        return list(plan), verifier

    def _replan(self, belief, tr, at_event: str) -> tuple:
        tr.autonomous_replans += 1
        tr.replan_steps.append(tr.steps)
        tr.events.append(f"replan({at_event}) @step{tr.steps}")
        return self._plan(belief, tr)

    def run(self) -> AgentTrace:
        tr = AgentTrace()
        belief = self.est.update(self.env.reset_percept())
        plan, verifier = self._plan(belief, tr)

        idx = 0
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
                try:
                    if si.done(belief):
                        break
                    action = si.act(belief)
                except KeyError:                    # track referenced by skill is gone
                    if self.cfg.allow_recovery and tr.autonomous_replans < self.cfg.max_replans:
                        plan, verifier = self._replan(belief, tr, "lost-track"); idx = 0
                        restart = True
                    break
                if self.cfg.use_world_model and not self.cfg.event_driven:
                    plan, verifier = self._plan(belief, tr)   # every-step replan ablation
                prev = belief
                belief = self.est.update(self.env.step(action))
                if self.cfg.adapt:
                    self.ctx.update(prev, action.target, belief)
                tr.steps += 1; sk_steps += 1
                tr.step_latencies_ms.append(1000 * (time.perf_counter() - t0))

                event = verifier.detect_failure(prev, belief, expected)
                if event:
                    tr.events.append(f"event:{event} @step{tr.steps}")
                    if tr.first_failure_step is None:
                        tr.first_failure_step = tr.steps
                    if self.cfg.allow_recovery and tr.autonomous_replans < self.cfg.max_replans:
                        plan, verifier = self._replan(belief, tr, event.split(":")[0])
                        idx = 0; restart = True
                    break
                if tr.steps >= self.cfg.max_total_steps:
                    idx = len(plan); break

            if restart:
                continue
            idx += 1
            try:
                if verifier.goal_satisfied(belief):
                    break
            except KeyError:
                pass

        try:
            tr.believed_success = verifier.goal_satisfied(belief)
            tr.believed_progress = verifier.progress(belief)
        except KeyError:
            pass
        return tr
