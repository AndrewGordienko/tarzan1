"""Stage D: closed-loop execution with event-driven replanning.

Execute only a short horizon of the selected plan, observe, update the world
model's dynamics context (online adaptation), and check the verifier. The
expensive planner (Stage C imagined search) is woken ONLY on an event:
  * a skill's precondition breaks,
  * the verifier detects a failure (drop / lost object),
  * a subgoal completes and the next needs grounding.
Otherwise the cheap skill controllers keep running. This is the "recompute only
on novelty / uncertainty / failure" principle, and it is what keeps latency low.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..compiler.task_graph import TaskGraph
from ..skills.grounding import ground_plan
from ..worldmodel.search import ImaginedSearch
from ..sim.base import SimState
from .verifier import Verifier


@dataclass
class EpisodeResult:
    success: bool = False
    first_attempt_success: bool = False        # succeeded with zero replans
    replans: int = 0                           # proxy for human interventions
    steps: int = 0
    collisions: int = 0
    force_violations: int = 0
    irreversible: int = 0
    planning_latency_s: float = 0.0            # total wall-clock in Stage C
    plan_calls: int = 0
    exec_time_s: float = 0.0
    events: list = field(default_factory=list)
    final_progress: float = 0.0

    @property
    def safety_violations(self) -> int:
        return self.force_violations + self.irreversible

    @property
    def mean_plan_latency_ms(self) -> float:
        return 1000 * self.planning_latency_s / max(1, self.plan_calls)


class ClosedLoopExecutor:
    def __init__(self, backend, graph: TaskGraph, search: ImaginedSearch,
                 max_steps_per_skill: int = 60, max_replans: int = 4,
                 max_total_steps: int = 600):
        self.backend = backend
        self.graph = graph
        self.search = search
        self.verifier = Verifier(graph.goal)
        self.max_steps_per_skill = max_steps_per_skill
        self.max_replans = max_replans
        self.max_total_steps = max_total_steps

    def _plan(self, state: SimState):
        t0 = time.perf_counter()
        base = ground_plan(self.graph)
        best = self.search.select(state, base, self.verifier.goal_satisfied)
        return best, time.perf_counter() - t0

    def run(self) -> EpisodeResult:
        res = EpisodeResult()
        wm = self.search.wm
        t_start = time.perf_counter()

        best, dt = self._plan(self.backend.state())
        res.planning_latency_s += dt; res.plan_calls += 1
        plan = list(best.plan)
        res.events.append(f"plan: {best.breakdown()}")

        idx = 0
        while idx < len(plan):
            si = plan[idx]
            # event: precondition broken -> wake planner
            if not si.skill.precondition(self.backend.state(), si.params):
                res.replans += 1
                res.events.append(f"replan(precond) before {si.label}")
                best, dt = self._plan(self.backend.state())
                res.planning_latency_s += dt; res.plan_calls += 1
                plan, idx = list(best.plan), 0
                if res.replans > self.max_replans:
                    break
                continue

            expected_grasp = si.params.get("object") if si.skill.name in ("move", "place") else None
            steps = 0
            while not si.done(self.backend.state()) and steps < self.max_steps_per_skill:
                prev = self.backend.state()
                action = si.act(prev)
                obs, info = self.backend.step(action)
                cur = self.backend.state()
                wm.update_context(prev, action.target, cur)   # online adaptation
                res.steps += 1; steps += 1
                res.collisions += int(info.collision)
                res.force_violations += int(info.force_violation)
                res.irreversible += int(info.irreversible)

                event = self.verifier.detect_failure(prev, cur, expected_grasp)
                if event:
                    res.events.append(f"event:{event} @step{res.steps}")
                    res.replans += 1
                    if res.replans > self.max_replans:
                        idx = len(plan); break
                    best, dt = self._plan(cur)
                    res.planning_latency_s += dt; res.plan_calls += 1
                    plan, idx = list(best.plan), 0
                    steps = -1  # signal outer loop to restart from new plan
                    break
                if res.steps >= self.max_total_steps:
                    idx = len(plan); break

            if steps == -1:
                continue
            idx += 1
            if self.verifier.goal_satisfied(self.backend.state()):
                break

        res.exec_time_s = time.perf_counter() - t_start
        res.success = self.verifier.goal_satisfied(self.backend.state())
        res.first_attempt_success = res.success and res.replans == 0
        res.final_progress = self.verifier.progress(self.backend.state())
        return res
