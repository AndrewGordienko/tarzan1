"""The benchmark metric suite.

Deliberately NOT average action error. These are the deployment-oriented numbers
from the research plan, aggregated over randomized episodes with a fixed seed set.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..execution.loop import EpisodeResult

# Assumed low-level control rate: converts simulated steps -> physical seconds so
# that "operating hours" and "completion time" are physical, not wall-clock (the
# toy sim runs far faster than real time). Planning latency stays wall-clock,
# since that is genuine compute cost.
CONTROL_HZ = 20.0


@dataclass
class BenchmarkReport:
    n_episodes: int
    demos_required: int
    unseen_env_success: float
    first_attempt_success: float
    eventual_success: float
    recovery_rate: float                 # of episodes needing a replan, how many still succeeded
    interventions_per_hour: float        # replans normalized to execution wall-clock
    mean_plan_latency_ms: float
    p95_plan_latency_ms: float
    mean_completion_time_s: float
    safety_violations_per_episode: float
    cost_per_success: float              # sim-steps + planning, per successful episode

    def pretty(self) -> str:
        return "\n".join([
            "=" * 58,
            "  BENCHMARK REPORT  (no fine-tuning after demonstration)",
            "=" * 58,
            f"  episodes                     : {self.n_episodes}",
            f"  demonstrations required      : {self.demos_required}",
            f"  unseen-environment success   : {self.unseen_env_success:6.1%}",
            f"  first-attempt success        : {self.first_attempt_success:6.1%}",
            f"  eventual success (w/ recovery): {self.eventual_success:6.1%}",
            f"  recovery rate                : {self.recovery_rate:6.1%}",
            f"  interventions / op-hour      : {self.interventions_per_hour:6.2f}",
            f"  mean planning latency        : {self.mean_plan_latency_ms:6.1f} ms",
            f"  p95  planning latency        : {self.p95_plan_latency_ms:6.1f} ms",
            f"  mean completion time         : {self.mean_completion_time_s:6.3f} s",
            f"  safety violations / episode  : {self.safety_violations_per_episode:6.2f}",
            f"  cost per success (steps)     : {self.cost_per_success:6.1f}",
            "=" * 58,
        ])


def aggregate(results: list[EpisodeResult], demos_required: int = 1) -> BenchmarkReport:
    n = len(results)
    succ = [r for r in results if r.success]
    needed_recovery = [r for r in results if r.replans > 0]
    recovered = [r for r in needed_recovery if r.success]
    total_sim_h = sum(r.steps / CONTROL_HZ for r in results) / 3600.0
    total_replans = sum(r.replans for r in results)
    plan_latencies = [r.mean_plan_latency_ms for r in results]
    total_cost = sum(r.steps + r.plan_calls * 5 for r in results)

    return BenchmarkReport(
        n_episodes=n,
        demos_required=demos_required,
        unseen_env_success=len(succ) / max(1, n),
        first_attempt_success=sum(r.first_attempt_success for r in results) / max(1, n),
        eventual_success=len(succ) / max(1, n),
        recovery_rate=len(recovered) / max(1, len(needed_recovery)),
        interventions_per_hour=total_replans / max(1e-6, total_sim_h),
        mean_plan_latency_ms=float(np.mean(plan_latencies)) if plan_latencies else 0.0,
        p95_plan_latency_ms=float(np.percentile(plan_latencies, 95)) if plan_latencies else 0.0,
        mean_completion_time_s=float(np.mean([r.steps / CONTROL_HZ for r in results])) if n else 0.0,
        safety_violations_per_episode=sum(r.safety_violations for r in results) / max(1, n),
        cost_per_success=total_cost / max(1, len(succ)),
    )
