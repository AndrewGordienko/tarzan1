"""Four paired ambiguity scenarios that exercise the ResolutionPolicy's action
selection, each hiding its discriminating evidence behind exactly the action that
should reveal it:

  1. noisy_identifiable  -- candidates differ in SIZE, but single-shot size is
     noisy. PASSIVE INSPECTION (average frames) resolves it; no human.
  2. occluded            -- sizes equal, a discriminating LABEL is hidden until a
     CHANGE_VIEWPOINT.
  3. interaction         -- look identical, differ in MASS; only a PROBE reveals it.
  4. fundamental         -- no physical observation distinguishes them (identical
     size/label/mass); only CLARIFICATION / SKU METADATA resolves it. Inspection
     must NOT pretend to.

This is a focused, deterministic test of the POLICY (does it pick the action that
carries information, and refuse to fake genuine ambiguity?) on an explicit evidence
model. Folding viewpoint/probe into the full ToyTabletopSim perception pipeline is
the follow-up; this proves the decision logic first.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from ..execution.resolution import ResolutionConfig, ResolutionPolicy, TaskContext

SIZE_NOISE = 0.006      # per-observation size measurement std (m)
COMMIT_P = 0.90         # posterior mass on one candidate required to commit
SCENARIOS = ("noisy_identifiable", "occluded", "interaction", "fundamental")


@dataclass
class Candidate:
    correct: bool
    size: float
    label: int
    mass: float
    sku: int


def make_scene(kind: str, seed: int):
    """Two candidates for one role; exactly one matches the demonstrated target."""
    rng = np.random.default_rng(seed + 4242)
    ci = int(rng.integers(0, 2))                       # which index is correct
    # defaults: identical on every channel (the fundamental case)
    base = dict(size=0.045, label=1, mass=0.2, sku=1)
    other = dict(base)
    if kind == "noisy_identifiable":
        base["size"], other["size"] = 0.052, 0.038     # a real, resolvable size gap
    elif kind == "occluded":
        other["label"] = 0                             # discriminating label, viewpoint-gated
    elif kind == "interaction":
        other["mass"] = 0.02                           # discriminating mass, probe-gated
    elif kind == "fundamental":
        other["sku"] = 2                               # only identity differs
    cands = [None, None]
    cands[ci] = Candidate(True, **base)
    cands[1 - ci] = Candidate(False, **other)
    return _Scene(kind, cands, ci, rng)


class _Scene:
    def __init__(self, kind, cands, correct_idx, rng):
        self.kind, self.cands, self.correct_idx, self.rng = kind, cands, correct_idx, rng
        self.target = cands[correct_idx]               # demo established this signature
        self.revealed = {"label": False, "mass": False, "sku": False}
        self.samples = [[] for _ in cands]

    # -- actions gather evidence about specific channels --
    def observe(self):
        for i, c in enumerate(self.cands):
            self.samples[i].append(c.size + float(self.rng.normal(0, SIZE_NOISE)))
    def change_viewpoint(self): self.revealed["label"] = True
    def probe(self): self.revealed["mass"] = True
    def request_metadata(self): self.revealed["sku"] = True

    def posterior(self):
        logp = [0.0 for _ in self.cands]
        # SIZE only discriminates when the candidates are statistically SEPARABLE.
        # Two objects of equal size stay a coin-flip however long you look -- so
        # more observation must never manufacture confidence out of measurement
        # noise (that would let inspection fake the fundamental case).
        if self.samples[0]:
            N = len(self.samples[0])
            means = [sum(s) / len(s) for s in self.samples]
            sem = SIZE_NOISE / math.sqrt(N)
            if abs(means[0] - means[1]) > 3 * sem:     # 3-sigma separable
                for i in range(len(self.cands)):
                    logp[i] += -((means[i] - self.target.size) ** 2) / (2 * sem ** 2)
        for i, c in enumerate(self.cands):             # hard channels, once revealed
            if self.revealed["label"] and c.label != self.target.label: logp[i] += -50.0
            if self.revealed["mass"] and abs(c.mass - self.target.mass) > 1e-6: logp[i] += -50.0
            if self.revealed["sku"] and c.sku != self.target.sku: logp[i] += -50.0
        m = max(logp)
        ps = [math.exp(x - m) for x in logp]
        z = sum(ps)
        return [p / z for p in ps]

    def best(self):
        p = self.posterior()
        idx = max(range(len(p)), key=lambda i: p[i])
        return p[idx], idx


def resolve_one(kind: str, seed: int, cfg: ResolutionConfig):
    """Drive the ResolutionPolicy against one scene; report how it was resolved."""
    scene = make_scene(kind, seed)
    policy = ResolutionPolicy(cfg)
    tried, insp, clar, last_gain = set(), 0, 0, None
    human = abstained = False
    p, idx = scene.best()
    for _ in range(20):                                # safety bound
        # single contested role; its confidence == current posterior mass.
        ra = SimpleNamespace(per_role_conf={"role0": p})
        action = policy.decide(ra, TaskContext(), insp, clar, last_gain, tried)
        if action.kind == "commit":
            break
        if action.kind == "observe":
            prev = p
            for _ in range(cfg.inspect_frames):
                scene.observe()
            p, idx = scene.best()
            last_gain = p - prev
            insp += 1
        elif action.kind in ("change_viewpoint", "probe", "request_metadata"):
            getattr(scene, action.kind)()
            tried.add(action.kind)
            p, idx = scene.best()
            last_gain = None                           # new modality: let observe retry too
        elif action.kind == "ask_user":
            idx, p, clar, human = scene.correct_idx, 1.0, clar + 1, True
            break
        else:                                          # abstain -- never guess
            abstained = True
            break
    committed = not abstained
    return dict(kind=kind, committed=committed, correct=committed and idx == scene.correct_idx,
                human=human, abstained=abstained, inspections=insp, clarifications=clar,
                used_physical=tuple(sorted(tried)))


def _cfg(**over):
    base = dict(allow_inspection=True, allow_clarification=False, commit_threshold=COMMIT_P,
                max_inspections=4, inspect_frames=3)
    base.update(over)
    return ResolutionConfig(**base)


CONFIGS = {
    "inspection-only": _cfg(),
    "+viewpoint": _cfg(allow_viewpoint=True),
    "+probe": _cfg(allow_probe=True),
    "+metadata": _cfg(allow_metadata=True),
    "clarification": _cfg(allow_clarification=True),
    "full": _cfg(allow_viewpoint=True, allow_probe=True, allow_metadata=True,
                 allow_clarification=True),
}


def run_scenarios(seeds=range(40)):
    """Returns {scenario: {config: {autonomous, human, abstain, correct}}}."""
    out = {}
    for kind in SCENARIOS:
        out[kind] = {}
        for cname, cfg in CONFIGS.items():
            rs = [resolve_one(kind, s, cfg) for s in seeds]
            n = len(rs)
            out[kind][cname] = dict(
                autonomous_correct=sum(r["correct"] and not r["human"] for r in rs) / n,
                human=sum(r["human"] for r in rs) / n,
                abstain=sum(r["abstained"] for r in rs) / n,
                correct=sum(r["correct"] for r in rs) / n)
    return out
