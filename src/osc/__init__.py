"""One-Shot Task Compiler (OSC).

A latency-first, modular pipeline that watches one demonstration, compiles it into
an object-centric task graph, grounds it onto reusable skill experts, searches
alternatives in an action-conditioned world model, and executes closed-loop with
event-driven replanning -- no gradient updates after the demonstration.
"""
__version__ = "0.0.1"
