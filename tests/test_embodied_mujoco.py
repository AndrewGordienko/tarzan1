import numpy as np

from osc.embodied import MujocoPackingAdapter, SkillCommand


def test_real_mujoco_observation_and_scripted_place_boundary():
    adapter = MujocoPackingAdapter()
    initial = adapter.reset()
    assert initial.rgb is not None and initial.rgb.size
    assert initial.depth is not None and initial.depth.size
    assert initial.masks["ordinary"].any()
    grasp = adapter.execute(SkillCommand("grasp", {"name": "ordinary"}))
    assert grasp.success
    assert grasp.contact_events, "grasp must expose simulator contact events"
    placed = adapter.execute(SkillCommand("place", {"name": "ordinary"},
                                          {"position": (0.0, 0.0, 0.14)}))
    assert placed.success
    assert placed.observation.masks["ordinary"].any()
    assert not hasattr(placed.observation, "qpos")
