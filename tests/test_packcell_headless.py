from osc.packcell.physical import PackCell


def test_geometry_layout_can_run_without_renderer(monkeypatch):
    import mujoco
    def fail(*args, **kwargs): raise AssertionError("OpenGL renderer created during geometry search")
    monkeypatch.setattr(mujoco, "Renderer", fail)
    for height in (0.0, 0.05, 0.10, 0.15):
        cell = PackCell(layout={"base_height": height}, render=False)
        cell.reset()
        state = cell.scorer_state()
        assert state["object_position"].shape == (3,)
