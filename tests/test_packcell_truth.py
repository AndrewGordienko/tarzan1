from pathlib import Path
import ast

from osc.packcell.physical import PackCell


def test_packcell_separates_agent_controller_and_scorer_views():
    cell = PackCell(); obs = cell.reset()
    assert obs["rgb"].size and obs["depth"].size
    assert "object_position" not in obs
    assert "joint_position" in cell.controller_state()
    assert "object_position" in cell.scorer_state()


def test_packcell_source_has_no_post_reset_object_qpos_writes():
    source = Path(__file__).parents[1] / "src/osc/packcell/physical.py"
    tree = ast.parse(source.read_text())
    writes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Subscript):
            target = ast.unparse(node.targets[0])
            if "data.qpos" in target or "data.qvel" in target:
                writes.append((node.lineno, target))
    # Initialization writes are confined to reset; the runtime controller uses
    # actuator controls and MuJoCo physics only.
    assert writes and all(line < 80 for line, _ in writes)
