from pathlib import Path
import inspect
import mujoco
from osc.packcell.light_sortable_pipeline import compile_item_execution
from osc.packcell.ur10e_adapter import UR10eAdapter

def test_generated_proxy_compiles_and_resolves_contract():
    robot=UR10eAdapter();robot.reset()
    assert robot.model.nu==8
    assert mujoco.mj_name2id(robot.model,mujoco.mjtObj.mjOBJ_SITE,'grasp_site')>=0
    assert robot.model.actuator_forcerange[6,1]==62.5
    assert robot.model.actuator_forcerange[7,1]==62.5

def test_pipeline_selects_axis_and_planner_placement():
    result=compile_item_execution((.12,.08,.06),1.0)
    assert result['status']=='execute'
    assert result['selected_grasp_axis']==2
    assert result['placement']

def test_unsupported_items_abstain():
    assert compile_item_execution((.12,.08,.06),2.1)['reason'].endswith('overweight')
    assert compile_item_execution((.20,.18,.16),1.)['reason'].endswith('aperture')

def test_adapter_has_no_post_reset_object_write_in_step():
    source=inspect.getsource(UR10eAdapter.step)
    assert 'qpos' not in source and 'qvel' not in source
