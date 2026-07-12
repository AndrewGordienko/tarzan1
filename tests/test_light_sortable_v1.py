from pathlib import Path
import inspect
import mujoco
from osc.packcell.light_sortable_pipeline import compile_item_execution
from osc.packcell.ur10e_adapter import UR10eAdapter
from osc.robot_api import ActuatorCommand

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

def test_light_sortable_reset_has_no_prohibited_robot_contact():
    robot=UR10eAdapter(telemetry=True);robot.reset()
    obs=robot.observe()
    robot.step(ActuatorCommand(obs.joint_position,(.075,-.075),'reset_audit'))
    assert not [c for c in robot.telemetry[-1]['contacts'] if c['prohibited']]

def test_transport_telemetry_has_one_gripper_writer_and_named_classes():
    robot=UR10eAdapter(telemetry=True);robot.reset();obs=robot.observe()
    robot.step(ActuatorCommand(obs.joint_position,(.075,-.075),'telemetry_audit'))
    row=robot.telemetry[-1]
    assert row['gripper_writer']=='UR10eAdapter.step/ActuatorCommand'
    assert all('collision_class' in c and 'body1' in c and 'body2' in c for c in row['contacts'])
