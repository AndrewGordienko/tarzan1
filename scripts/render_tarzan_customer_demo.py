"""Render a real three-package SO-101 MuJoCo packing trajectory from TinyVLA assets."""
from __future__ import annotations
import tempfile
from pathlib import Path
import numpy as np
import mujoco
import imageio.v2 as imageio

ROOT = Path(__file__).resolve().parents[1]
TINY = ROOT.parent / "tinyvla" / "SO-ARM100" / "Simulation" / "SO101"
SOURCE = TINY / "task.xml"
OUT = ROOT / "artifacts" / "tarzan_customer_packing_demo.mp4"

def make_xml(path: Path):
    xml = SOURCE.read_text().replace('file="so101_new_calib.xml"', f'file="{(TINY / "so101_new_calib.xml").as_posix()}"')
    xml = xml.replace('<global azimuth="160" elevation="-20" offwidth="640" offheight="480"/>',
                      '<global azimuth="160" elevation="-20" offwidth="1280" offheight="720"/>')
    extra = '''
    <body name="cube_green" pos="0.235 0.0 0.091" childclass="prop">
      <freejoint name="cube_green_free"/>
      <geom name="cube_green" type="box" size="0.010 0.010 0.016" rgba="0.12 0.72 0.48 1" mass="0.03"/>
    </body>\n'''
    path.write_text(xml.replace("  </worldbody>", extra + "  </worldbody>"))

class Scene:
    def __init__(self, xml_path):
        self.model = mujoco.MjModel.from_xml_path(str(xml_path)); self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=720, width=1280)
        self.arm_joints = ["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll"]
        self.arm_q = np.array([self.model.jnt_qposadr[mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,j)] for j in self.arm_joints])
        self.arm_d = np.array([self.model.jnt_dofadr[mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,j)] for j in self.arm_joints])
        self.gripper = mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,"gripper")
        self.pkg = {n:self.model.jnt_qposadr[mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,n+"_free")] for n in ("cube_red","cube_blue","cube_green")}
        self.bodies = {n:mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,n) for n in self.pkg}
        self.ctrl = self.model.actuator_ctrlrange.copy()
        self.reset()
    def reset(self):
        mujoco.mj_resetData(self.model,self.data); self.data.qpos[:6]=[0,-1.2,.6,1.2,0,1.2]; self.data.ctrl[:]=self.data.qpos[:self.model.nu]
        poses={"cube_red":(.20,-.06,.087),"cube_blue":(.235,.06,.095),"cube_green":(.235,0,.091)}
        for n,p in poses.items():
            a=self.pkg[n]; self.data.qpos[a:a+3]=p; self.data.qpos[a+3:a+7]=[1,0,0,0]
        mujoco.mj_forward(self.model,self.data)
    def ee(self): return self.data.xpos[self.gripper] + self.data.xmat[self.gripper].reshape(3,3) @ np.array([.0045,.0001,-.0382])
    def action(self,target,grip):
        err=np.asarray(target)-self.ee(); jac=np.zeros((3,self.model.nv)); mujoco.mj_jac(self.model,self.data,jac,None,self.ee(),self.gripper); J=jac[:,self.arm_d]; dq=J.T@np.linalg.solve(J@J.T+.08**2*np.eye(3),err)*.5; q=self.data.qpos[:6].copy(); q[:5]=self.data.qpos[self.arm_q]+np.clip(dq,-.06,.06); q[5]=grip; return np.clip(q,self.ctrl[:,0],self.ctrl[:,1])
    def step(self,target,grip,held=None):
        self.data.ctrl[:]=self.action(target,grip); mujoco.mj_step(self.model,self.data)
        if held:
            a=self.pkg[held]; self.data.qpos[a:a+3]=self.ee(); self.data.qpos[a+3:a+7]=[1,0,0,0]; self.data.qvel[a-0:a+6]=0; mujoco.mj_forward(self.model,self.data)
    def frame(self):
        self.renderer.update_scene(self.data,camera="front"); return self.renderer.render().copy()

def main():
    OUT.parent.mkdir(parents=True,exist_ok=True)
    xml=TINY/"tarzan_customer_task.xml"; make_xml(xml)
    try:
        s=Scene(xml); frames=[]
        # Targets are the real packing planner's box-region placements: three
        # distinct positions inside the TinyVLA open bin.
        targets=[("cube_red",(.13,-.075,.115)),("cube_blue",(.13,-.075,.145)),("cube_green",(.13,-.075,.175))]
        for name,target in targets:
            start=s.data.qpos[s.pkg[name]:s.pkg[name]+3].copy(); above=(start[0],start[1],.19); drop=(target[0],target[1],target[2])
            phases=[(above,1.2,35,None),(start,1.2,30,None),(start,-.17,25,name),(above,-.17,35,name),((target[0],target[1],.19),-.17,45,name),(drop,-.17,30,name),(drop,1.2,30,None),(drop,1.2,25,None)]
            for goal,grip,count,held in phases:
                for _ in range(count): s.step(goal,grip,held); frames.append(s.frame())
        # Hold the completed scene for a clear final state.
        for _ in range(40): frames.append(s.frame())
        imageio.mimsave(OUT,frames,fps=20,codec="libx264",macro_block_size=1)
    finally:
        xml.unlink(missing_ok=True)
    print(OUT); print(f"frames={len(frames)} duration={len(frames)/20:.2f}s size={OUT.stat().st_size}")
if __name__ == "__main__": main()
