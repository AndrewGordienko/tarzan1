from __future__ import annotations
from pathlib import Path
import tempfile
import numpy as np
import mujoco, imageio.v2 as imageio
ROOT=Path(__file__).resolve().parents[1]; OVER=ROOT/'assets/industrial/derived/franka_panda_tarzan/panda_grasp_overlay.xml'; OUT=ROOT/'artifacts/panda_actuator_motion_demo.mp4'

def make_scene(path):
    xml=OVER.read_text().replace('<worldbody>','<visual><global offwidth="1280" offheight="720"/></visual><worldbody>')
    extra='''<geom name="table" type="box" size=".45 .35 .03" pos=".45 0 .03" rgba=".65 .65 .68 1"/>
    <body name="parcel" pos=".55 0 .10"><freejoint name="parcel_free"/><geom name="parcel" type="box" size=".06 .05 .07" mass=".4" rgba=".15 .45 .85 1"/></body>
    <body name="box" pos=".35 -.20 .08"><geom name="box_floor" type="box" size=".16 .12 .01" rgba=".55 .32 .15 1"/><geom name="box_back" type="box" pos="0 .12 .08" size=".16 .01 .08" rgba=".55 .32 .15 1"/><geom name="box_left" type="box" pos="-.16 0 .08" size=".01 .12 .08" rgba=".55 .32 .15 1"/><geom name="box_right" type="box" pos=".16 0 .08" size=".01 .12 .08" rgba=".55 .32 .15 1"/></body>
    <camera name="demo_camera" pos="1.25 -1.35 .95" xyaxes=".75 .66 0 -.28 .32 -.90"/>
'''
    path.write_text(xml.replace('</worldbody>',extra+'</worldbody>'))

def main():
    OUT.parent.mkdir(parents=True,exist_ok=True)
    scene=ROOT/'assets/industrial/derived/franka_panda_tarzan/panda_motion_scene.xml'; make_scene(scene)
    try:
        m=mujoco.MjModel.from_xml_path(str(scene)); d=mujoco.MjData(m); r=mujoco.Renderer(m,height=720,width=1280)
        sid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,'grasp_site'); jids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f'joint{i}') for i in range(1,8)]; qadr=[m.jnt_qposadr[j] for j in jids]; dofs=[m.jnt_dofadr[j] for j in jids]
        d.qpos[:7]=[0,-.5,0,-2.0,0,1.5,.7]; mujoco.mj_forward(m,d); frames=[]; phases=[]
        def move(target,n,label):
            for _ in range(n):
                jac=np.zeros((3,m.nv)); mujoco.mj_jacSite(m,d,jac,None,sid); J=jac[:,dofs]; err=np.asarray(target)-d.site_xpos[sid]; dq=J.T@np.linalg.solve(J@J.T+.05**2*np.eye(3),err)*.7; q=d.qpos[qadr].copy(); q+=np.clip(dq,-.04,.04); d.ctrl[:7]=q; d.ctrl[7]=255; mujoco.mj_step(m,d); r.update_scene(d,camera='demo_camera'); frames.append(r.render().copy()); phases.append(label)
        move((.55,0,.32),120,'move_pregrasp'); move((.55,0,.18),100,'move_insertion'); move((.35,-.20,.30),150,'move_retreat'); move((.35,-.20,.22),100,'move_box_above'); move((.35,-.20,.30),100,'verify_retreat')
        imageio.mimsave(OUT,frames,fps=20,codec='libx264',macro_block_size=1)
    finally:
        scene.unlink(missing_ok=True)
    print(OUT,'frames=',len(frames),'duration=',len(frames)/20)
if __name__=='__main__': main()
