from __future__ import annotations
import hashlib,json
from pathlib import Path
import re
import mujoco

ROOT=Path(__file__).resolve().parents[3]; ASSET=ROOT/'assets/industrial/franka_emika_panda'
def validate():
    files={str(p.relative_to(ASSET)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(x for x in ASSET.rglob('*') if x.is_file())}
    tree=hashlib.sha256('\n'.join(f'{k} {v}' for k,v in files.items()).encode()).hexdigest(); result={'asset_tree_sha256':tree,'file_hashes':files,'checks':{}}
    refs=[]; missing=[]; empty=[]; invalid_mesh=[]
    for xml in ASSET.rglob('*.xml'):
        text=xml.read_text()
        for rel in re.findall(r'(?:mesh|file)="([^"]+\.(?:obj|stl|png|jpg|jpeg))"',text):
            q=(xml.parent/rel).resolve()
            if not q.exists(): q=(xml.parent/'assets'/rel).resolve()
            refs.append(rel)
            if not q.exists(): missing.append(str(q.relative_to(ASSET.parent)))
            elif q.stat().st_size==0: empty.append(str(q.relative_to(ASSET.parent)))
            elif q.suffix=='.obj':
                s=q.read_text(errors='ignore');
                if s.count('\nv ')<3 or s.count('\nf ')<1: invalid_mesh.append(str(q.relative_to(ASSET.parent)))
    result['integrity']={'references':len(refs),'missing':missing,'empty':empty,'invalid_mesh':invalid_mesh}
    try:
        m=mujoco.MjModel.from_xml_path(str(ASSET/'panda.xml')); result['checks']['compile']=True; result['joints']=m.njnt; result['actuators']=m.nu; result['sites']=m.nsite; result['checks']['seven_arm_joints']=all(mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n)>=0 for n in [f'joint{i}' for i in range(1,8)]); result['checks']['gripper_joints']=all(mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n)>=0 for n in ('finger_joint1','finger_joint2')); result['checks']['grasp_site']=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,'grasp_site')>=0; result['checks']['jacobian_site']=result['checks']['grasp_site']
    except Exception as exc: result['checks']['compile']=False; result['error']=str(exc)
    return result
