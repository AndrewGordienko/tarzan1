from __future__ import annotations
import hashlib,json
from pathlib import Path
import mujoco

ROOT=Path(__file__).resolve().parents[3]; ASSET=ROOT/'assets/industrial/franka_emika_panda'
def validate():
    files={str(p.relative_to(ASSET)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(x for x in ASSET.rglob('*') if x.is_file())}
    tree=hashlib.sha256('\n'.join(f'{k} {v}' for k,v in files.items()).encode()).hexdigest(); result={'asset_tree_sha256':tree,'file_hashes':files,'checks':{}}
    try:
        m=mujoco.MjModel.from_xml_path(str(ASSET/'panda.xml')); result['checks']['compile']=True; result['joints']=m.njnt; result['actuators']=m.nu; result['sites']=m.nsite
    except Exception as exc: result['checks']['compile']=False; result['error']=str(exc)
    return result
