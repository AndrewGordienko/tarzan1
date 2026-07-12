from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json

@dataclass(frozen=True)
class IndustrialArmContract:
    asset_family: str
    asset_sha256: str | None
    mesh_sha256: str | None
    joints: tuple[str, ...]
    actuators: tuple[str, ...]
    grasp_site: str
    finger_collision_geoms: tuple[str, ...]
    payload_limit_kg: float | None
    aperture_m: float | None
    force_limit_n: float | None

def load_contract(path=None):
    p=Path(path or Path(__file__).resolve().parents[3]/'configs'/'industrial_arm_panda_v1.json')
    x=json.loads(p.read_text()); g=x['gripper']; a=x['asset']
    return IndustrialArmContract(a['family'],a['asset_sha256'],a['mesh_sha256'],tuple(x['joints']),tuple(x['actuators']),g['grasp_site'],tuple(g['finger_collision_geoms']),x['payload_limit_kg'],g['aperture_m'],g['force_limit_n'])
