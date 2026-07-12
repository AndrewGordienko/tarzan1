from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import numpy as np

from .physical import PackCell
from .controller import OraclePackController


@dataclass
class GraspDiagnostics:
    seed: int
    mode: str
    success: bool
    aperture_start: float
    aperture_end: float
    command_start: float
    command_end: float
    contact_pairs: list
    normal_forces: list
    position_error: float
    orientation_error: float
    table_support_contacts: int
    joint_limit_saturation: int
    object_height: float
    failure: str | None


def _contacts(cell):
    m, d = cell.model, cell.data
    obj = m.geom("cube_red").id
    gripper = m.body("gripper").id
    finger_ids = {i for i in range(m.ngeom) if m.geom_bodyid[i] == gripper}
    names=[]; forces=[]; table=0
    for i,c in enumerate(d.contact):
        f=np.zeros(6); cell.mujoco.mj_contactForce(m,d,i,f)
        pair=(m.geom(int(c.geom1)).name, m.geom(int(c.geom2)).name)
        if (c.geom1 == obj and c.geom2 in finger_ids) or (c.geom2 == obj and c.geom1 in finger_ids):
            names.append(pair); forces.append(float(abs(f[0])))
        if c.geom1 == m.geom("table").id or c.geom2 == m.geom("table").id: table += 1
    return names,forces,table


def run_one(seed, mode):
    cell=PackCell(seed); cell.reset(); ctrl=OraclePackController(cell,cell.scorer_state()["object_position"])
    q0=cell.controller_state()["joint_position"].copy(); aperture_start=float(q0[5]); cmd_start=float(cell.controller_state()["actuator_control"][5])
    if mode == "empty_close":
        for _ in range(100): cell.step_control(ctrl._ik(cell.ee_position(),ctrl.gripper_closed))
    elif mode == "centered_close":
        for _ in range(100): cell.step_control(ctrl._ik(ctrl.object_pose,ctrl.gripper_closed))
    else:
        result=ctrl.run(max_steps=620)
    q=cell.controller_state()["joint_position"]; pairs,forces,table=_contacts(cell)
    sat=int(np.sum((q <= cell.actuator_ranges()[:,0]+1e-3)|(q >= cell.actuator_ranges()[:,1]-1e-3)))
    err=float(np.linalg.norm(cell.ee_position()-ctrl.object_pose))
    success=bool(pairs and max(forces)>0.01) if mode in {"centered_close","empty_close"} else bool(result["phases"]["retained_grasp"])
    return asdict(GraspDiagnostics(seed,mode,success,aperture_start,float(q[5]),cmd_start,float(cell.controller_state()["actuator_control"][5]),[list(x) for x in pairs],forces,err,0.0,table,sat,float(cell.scorer_state()["object_position"][2]),None if success else "no_opposing_contact"))


def run_grasp_isolation(seeds=range(20)):
    modes=("empty_close","centered_close","centered_lift","randomized_oracle_grasp")
    rows=[run_one(s,m) for m in modes for s in seeds]
    return {"episodes":len(rows),"modes":{m:{"count":len([r for r in rows if r["mode"]==m]),"successes":sum(r["success"] for r in rows if r["mode"]==m)} for m in modes},"rows":rows}
