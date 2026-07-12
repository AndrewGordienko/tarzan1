from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class PhaseLadder:
    reach_success: bool = False
    opposing_contact: bool = False
    lift_success: bool = False
    retained_grasp: bool = False
    transport_success: bool = False
    released: bool = False
    stable_inside_box: bool = False


class OraclePackController:
    """Scripted actuator-only controller using an oracle object pose input."""

    def __init__(self, cell, object_pose):
        self.cell = cell
        self.object_pose = np.asarray(object_pose, dtype=float).copy()
        self.m = cell.mujoco
        self.site = self.m.mj_name2id(cell.model, self.m.mjtObj.mjOBJ_SITE, "grasp_site")
        self.arm_dof = np.array([cell.model.jnt_dofadr[self.m.mj_name2id(cell.model, self.m.mjtObj.mjOBJ_JOINT, n)]
                                 for n in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")])
        self.arm_q = np.array([cell.model.jnt_qposadr[self.m.mj_name2id(cell.model, self.m.mjtObj.mjOBJ_JOINT, n)]
                               for n in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")])
        self.ladder = PhaseLadder()
        self.gripper_open = 1.2; self.gripper_closed = -0.17

    def _ik(self, target, grip):
        err = np.asarray(target) - self.cell.ee_position()
        jac = np.zeros((3, self.cell.model.nv)); self.m.mj_jacSite(self.cell.model, self.cell.data, jac, None, self.site)
        J = jac[:, self.arm_dof]
        dq = J.T @ np.linalg.solve(J @ J.T + .08**2 * np.eye(3), err) * .5
        q = self.cell.controller_state()["joint_position"].copy()
        q[:5] = self.cell.data.qpos[self.arm_q] + np.clip(dq, -.06, .06); q[5] = grip
        return np.clip(q, self.cell.actuator_ranges()[:, 0], self.cell.actuator_ranges()[:, 1])

    def _contacts(self):
        obj = self.cell.model.geom("cube_red").id
        gripper_body = self.cell.model.body("gripper").id
        moving_body = self.cell.model.body("moving_jaw_so101_v1").id
        fingers = {i for i in range(self.cell.model.ngeom)
                   if self.cell.model.geom_bodyid[i] in {gripper_body, moving_body} and i >= 29}
        hits=[]; forces=[]
        for c in self.cell.data.contact:
            if (c.geom1 == obj and c.geom2 in fingers) or (c.geom2 == obj and c.geom1 in fingers):
                f=np.zeros(6); self.m.mj_contactForce(self.cell.model,self.cell.data,int(list(self.cell.data.contact).index(c)),f); hits.append((c.geom1,c.geom2)); forces.append(abs(float(f[0])))
        return hits, forces

    def _metrics(self, target):
        jac=np.zeros((3,self.cell.model.nv)); self.m.mj_jacSite(self.cell.model,self.cell.data,jac,None,self.site); J=jac[:,self.arm_dof]
        q=self.cell.controller_state()["joint_position"]; return {"desired":np.asarray(target).tolist(),"actual":self.cell.ee_position().tolist(),"error":(np.asarray(target)-self.cell.ee_position()).tolist(),"condition_number":float(np.linalg.cond(J)),"damping":.08,"joint_update_norm":float(np.linalg.norm(self._ik(target,self.gripper_open)[:5]-q[:5])),"saturation":int(np.sum((q<=self.cell.actuator_ranges()[:,0]+1e-3)|(q>=self.cell.actuator_ranges()[:,1]-1e-3)))}

    def run(self, max_steps=500):
        phases=[("move_pregrasp", np.array([*self.object_pose[:2], .19]), self.gripper_open, 90),
                ("move_insertion", self.object_pose, self.gripper_open, 120),
                ("close_gripper", self.object_pose, self.gripper_closed, 80),
                ("lift", np.array([*self.object_pose[:2], .20]), self.gripper_closed, 100),
                ("transport", np.array([.13,-.075,.20]), self.gripper_closed, 100),
                ("lower", np.array([.13,-.075,.12]), self.gripper_closed, 80),
                ("release", np.array([.13,-.075,.12]), self.gripper_open, 50),
                ("retreat", np.array([.13,-.075,.20]), self.gripper_open, 40)]
        steps=0; trace=[]; insertion_reached=False
        for name,target,grip,count in phases:
            for _ in range(count):
                self.cell.step_control(self._ik(target,grip)); steps += 1
                hits, forces = self._contacts()
                if name == "move_pregrasp" and np.linalg.norm(self.cell.ee_position()-target) < .005: self.ladder.reach_success=True
                if name == "move_insertion" and np.linalg.norm(self.cell.ee_position()-target) < .005: insertion_reached=True
                if name == "close_gripper" and insertion_reached and len(set(h[0] for h in hits+[(b,a) for a,b in hits])) >= 2 and max(forces or [0]) > .01: self.ladder.opposing_contact=True
                if name == "lift" and self.cell.ee_position()[2] > .16: self.ladder.lift_success=True
                if name == "lift" and self.ladder.opposing_contact and self.cell.ee_position()[2] > .16: self.ladder.retained_grasp=True
                if name == "transport" and np.linalg.norm(self.cell.ee_position()[:2]-np.array([.13,-.075])) < .025: self.ladder.transport_success=True
                if name == "release" and grip > .7: self.ladder.released=True
            if name in {"move_pregrasp","move_insertion","close_gripper"}:
                trace.append({"phase":name,"metrics":self._metrics(target),"transition":"reached" if (name!="move_insertion" or insertion_reached) else "timeout"})
            if name == "move_insertion" and not insertion_reached:
                return {"phases":asdict(self.ladder),"steps":steps,"success":False,"failure_phase":"insertion_reach","trace":trace}
        self.ladder.stable_inside_box=self.cell.verify().success
        return {"phases":asdict(self.ladder),"steps":steps,"success":self.ladder.stable_inside_box,"failure_phase":None if self.ladder.stable_inside_box else "grasp_contact","trace":trace}
