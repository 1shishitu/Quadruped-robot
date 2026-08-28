"""Single-leg lift — FL swing IK; stance: Balance QP + WBC (−JᵀF), Kp=0."""

from __future__ import annotations

import numpy as np

from quadruped.controllers.low_level import (
    AttitudeStandAssist,
    BalanceController,
    JointController,
    WBCController,
)
from quadruped.models.leg_ik import LegIK
from quadruped.config_loader import gait_control_block
from quadruped.planners.fl_lift_planner import FlLiftPlanner
from quadruped.planners.trunk_planner import TrunkPlanner
from quadruped.sim.mujoco_robot import joint_gravity_torques, read_foot_positions
from quadruped.types import JointCommand, RobotState


class FlLiftController:
    """Stance (3 or 4 feet): Balance + WBC. Swing: foot IK + joint PD only."""

    def __init__(
        self,
        robot_cfg: dict,
        *,
        planner: FlLiftPlanner,
        balance: BalanceController,
        wbc: WBCController,
        trunk_planner: TrunkPlanner,
        joint: JointController,
        leg_ik: LegIK,
        attitude: AttitudeStandAssist | None = None,
        swing_kp: np.ndarray | None = None,
        swing_kd: np.ndarray | None = None,
        joint_kd: np.ndarray | None = None,
        use_balance: bool = True,
        swing_leg: str = "FL",
        gravity_compensation: bool = True,
    ) -> None:
        self.robot_cfg = robot_cfg
        self.planner = planner
        self.balance = balance
        self.wbc = wbc
        self.trunk_planner = trunk_planner
        self.joint = joint
        self.leg_ik = leg_ik
        self.attitude = attitude
        self.swing_leg = swing_leg
        self.use_balance = use_balance
        self.gravity_compensation = gravity_compensation
        self._joint_kd = joint_kd if joint_kd is not None else np.tile(
            np.asarray([2.0, 2.0, 2.0], dtype=float), 4
        )
        self._swing_kp = swing_kp if swing_kp is not None else np.tile(
            np.asarray([40.0, 60.0, 60.0], dtype=float), 4
        )
        self._swing_kd = swing_kd if swing_kd is not None else np.tile(
            np.asarray([2.0, 3.0, 3.0], dtype=float), 4
        )

    def begin(self, state: RobotState, foot_pos: dict[str, np.ndarray]) -> None:
        height = float(state.base_pos[2])
        self.trunk_planner.default_height = height
        self.trunk_planner.preview_horizon = 0.0
        self.trunk_planner.reset(
            p_com=np.array([state.base_pos[0], state.base_pos[1], height]),
            yaw=float(state.base_rpy[2]),
        )
        self.planner.set_stance_positions(
            foot_pos,
            base_pos=state.base_pos,
            base_rpy=state.base_rpy,
        )
        self.balance.reset()

    def compute(
        self,
        model,
        data,
        state: RobotState,
        *,
        gait_t: float,
        dt: float,
    ) -> np.ndarray:
        tau_g = (
            joint_gravity_torques(model, data, self.robot_cfg)
            if self.gravity_compensation
            else np.zeros(12, dtype=float)
        )
        foot_refs = self.planner.update(
            gait_t, base_pos=state.base_pos, base_rpy=state.base_rpy
        )
        foot_pos = read_foot_positions(model, data, self.robot_cfg)
        trunk_ref = self.trunk_planner.update(
            gait_t,
            dt,
            np.zeros(3),
            omega_cmd=0.0,
            base_pos=state.base_pos,
        )
        stance_legs = [
            leg
            for leg in self.robot_cfg.get("leg_names", [])
            if foot_refs.by_leg(leg).contact
        ]
        contact_forces = (
            self.balance.compute(state, trunk_ref, stance_legs, foot_pos)
            if self.use_balance and stance_legs
            else np.zeros(12, dtype=float)
        )
        stance_tau, _, _ = self.wbc.compute_task_torques(
            model,
            data,
            self.robot_cfg,
            state,
            contact_forces,
            foot_refs,
        )
        q_des, dq_des = self.leg_ik.solve_all_swing(
            model, data, foot_refs, q_seed=state.q
        )
        kp = np.zeros(12, dtype=float)
        kd = self._joint_kd.copy()

        for leg_idx, leg in enumerate(self.robot_cfg.get("leg_names", [])):
            j0 = leg_idx * 3
            if not foot_refs.by_leg(leg).contact:
                kp[j0 : j0 + 3] = self._swing_kp[j0 : j0 + 3]
                kd[j0 : j0 + 3] = self._swing_kd[j0 : j0 + 3]

        cmd = JointCommand(
            q_des=q_des,
            dq_des=dq_des,
            tau_ff=tau_g + stance_tau,
            kp=kp,
            kd=kd,
        )
        return self.joint.compute(state, cmd)

    @classmethod
    def from_config(
        cls,
        robot_cfg: dict,
        loco_cfg: dict,
        gait_cfg: dict,
        *,
        planner: FlLiftPlanner,
        balance: BalanceController,
        wbc: WBCController,
        trunk_planner: TrunkPlanner,
        attitude: AttitudeStandAssist | None = None,
    ) -> FlLiftController:
        loco = loco_cfg.get("locomotion", loco_cfg)
        wbc_yaml = loco.get("wbc", {})
        block = gait_control_block(gait_cfg)
        swing = block.get("swing_joint", {})
        trunk = block.get("trunk", loco.get("trunk", {}))
        if trunk:
            balance.set_trunk_gains(
                kp_pos=np.asarray(trunk.get("kp_pos", balance.kp_pos)),
                kd_pos=np.asarray(trunk.get("kd_pos", balance.kd_pos)),
                kp_rpy=np.asarray(trunk.get("kp_rpy", balance.kp_rpy)),
                kd_rpy=np.asarray(trunk.get("kd_rpy", balance.kd_rpy)),
            )
        swkp = np.tile(
            np.asarray(swing.get("kp", [40.0, 60.0, 60.0]), dtype=float), 4
        )
        swkd = np.tile(
            np.asarray(swing.get("kd", [2.0, 3.0, 3.0]), dtype=float), 4
        )
        jkd = np.tile(
            np.asarray(wbc_yaml.get("joint_kd", [2.0, 2.0, 2.0]), dtype=float), 4
        )
        return cls(
            robot_cfg,
            planner=planner,
            balance=balance,
            wbc=wbc,
            trunk_planner=trunk_planner,
            joint=JointController.from_robot_config({"robot": robot_cfg}),
            leg_ik=LegIK.from_config({"robot": robot_cfg}, loco_cfg),
            attitude=attitude,
            swing_kp=swkp,
            swing_kd=swkd,
            joint_kd=jkd,
            use_balance=bool(block.get("use_balance", True)),
            swing_leg=planner.swing_leg,
            gravity_compensation=bool(wbc_yaml.get("gravity_compensation", True)),
        )
