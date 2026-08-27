"""Locomotion control — march in place or walk.

march_in_place (默认):
    进入 walk 后跑 trot 相位，落足=stance 锚点，v_cmd=0，躯干原地保持

walk (后续):
    Raibert + 键盘速度；|v|<deadband 时四足支撑
"""

from __future__ import annotations

import numpy as np

from quadruped.controllers.low_level.attitude_stand_assist import AttitudeStandAssist
from quadruped.controllers.low_level.balance_controller import BalanceController
from quadruped.controllers.low_level.joint_controller import JointController
from quadruped.controllers.low_level.wbc_controller import WBCController
from quadruped.models.leg_ik import LegIK
from quadruped.planners.foot_planner import FootPlanner
from quadruped.planners.trunk_planner import TrunkPlanner
from quadruped.sim.mujoco_robot import (
    joint_gravity_torques,
    read_foot_positions,
    read_hip_positions,
)
from quadruped.types import FootPlannerDebug, FootRef, FootRefs, JointCommand, LegFootDebug, RobotState


class LocomotionController:
    """Walk / trot: τ_g + stance wrench WBC + swing foot Cartesian tracking."""

    def __init__(
        self,
        robot_cfg: dict,
        *,
        balance: BalanceController,
        wbc: WBCController,
        foot_planner: FootPlanner,
        trunk_planner: TrunkPlanner,
        joint: JointController,
        leg_ik: LegIK,
        gravity_compensation: bool = True,
        attitude: AttitudeStandAssist | None = None,
    ) -> None:
        self.robot_cfg = robot_cfg
        self.balance = balance
        self.wbc = wbc
        self.foot_planner = foot_planner
        self.trunk_planner = trunk_planner
        self.joint = joint
        self.leg_ik = leg_ik
        self.gravity_compensation = gravity_compensation
        self.attitude = attitude
        self._joint_kd = np.tile(np.asarray(wbc.joint_kd, dtype=float), 4)
        self._swing_kp = np.tile(np.asarray([80.0, 100.0, 100.0], dtype=float), 4)
        self._swing_kd = np.tile(np.asarray([2.0, 2.0, 2.0], dtype=float), 4)
        self._cmd_deadband = float(
            balance.v_cmd_deadband if hasattr(balance, "v_cmd_deadband") else 0.04
        )
        self._omega_deadband = 0.05
        self._march_in_place = False
        self._march_cfg: dict = {}
        self._march_q_nom = np.zeros(12, dtype=float)
        self._march_use_balance = True
        self._stance_kp = np.zeros(12, dtype=float)
        self._stance_kd = np.zeros(12, dtype=float)

    @property
    def march_in_place(self) -> bool:
        return self._march_in_place

    def begin_march(self, state: RobotState) -> None:
        """Latch standing joint pose and march-specific gains at entry."""
        self._march_q_nom = np.asarray(state.q, dtype=float).copy()
        march = self._march_cfg
        if not march:
            return
        trunk = march.get("trunk", {})
        if trunk:
            self.balance.set_trunk_gains(
                kp_pos=np.asarray(trunk.get("kp_pos", self.balance.kp_pos)),
                kd_pos=np.asarray(trunk.get("kd_pos", self.balance.kd_pos)),
                kp_rpy=np.asarray(trunk.get("kp_rpy", self.balance.kp_rpy)),
                kd_rpy=np.asarray(trunk.get("kd_rpy", self.balance.kd_rpy)),
            )
        self._march_use_balance = bool(march.get("use_balance", True))
        stance = march.get("stance_joint", {})
        swing = march.get("swing_joint", {})
        stand = self.robot_cfg.get("stand_joint", {})
        default_stance_kp = stand.get("kp", [200.0, 200.0, 200.0])
        default_stance_kd = stand.get("kd", [1.0, 1.0, 1.0])
        self._stance_kp = np.tile(
            np.asarray(stance.get("kp", default_stance_kp), dtype=float), 4
        )
        self._stance_kd = np.tile(
            np.asarray(stance.get("kd", default_stance_kd), dtype=float), 4
        )
        if swing:
            self._swing_kp = np.tile(
                np.asarray(swing.get("kp", [40.0, 90.0, 110.0]), dtype=float), 4
            )
            self._swing_kd = np.tile(
                np.asarray(swing.get("kd", [2.5, 4.0, 5.0]), dtype=float), 4
            )
        self.balance.reset()

    @staticmethod
    def _hold_stance_foot_refs(
        leg_names: list[str], foot_pos: dict[str, np.ndarray]
    ) -> FootRefs:
        """Idle walk: 四足支撑在当前落足点，不跑 trot 相位."""
        refs = {
            leg: FootRef(
                position=np.asarray(foot_pos[leg], dtype=float).copy(),
                velocity=np.zeros(3, dtype=float),
                acceleration=np.zeros(3, dtype=float),
                contact=True,
            )
            for leg in leg_names
        }
        return FootRefs(**refs)

    @staticmethod
    def _idle_foot_debug(foot_pos: dict[str, np.ndarray]) -> FootPlannerDebug:
        legs = {}
        for leg, p in foot_pos.items():
            pos = np.asarray(p, dtype=float).copy()
            ground = pos.copy()
            ground[2] = 0.0
            legs[leg] = LegFootDebug(
                leg=leg,
                phase=0.0,
                in_stance=True,
                stance_anchor=pos.copy(),
                raibert_target=ground,
                swing_start=pos.copy(),
                ref_position=pos.copy(),
                ref_contact=True,
            )
        return FootPlannerDebug(legs=legs, swing_paths={})

    def _is_idle_command(self, v_cmd: np.ndarray, omega_cmd: float) -> bool:
        return np.linalg.norm(v_cmd[:2]) < self._cmd_deadband and abs(
            omega_cmd
        ) < self._omega_deadband

    def compute(
        self,
        model,
        data,
        state: RobotState,
        *,
        gait_t: float,
        dt: float,
        v_cmd: np.ndarray,
        omega_cmd: float = 0.0,
    ) -> np.ndarray:
        if self.gravity_compensation:
            tau_g = joint_gravity_torques(model, data, self.robot_cfg)
        else:
            tau_g = np.zeros(12, dtype=float)

        if self._march_in_place:
            v_cmd = np.zeros(3, dtype=float)
            omega_cmd = 0.0

        trunk_ref = self.trunk_planner.update(
            gait_t,
            dt,
            v_cmd,
            omega_cmd=omega_cmd,
            base_pos=state.base_pos,
        )
        foot_pos = read_foot_positions(model, data, self.robot_cfg)
        run_gait = self._march_in_place or not self._is_idle_command(
            v_cmd, omega_cmd
        )
        if run_gait:
            hip_pos = read_hip_positions(model, data, self.robot_cfg)
            foot_refs = self.foot_planner.update(
                gait_t,
                state.base_vel,
                v_cmd,
                hip_pos,
                yaw=float(state.base_rpy[2]),
                base_pos=state.base_pos,
                base_rpy=state.base_rpy,
                foot_pos=foot_pos,
            )
        else:
            foot_refs = self._hold_stance_foot_refs(
                self.robot_cfg.get("leg_names", []), foot_pos
            )
            self.foot_planner.debug = self._idle_foot_debug(foot_pos)
        stance_legs = [
            leg
            for leg in self.robot_cfg.get("leg_names", [])
            if foot_refs.by_leg(leg).contact
        ]
        contact_forces = self.balance.compute(
            state, trunk_ref, stance_legs, foot_pos
        )
        if self._march_in_place and not self._march_use_balance:
            contact_forces = np.zeros(12, dtype=float)

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
            if foot_refs.by_leg(leg).contact:
                if self._march_in_place:
                    q_des[j0 : j0 + 3] = self._march_q_nom[j0 : j0 + 3]
                    kp[j0 : j0 + 3] = self._stance_kp[j0 : j0 + 3]
                    kd[j0 : j0 + 3] = self._stance_kd[j0 : j0 + 3]
                continue
            kp[j0 : j0 + 3] = self._swing_kp[j0 : j0 + 3]
            kd[j0 : j0 + 3] = self._swing_kd[j0 : j0 + 3]

        if self._march_in_place and self.attitude is not None:
            assist = self.attitude.compute(
                state,
                "march",
                min_feet=2,
                require_all_feet=False,
            )
            q_des = q_des + assist.dq_corr

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
        *,
        balance: BalanceController,
        wbc: WBCController,
        foot_planner: FootPlanner,
        trunk_planner: TrunkPlanner,
        attitude: AttitudeStandAssist | None = None,
    ) -> LocomotionController:
        loco = loco_cfg.get("locomotion", loco_cfg)
        wbc_yaml = loco.get("wbc", {})
        swing_joint = loco.get("swing_joint", {})
        joint = JointController.from_robot_config({"robot": robot_cfg})
        leg_ik = LegIK.from_config({"robot": robot_cfg}, loco_cfg)
        inst = cls(
            robot_cfg,
            balance=balance,
            wbc=wbc,
            foot_planner=foot_planner,
            trunk_planner=trunk_planner,
            joint=joint,
            leg_ik=leg_ik,
            gravity_compensation=bool(wbc_yaml.get("gravity_compensation", True)),
            attitude=attitude,
        )
        inst._swing_kp = np.tile(
            np.asarray(swing_joint.get("kp", [80.0, 100.0, 100.0]), dtype=float), 4
        )
        inst._swing_kd = np.tile(
            np.asarray(swing_joint.get("kd", [2.0, 2.0, 2.0]), dtype=float), 4
        )
        mode = str(loco.get("mode", "march_in_place"))
        inst._march_in_place = mode == "march_in_place"
        inst._march_cfg = loco.get("march", {})
        return inst
