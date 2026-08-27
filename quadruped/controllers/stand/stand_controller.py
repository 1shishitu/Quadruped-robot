"""MIT-style standing: fixed default pose + joint PD command (τ_ff supplied externally)."""

from __future__ import annotations

import numpy as np

from quadruped.types import JointCommand


class StandController:
    """
    固定默认站立角 + 关节 PD（MIT / Unitree 低层同款结构）.

    τ = τ_ff + Kp(q_des - q) + Kd(dq_des - dq)
    τ_ff 由动力学模型提供（仿真: MuJoCo bias；实机: Pinocchio RNEA）.
    """

    def __init__(
        self,
        q_des: np.ndarray,
        kp: np.ndarray,
        kd: np.ndarray,
    ) -> None:
        self.q_des = np.asarray(q_des, dtype=float)
        self.kp = np.asarray(kp, dtype=float)
        self.kd = np.asarray(kd, dtype=float)
        if self.q_des.shape != (12,):
            raise ValueError(f"q_des must be shape (12,), got {self.q_des.shape}")

    @classmethod
    def from_robot_config(cls, robot_cfg: dict) -> StandController:
        r = robot_cfg["robot"]
        stand = r["stand_joint"]

        q_des = []
        per_leg_kp = stand["kp"]
        per_leg_kd = stand["kd"]
        kp = []
        kd = []
        for leg in r["leg_names"]:
            q_des.extend(r["default_joint_angles"][leg])
            kp.extend(per_leg_kp)
            kd.extend(per_leg_kd)

        return cls(q_des=np.asarray(q_des, dtype=float), kp=np.asarray(kp), kd=np.asarray(kd))

    def command(self, tau_ff: np.ndarray) -> JointCommand:
        return JointCommand(
            q_des=self.q_des.copy(),
            dq_des=np.zeros(12, dtype=float),
            tau_ff=np.asarray(tau_ff, dtype=float),
            kp=self.kp.copy(),
            kd=self.kd.copy(),
        )
