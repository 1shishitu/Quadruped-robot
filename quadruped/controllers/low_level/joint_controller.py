"""Joint-level PD + torque limiting."""

from __future__ import annotations

import numpy as np

from quadruped.types import JointCommand, RobotState


class JointController:
    """关节 PD 控制器 + 力矩限幅."""

    def __init__(
        self,
        kp: float | np.ndarray = 80.0,
        kd: float | np.ndarray = 2.0,
        torque_limits: np.ndarray | None = None,
    ) -> None:
        self.kp = np.atleast_1d(kp).astype(float)
        self.kd = np.atleast_1d(kd).astype(float)
        self.torque_limits = torque_limits

    def compute(
        self,
        state: RobotState,
        cmd: JointCommand,
    ) -> np.ndarray:
        """
        tau = tau_ff + kp * (q_des - q) + kd * (dq_des - dq)
        """
        q = state.q
        dq = state.dq
        tau = cmd.tau_ff.copy()
        kp = cmd.kp if len(cmd.kp) else self.kp
        kd = cmd.kd if len(cmd.kd) else self.kd

        if len(kp) == 1:
            tau += kp[0] * (cmd.q_des - q) + kd[0] * (cmd.dq_des - dq)
        else:
            tau += kp * (cmd.q_des - q) + kd * (cmd.dq_des - dq)

        if self.torque_limits is not None:
            tau = np.clip(tau, -self.torque_limits, self.torque_limits)
        return tau

    @classmethod
    def from_robot_config(cls, robot_cfg: dict) -> JointController:
        r = robot_cfg["robot"]
        limits = np.tile(r["torque_limits"], 4)
        return cls(torque_limits=limits)
