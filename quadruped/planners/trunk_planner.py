"""Trunk / CoM trajectory planner."""

from __future__ import annotations

import numpy as np

from quadruped.types import TrunkRef


class TrunkPlanner:
    """
    躯干轨迹规划器.

    - 水平: v_xy 跟踪速度指令
    - 高度: 常值 + 可选 quintic 过渡
    - 姿态: roll/pitch=0, yaw 积分
    """

    def __init__(
        self,
        default_height: float = 0.28,
        gravity: float = 9.81,
    ) -> None:
        self.default_height = default_height
        self.gravity = gravity
        self.preview_horizon = 0.08
        self._p_com = np.array([0.0, 0.0, default_height])
        self._yaw = 0.0

    def reset(self, p_com: np.ndarray | None = None, yaw: float = 0.0) -> None:
        self._p_com = (
            np.asarray(p_com, dtype=float)
            if p_com is not None
            else np.array([0.0, 0.0, self.default_height])
        )
        self._yaw = yaw

    def update(
        self,
        t: float,
        dt: float,
        v_cmd: np.ndarray,
        omega_cmd: float = 0.0,
        *,
        base_pos: np.ndarray | None = None,
    ) -> TrunkRef:
        """
        Args:
            t: current time
            dt: control period
            v_cmd: [vx, vy, vz] desired CoM velocity
            omega_cmd: desired yaw rate
            base_pos: actual base xy; anchors preview (avoids runaway p_ref)

        Returns:
            TrunkRef with CoM and base orientation references
        """
        if base_pos is not None:
            self._p_com[0] = float(base_pos[0]) + float(v_cmd[0]) * self.preview_horizon
            self._p_com[1] = float(base_pos[1]) + float(v_cmd[1]) * self.preview_horizon
        else:
            self._p_com[:2] += v_cmd[:2] * dt
        self._p_com[2] = self.default_height + v_cmd[2] * dt
        self._yaw += omega_cmd * dt

        v_com = np.array([v_cmd[0], v_cmd[1], 0.0])
        a_com = np.zeros(3)
        rpy = np.array([0.0, 0.0, self._yaw])
        omega = np.array([0.0, 0.0, omega_cmd])

        return TrunkRef(
            p_com=self._p_com.copy(),
            v_com=v_com,
            a_com=a_com,
            rpy=rpy,
            omega=omega,
        )
