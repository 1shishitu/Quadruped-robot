"""SRBM model predictive control."""

from __future__ import annotations

import numpy as np

from quadruped.models.srbm import SingleRigidBodyModel
from quadruped.types import TrunkRef


class MPCController:
    """
    质心 MPC (SRBM + QP).

    输入: 当前状态, TrunkRef, 接触序列
    输出: 四足接触力 (12,)
    """

    def __init__(self, model: SingleRigidBodyModel, cfg: dict) -> None:
        self.model = model
        self.cfg = cfg["mpc"]
        self.horizon = self.cfg["horizon"]
        self.dt = self.cfg["dt"]
        self._last_forces: np.ndarray | None = None

    def reset(self) -> None:
        self._last_forces = None

    def compute(
        self,
        state: np.ndarray,
        ref: TrunkRef,
        contact_horizon: np.ndarray,
    ) -> np.ndarray:
        """
        Solve MPC QP.

        Args:
            state: (12,) SRBM state
            ref: trunk reference at current step
            contact_horizon: (N, 4) bool stance flags

        Returns:
            contact forces (12,) for current step
        """
        raise NotImplementedError("MPC QP solve — Phase 4")

    @classmethod
    def from_config(cls, robot_cfg: dict, mpc_cfg: dict) -> MPCController:
        r = robot_cfg["robot"]
        model = SingleRigidBodyModel(mass=r["mass"], gravity=r["gravity"])
        return cls(model, mpc_cfg)
