"""Single rigid body (centroidal) model for MPC."""

from __future__ import annotations

import numpy as np


class SingleRigidBodyModel:
    """
    单刚体质心模型 (SRBM).

    状态: [p_com(3), v_com(3), rpy(3), omega(3)] = 12D
    控制: 四足接触力 f_i ∈ R^3, i=1..4 → 12D
    """

    STATE_DIM = 12
    CONTROL_DIM = 12

    def __init__(
        self,
        mass: float,
        gravity: float = 9.81,
        inertia: np.ndarray | None = None,
        foot_positions_body: np.ndarray | None = None,
    ) -> None:
        self.mass = mass
        self.gravity = gravity
        self.inertia = inertia if inertia is not None else np.eye(3) * 0.05
        # (4, 3) foot positions in body frame relative to CoM
        self.foot_positions_body = (
            foot_positions_body
            if foot_positions_body is not None
            else np.zeros((4, 3))
        )

    def discrete_dynamics(
        self, x: np.ndarray, u: np.ndarray, dt: float, contact: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        离散化 x_{k+1} = A x_k + B u_k.

        Args:
            x: state (12,)
            u: stacked contact forces (12,)
            dt: time step
            contact: (4,) bool, True if stance

        Returns:
            x_next, A, B  (B depends on contact; placeholder linearization)
        """
        raise NotImplementedError("SRBM discrete dynamics — Phase 4")

    def friction_cone_constraints(self, mu: float) -> dict:
        """Return constraint matrices for QP formulation."""
        raise NotImplementedError("SRBM friction cone — Phase 4")
