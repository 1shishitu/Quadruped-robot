"""Whole-body control — stance force mapping (unitree_guide).

Stance: Balance GRF  F  →  τ = −JᵀF

Swing 腿由 ``LegIK`` + 关节 PD 执行（quintic 足端轨迹 → q_des），不在此模块。
"""

from __future__ import annotations

import numpy as np

from quadruped.types import FootRefs, RobotState

LEG_ORDER = ("FL", "FR", "RL", "RR")


class WBCController:
    """Map stance contact forces to joint torques (−JᵀF)."""

    def __init__(self, cfg: dict | None = None) -> None:
        loco = (cfg or {}).get("locomotion", cfg or {})
        wbc = loco.get("wbc", loco) if isinstance(loco, dict) else (cfg or {})
        self.joint_kd = np.asarray(wbc.get("joint_kd", [1.0, 1.0, 1.0]), dtype=float)

    def compute_task_torques(
        self,
        model,
        data,
        robot_cfg: dict,
        state: RobotState,
        contact_forces: np.ndarray,
        foot_refs: FootRefs,
        *,
        foot_pos: dict[str, np.ndarray] | None = None,
        foot_vel: dict[str, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        from quadruped.sim.mujoco_robot import foot_jacobian

        del state, foot_pos, foot_vel
        task_tau = np.zeros(12, dtype=float)
        swing_mask = np.zeros(12, dtype=float)
        stance_mask = np.zeros(12, dtype=float)

        for leg_idx, leg in enumerate(LEG_ORDER):
            j_start = leg_idx * 3
            ref = foot_refs.by_leg(leg)

            if ref.contact:
                jac = foot_jacobian(model, data, robot_cfg, leg)
                f = contact_forces[j_start : j_start + 3]
                task_tau[j_start : j_start + 3] = -(jac.T @ f)
                stance_mask[j_start : j_start + 3] = 1.0
            else:
                swing_mask[j_start : j_start + 3] = 1.0

        return task_tau, swing_mask, stance_mask

    @classmethod
    def from_config(cls, loco_cfg: dict) -> WBCController:
        return cls(loco_cfg)
