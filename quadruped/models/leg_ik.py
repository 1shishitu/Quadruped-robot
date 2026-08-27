"""Per-leg inverse kinematics (unitree_guide-style swing execution).

Foot target is in world frame; solves [hip, thigh, calf] with damped least squares
on the MuJoCo model (matches URDF kinematics without a separate analytic chain).
"""

from __future__ import annotations

import numpy as np

from quadruped.sim.mujoco_robot import build_joint_maps, foot_jacobian, read_foot_positions


class LegIK:
    """3-DoF foot IK for one leg using iterative DLS on MuJoCo FK."""

    def __init__(
        self,
        robot_cfg: dict,
        *,
        max_iters: int = 12,
        tol: float = 1.0e-4,
        damping: float = 0.08,
        max_step: float = 0.35,
    ) -> None:
        self.robot_cfg = robot_cfg
        self.max_iters = int(max_iters)
        self.tol = float(tol)
        self.damping = float(damping)
        self.max_step = float(max_step)
        self._limits = _joint_limits(robot_cfg)

    def solve(
        self,
        model,
        data,
        leg: str,
        foot_target_world: np.ndarray,
        *,
        q_seed: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return joint angles [hip, thigh, calf] for ``leg``."""
        q, _ = self.solve_with_velocity(
            model,
            data,
            leg,
            foot_target_world,
            foot_velocity_world=np.zeros(3, dtype=float),
            q_seed=q_seed,
        )
        return q

    def solve_with_velocity(
        self,
        model,
        data,
        leg: str,
        foot_target_world: np.ndarray,
        *,
        foot_velocity_world: np.ndarray,
        q_seed: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        import mujoco

        target = np.asarray(foot_target_world, dtype=float).reshape(3)
        v_target = np.asarray(foot_velocity_world, dtype=float).reshape(3)
        qa_list, _, _ = build_joint_maps(model, self.robot_cfg)
        leg_names = self.robot_cfg.get("leg_names", [])
        leg_idx = leg_names.index(leg)
        leg_qa = [qa_list[leg_idx * 3 + j] for j in range(3)]

        if q_seed is None:
            q_leg = np.array([data.qpos[qa] for qa in leg_qa], dtype=float)
        else:
            q_leg = np.asarray(q_seed, dtype=float).reshape(3).copy()

        saved = [float(data.qpos[qa]) for qa in leg_qa]
        try:
            for _ in range(self.max_iters):
                for i, qa in enumerate(leg_qa):
                    data.qpos[qa] = float(q_leg[i])
                mujoco.mj_forward(model, data)

                foot_pos = read_foot_positions(model, data, self.robot_cfg)[leg]
                err = target - foot_pos
                if float(np.linalg.norm(err)) < self.tol:
                    break

                jac = foot_jacobian(model, data, self.robot_cfg, leg)
                dq = _damped_least_squares(jac, err, self.damping)
                dq = np.clip(dq, -self.max_step, self.max_step)
                q_leg = q_leg + dq
                q_leg = _clip_leg_q(q_leg, self._limits)

            for i, qa in enumerate(leg_qa):
                data.qpos[qa] = float(q_leg[i])
            mujoco.mj_forward(model, data)
            jac = foot_jacobian(model, data, self.robot_cfg, leg)
            dq_leg = _damped_least_squares(jac, v_target, self.damping)
        finally:
            for qa, q_saved in zip(leg_qa, saved):
                data.qpos[qa] = q_saved
            mujoco.mj_forward(model, data)

        q_leg = _clip_leg_q(q_leg, self._limits)
        return q_leg, dq_leg

    def solve_all_swing(
        self,
        model,
        data,
        foot_refs,
        *,
        q_seed: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Solve IK for all swing legs; return full 12-dof q_des, dq_des."""
        leg_names = self.robot_cfg.get("leg_names", [])
        q_des = (
            np.asarray(q_seed, dtype=float).copy()
            if q_seed is not None
            else np.array([data.qpos[qa] for qa in build_joint_maps(model, self.robot_cfg)[0]])
        )
        dq_des = np.zeros(12, dtype=float)

        for leg_idx, leg in enumerate(leg_names):
            ref = foot_refs.by_leg(leg)
            if ref.contact:
                continue
            j0 = leg_idx * 3
            q_leg, dq_leg = self.solve_with_velocity(
                model,
                data,
                leg,
                ref.position,
                foot_velocity_world=ref.velocity,
                q_seed=q_des[j0 : j0 + 3],
            )
            q_des[j0 : j0 + 3] = q_leg
            dq_des[j0 : j0 + 3] = dq_leg
        return q_des, dq_des

    @classmethod
    def from_config(cls, robot_cfg: dict, loco_cfg: dict | None = None) -> LegIK:
        r = robot_cfg.get("robot", robot_cfg)
        ik = {}
        if loco_cfg:
            loco = loco_cfg.get("locomotion", loco_cfg)
            ik = loco.get("swing_ik", {})
        return cls(
            r,
            max_iters=int(ik.get("max_iters", 12)),
            tol=float(ik.get("tol", 1.0e-4)),
            damping=float(ik.get("damping", 0.08)),
            max_step=float(ik.get("max_step", 0.35)),
        )


def _damped_least_squares(jac: np.ndarray, target: np.ndarray, damping: float) -> np.ndarray:
    jjt = jac @ jac.T + (damping**2) * np.eye(3)
    return jac.T @ np.linalg.solve(jjt, target)


def _joint_limits(robot_cfg: dict) -> dict[str, tuple[float, float]]:
    limits = robot_cfg.get("joint_limits", {})
    return {
        name: (float(lo), float(hi))
        for name, (lo, hi) in limits.items()
    }


def _clip_leg_q(q_leg: np.ndarray, limits: dict[str, tuple[float, float]]) -> np.ndarray:
    names = ("hip", "thigh", "calf")
    out = q_leg.copy()
    for i, name in enumerate(names):
        if name in limits:
            lo, hi = limits[name]
            out[i] = float(np.clip(out[i], lo, hi))
    return out
