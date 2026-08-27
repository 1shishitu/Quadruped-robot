"""Stance force allocation — unitree_guide BalanceCtrl QP.

Soft wrench tracking (not hard equality)::

    min  0.5 Fᵀ G F + g₀ᵀ F
    G = Aᵀ S A + α W + β U

    s.t.  swing: F_i = 0   (eliminated — only stance vars are optimized)
          stance: friction pyramid (5 rows / foot)

``A`` maps stance contact forces (world frame) to [ΣF, Σ r×F].
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import LinearConstraint, minimize

from quadruped.types import RobotState, TrunkRef

LEG_ORDER = ("FL", "FR", "RL", "RR")


class BalanceController:
    """unitree_guide BalanceCtrl — reduced QP on stance feet only."""

    def __init__(self, cfg: dict, *, mass: float, gravity: float) -> None:
        bal = cfg.get("balance", cfg)
        self.mass = mass
        self.gravity = gravity
        self.mu = float(bal.get("friction_coeff", 0.4))
        self.include_gravity = bool(bal.get("include_gravity", True))
        self.v_cmd_deadband = float(bal.get("v_cmd_deadband", 0.05))

        self.alpha = float(bal.get("alpha", 0.001))
        self.beta = float(bal.get("beta", 0.1))
        self.com_offset = np.asarray(
            bal.get("com_offset", [0.0, 0.0, 0.0]), dtype=float
        )

        inertia = bal.get("inertia", [0.0792, 0.2085, 0.2265])
        self.inertia = np.diag(np.asarray(inertia, dtype=float))

        w_s = np.asarray(
            bal.get("weight_wrench", [20.0, 20.0, 50.0, 450.0, 450.0, 450.0]),
            dtype=float,
        )
        w_f = np.asarray(bal.get("weight_force", [10.0, 10.0, 4.0]), dtype=float)
        w_u = np.asarray(bal.get("weight_smooth", [3.0, 3.0, 3.0]), dtype=float)
        self._S = np.diag(w_s)
        self._w_f = w_f
        self._w_u = w_u

        trunk = cfg.get("trunk", {})
        self.kp_pos = np.asarray(trunk.get("kp_pos", [40.0, 40.0, 80.0]), dtype=float)
        self.kd_pos = np.asarray(trunk.get("kd_pos", [8.0, 8.0, 8.0]), dtype=float)
        self.kp_rpy = np.asarray(trunk.get("kp_rpy", [80.0, 80.0, 20.0]), dtype=float)
        self.kd_rpy = np.asarray(trunk.get("kd_rpy", [8.0, 8.0, 4.0]), dtype=float)

        self._friction_rows = np.array(
            [
                [1.0, 0.0, self.mu],
                [-1.0, 0.0, self.mu],
                [0.0, 1.0, self.mu],
                [0.0, -1.0, self.mu],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        self._F_prev = np.zeros(12, dtype=float)

    def set_trunk_gains(
        self,
        *,
        kp_pos: np.ndarray | None = None,
        kd_pos: np.ndarray | None = None,
        kp_rpy: np.ndarray | None = None,
        kd_rpy: np.ndarray | None = None,
    ) -> None:
        if kp_pos is not None:
            self.kp_pos = np.asarray(kp_pos, dtype=float)
        if kd_pos is not None:
            self.kd_pos = np.asarray(kd_pos, dtype=float)
        if kp_rpy is not None:
            self.kp_rpy = np.asarray(kp_rpy, dtype=float)
        if kd_rpy is not None:
            self.kd_rpy = np.asarray(kd_rpy, dtype=float)

    def desired_wrench(
        self, state: RobotState, ref: TrunkRef
    ) -> tuple[np.ndarray, np.ndarray]:
        p_err = ref.p_com - state.base_pos
        v_err = ref.v_com - state.base_vel
        dd_p = ref.a_com + self.kp_pos * p_err + self.kd_pos * v_err

        rpy_err = _angle_diff(ref.rpy, state.base_rpy)
        omega_err = ref.omega - state.base_omega
        d_omega = self.kp_rpy * rpy_err + self.kd_rpy * omega_err

        if self.include_gravity:
            f_des = self.mass * (dd_p + np.array([0.0, 0.0, self.gravity]))
        else:
            f_des = self.mass * dd_p

        rot = _rotation_matrix(state.base_rpy)
        tau_des = rot @ self.inertia @ rot.T @ d_omega
        return f_des, tau_des

    def _apply_velocity_deadband(
        self,
        state: RobotState,
        ref: TrunkRef,
        f_des: np.ndarray,
    ) -> np.ndarray:
        if np.linalg.norm(ref.v_com[:2]) >= self.v_cmd_deadband:
            return f_des

        p_err = ref.p_com - state.base_pos
        v_err = ref.v_com - state.base_vel
        dd_z = ref.a_com[2] + self.kp_pos[2] * p_err[2] + self.kd_pos[2] * v_err[2]
        if self.include_gravity:
            return np.array([0.0, 0.0, self.mass * (dd_z + self.gravity)])
        return np.array([0.0, 0.0, self.mass * dd_z])

    def compute(
        self,
        state: RobotState,
        ref: TrunkRef,
        stance_legs: list[str],
        foot_positions: dict[str, np.ndarray],
    ) -> np.ndarray:
        forces = np.zeros(12, dtype=float)
        if not stance_legs:
            self._F_prev[:] = 0.0
            return forces

        f_des, tau_des = self.desired_wrench(state, ref)
        f_des = self._apply_velocity_deadband(state, ref, f_des)
        bd = np.concatenate([f_des, tau_des])

        contact = {leg: leg in stance_legs for leg in LEG_ORDER}
        stance_idx = [i for i, leg in enumerate(LEG_ORDER) if contact[leg]]
        n_stance = len(stance_idx)
        n_vars = 3 * n_stance

        rot = _rotation_matrix(state.base_rpy)
        p_com = state.base_pos + rot @ self.com_offset

        a_red = np.zeros((6, n_vars), dtype=float)
        for i, leg_idx in enumerate(stance_idx):
            leg = LEG_ORDER[leg_idx]
            foot = np.asarray(
                foot_positions.get(leg, state.base_pos), dtype=float
            )
            r = foot - p_com
            col = 3 * i
            a_red[0:3, col : col + 3] = np.eye(3)
            a_red[3:6, col : col + 3] = _skew(r)

        w_red = np.diag(np.tile(self._w_f, n_stance))
        u_red = np.diag(np.tile(self._w_u, n_stance))
        f_prev_red = np.concatenate(
            [self._F_prev[3 * idx : 3 * idx + 3] for idx in stance_idx]
        )

        g_mat = a_red.T @ self._S @ a_red + self.alpha * w_red + self.beta * u_red
        g0 = -(bd @ self._S @ a_red) - self.beta * (u_red @ f_prev_red)

        ineq_rows: list[np.ndarray] = []
        for i in range(n_stance):
            col = 3 * i
            for fr in self._friction_rows:
                row = np.zeros(n_vars, dtype=float)
                row[col : col + 3] = fr
                ineq_rows.append(row)

        constraints: list[LinearConstraint] = []
        if ineq_rows:
            a_ineq = np.vstack(ineq_rows)
            constraints.append(
                LinearConstraint(a_ineq, lb=0.0, ub=np.full(len(ineq_rows), np.inf))
            )

        sol_red = _solve_qp(g_mat, g0, f_prev_red, constraints, mu=self.mu)

        for i, leg_idx in enumerate(stance_idx):
            forces[3 * leg_idx : 3 * leg_idx + 3] = sol_red[3 * i : 3 * i + 3]

        self._F_prev = forces.copy()
        return forces

    def reset(self) -> None:
        self._F_prev[:] = 0.0

    @classmethod
    def from_config(cls, loco_cfg: dict, robot_cfg: dict) -> BalanceController:
        r = robot_cfg.get("robot", robot_cfg)
        return cls(
            loco_cfg.get("locomotion", loco_cfg),
            mass=float(r.get("mass", 12.0)),
            gravity=float(r.get("gravity", 9.81)),
        )


def _solve_qp(
    g_mat: np.ndarray,
    g0: np.ndarray,
    x0: np.ndarray,
    constraints: list[LinearConstraint],
    *,
    mu: float,
) -> np.ndarray:
    def objective(x: np.ndarray) -> float:
        return float(0.5 * x @ g_mat @ x + g0 @ x)

    def jacobian(x: np.ndarray) -> np.ndarray:
        return g_mat @ x + g0

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        jac=jacobian,
        constraints=constraints,
        options={"maxiter": 80, "ftol": 1e-9},
    )
    if result.success:
        return np.asarray(result.x, dtype=float)

    return _project_friction_pyramid(x0, mu)


def _rotation_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def _skew(r: np.ndarray) -> np.ndarray:
    x, y, z = r
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)


def _angle_diff(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    diff = target - current
    diff[2] = (diff[2] + np.pi) % (2 * np.pi) - np.pi
    return diff


def _project_friction_pyramid(forces: np.ndarray, mu: float) -> np.ndarray:
    out = forces.copy()
    n_feet = len(out) // 3
    for i in range(n_feet):
        f = out[3 * i : 3 * i + 3]
        fx, fy, fz = f
        if fz <= 0.0:
            out[3 * i : 3 * i + 3] = 0.0
            continue
        lim = mu * fz
        fx = float(np.clip(fx, -lim, lim))
        fy = float(np.clip(fy, -lim, lim))
        out[3 * i : 3 * i + 3] = np.array([fx, fy, fz])
    return out
