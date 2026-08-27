"""Raibert foot placement heuristic (stance-relative, body-frame preview).

MIT / unitree 标准形式：落足 = 当前支撑足位置 + R_yaw · (T/2 · v_cmd + kv · (v − v_cmd))，
**不是** hip 投影到地面（Go1 站立时足端比 hip 外展 ~0.15 m，用 hip 会把第一步向内收）。
"""

from __future__ import annotations

import numpy as np


def yaw_rotation(yaw: float) -> np.ndarray:
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def rpy_to_rotation(rpy: np.ndarray) -> np.ndarray:
    """Body→world rotation matching ``mujoco_robot._quat_to_rpy`` convention."""
    roll, pitch, yaw = np.asarray(rpy, dtype=float).reshape(3)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return rz @ ry @ rx


def _yaw_rotation(yaw: float) -> np.ndarray:
    return yaw_rotation(yaw)


def raibert_foot_placement(
    stance_foot_pos: np.ndarray,
    v_body: np.ndarray,
    v_cmd: np.ndarray,
    step_period: float,
    *,
    yaw: float = 0.0,
    kv: float = 0.04,
    kp: float = 0.0,
    p_des: np.ndarray | None = None,
    hip_pos: np.ndarray | None = None,
) -> np.ndarray:
    """
    Raibert touchdown preview in world frame.

    Args:
        stance_foot_pos: support foot position at swing start (world, 3,)
        v_body: body / CoM velocity (world, 3,)
        v_cmd: commanded velocity (world, 3,)
        step_period: gait period T [s] (trot 下 T/2 ≈ stance 时长)
        yaw: body yaw for horizontal preview (rad)
        kv: velocity error gain (body horizontal)
        kp: optional position feedback on hip xy (legacy, usually 0)
        hip_pos: only used when kp > 0

    Returns:
        desired foot placement on ground (z=0)
    """
    p0 = np.asarray(stance_foot_pos, dtype=float).reshape(3)
    v_body = np.asarray(v_body, dtype=float).reshape(3)
    v_cmd = np.asarray(v_cmd, dtype=float).reshape(3)

    rot = _yaw_rotation(yaw)
    v_body_b = rot.T @ v_body
    v_cmd_b = rot.T @ v_cmd

    delta_b = 0.5 * step_period * v_cmd_b.copy()
    if kv > 0.0:
        delta_b[:2] += kv * (v_body_b[:2] - v_cmd_b[:2])

    placement = p0 + rot @ delta_b

    if kp > 0.0 and p_des is not None and hip_pos is not None:
        placement[:2] += kp * (np.asarray(p_des)[:2] - np.asarray(hip_pos)[:2])

    placement[2] = 0.0
    return placement


def in_place_touchdown(
    stance_foot_pos: np.ndarray,
    *,
    ground_z: float | None = None,
) -> np.ndarray:
    """March in place: land back on the recorded stance anchor (same xyz)."""
    p = np.asarray(stance_foot_pos, dtype=float).reshape(3).copy()
    if ground_z is not None:
        p[2] = float(ground_z)
    return p
