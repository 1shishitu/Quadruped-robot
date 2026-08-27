"""Unitree low-level stand-up: init → Fuse → Tachi (linear joint interpolation)."""

from __future__ import annotations

import numpy as np

# Go1 URDF: FL/RL vs FR/RR hip joint mirror (leg.xacro mirror_dae)
_HIP_MIRROR = {"FL": 1, "FR": -1, "RL": 1, "RR": -1}


def _hip_signs(robot_cfg: dict) -> dict[str, int]:
    custom = robot_cfg.get("hip_mirror_sign")
    if custom:
        return {str(k): int(v) for k, v in custom.items()}
    legs = robot_cfg.get("leg_names", [])
    if set(legs) == set(_HIP_MIRROR):
        return dict(_HIP_MIRROR)
    return {leg: 1 for leg in legs}


def motor_pose_vector(robot_cfg: dict, motor_triple: list[float]) -> np.ndarray:
    """Unitree motor [hip, thigh, calf] → URDF q (L/R hip mirrored)."""
    hip, thigh, calf = (float(x) for x in motor_triple)
    signs = _hip_signs(robot_cfg)
    q = []
    for leg in robot_cfg["leg_names"]:
        q.extend([signs.get(leg, 1) * abs(hip), thigh, calf])
    return np.asarray(q, dtype=float)


def fuse_pose_vector(robot_cfg: dict) -> np.ndarray:
    stand = robot_cfg.get("stand_joint", {})
    per_leg = stand.get("stand_poses", {}).get("fuse", [-0.02, 1.3, -2.8])
    return motor_pose_vector(robot_cfg, per_leg)


def tachi_pose_vector(robot_cfg: dict) -> np.ndarray:
    """Tachi = default_joint_angles in robot.yaml (URDF, per leg)."""
    angles = robot_cfg.get("default_joint_angles", {})
    legs = robot_cfg.get("leg_names", [])
    if angles and legs:
        q = []
        for leg in legs:
            q.extend(float(a) for a in angles[leg])
        return np.asarray(q, dtype=float)
    stand = robot_cfg.get("stand_joint", {})
    per_leg = stand.get("stand_poses", {}).get("tachi", [-0.27, 0.8, -1.6])
    return motor_pose_vector(robot_cfg, per_leg)


def linear_joint_interpolation(q0: np.ndarray, q1: np.ndarray, rate: float) -> np.ndarray:
    rate = float(np.clip(rate, 0.0, 1.0))
    return (1.0 - rate) * np.asarray(q0, dtype=float) + rate * np.asarray(q1, dtype=float)


def stand_up_q_des(
    q_init: np.ndarray,
    q_fuse: np.ndarray,
    q_tachi: np.ndarray,
    elapsed: float,
    phase_duration: float,
) -> tuple[np.ndarray, str]:
    """
    Unitree SDK tutorial schedule (linear positionCurve):
      phase A: q_init → Fuse
      phase B: Fuse → Tachi (Tachi = default_joint_angles / hold)
    """
    dur = float(phase_duration)
    if dur <= 0.0:
        return q_tachi.copy(), "hold"

    elapsed = max(0.0, float(elapsed))
    if elapsed < dur:
        rate = elapsed / dur
        return linear_joint_interpolation(q_init, q_fuse, rate), "fuse"
    if elapsed < 2.0 * dur:
        rate = (elapsed - dur) / dur
        return linear_joint_interpolation(q_fuse, q_tachi, rate), "tachi"
    return q_tachi.copy(), "hold"


def stand_up_total_duration(robot_cfg: dict) -> float:
    stand = robot_cfg.get("stand_joint", {})
    dur = float(stand.get("stand_up_phase_duration", 10.0))
    return 2.0 * dur
