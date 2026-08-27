"""MuJoCo viewer overlays for FootPlanner / Raibert debug."""

from __future__ import annotations

import numpy as np

from quadruped.types import FootPlannerDebug

LEG_RGBA = {
    "FL": np.array([1.0, 0.25, 0.25, 0.95], dtype=float),
    "FR": np.array([0.25, 1.0, 0.35, 0.95], dtype=float),
    "RL": np.array([0.35, 0.55, 1.0, 0.95], dtype=float),
    "RR": np.array([1.0, 0.85, 0.2, 0.95], dtype=float),
}

RAIBERT_RGBA = np.array([1.0, 0.95, 0.1, 1.0], dtype=float)
REF_RGBA = np.array([0.95, 0.95, 0.95, 0.85], dtype=float)
ACTUAL_RGBA = np.array([0.1, 0.1, 0.1, 0.9], dtype=float)
PATH_RGBA = np.array([0.7, 0.85, 1.0, 0.55], dtype=float)
GROUND_Z = 0.015


def _ground(p: np.ndarray) -> np.ndarray:
    out = np.asarray(p, dtype=float).copy()
    out[2] = GROUND_Z
    return out


def _add_geom(scn, mujoco) -> int | None:
    if scn.ngeom >= scn.maxgeom:
        return None
    idx = scn.ngeom
    scn.ngeom += 1
    return idx


def _init_sphere(scn, mujoco, pos: np.ndarray, radius: float, rgba: np.ndarray) -> None:
    idx = _add_geom(scn, mujoco)
    if idx is None:
        return
    mujoco.mjv_initGeom(
        scn.geoms[idx],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=np.array([radius, 0.0, 0.0], dtype=float),
        pos=np.asarray(pos, dtype=float),
        mat=np.eye(3, dtype=float).reshape(-1),
        rgba=np.asarray(rgba, dtype=float),
    )


def _init_capsule(
    scn,
    mujoco,
    p0: np.ndarray,
    p1: np.ndarray,
    radius: float,
    rgba: np.ndarray,
) -> None:
    idx = _add_geom(scn, mujoco)
    if idx is None:
        return
    mujoco.mjv_initGeom(
        scn.geoms[idx],
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        size=np.array([radius, 1e-4, 0.0], dtype=float),
        pos=np.zeros(3, dtype=float),
        mat=np.eye(3, dtype=float).reshape(-1),
        rgba=np.asarray(rgba, dtype=float),
    )
    mujoco.mjv_connector(
        scn.geoms[idx],
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        float(radius),
        np.asarray(p0, dtype=float),
        np.asarray(p1, dtype=float),
    )


def draw_foot_planner_debug(
    viewer,
    debug: FootPlannerDebug,
    foot_pos: dict[str, np.ndarray],
    *,
    enabled: bool = True,
    show_swing_path: bool = True,
    show_raibert: bool = True,
    show_actual: bool = True,
) -> None:
    """
    在 passive viewer.user_scn 中绘制足端规划调试量.

    - 黄色大球: Raibert 落足目标 ``p_end``（z=0）
    - 彩色小球: 当前支撑锁定点 ``stance_anchor``
    - 浅色折线: swing quintic 采样路径
    - 白球: 当前 ``FootRef.position``（含抬脚高度）
    - 黑球: 仿真实际足端位置
    """
    if not enabled or viewer is None:
        return

    import mujoco

    with viewer.lock():
        scn = viewer.user_scn
        scn.ngeom = 0

        for leg, info in debug.legs.items():
            color = LEG_RGBA.get(leg, REF_RGBA)

            if show_raibert:
                _init_sphere(scn, mujoco, _ground(info.raibert_target), 0.022, RAIBERT_RGBA)
                _init_capsule(
                    scn,
                    mujoco,
                    _ground(info.swing_start if not info.in_stance else info.stance_anchor),
                    _ground(info.raibert_target),
                    0.004,
                    color * np.array([1, 1, 1, 0.35]),
                )

            _init_sphere(scn, mujoco, _ground(info.stance_anchor), 0.014, color * np.array([1, 1, 1, 0.75]))

            if show_swing_path and leg in debug.swing_paths:
                path = debug.swing_paths[leg]
                for i in range(len(path) - 1):
                    _init_capsule(scn, mujoco, path[i], path[i + 1], 0.005, PATH_RGBA)

            _init_sphere(scn, mujoco, info.ref_position, 0.012, REF_RGBA)

            if show_actual and leg in foot_pos:
                _init_sphere(scn, mujoco, foot_pos[leg], 0.010, ACTUAL_RGBA)
