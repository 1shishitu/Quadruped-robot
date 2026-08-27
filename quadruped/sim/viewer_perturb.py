"""MuJoCo passive viewer mouse perturbation (Simulate-style push/pull)."""

from __future__ import annotations

import mujoco


def perturb_config(robot_cfg: dict) -> dict:
    return robot_cfg.get("sim_perturb", {})


def perturb_enabled(robot_cfg: dict) -> bool:
    return bool(perturb_config(robot_cfg).get("enabled", True))


def init_default_perturb_target(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    pert: mujoco.MjvPerturb,
    robot_cfg: dict,
) -> int:
    """Select default body so Ctrl+drag works without double-click."""
    cfg = perturb_config(robot_cfg)
    target = str(cfg.get("default_body", "trunk"))
    candidates = (target, f"/{target}", f"{target}_link")
    body_id = -1
    for name in candidates:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id >= 0:
            break
    if body_id < 0:
        return -1

    mujoco.mj_forward(model, data)
    pert.select = body_id
    pert.refpos[:] = data.xpos[body_id]
    pert.refselpos[:] = data.xpos[body_id]
    mujoco.mju_mat2Quat(pert.refquat, data.xmat[body_id])
    pert.active = 0
    pert.active2 = 0
    return body_id


def apply_viewer_perturbation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    pert: mujoco.MjvPerturb,
) -> None:
    """
    Write external wrench from viewer drag into ``data.xfrc_applied``.

    Matches MuJoCo Simulate: virtual spring from mouse → force on selected body.
    """
    data.xfrc_applied[:] = 0.0
    if int(pert.select) <= 0:
        return
    mujoco.mjv_applyPerturbPose(model, data, pert, 0)
    mujoco.mjv_applyPerturbForce(model, data, pert)


def selected_body_name(model: mujoco.MjModel, pert: mujoco.MjvPerturb) -> str:
    sel = int(pert.select)
    if sel <= 0:
        return "none"
    return model.body(sel).name or f"body_{sel}"
