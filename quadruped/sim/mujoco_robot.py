"""MuJoCo robot I/O helpers (joint order aligned with real pipeline)."""

from __future__ import annotations

import numpy as np

from quadruped.types import RobotState


def joint_id(model, name: str) -> int:
    import mujoco

    for candidate in (name, f"/{name}", name.lstrip("/")):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, candidate)
        if jid >= 0:
            return jid
    return -1


def actuator_id(model, name: str) -> int:
    import mujoco

    for candidate in (name, f"/{name}", name.lstrip("/")):
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, candidate)
        if aid >= 0:
            return aid
    return -1


def build_joint_maps(model, robot_cfg: dict) -> tuple[list[int], list[int], list[int]]:
    """Return (qpos indices, dof indices, actuator ids) in leg order."""
    qa_list: list[int] = []
    da_list: list[int] = []
    act_list: list[int] = []

    for leg in robot_cfg.get("leg_names", []):
        for suffix in robot_cfg.get("joint_suffix", []):
            jname = f"{leg}_{suffix}_joint"
            jid = joint_id(model, jname)
            if jid < 0:
                raise RuntimeError(f"Joint not found: {jname}")
            qa_list.append(int(model.jnt_qposadr[jid]))
            da_list.append(int(model.jnt_dofadr[jid]))
            aid = actuator_id(model, f"{jname}_motor")
            act_list.append(aid)

    return qa_list, da_list, act_list


def default_joint_vector(robot_cfg: dict) -> np.ndarray:
    q_des = []
    for leg in robot_cfg.get("leg_names", []):
        angles = robot_cfg.get("default_joint_angles", {}).get(leg)
        if angles is None:
            raise KeyError(f"Missing default_joint_angles for {leg}")
        q_des.extend(float(a) for a in angles)
    return np.asarray(q_des, dtype=float)


def _is_foot_contact_geom(model, geom_id: int) -> bool:
    import mujoco

    body_name = model.body(model.geom_bodyid[geom_id]).name
    if body_name.endswith("_foot"):
        return True
    if not body_name.endswith("_calf"):
        return False
    gtype = int(model.geom_type[geom_id])
    return gtype in (
        int(mujoco.mjtGeom.mjGEOM_SPHERE),
        int(mujoco.mjtGeom.mjGEOM_CYLINDER),
        int(mujoco.mjtGeom.mjGEOM_CAPSULE),
    )


def lowest_foot_z(model, data, *, skip: frozenset[str] = frozenset({"floor"})) -> float:
    import mujoco

    mujoco.mj_forward(model, data)
    min_z = float("inf")

    for i in range(model.ngeom):
        if model.geom(i).name in skip:
            continue
        if not _is_foot_contact_geom(model, i):
            continue

        pos = data.geom_xpos[i]
        gtype = int(model.geom_type[i])
        if gtype == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            bottom = float(pos[2] - model.geom_size[i][0])
        elif gtype in (
            int(mujoco.mjtGeom.mjGEOM_CYLINDER),
            int(mujoco.mjtGeom.mjGEOM_CAPSULE),
        ):
            bottom = float(pos[2] - model.geom_size[i][1])
        else:
            bottom = float(pos[2] - model.geom_rbound[i])
        min_z = min(min_z, bottom)

    return min_z


def reset_joint_vector(robot_cfg: dict) -> np.ndarray:
    stand = robot_cfg.get("stand_joint", {})
    per_leg = stand.get(
        "reset_joint_angles",
        stand.get("collapsed_angles", [0.0, 0.0, 0.0]),
    )
    q = []
    for _leg in robot_cfg.get("leg_names", []):
        q.extend(float(a) for a in per_leg)
    return np.asarray(q, dtype=float)


def set_joint_vector(model, data, robot_cfg: dict, q: np.ndarray) -> None:
    qa_list, _, _ = build_joint_maps(model, robot_cfg)
    for qa, angle in zip(qa_list, q):
        data.qpos[qa] = float(angle)


def set_default_pose(model, data, robot_cfg: dict) -> None:
    set_joint_vector(model, data, robot_cfg, default_joint_vector(robot_cfg))


def set_reset_joint_pose(model, data, robot_cfg: dict) -> None:
    set_joint_vector(model, data, robot_cfg, reset_joint_vector(robot_cfg))


def trunk_body_id(model) -> int:
    import mujoco

    for name in ("trunk", "/trunk", "base", "/base", "robot_root"):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            return bid
    return 1 if model.nbody > 1 else 0


def _hip_body_ids(model, robot_cfg: dict) -> dict[str, int]:
    import mujoco

    ids: dict[str, int] = {}
    for leg in robot_cfg.get("leg_names", []):
        for name in (f"{leg}_hip", f"/{leg}_hip"):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                ids[leg] = bid
                break
    return ids


def ensure_mj_forward(model, data) -> None:
    """One kinematics pass per control step (after previous mj_step, before reads)."""
    import mujoco

    mujoco.mj_forward(model, data)


def read_hip_positions(model, data, robot_cfg: dict) -> dict[str, np.ndarray]:
    positions: dict[str, np.ndarray] = {}
    for leg, bid in _hip_body_ids(model, robot_cfg).items():
        positions[leg] = data.xpos[bid].copy()
    return positions


def read_foot_positions(model, data, robot_cfg: dict) -> dict[str, np.ndarray]:
    positions: dict[str, np.ndarray] = {}
    for leg, bid in _foot_body_ids(model, robot_cfg).items():
        positions[leg] = data.xpos[bid].copy()
    return positions


def read_foot_velocities(model, data, robot_cfg: dict) -> dict[str, np.ndarray]:
    import mujoco

    velocities: dict[str, np.ndarray] = {}
    foot_bodies = _foot_body_ids(model, robot_cfg)
    for leg, bid in foot_bodies.items():
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, jacr, bid)
        velocities[leg] = jacp @ data.qvel
    return velocities


def foot_jacobian(model, data, robot_cfg: dict, leg: str) -> np.ndarray:
    """3×3 foot Jacobian w.r.t. leg joint velocities (hip, thigh, calf)."""
    import mujoco

    foot_bodies = _foot_body_ids(model, robot_cfg)
    if leg not in foot_bodies:
        raise KeyError(f"Foot body not found for leg {leg}")
    _, da_list, _ = build_joint_maps(model, robot_cfg)
    leg_idx = robot_cfg.get("leg_names", []).index(leg)
    dof_cols = [da_list[leg_idx * 3 + j] for j in range(3)]

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jacp, jacr, foot_bodies[leg])
    return jacp[:, dof_cols].copy()


def _foot_body_ids(model, robot_cfg: dict) -> dict[str, int]:
    import mujoco

    ids: dict[str, int] = {}
    for leg in robot_cfg.get("leg_names", []):
        for name in (f"{leg}_foot", f"/{leg}_foot"):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                ids[leg] = bid
                break
    return ids


def read_foot_contact(
    model,
    data,
    robot_cfg: dict,
    *,
    min_normal_force: float = 1.0,
) -> dict[str, bool]:
    """Per-leg ground contact from MuJoCo contact normals (sim stand-in for foot FSR)."""
    import mujoco

    legs = robot_cfg.get("leg_names", [])
    contact = {leg: False for leg in legs}
    foot_bodies = _foot_body_ids(model, robot_cfg)
    if not foot_bodies:
        return contact

    force6 = np.zeros(6, dtype=float)
    for i in range(data.ncon):
        con = data.contact[i]
        b1 = int(model.geom_bodyid[con.geom1])
        b2 = int(model.geom_bodyid[con.geom2])
        leg_hit = None
        for leg, bid in foot_bodies.items():
            if bid in (b1, b2):
                leg_hit = leg
                break
        if leg_hit is None:
            continue
        mujoco.mj_contactForce(model, data, i, force6)
        if float(force6[0]) >= min_normal_force:
            contact[leg_hit] = True
    return contact


def read_joint_state(
    model,
    data,
    robot_cfg: dict,
    *,
    t: float = 0.0,
    contact_min_force: float = 1.0,
    include_contact: bool = True,
) -> RobotState:
    qa_list, da_list, _ = build_joint_maps(model, robot_cfg)
    q = np.array([data.qpos[qa] for qa in qa_list], dtype=float)
    dq = np.array([data.qvel[da] for da in da_list], dtype=float)

    base_pos = data.qpos[0:3].copy()
    base_vel = data.qvel[0:3].copy()
    base_rpy = _quat_to_rpy(data.qpos[3:7])
    base_omega = data.qvel[3:6].copy()
    if include_contact:
        contact = read_foot_contact(
            model, data, robot_cfg, min_normal_force=contact_min_force
        )
    else:
        contact = {leg: False for leg in robot_cfg.get("leg_names", [])}

    return RobotState(
        t=t,
        q=q,
        dq=dq,
        base_pos=base_pos,
        base_vel=base_vel,
        base_rpy=base_rpy,
        base_omega=base_omega,
        contact=contact,
    )


def joint_gravity_torques(model, data, robot_cfg: dict) -> np.ndarray:
    """MIT-style gravity feedforward: bias forces on joint dofs (τ_g + Coriolis)."""
    _, da_list, _ = build_joint_maps(model, robot_cfg)
    return np.array([data.qfrc_bias[da] for da in da_list], dtype=float)


def apply_joint_torques(model, data, tau: np.ndarray, act_ids: list[int]) -> None:
    data.qfrc_applied[:] = 0.0
    if model.nu > 0:
        data.ctrl[:] = 0.0
    for aid, torque in zip(act_ids, tau):
        if aid >= 0:
            data.ctrl[aid] = float(torque)
        else:
            raise RuntimeError("Motor actuators required for real-aligned stand control")


def zero_joint_torques(model, data) -> None:
    """断电：电机无输出（与实机未使能一致）。"""
    data.qfrc_applied[:] = 0.0
    if model.nu > 0:
        data.ctrl[:] = 0.0


def _quat_to_rpy(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([roll, pitch, yaw], dtype=float)
