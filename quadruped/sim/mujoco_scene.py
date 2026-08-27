"""MuJoCo scene: URDF prep, floor, floating base, joint motors."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from quadruped.config_loader import PROJECT_ROOT
from quadruped.sim.mujoco_robot import (
    reset_joint_vector,
    lowest_foot_z,
    set_joint_vector,
)

PACKAGE_URI = re.compile(r"package://[^/]+/")


def resolve_urdf(urdf_rel: str) -> Path:
    path = (PROJECT_ROOT / urdf_rel).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"URDF not found: {path}")
    return path


def _package_root(urdf_path: Path) -> Path:
    if urdf_path.parent.name == "urdf":
        return urdf_path.parent.parent
    return urdf_path.parent


def prepare_urdf(urdf_path: Path) -> Path:
    """Resolve package:// paths for MuJoCo."""
    pkg_root = _package_root(urdf_path)
    text = urdf_path.read_text(encoding="utf-8")
    original = text

    text = PACKAGE_URI.sub(f"{pkg_root.as_posix()}/", text)
    if 'meshdir="meshes"' in text or "meshdir='meshes'" in text:
        text = text.replace('filename="meshes/', 'filename="')
        text = text.replace("filename='meshes/", "filename='")

    if text == original:
        return urdf_path

    out = urdf_path.parent / f".{urdf_path.stem}_mujoco.urdf"
    out.write_text(text, encoding="utf-8")
    return out


def compute_robot_lift(
    urdf_path: Path,
    robot_cfg: dict,
    q: np.ndarray | None = None,
    margin: float = 0.002,
) -> float:
    import mujoco

    if q is None:
        q = reset_joint_vector(robot_cfg)
    robot = mujoco.MjSpec.from_file(str(urdf_path)).compile()
    data = mujoco.MjData(robot)
    set_joint_vector(robot, data, robot_cfg, q)
    min_z = lowest_foot_z(robot, data)
    if not np.isfinite(min_z):
        raise RuntimeError("No foot collision geoms found for lift computation")
    return max(0.0, -min_z + margin)


def _joint_torque_limit(joint_name: str, robot_cfg: dict) -> float:
    suffixes = robot_cfg.get("joint_suffix", ["hip", "thigh", "calf"])
    limits = robot_cfg["torque_limits"]
    name = joint_name.lstrip("/").lower()
    for suffix, limit in zip(suffixes, limits):
        if f"_{suffix}_joint" in name:
            return float(limit)
    return float(limits[0])


def _add_joint_motors(scene, robot_cfg: dict) -> None:
    import mujoco

    for joint in scene.joints:
        if not joint.name or "floating" in joint.name:
            continue
        torque_limit = _joint_torque_limit(joint.name, robot_cfg)
        scene.add_actuator(
            name=f"{joint.name.lstrip('/')}_motor",
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            target=joint.name,
            gear=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            forcelimited=1,
            forcerange=[-torque_limit, torque_limit],
        )


def load_model_with_floor(urdf_path: Path, robot_cfg: dict):
    import mujoco

    gravity = float(robot_cfg.get("gravity", 9.81))
    dt = float(robot_cfg.get("sim_dt", 0.002))
    lift = compute_robot_lift(urdf_path, robot_cfg)

    scene = mujoco.MjSpec()
    scene.option.gravity = [0.0, 0.0, -gravity]
    scene.option.timestep = dt

    scene.add_texture(
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        name="grid",
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        mark=mujoco.mjtMark.mjMARK_EDGE,
        markrgb=[0.8, 0.8, 0.8],
        rgb1=[1.0, 1.0, 1.0],
        rgb2=[0.88, 0.88, 0.88],
        width=512,
        height=512,
    )
    mat = scene.add_material(name="grid", texrepeat=[12, 12], texuniform=True, reflectance=0.05)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    scene.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        pos=[0, 0, 0],
        size=[0, 0, 0.05],
        material="grid",
        condim=3,
        friction=[1.0, 0.005, 0.0001],
    )

    robot = mujoco.MjSpec.from_file(str(urdf_path))
    root = scene.worldbody.add_body(name="robot_root", pos=[0, 0, lift])
    root.add_freejoint(name="floating_base_joint")
    anchor = root.add_site(name="robot_anchor", pos=[0, 0, 0], size=[0.001, 0.001, 0.001])
    scene.attach(robot, site=anchor)
    _add_joint_motors(scene, robot_cfg)
    return scene.compile(), lift, gravity, dt
