"""Leg inverse kinematics tests."""

import mujoco
import numpy as np
import pytest

from quadruped.config_loader import load_robot_config
from quadruped.models.leg_ik import LegIK
from quadruped.sim.mujoco_robot import (
    build_joint_maps,
    default_joint_vector,
    read_foot_positions,
)
from quadruped.sim.mujoco_scene import load_model_with_floor, prepare_urdf, resolve_urdf


@pytest.fixture
def standing_sim():
    cfg = load_robot_config()["robot"]
    urdf = prepare_urdf(resolve_urdf(cfg["urdf_path"]))
    model, lift, _, _ = load_model_with_floor(urdf, cfg)
    data = mujoco.MjData(model)
    qa, _, _ = build_joint_maps(model, cfg)
    q_des = default_joint_vector(cfg)
    data.qpos[0:3] = [0.0, 0.0, lift]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    for qa_i, q in zip(qa, q_des):
        data.qpos[qa_i] = q
    mujoco.mj_forward(model, data)
    return model, data, cfg


def _leg_q(data, qa_list, leg_idx: int) -> np.ndarray:
    return np.array([data.qpos[qa_list[leg_idx * 3 + j]] for j in range(3)])


class TestLegIK:
    def test_recovers_current_foot_pose(self, standing_sim):
        model, data, cfg = standing_sim
        ik = LegIK(cfg)
        foot_pos = read_foot_positions(model, data, cfg)
        qa_list = build_joint_maps(model, cfg)[0]
        for leg_idx, leg in enumerate(cfg["leg_names"]):
            q_now = _leg_q(data, qa_list, leg_idx)
            q_sol = ik.solve(model, data, leg, foot_pos[leg], q_seed=q_now)
            assert np.allclose(q_sol, q_now, atol=0.05)

    def test_lifted_target_changes_calf(self, standing_sim):
        model, data, cfg = standing_sim
        ik = LegIK(cfg)
        leg = "FL"
        foot_pos = read_foot_positions(model, data, cfg)[leg].copy()
        target = foot_pos.copy()
        target[2] += 0.05
        leg_idx = cfg["leg_names"].index(leg)
        qa_list = build_joint_maps(model, cfg)[0]
        q_seed = _leg_q(data, qa_list, leg_idx)
        q_sol = ik.solve(model, data, leg, target, q_seed=q_seed)

        saved = _leg_q(data, qa_list, leg_idx).copy()
        for j in range(3):
            data.qpos[qa_list[leg_idx * 3 + j]] = q_sol[j]
        mujoco.mj_forward(model, data)
        foot_new = read_foot_positions(model, data, cfg)[leg]
        for j in range(3):
            data.qpos[qa_list[leg_idx * 3 + j]] = saved[j]
        mujoco.mj_forward(model, data)

        assert foot_new[2] > foot_pos[2] + 0.02
        assert abs(float(foot_new[2] - target[2])) < 0.02
