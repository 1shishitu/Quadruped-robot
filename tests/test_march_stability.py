"""Locomotion stability — march in place (default mode)."""

import numpy as np
import mujoco
import pytest

from quadruped.config_loader import (
    load_gait_config_for_locomotion,
    load_locomotion_config,
    load_robot_config,
)
from quadruped.controllers.low_level import AttitudeStandAssist, BalanceController, JointController
from quadruped.controllers.gait_controller import LocomotionController
from quadruped.controllers.stand import StandController
from quadruped.planners.foot_planner import FootPlanner
from quadruped.planners.gait_scheduler import GaitScheduler
from quadruped.planners.trunk_planner import TrunkPlanner
from quadruped.sim.mujoco_robot import (
    apply_joint_torques,
    build_joint_maps,
    default_joint_vector,
    joint_gravity_torques,
    read_foot_positions,
    read_joint_state,
)
from quadruped.sim.mujoco_scene import load_model_with_floor, prepare_urdf, resolve_urdf


def _settle_standing(model, data, cfg, act_ids, steps: int = 4000) -> None:
    q_des = default_joint_vector(cfg)
    qa, _, _ = build_joint_maps(model, cfg)
    for qa_i, q in zip(qa, q_des):
        data.qpos[qa_i] = q
    mujoco.mj_forward(model, data)
    stand = StandController.from_robot_config({"robot": cfg})
    joint = JointController.from_robot_config({"robot": cfg})
    for _ in range(steps):
        state = read_joint_state(model, data, cfg)
        cmd = stand.command(joint_gravity_torques(model, data, cfg))
        tau = joint.compute(state, cmd)
        apply_joint_torques(model, data, tau, act_ids)
        mujoco.mj_step(model, data)


def _loco_step(model, data, cfg, act_ids, *, dt, gait_t, loco) -> None:
    state = read_joint_state(model, data, cfg)
    tau = loco.compute(
        model,
        data,
        state,
        gait_t=gait_t,
        dt=dt,
        v_cmd=np.zeros(3),
        omega_cmd=0.0,
    )
    apply_joint_torques(model, data, tau, act_ids)
    mujoco.mj_step(model, data)


@pytest.fixture
def march_sim():
    cfg = load_robot_config()["robot"]
    urdf = prepare_urdf(resolve_urdf(cfg["urdf_path"]))
    model, lift, _, dt = load_model_with_floor(urdf, cfg)
    data = mujoco.MjData(model)
    _, _, act_ids = build_joint_maps(model, cfg)
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [0.0, 0.0, lift]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    _settle_standing(model, data, cfg, act_ids)
    loco_yaml = load_locomotion_config()
    loco_yaml = {
        "locomotion": {
            **loco_yaml["locomotion"],
            "mode": "march_in_place",
            "gait_config": "march",
        }
    }
    gait_cfg = load_gait_config_for_locomotion(loco_yaml)
    gait = GaitScheduler.from_config(gait_cfg)
    fp = FootPlanner.from_config(gait_cfg, gait)
    state = read_joint_state(model, data, cfg)
    fp.set_stance_positions(
        read_foot_positions(model, data, cfg),
        base_pos=state.base_pos,
        base_rpy=state.base_rpy,
        yaw=float(state.base_rpy[2]),
    )
    height = float(state.base_pos[2])
    tp = TrunkPlanner(default_height=height)
    tp.preview_horizon = 0.0
    tp.reset(
        p_com=np.array([state.base_pos[0], state.base_pos[1], height]),
        yaw=float(state.base_rpy[2]),
    )
    from quadruped.controllers.low_level import WBCController

    loco = LocomotionController.from_config(
        cfg,
        loco_yaml,
        gait_cfg,
        balance=BalanceController.from_config(loco_yaml, cfg),
        wbc=WBCController.from_config(loco_yaml),
        foot_planner=fp,
        trunk_planner=tp,
        attitude=AttitudeStandAssist(cfg),
    )
    assert loco.march_in_place
    loco.begin_march(state)
    return model, data, cfg, act_ids, dt, loco, gait


class TestMarchStability:
    def test_march_no_xy_drift(self, march_sim):
        model, data, cfg, act_ids, dt, loco, gait = march_sim
        x0, y0 = float(data.qpos[0]), float(data.qpos[1])
        z0 = float(data.qpos[2])
        steps = int(2.0 / gait.period / dt)  # ~2 s of marching
        for step in range(steps):
            _loco_step(model, data, cfg, act_ids, dt=dt, gait_t=step * dt, loco=loco)
        assert abs(float(data.qpos[0]) - x0) < 0.35
        assert abs(float(data.qpos[1]) - y0) < 0.15
        assert float(data.qpos[2]) > z0 * 0.85

    def test_march_keeps_height_over_one_cycle(self, march_sim):
        model, data, cfg, act_ids, dt, loco, gait = march_sim
        z0 = float(data.qpos[2])
        steps = int(1.5 / gait.period / dt)
        for step in range(steps):
            _loco_step(model, data, cfg, act_ids, dt=dt, gait_t=step * dt, loco=loco)
        assert float(data.qpos[2]) > z0 * 0.82
        assert float(data.qpos[2]) > 0.14
