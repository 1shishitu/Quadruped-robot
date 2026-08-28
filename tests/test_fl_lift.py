"""FL single-leg vertical lift cycle."""

import numpy as np
import mujoco
import pytest

from quadruped.config_loader import load_gait_config, load_locomotion_config, load_robot_config
from quadruped.controllers.gait_controller import FlLiftController
from quadruped.controllers.low_level import (
    AttitudeStandAssist,
    BalanceController,
    JointController,
    WBCController,
)
from quadruped.controllers.stand import StandController
from quadruped.planners.fl_lift_planner import FlLiftPlanner
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


def _settle(model, data, cfg, act_ids, steps=4000):
    qa, _, _ = build_joint_maps(model, cfg)
    for qa_i, q in zip(qa, default_joint_vector(cfg)):
        data.qpos[qa_i] = q
    mujoco.mj_forward(model, data)
    stand = StandController.from_robot_config({"robot": cfg})
    joint = JointController.from_robot_config({"robot": cfg})
    for _ in range(steps):
        st = read_joint_state(model, data, cfg)
        tau = joint.compute(st, stand.command(joint_gravity_torques(model, data, cfg)))
        apply_joint_torques(model, data, tau, act_ids)
        mujoco.mj_step(model, data)


@pytest.fixture
def fl_lift_sim():
    cfg = load_robot_config()["robot"]
    urdf = prepare_urdf(resolve_urdf(cfg["urdf_path"]))
    model, lift, _, dt = load_model_with_floor(urdf, cfg)
    data = mujoco.MjData(model)
    _, _, act_ids = build_joint_maps(model, cfg)
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [0.0, 0.0, lift]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    _settle(model, data, cfg, act_ids)
    loco_yaml = load_locomotion_config()
    gait_yaml = load_gait_config("fl_lift")
    planner = FlLiftPlanner.from_config(gait_yaml, cfg)
    st = read_joint_state(model, data, cfg, include_contact=True)
    tp = TrunkPlanner(default_height=float(st.base_pos[2]))
    tp.preview_horizon = 0.0
    ctrl = FlLiftController.from_config(
        cfg,
        loco_yaml,
        gait_yaml,
        planner=planner,
        balance=BalanceController.from_config(loco_yaml, cfg),
        wbc=WBCController.from_config(loco_yaml),
        trunk_planner=tp,
        attitude=AttitudeStandAssist(cfg),
    )
    ctrl.begin(st, read_foot_positions(model, data, cfg))
    return model, data, cfg, act_ids, dt, ctrl, planner


class TestFlLift:
    def test_planner_hold_raises_foot(self):
        g = load_gait_config("fl_lift")
        p = FlLiftPlanner.from_config(g)
        base = np.array([0.0, 0.0, 0.28])
        rpy = np.zeros(3)
        foot = {"FL": np.array([0.19, 0.20, 0.05])}
        p.set_stance_positions(foot, base_pos=base, base_rpy=rpy)
        hold_t = (p._edges[2] + 0.05) / p.frequency
        refs = p.update(hold_t, base_pos=base, base_rpy=rpy)
        fl = refs.by_leg("FL")
        assert not fl.contact
        assert fl.position[2] > foot["FL"][2] + p.clearance * 0.8

    def test_sim_stays_upright_two_cycles(self, fl_lift_sim):
        model, data, cfg, act_ids, dt, ctrl, planner = fl_lift_sim
        z0 = float(data.qpos[2])
        steps = int(2.0 * planner.period / dt)
        max_roll = 0.0
        for i in range(steps):
            t = i * dt
            st = read_joint_state(model, data, cfg, t=t, include_contact=True)
            tau = ctrl.compute(model, data, st, gait_t=t, dt=dt)
            apply_joint_torques(model, data, tau, act_ids)
            mujoco.mj_step(model, data)
            max_roll = max(max_roll, abs(float(st.base_rpy[0])))
        assert float(data.qpos[2]) > z0 * 0.88
        assert max_roll < 0.25
