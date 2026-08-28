"""Locomotion: forward walk (mode=walk, gait_config/trot)."""

import numpy as np
import mujoco
import pytest

from quadruped.config_loader import load_gait_config, load_locomotion_config, load_robot_config
from quadruped.controllers.low_level import JointController
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


def _loco_step(model, data, cfg, act_ids, *, dt, gait_t, v_cmd, omega, loco) -> None:
    state = read_joint_state(model, data, cfg)
    tau = loco.compute(
        model,
        data,
        state,
        gait_t=gait_t,
        dt=dt,
        v_cmd=v_cmd,
        omega_cmd=omega,
    )
    apply_joint_torques(model, data, tau, act_ids)
    mujoco.mj_step(model, data)


class TestLocomotionStability:
    @pytest.fixture
    def sim(self):
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
                "mode": "walk",
                "gait_config": "trot",
            }
        }
        gait_cfg = load_gait_config("trot")
        gait = GaitScheduler.from_config(gait_cfg)
        fp = FootPlanner.from_config(gait_cfg, gait)
        fp.set_stance_positions(read_foot_positions(model, data, cfg))
        state = read_joint_state(model, data, cfg)
        tp = TrunkPlanner(default_height=0.28)
        tp.preview_horizon = float(loco_yaml["locomotion"]["trunk"].get("preview_horizon", 0.08))
        tp.reset(
            p_com=np.array([state.base_pos[0], state.base_pos[1], 0.28]),
            yaw=float(state.base_rpy[2]),
        )
        from quadruped.controllers.low_level import BalanceController, WBCController

        loco = LocomotionController.from_config(
            cfg,
            loco_yaml,
            gait_cfg,
            balance=BalanceController.from_config(loco_yaml, cfg),
            wbc=WBCController.from_config(loco_yaml),
            foot_planner=fp,
            trunk_planner=tp,
        )
        return model, data, cfg, act_ids, dt, loco

    def test_idle_walk_no_drift(self, sim):
        """Walk mode with v=0 uses 4-stance hold (not trot)."""
        model, data, cfg, act_ids, dt, loco = sim
        x0, y0 = float(data.qpos[0]), float(data.qpos[1])
        z0 = float(data.qpos[2])
        v_zero = np.zeros(3)
        for step in range(800):
            _loco_step(
                model,
                data,
                cfg,
                act_ids,
                dt=dt,
                gait_t=step * dt,
                v_cmd=v_zero,
                omega=0.0,
                loco=loco,
            )
        assert abs(float(data.qpos[0]) - x0) < 0.08
        assert abs(float(data.qpos[1]) - y0) < 0.08
        assert float(data.qpos[2]) > z0 * 0.92

    def test_locomotion_keeps_height(self, sim):
        model, data, cfg, act_ids, dt, loco = sim
        z0 = float(data.qpos[2])
        v_cmd = np.array([0.4, 0.0, 0.0])
        for step in range(500):
            _loco_step(
                model,
                data,
                cfg,
                act_ids,
                dt=dt,
                gait_t=step * dt,
                v_cmd=v_cmd,
                omega=0.0,
                loco=loco,
            )
        assert float(data.qpos[2]) > z0 * 0.75
        assert float(data.qpos[2]) > 0.15

    def test_locomotion_moves_forward(self, sim):
        model, data, cfg, act_ids, dt, loco = sim
        x0 = float(data.qpos[0])
        v_cmd = np.array([0.35, 0.0, 0.0])
        for step in range(1200):
            _loco_step(
                model,
                data,
                cfg,
                act_ids,
                dt=dt,
                gait_t=step * dt,
                v_cmd=v_cmd,
                omega=0.0,
                loco=loco,
            )
        assert float(data.qpos[0]) - x0 > 0.02
