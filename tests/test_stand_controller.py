"""Tests for MIT-style stand controller."""

import numpy as np

from quadruped.config_loader import load_robot_config
from quadruped.controllers.low_level import JointController
from quadruped.controllers.stand import StandController
from quadruped.types import RobotState


class TestStandController:
    def test_default_pose_shape(self):
        cfg = load_robot_config()
        stand = StandController.from_robot_config(cfg)
        cmd = stand.command(np.zeros(12))
        assert cmd.q_des.shape == (12,)
        assert cmd.kp.shape == (12,)
        assert cmd.kd.shape == (12,)
        assert np.allclose(cmd.dq_des, 0.0)

    def test_joint_pd_with_gravity_ff(self):
        cfg = load_robot_config()
        stand = StandController.from_robot_config(cfg)
        joint = JointController.from_robot_config(cfg)

        state = RobotState(
            t=0.0,
            q=stand.q_des.copy(),
            dq=np.zeros(12),
            base_pos=np.zeros(3),
            base_vel=np.zeros(3),
            base_rpy=np.zeros(3),
            base_omega=np.zeros(3),
        )
        tau_g = np.array([0.5, -2.0, 1.0] * 4)
        cmd = stand.command(tau_g)
        tau = joint.compute(state, cmd)
        assert tau.shape == (12,)
        assert np.allclose(tau, tau_g)
