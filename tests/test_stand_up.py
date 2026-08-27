"""Tests for Unitree-style two-phase stand-up."""

import numpy as np

from quadruped.config_loader import load_robot_config
from quadruped.controllers.stand.stand_up import (
    fuse_pose_vector,
    stand_up_q_des,
    stand_up_total_duration,
    tachi_pose_vector,
)


class TestStandUp:
    def test_pose_vectors_length(self):
        cfg = load_robot_config()["robot"]
        assert fuse_pose_vector(cfg).shape == (12,)
        assert tachi_pose_vector(cfg).shape == (12,)

    def test_fuse_and_tachi_hip_mirror(self):
        cfg = load_robot_config()["robot"]
        q_fuse = fuse_pose_vector(cfg)
        q_tachi = tachi_pose_vector(cfg)
        # FL/RL hip > 0, FR/RR hip < 0 (order FL, FR, RL, RR)
        assert q_fuse[0] > 0 and q_fuse[6] > 0
        assert q_fuse[3] < 0 and q_fuse[9] < 0
        np.testing.assert_allclose(q_tachi[0:3], [0.27, 0.8, -1.6])
        np.testing.assert_allclose(q_tachi[3:6], [-0.27, 0.8, -1.6])

    def test_two_phase_schedule(self):
        q0 = np.zeros(12)
        q_fuse = np.ones(12)
        q_tachi = np.full(12, 2.0)
        dur = 10.0

        q, phase = stand_up_q_des(q0, q_fuse, q_tachi, 0.0, dur)
        assert phase == "fuse"
        assert np.allclose(q, q0)

        q, phase = stand_up_q_des(q0, q_fuse, q_tachi, 5.0, dur)
        assert phase == "fuse"
        assert np.allclose(q, 0.5 * q_fuse)

        q, phase = stand_up_q_des(q0, q_fuse, q_tachi, 10.0, dur)
        assert phase == "tachi"
        assert np.allclose(q, q_fuse)

        q, phase = stand_up_q_des(q0, q_fuse, q_tachi, 15.0, dur)
        assert phase == "tachi"
        assert np.allclose(q, 0.5 * q_fuse + 0.5 * q_tachi)

        q, phase = stand_up_q_des(q0, q_fuse, q_tachi, 20.0, dur)
        assert phase == "hold"
        assert np.allclose(q, q_tachi)

    def test_total_duration_from_config(self):
        cfg = load_robot_config()["robot"]
        assert stand_up_total_duration(cfg) == 10.0
