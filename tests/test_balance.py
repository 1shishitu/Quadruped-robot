"""Tests for balance stance force QP."""

import numpy as np
import pytest

from quadruped.controllers.low_level import BalanceController
from quadruped.config_loader import load_locomotion_config, load_robot_config
from quadruped.types import RobotState, TrunkRef


class TestBalanceController:
    @pytest.fixture
    def balance(self):
        return BalanceController.from_config(load_locomotion_config(), load_robot_config())

    @pytest.fixture
    def state(self):
        return RobotState(
            t=0.0,
            q=np.zeros(12),
            dq=np.zeros(12),
            base_pos=np.array([0.0, 0.0, 0.28]),
            base_vel=np.zeros(3),
            base_rpy=np.zeros(3),
            base_omega=np.zeros(3),
        )

    @pytest.fixture
    def ref(self):
        return TrunkRef(
            p_com=np.array([0.0, 0.0, 0.28]),
            v_com=np.zeros(3),
            a_com=np.zeros(3),
            rpy=np.zeros(3),
            omega=np.zeros(3),
        )

    def test_four_stance_with_gravity_enabled(self):
        cfg = load_locomotion_config()
        cfg["locomotion"]["balance"]["include_gravity"] = True
        balance = BalanceController.from_config(cfg, load_robot_config())
        state = RobotState(
            t=0.0,
            q=np.zeros(12),
            dq=np.zeros(12),
            base_pos=np.array([0.0, 0.0, 0.28]),
            base_vel=np.zeros(3),
            base_rpy=np.zeros(3),
            base_omega=np.zeros(3),
        )
        ref = TrunkRef(
            p_com=np.array([0.0, 0.0, 0.28]),
            v_com=np.zeros(3),
            a_com=np.zeros(3),
            rpy=np.zeros(3),
            omega=np.zeros(3),
        )
        feet = {
            "FL": np.array([0.19, 0.05, 0.0]),
            "FR": np.array([0.19, -0.05, 0.0]),
            "RL": np.array([-0.19, 0.05, 0.0]),
            "RR": np.array([-0.19, -0.05, 0.0]),
        }
        f = balance.compute(state, ref, list(feet.keys()), feet)
        total_z = sum(f[3 * i + 2] for i in range(4))
        assert total_z == pytest.approx(balance.mass * balance.gravity, rel=0.05)

    def test_propulsion_produces_horizontal_force(self, balance, state):
        ref = TrunkRef(
            p_com=state.base_pos + np.array([0.05, 0.0, 0.0]),
            v_com=np.array([0.35, 0.0, 0.0]),
            a_com=np.zeros(3),
            rpy=np.zeros(3),
            omega=np.zeros(3),
        )
        feet = {
            "FL": np.array([0.19, 0.05, 0.0]),
            "FR": np.array([0.19, -0.05, 0.0]),
            "RL": np.array([-0.19, 0.05, 0.0]),
            "RR": np.array([-0.19, -0.05, 0.0]),
        }
        f = balance.compute(state, ref, ["FR", "RL"], feet)
        fxy = f.reshape(4, 3)[:, :2]
        assert np.linalg.norm(fxy) > 1.0

    def test_deadband_zeros_horizontal_force(self, balance, state, ref):
        feet = {
            "FL": np.array([0.19, 0.05, 0.0]),
            "FR": np.array([0.19, -0.05, 0.0]),
            "RL": np.array([-0.19, 0.05, 0.0]),
            "RR": np.array([-0.19, -0.05, 0.0]),
        }
        f = balance.compute(state, ref, list(feet.keys()), feet)
        assert abs(f[0]) + abs(f[1]) + abs(f[3]) + abs(f[4]) < 1e-3
        total_z = sum(f[3 * i + 2] for i in range(4))
        assert total_z == pytest.approx(balance.mass * balance.gravity, rel=0.05)

    def test_swing_legs_zero_force(self, balance, state, ref):
        feet = {
            "FL": np.array([0.19, 0.05, 0.0]),
            "FR": np.array([0.19, -0.05, 0.0]),
            "RL": np.array([-0.19, 0.05, 0.0]),
            "RR": np.array([-0.19, -0.05, 0.0]),
        }
        f = balance.compute(state, ref, ["FL"], feet)
        assert np.allclose(f[3:12], 0.0)

    def test_two_stance_pitch_splits_normal_force(self, balance):
        """QP should use differential Fz to resist pitch (not equal split)."""
        state = RobotState(
            t=0.0,
            q=np.zeros(12),
            dq=np.zeros(12),
            base_pos=np.array([0.0, 0.0, 0.28]),
            base_vel=np.zeros(3),
            base_rpy=np.array([0.0, 0.12, 0.0]),
            base_omega=np.zeros(3),
        )
        ref = TrunkRef(
            p_com=np.array([0.0, 0.0, 0.28]),
            v_com=np.zeros(3),
            a_com=np.zeros(3),
            rpy=np.zeros(3),
            omega=np.zeros(3),
        )
        feet = {
            "FL": np.array([0.19, 0.05, 0.0]),
            "FR": np.array([0.19, -0.05, 0.0]),
            "RL": np.array([-0.19, 0.05, 0.0]),
            "RR": np.array([-0.19, -0.05, 0.0]),
        }
        f = balance.compute(state, ref, ["FR", "RL"], feet)
        fr_z = f[3 * 1 + 2]
        rl_z = f[3 * 2 + 2]
        assert fr_z + rl_z == pytest.approx(balance.mass * balance.gravity, rel=0.12)
        # Nose-up pitch error → more load on front than rear (pitch-down moment)
        assert fr_z > rl_z + 0.5
