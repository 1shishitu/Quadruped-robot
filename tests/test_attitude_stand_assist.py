"""Tests for IMU attitude stand assist."""

import numpy as np

from quadruped.config_loader import load_robot_config
from quadruped.controllers.low_level import AttitudeStandAssist
from quadruped.types import RobotState


def _state(
    roll: float = 0.0,
    pitch: float = 0.0,
    *,
    contact: dict[str, bool] | None = None,
) -> RobotState:
    legs = ["FL", "FR", "RL", "RR"]
    if contact is None:
        contact = {leg: True for leg in legs}
    return RobotState(
        t=0.0,
        q=np.zeros(12),
        dq=np.zeros(12),
        base_pos=np.zeros(3),
        base_vel=np.zeros(3),
        base_rpy=np.array([roll, pitch, 0.0]),
        base_omega=np.zeros(3),
        contact=contact,
    )


class TestAttitudeStandAssist:
    def test_zero_tilt_no_correction(self):
        cfg = load_robot_config()["robot"]
        assist = AttitudeStandAssist(cfg)
        out = assist.compute(_state(), "hold")
        assert np.allclose(out.dq_corr, 0.0)
        assert out.active is False

    def test_pitch_produces_front_rear_opposite(self):
        cfg = load_robot_config()["robot"]
        assist = AttitudeStandAssist(cfg)
        out = assist.compute(_state(pitch=0.2), "hold")
        # +pitch (nose up) → front/rear thigh corrections oppose
        assert out.dq_corr[1] * out.dq_corr[7] < 0.0
        assert abs(out.dq_corr[1]) > 0.01

    def test_gate_freezes_trajectory(self):
        cfg = load_robot_config()["robot"]
        assist = AttitudeStandAssist(cfg)
        out = assist.compute(_state(roll=0.3), "fuse")
        assert out.freeze_trajectory is True

    def test_requires_feet_when_configured(self):
        cfg = load_robot_config()["robot"]
        assist = AttitudeStandAssist(cfg)
        out = assist.compute(_state(pitch=0.2, contact={"FL": False, "FR": False, "RL": False, "RR": False}), "hold")
        assert out.active is False
        assert np.allclose(out.dq_corr, 0.0)

    def test_reset_clears_gate_latch(self):
        cfg = load_robot_config()["robot"]
        assist = AttitudeStandAssist(cfg)
        assist.compute(_state(roll=0.3), "fuse")
        assist.reset()
        out = assist.compute(_state(roll=0.05), "fuse")
        assert out.freeze_trajectory is False
