"""Tests for sim timing configuration."""

from quadruped.config_loader import load_robot_config
from quadruped.sim.sim_timing import SimTiming


class TestSimTiming:
    def test_from_robot_config(self):
        cfg = load_robot_config()["robot"]
        timing = SimTiming.from_robot_config(cfg, sim_dt=0.002)
        assert timing.control_rate_hz == 500.0
        assert timing.viewer_rate_hz == 30.0
        assert timing.print_rate_hz == 0.25
        assert timing.realtime_scale == 1.0
        assert timing.steps_per_viewer_frame >= 1
