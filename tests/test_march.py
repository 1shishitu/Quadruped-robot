"""March-in-place gait tests."""

import numpy as np
import pytest

from quadruped.config_loader import load_gait_config, load_locomotion_config
from quadruped.planners.foot_planner import FootPlanner
from quadruped.planners.gait_scheduler import GaitScheduler
from quadruped.utils.raibert import in_place_touchdown


class TestMarchInPlace:
    def test_in_place_touchdown_matches_stance_xy(self):
        foot = np.array([0.19, 0.20, 0.18])
        target = in_place_touchdown(foot)
        np.testing.assert_allclose(target, foot)

    def test_foot_planner_in_place_swing_target(self):
        gait = GaitScheduler.from_config(load_gait_config("march"))
        fp = FootPlanner.from_config(load_gait_config("march"), gait)
        stance = {
            "FL": np.array([0.1881, 0.203, 0.18]),
            "FR": np.array([0.1881, -0.203, 0.18]),
            "RL": np.array([-0.1881, 0.203, 0.18]),
            "RR": np.array([-0.1881, -0.203, 0.18]),
        }
        fp.set_stance_positions(stance)
        hip = {leg: p + np.array([0, 0, 0.28]) for leg, p in stance.items()}
        fp.update(0.0, np.zeros(3), np.zeros(3), hip)
        fl = fp.debug.legs["FL"]
        assert fl.in_stance is False
        np.testing.assert_allclose(fl.raibert_target, fl.stance_anchor, atol=1e-6)

    def test_locomotion_default_is_march(self):
        loco = load_locomotion_config()["locomotion"]
        assert loco.get("mode") == "march_in_place"
        assert loco.get("gait_config") == "march"
