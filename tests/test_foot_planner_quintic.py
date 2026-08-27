"""Foot planner quintic swing tests."""

import numpy as np
import pytest

from quadruped.planners.foot_planner import FootPlanner
from quadruped.planners.gait_scheduler import GaitScheduler


class TestFootPlannerQuintic:
    def test_swing_midpoint_raises_foot(self):
        gait = GaitScheduler(frequency=2.0, swing_ratio=0.5)
        planner = FootPlanner(gait, clearance=0.05)
        stance = {
            "FL": np.array([0.2, 0.05, 0.0]),
            "FR": np.array([0.2, -0.05, 0.0]),
            "RL": np.array([-0.2, 0.05, 0.0]),
            "RR": np.array([-0.2, -0.05, 0.0]),
        }
        planner.set_stance_positions(stance)
        hip = {leg: p + np.array([0.0, 0.0, 0.28]) for leg, p in stance.items()}
        v = np.zeros(3)
        v_cmd = np.zeros(3)

        # t=0: FR in stance for diagonal trot
        t0 = planner.update(0.0, v, v_cmd, hip)
        assert t0.FR.contact is True

        mid_t = gait.swing_duration * 0.5
        mid = planner.update(mid_t, v, v_cmd, hip)
        assert mid.FL.contact is False
        assert mid.FL.position[2] >= stance["FL"][2]

        late = planner.update(gait.swing_duration * 0.95, v, v_cmd, hip)
        assert late.FL.contact is False

    def test_forward_cmd_does_not_pull_feet_inward(self):
        gait = GaitScheduler(frequency=1.5, swing_ratio=0.5)
        planner = FootPlanner(
            gait,
            clearance=0.05,
            placement="raibert",
            raibert_kv=0.03,
        )
        stance = {
            "FL": np.array([0.1881, 0.203, 0.0]),
            "FR": np.array([0.1881, -0.203, 0.0]),
            "RL": np.array([-0.1881, 0.203, 0.0]),
            "RR": np.array([-0.1881, -0.203, 0.0]),
        }
        planner.set_stance_positions(stance)
        hip = {leg: p + np.array([0.0, 0.0, 0.28]) for leg, p in stance.items()}
        v_cmd = np.array([0.15, 0.0, 0.0])
        planner.update(0.0, np.zeros(3), v_cmd, hip)
        fl = planner.debug.legs["FL"]
        assert fl.in_stance is False
        assert fl.raibert_target[0] > fl.stance_anchor[0]
        assert fl.raibert_target[1] == pytest.approx(fl.stance_anchor[1], abs=0.01)
