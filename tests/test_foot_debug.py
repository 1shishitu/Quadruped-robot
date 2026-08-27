"""Foot planner debug + swing path sampling."""

import numpy as np

from quadruped.planners.foot_planner import FootPlanner
from quadruped.planners.gait_scheduler import GaitScheduler


def test_foot_planner_populates_raibert_debug():
    gait = GaitScheduler(frequency=2.0, swing_ratio=0.5)
    fp = FootPlanner(gait, clearance=0.08, raibert_kv=0.03)
    fp.set_stance_positions({
        "FL": np.array([0.2, 0.15, 0.0]),
        "FR": np.array([0.2, -0.15, 0.0]),
        "RL": np.array([-0.2, 0.15, 0.0]),
        "RR": np.array([-0.2, -0.15, 0.0]),
    })
    hip = {
        "FL": np.array([0.19, 0.14, 0.0]),
        "FR": np.array([0.19, -0.14, 0.0]),
        "RL": np.array([-0.19, 0.14, 0.0]),
        "RR": np.array([-0.19, -0.14, 0.0]),
    }
    v_cmd = np.array([0.2, 0.0, 0.0])
    fp.update(0.05, np.zeros(3), v_cmd, hip)
    dbg = fp.debug.legs["FL"]
    assert dbg.raibert_target[2] == 0.0
    assert dbg.raibert_target[0] != dbg.stance_anchor[0] or dbg.in_stance is False


def test_sample_swing_path_has_lift():
    gait = GaitScheduler(frequency=2.0, swing_ratio=0.5)
    fp = FootPlanner(gait, clearance=0.08)
    p0 = np.array([0.0, 0.0, 0.0])
    p1 = np.array([0.1, 0.0, 0.0])
    path = fp.sample_swing_path(p0, p1, n=9)
    assert path.shape == (9, 3)
    assert np.max(path[:, 2]) >= 0.07
