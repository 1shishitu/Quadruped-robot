"""Tests for Raibert foot placement."""

import numpy as np
import pytest

from quadruped.utils.raibert import raibert_foot_placement


class TestRaibert:
    def test_stationary_preserves_lateral_offset(self):
        foot = np.array([0.19, 0.20, 0.18])
        v = np.zeros(3)
        p = raibert_foot_placement(foot, v, v, step_period=0.5)
        assert p[2] == 0.0
        np.testing.assert_allclose(p[:2], foot[:2], atol=1e-9)

    def test_forward_velocity_from_stance_foot(self):
        foot = np.array([0.19, 0.20, 0.18])
        v = np.array([0.5, 0.0, 0.0])
        p = raibert_foot_placement(foot, v, v, step_period=0.5)
        assert p[0] > foot[0]
        assert p[1] == pytest.approx(foot[1], abs=1e-9)

    def test_velocity_error_correction(self):
        foot = np.array([0.0, 0.20, 0.0])
        v_body = np.array([0.6, 0.0, 0.0])
        v_cmd = np.array([0.3, 0.0, 0.0])
        p = raibert_foot_placement(foot, v_body, v_cmd, step_period=0.5, kv=0.04)
        p_nom = raibert_foot_placement(foot, v_body, v_body, step_period=0.5, kv=0.04)
        assert p[0] < p_nom[0]  # too fast → shorter stride to decelerate
        assert p[1] == pytest.approx(foot[1], abs=1e-9)

    def test_not_anchored_to_hip(self):
        """Hip-based placement would pull foot inward on Go1 stand."""
        foot = np.array([0.1881, 0.203, 0.0])
        hip = np.array([0.1881, 0.04675, 0.448])
        v_cmd = np.array([0.15, 0.0, 0.0])
        p = raibert_foot_placement(foot, np.zeros(3), v_cmd, step_period=1.0 / 1.5)
        assert abs(p[1] - foot[1]) < 0.01
        assert p[0] > foot[0]
        assert abs(p[1] - hip[1]) > 0.1
