"""Tests for quintic polynomial."""

import numpy as np
import pytest

from quadruped.utils.quintic import QuinticPolynomial


class TestQuinticPolynomial:
    def test_boundary_position(self):
        q = QuinticPolynomial(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, T=1.0)
        assert q.position(0.0) == pytest.approx(0.0)
        assert q.position(1.0) == pytest.approx(1.0)

    def test_boundary_velocity(self):
        q = QuinticPolynomial(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, T=1.0)
        assert q.velocity(0.0) == pytest.approx(0.0)
        assert q.velocity(1.0) == pytest.approx(0.0)

    def test_boundary_acceleration(self):
        q = QuinticPolynomial(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, T=1.0)
        assert q.acceleration(0.0) == pytest.approx(0.0)
        assert q.acceleration(1.0) == pytest.approx(0.0)

    def test_midpoint_smooth(self):
        q = QuinticPolynomial(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, T=2.0)
        assert 0.0 < q.position(1.0) < 1.0

    def test_invalid_duration(self):
        with pytest.raises(ValueError):
            QuinticPolynomial(0, 0, 0, 1, 0, 0, T=0)

    def test_vector3(self):
        p0 = np.zeros(3)
        p1 = np.ones(3)
        qx, qy, qz = QuinticPolynomial.vector3(p0, p0, p0, p1, p0, p0, T=1.0)
        assert qx.position(1.0) == pytest.approx(1.0)
        assert qy.position(1.0) == pytest.approx(1.0)
        assert qz.position(1.0) == pytest.approx(1.0)
