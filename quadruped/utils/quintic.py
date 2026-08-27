"""Quintic polynomial for smooth trajectory interpolation."""

from __future__ import annotations

import numpy as np


class QuinticPolynomial:
    """
    五次多项式 p(t) = a0 + a1*t + ... + a5*t^5

    边界条件: (p0, v0, a0) at t=0, (p1, v1, a1) at t=T
    """

    def __init__(
        self,
        p0: float,
        v0: float,
        a0: float,
        p1: float,
        v1: float,
        a1: float,
        T: float,
    ) -> None:
        if T <= 0:
            raise ValueError("Duration T must be positive")
        self.T = T
        self.coeffs = self._solve_coeffs(p0, v0, a0, p1, v1, a1, T)

    @staticmethod
    def _solve_coeffs(
        p0: float, v0: float, a0: float,
        p1: float, v1: float, a1: float,
        T: float,
    ) -> np.ndarray:
        """Solve 6x6 linear system for quintic coefficients."""
        T2, T3, T4, T5 = T**2, T**3, T**4, T**5
        A = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 2, 0, 0, 0],
            [1, T, T2, T3, T4, T5],
            [0, 1, 2*T, 3*T2, 4*T3, 5*T4],
            [0, 0, 2, 6*T, 12*T2, 20*T3],
        ], dtype=float)
        b = np.array([p0, v0, a0, p1, v1, a1], dtype=float)
        return np.linalg.solve(A, b)

    def position(self, t: float) -> float:
        t = np.clip(t, 0.0, self.T)
        c = self.coeffs
        return c[0] + c[1]*t + c[2]*t**2 + c[3]*t**3 + c[4]*t**4 + c[5]*t**5

    def velocity(self, t: float) -> float:
        t = np.clip(t, 0.0, self.T)
        c = self.coeffs
        return c[1] + 2*c[2]*t + 3*c[3]*t**2 + 4*c[4]*t**3 + 5*c[5]*t**4

    def acceleration(self, t: float) -> float:
        t = np.clip(t, 0.0, self.T)
        c = self.coeffs
        return 2*c[2] + 6*c[3]*t + 12*c[4]*t**2 + 20*c[5]*t**3

    @classmethod
    def vector3(
        cls,
        p0: np.ndarray,
        v0: np.ndarray,
        a0: np.ndarray,
        p1: np.ndarray,
        v1: np.ndarray,
        a1: np.ndarray,
        T: float,
    ) -> tuple[QuinticPolynomial, QuinticPolynomial, QuinticPolynomial]:
        """Create independent quintics for x, y, z."""
        return (
            cls(p0[0], v0[0], a0[0], p1[0], v1[0], a1[0], T),
            cls(p0[1], v0[1], a0[1], p1[1], v1[1], a1[1], T),
            cls(p0[2], v0[2], a0[2], p1[2], v1[2], a1[2], T),
        )

    def sample(self, t: float) -> tuple[float, float, float]:
        return self.position(t), self.velocity(t), self.acceleration(t)
