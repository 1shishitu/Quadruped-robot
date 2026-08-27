"""Live matplotlib plots for stand debugging (joint angles + torques)."""

from __future__ import annotations

from collections import deque

import numpy as np


def joint_labels(robot_cfg: dict) -> list[str]:
    return [
        f"{leg}_{suffix}"
        for leg in robot_cfg.get("leg_names", [])
        for suffix in robot_cfg.get("joint_suffix", [])
    ]


class StandDebugPlotter:
    """
    Two rolling plots:
      1. Actual joint angle vs time; default_joint_angles as horizontal lines
      2. Joint torques vs time
    """

    def __init__(
        self,
        joint_labels: list[str],
        q_target: np.ndarray,
        *,
        window_s: float = 15.0,
        torque_limits: np.ndarray | None = None,
    ) -> None:
        import matplotlib.pyplot as plt

        self._labels = list(joint_labels)
        self._q_target = np.asarray(q_target, dtype=float)
        self._window_s = float(window_s)
        self._torque_limits = (
            np.asarray(torque_limits, dtype=float)
            if torque_limits is not None
            else None
        )

        n = len(self._labels)
        self._t: deque[float] = deque()
        self._q = [deque() for _ in range(n)]
        self._tau = [deque() for _ in range(n)]

        self._colors = plt.cm.tab20(np.linspace(0.0, 1.0, n, endpoint=False))
        self._fig, (self._ax_q, self._ax_tau) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        self._fig.canvas.manager.set_window_title("Go1 stand debug")
        self._fig.tight_layout(pad=2.0)

        self._q_lines = []
        self._q_refs = []
        for i, label in enumerate(self._labels):
            color = self._colors[i]
            (line,) = self._ax_q.plot([], [], color=color, linewidth=1.2, label=label)
            self._q_lines.append(line)
            ref = self._ax_q.axhline(
                self._q_target[i],
                color=color,
                linestyle="--",
                linewidth=0.9,
                alpha=0.55,
            )
            self._q_refs.append(ref)

        self._tau_lines = []
        for i, label in enumerate(self._labels):
            (line,) = self._ax_tau.plot([], [], color=self._colors[i], linewidth=1.2)
            self._tau_lines.append(line)

        self._ax_q.set_ylabel("q [rad]")
        self._ax_q.set_title("Joint angle (solid) vs default_joint_angles (dashed)")
        self._ax_q.grid(True, alpha=0.3)
        self._ax_q.legend(
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            fontsize=7,
            ncol=1,
            frameon=False,
        )

        self._ax_tau.set_ylabel("τ [N·m]")
        self._ax_tau.set_xlabel("t [s]")
        self._ax_tau.set_title("Joint torque")
        self._ax_tau.grid(True, alpha=0.3)
        if self._torque_limits is not None:
            limit = float(np.max(self._torque_limits))
            self._ax_tau.axhline(limit, color="0.4", linestyle=":", linewidth=1.0)
            self._ax_tau.axhline(-limit, color="0.4", linestyle=":", linewidth=1.0)

        plt.ion()
        plt.show(block=False)

    def is_open(self) -> bool:
        import matplotlib.pyplot as plt

        return plt.fignum_exists(self._fig.number)

    def clear(self) -> None:
        self._t.clear()
        for series in self._q:
            series.clear()
        for series in self._tau:
            series.clear()
        self._refresh_axes()

    def append(self, t: float, q: np.ndarray, tau: np.ndarray) -> None:
        q = np.asarray(q, dtype=float).reshape(-1)
        tau = np.asarray(tau, dtype=float).reshape(-1)
        if q.shape[0] != len(self._labels) or tau.shape[0] != len(self._labels):
            raise ValueError("q/tau length must match joint count")

        if self._t and t < self._t[-1] - 1e-9:
            self.clear()

        self._t.append(float(t))
        for i in range(len(self._labels)):
            self._q[i].append(float(q[i]))
            self._tau[i].append(float(tau[i]))

        t_min = self._t[-1] - self._window_s
        while self._t and self._t[0] < t_min:
            self._t.popleft()
            for series in self._q:
                series.popleft()
            for series in self._tau:
                series.popleft()

    def refresh(self) -> None:
        self._refresh_axes()
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

    def _refresh_axes(self) -> None:
        if not self._t:
            t_arr = np.array([])
        else:
            t_arr = np.fromiter(self._t, dtype=float)

        for i, line in enumerate(self._q_lines):
            line.set_data(t_arr, np.fromiter(self._q[i], dtype=float) if self._t else [])
        for i, line in enumerate(self._tau_lines):
            line.set_data(t_arr, np.fromiter(self._tau[i], dtype=float) if self._t else [])

        if self._t:
            t_end = self._t[-1]
            t_start = max(0.0, t_end - self._window_s)
            self._ax_q.set_xlim(t_start, max(t_end, t_start + 0.1))
            self._ax_tau.set_xlim(t_start, max(t_end, t_start + 0.1))

            q_vals = [v for series in self._q for v in series]
            q_vals.extend(float(v) for v in self._q_target)
            q_pad = max(0.05, 0.05 * (max(q_vals) - min(q_vals) + 1e-6))
            self._ax_q.set_ylim(min(q_vals) - q_pad, max(q_vals) + q_pad)

            tau_vals = [v for series in self._tau for v in series]
            if tau_vals:
                tau_pad = max(1.0, 0.05 * (max(tau_vals) - min(tau_vals) + 1e-6))
                self._ax_tau.set_ylim(min(tau_vals) - tau_pad, max(tau_vals) + tau_pad)
            elif self._torque_limits is not None:
                limit = float(np.max(self._torque_limits))
                self._ax_tau.set_ylim(-limit * 1.1, limit * 1.1)

    def close(self) -> None:
        import matplotlib.pyplot as plt

        if self.is_open():
            plt.close(self._fig)
