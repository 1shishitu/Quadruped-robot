"""Simulation timing: decouple control rate from viewer / print (real-robot model)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SimTiming:
    control_rate_hz: float
    viewer_rate_hz: float
    print_rate_hz: float
    plot_rate_hz: float
    realtime_scale: float
    max_steps_per_frame: int

    @classmethod
    def from_robot_config(cls, robot_cfg: dict, *, sim_dt: float) -> SimTiming:
        timing = robot_cfg.get("sim_timing", {})
        control_rate_hz = float(timing.get("control_rate_hz", 1.0 / sim_dt))
        viewer_rate_hz = float(timing.get("viewer_rate_hz", 30.0))
        print_rate_hz = float(timing.get("print_rate_hz", 1.0))
        plot_rate_hz = float(timing.get("plot_rate_hz", 20.0))
        realtime_scale = float(timing.get("realtime_scale", 1.0))
        max_steps_per_frame = int(timing.get("max_steps_per_frame", 500))
        return cls(
            control_rate_hz=control_rate_hz,
            viewer_rate_hz=viewer_rate_hz,
            print_rate_hz=print_rate_hz,
            plot_rate_hz=plot_rate_hz,
            realtime_scale=realtime_scale,
            max_steps_per_frame=max(1, max_steps_per_frame),
        )

    @property
    def steps_per_viewer_frame(self) -> int:
        if self.viewer_rate_hz <= 0.0:
            return self.max_steps_per_frame
        return max(1, int(round(self.control_rate_hz / self.viewer_rate_hz)))
