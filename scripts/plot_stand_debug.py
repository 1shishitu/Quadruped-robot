#!/usr/bin/env python3
"""Stand tuning: MuJoCo viewer + live joint / torque plots (debug only)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quadruped.config_loader import load_robot_config
from quadruped.sim.mujoco_env import MuJoCoEnv
from quadruped.sim.mujoco_robot import read_joint_state
from quadruped.sim.stand_debug_plot import StandDebugPlotter, joint_labels


def _torque_limit_vector(robot_cfg: dict) -> np.ndarray:
    import numpy as np

    per_leg = robot_cfg["torque_limits"]
    return np.tile(per_leg, len(robot_cfg["leg_names"]))


def run_viewer_with_plots(env: MuJoCoEnv, plotter: StandDebugPlotter) -> None:
    import mujoco.viewer

    env.reset_collapsed()
    env.reset_wall_clock()
    key = chr(env._power_keycode)  # noqa: SLF001 — debug entry only
    print(f"[t=0] 电机断电 (τ=0)，按 [{key}] 通/断电，[R] 重置")
    print(f"      {env.format_timing_banner()}")
    print("      关闭曲线窗口即结束调试")

    last_plot_wall = 0.0
    try:
        with mujoco.viewer.launch_passive(
            env.model,
            env.data,
            key_callback=env._make_key_callback(),  # noqa: SLF001
            show_left_ui=True,
            show_right_ui=True,
        ) as viewer:
            env.attach_viewer(viewer)
            viewer.cam.lookat[0] = 0.0
            viewer.cam.lookat[1] = 0.0
            viewer.cam.lookat[2] = 0.15
            viewer.cam.distance = 2.5

            while viewer.is_running() and plotter.is_open():
                wall_now = time.perf_counter()
                tau = env.tick(wall_now)

                if env.should_refresh_plot(wall_now, last_plot_wall) and tau is not None:
                    state = read_joint_state(
                        env.model, env.data, env.robot_cfg, t=env._t  # noqa: SLF001
                    )
                    plotter.append(env._t, state.q, tau)  # noqa: SLF001
                    plotter.refresh()
                    last_plot_wall = wall_now

                if env.should_sync_viewer(wall_now):
                    env._viewer_overlay(viewer)  # noqa: SLF001
                    viewer.sync()
                    env.mark_viewer_synced(wall_now)
                else:
                    time.sleep(0.001)
    finally:
        env.detach_viewer()

    plotter.close()


def main() -> int:
    try:
        import matplotlib  # noqa: F401
        import mujoco  # noqa: F401
    except ImportError as exc:
        print("请先安装: pip install mujoco matplotlib", file=sys.stderr)
        print(f"  ({exc})", file=sys.stderr)
        return 1

    import numpy as np

    parser = argparse.ArgumentParser(description="MuJoCo stand debug with live plots")
    parser.add_argument(
        "--window",
        type=float,
        default=15.0,
        help="rolling plot window in seconds (default: 15)",
    )
    args = parser.parse_args()

    cfg = load_robot_config()["robot"]
    env = MuJoCoEnv(cfg)
    env.load()

    labels = joint_labels(cfg)
    q_target = env.stand.q_des
    plotter = StandDebugPlotter(
        labels,
        q_target,
        window_s=args.window,
        torque_limits=_torque_limit_vector(cfg),
    )

    mass = float(cfg.get("mass", 12.0))
    g = float(cfg.get("gravity", 9.81))
    print(f"URDF: {env.urdf_path}")
    print(f"Lift (reset pose): {env.lift:.3f} m  weight ~ {mass * g:.1f} N  dt: {env.dt}s")
    print(f"目标角 default_joint_angles: {np.round(q_target, 3).tolist()}")

    try:
        run_viewer_with_plots(env, plotter)
    except KeyboardInterrupt:
        plotter.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
