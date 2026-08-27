#!/usr/bin/env python3
"""MuJoCo viewer entry: MIT stand (default pose + joint PD + gravity feedforward)."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quadruped.config_loader import load_robot_config
from quadruped.sim.mujoco_env import MuJoCoEnv


def main() -> int:
    try:
        import mujoco  # noqa: F401
    except ImportError:
        print("请先安装: pip install mujoco", file=sys.stderr)
        return 1

    cfg = load_robot_config()["robot"]
    env = MuJoCoEnv(cfg)
    env.load()

    mass = float(cfg.get("mass", 12.0))
    g = float(cfg.get("gravity", 9.81))
    print(f"URDF: {env.urdf_path}")
    stand = cfg["stand_joint"]
    print(f"Lift (reset pose): {env.lift:.3f} m  weight ~ {mass * g:.1f} N  dt: {env.dt}s")
    print(f"快捷键: [{stand.get('power_key', '9')}] 通/断电  "
          f"[{stand.get('reset_key', 'HOME')}] 重置  "
          f"[8] 原地踏步")
    grav = stand.get("gravity_compensation", True)
    print(f"重力前馈: {'开' if grav else '关'}  (stand_joint.gravity_compensation)")
    if float(stand.get("power_off_duration", 0)) > 0:
        print(f"        或 {stand['power_off_duration']}s 后自动通电")

    env.run_viewer()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
