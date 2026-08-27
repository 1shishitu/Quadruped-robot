"""Keyboard velocity command with smoothing (arrow keys by default)."""

from __future__ import annotations

import numpy as np

from quadruped.input.mujoco_keys import key_matches, parse_key


class LocomotionKeyboard:
    """
    Configurable keys → v_cmd = [vx, vy, vz].

    ``key_mode: remote`` (default, MuJoCo viewer):
        方向键设定巡航速度，**End** 或反向键停止/换向。
        Passive viewer 只有 key-down/repeat、没有 key-up，无法做到真·松手即停；
        若用短 timeout 会表现为“点一下动一下”（Linux 首 repeat ~660ms）。

    ``key_mode: repeat``: 靠 OS 按键 repeat 刷新 ``key_hold_timeout``（易成点按感，不推荐）。

    ``key_mode: latch``: 同 remote（别名）。

    ``cmd_vx/cmd_vy/cmd_omega``: 按住方向键时的**目标巡航速度**（匀速）。
    ``max_vx/max_vy/max_omega``: 硬上限；``smoothing_tau`` 做加减速平滑。
    """

    _DEFAULT_KEYS = {
        "forward": "UP",
        "back": "DOWN",
        "strafe_left": "LEFT",
        "strafe_right": "RIGHT",
        "yaw_left": "INSERT",
        "yaw_right": "DELETE",
    }

    def __init__(self, cfg: dict) -> None:
        loco = cfg.get("locomotion", cfg)
        self.max_vx = float(loco.get("max_vx", 0.4))
        self.max_vy = float(loco.get("max_vy", 0.25))
        self.max_omega = float(loco.get("max_omega", 0.8))
        self.cmd_vx = float(loco.get("cmd_vx", self.max_vx))
        self.cmd_vy = float(loco.get("cmd_vy", self.max_vy))
        self.cmd_omega = float(loco.get("cmd_omega", self.max_omega))
        self.smoothing_tau = float(loco.get("smoothing_tau", 0.12))
        self.hold_timeout = float(loco.get("key_hold_timeout", 0.12))
        self.key_mode = str(loco.get("key_mode", "remote")).lower()

        stop = loco.get("stop_key", "END")
        self._stop_name = str(stop)
        self._stop_code = parse_key(self._stop_name)

        key_cfg = loco.get("keys", {})
        self._key_names: dict[str, str] = {
            name: str(key_cfg.get(name, default))
            for name, default in self._DEFAULT_KEYS.items()
        }
        self._keys = {name: parse_key(label) for name, label in self._key_names.items()}

        self._v_target = np.zeros(3, dtype=float)
        self._v_cmd = np.zeros(3, dtype=float)
        self._omega_cmd = 0.0
        self._vx_sign = 0
        self._vy_sign = 0
        self._omega_sign = 0
        self._last_wall: dict[str, float] = {}

    def reset(self) -> None:
        self._v_target[:] = 0.0
        self._v_cmd[:] = 0.0
        self._omega_cmd = 0.0
        self._vx_sign = 0
        self._vy_sign = 0
        self._omega_sign = 0
        self._last_wall.clear()

    def _hit(self, keycode: int, name: str) -> bool:
        code = self._keys[name]
        return keycode == code or key_matches(keycode, self._key_names[name])

    def on_key(self, keycode: int, wall_now: float) -> None:
        if keycode == self._stop_code or key_matches(keycode, self._stop_name):
            self.reset()
            return

        if self._hit(keycode, "forward"):
            self._vx_sign = 1
            self._last_wall["vx"] = wall_now
        elif self._hit(keycode, "back"):
            self._vx_sign = -1
            self._last_wall["vx"] = wall_now
        elif self._hit(keycode, "strafe_left"):
            self._vy_sign = 1
            self._last_wall["vy"] = wall_now
        elif self._hit(keycode, "strafe_right"):
            self._vy_sign = -1
            self._last_wall["vy"] = wall_now
        elif self._hit(keycode, "yaw_left"):
            self._omega_sign = 1
            self._last_wall["omega"] = wall_now
        elif self._hit(keycode, "yaw_right"):
            self._omega_sign = -1
            self._last_wall["omega"] = wall_now

    def _uses_latch_semantics(self) -> bool:
        return self.key_mode in ("remote", "hold", "latch")

    def _axis_active(self, axis: str, wall_now: float) -> bool:
        if self._uses_latch_semantics():
            if axis == "vx":
                return self._vx_sign != 0
            if axis == "vy":
                return self._vy_sign != 0
            if axis == "omega":
                return self._omega_sign != 0
            return False
        t = self._last_wall.get(axis)
        return t is not None and (wall_now - t) <= self.hold_timeout

    def _axis_speed(self, axis: str, sign: int) -> float:
        if sign == 0:
            return 0.0
        if axis == "vx":
            return float(np.clip(self.cmd_vx * sign, -self.max_vx, self.max_vx))
        if axis == "vy":
            return float(np.clip(self.cmd_vy * sign, -self.max_vy, self.max_vy))
        return 0.0

    def update(self, wall_now: float, dt: float) -> tuple[np.ndarray, float]:
        """Return smoothed (v_cmd, omega_cmd)."""
        if not self._uses_latch_semantics():
            if not self._axis_active("vx", wall_now):
                self._vx_sign = 0
            if not self._axis_active("vy", wall_now):
                self._vy_sign = 0
            if not self._axis_active("omega", wall_now):
                self._omega_sign = 0

        vx = self._axis_speed("vx", self._vx_sign) if self._axis_active("vx", wall_now) else 0.0
        vy = self._axis_speed("vy", self._vy_sign) if self._axis_active("vy", wall_now) else 0.0
        self._v_target[0] = vx
        self._v_target[1] = vy
        self._v_target[2] = 0.0

        omega_active = self._axis_active("omega", wall_now)
        if omega_active and self._omega_sign != 0:
            omega_target = float(
                np.clip(
                    self.cmd_omega * self._omega_sign,
                    -self.max_omega,
                    self.max_omega,
                )
            )
        else:
            omega_target = 0.0

        if self.smoothing_tau <= 0.0:
            self._v_cmd = self._v_target.copy()
            self._omega_cmd = omega_target
        else:
            alpha = np.exp(-dt / self.smoothing_tau)
            self._v_cmd = alpha * self._v_cmd + (1.0 - alpha) * self._v_target
            self._omega_cmd = alpha * self._omega_cmd + (1.0 - alpha) * omega_target

        return self._v_cmd.copy(), float(self._omega_cmd)
