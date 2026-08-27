"""IMU roll/pitch outer loop for stand-up / hold disturbance rejection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quadruped.types import RobotState


@dataclass(frozen=True)
class AttitudeAssistResult:
    """Correction added on top of nominal stand-up joint trajectory."""

    dq_corr: np.ndarray
    freeze_trajectory: bool
    active: bool
    roll_err: float
    pitch_err: float


def _vec3(cfg: dict, key: str, default: list[float]) -> np.ndarray:
    return np.asarray(cfg.get(key, default), dtype=float)


class AttitudeStandAssist:
    """
    Virtual spring on roll/pitch → differential leg joint offsets.

    Still MIT low-level: only modifies ``q_des`` before joint PD.
    """

    def __init__(self, robot_cfg: dict) -> None:
        cfg = robot_cfg.get("attitude_assist", {})
        self.enabled = bool(cfg.get("enabled", True))
        self.kp = _vec3(cfg, "kp", [0.6, 0.6])[:2]
        self.kd = _vec3(cfg, "kd", [0.04, 0.04])[:2]
        self.max_correction = float(cfg.get("max_correction", 0.3))
        self.gate_tilt = float(cfg.get("gate_tilt", 0.26))
        self.release_tilt = float(cfg.get("release_tilt", 0.15))
        self.require_all_feet = bool(cfg.get("require_all_feet", True))
        self.min_feet = int(cfg.get("min_feet", 4))

        pitch = cfg.get("pitch_coupling", {})
        roll = cfg.get("roll_coupling", {})
        self.pitch_hip = float(pitch.get("hip", 0.2))
        self.pitch_thigh = float(pitch.get("thigh", 0.8))
        self.pitch_calf = float(pitch.get("calf", 0.0))
        self.roll_hip = float(roll.get("hip", 0.6))
        self.roll_thigh = float(roll.get("thigh", 0.3))
        self.roll_calf = float(roll.get("calf", 0.0))

        phase_scale = cfg.get("phase_scale", {})
        self.phase_scale = {
            "fuse": float(phase_scale.get("fuse", 0.6)),
            "tachi": float(phase_scale.get("tachi", 0.85)),
            "hold": float(phase_scale.get("hold", 1.0)),
            "march": float(phase_scale.get("march", 1.0)),
            "collapsed": 0.0,
        }

        self.legs = list(robot_cfg.get("leg_names", ["FL", "FR", "RL", "RR"]))
        self._frozen = False

    def reset(self) -> None:
        self._frozen = False

    def _feet_ok(
        self,
        state: RobotState,
        *,
        min_feet: int | None = None,
        require_all_feet: bool | None = None,
    ) -> bool:
        req_all = (
            self.require_all_feet
            if require_all_feet is None
            else require_all_feet
        )
        min_n = self.min_feet if min_feet is None else min_feet
        if not state.contact:
            return not req_all
        n = sum(1 for leg in self.legs if state.contact.get(leg, False))
        if req_all:
            return n >= min_n
        return n >= max(1, min(min_n, len(self.legs)))

    def _update_freeze(self, roll: float, pitch: float) -> bool:
        tilt = max(abs(roll), abs(pitch))
        if self._frozen:
            if tilt <= self.release_tilt:
                self._frozen = False
            return self._frozen
        if tilt >= self.gate_tilt:
            self._frozen = True
        return self._frozen

    def compute(
        self,
        state: RobotState,
        phase: str,
        *,
        min_feet: int | None = None,
        require_all_feet: bool | None = None,
    ) -> AttitudeAssistResult:
        zero = AttitudeAssistResult(
            dq_corr=np.zeros(12, dtype=float),
            freeze_trajectory=False,
            active=False,
            roll_err=0.0,
            pitch_err=0.0,
        )
        if not self.enabled or phase == "collapsed":
            return zero

        roll = float(state.base_rpy[0])
        pitch = float(state.base_rpy[1])
        roll_err = -roll
        pitch_err = -pitch
        freeze = self._update_freeze(roll, pitch)

        if not self._feet_ok(
            state, min_feet=min_feet, require_all_feet=require_all_feet
        ):
            return AttitudeAssistResult(
                dq_corr=np.zeros(12, dtype=float),
                freeze_trajectory=freeze,
                active=False,
                roll_err=roll_err,
                pitch_err=pitch_err,
            )

        scale = self.phase_scale.get(phase, 1.0)
        roll_cmd = scale * (
            self.kp[0] * roll_err - self.kd[0] * float(state.base_omega[0])
        )
        pitch_cmd = scale * (
            self.kp[1] * pitch_err - self.kd[1] * float(state.base_omega[1])
        )

        dq = np.zeros(12, dtype=float)
        for i, leg in enumerate(self.legs):
            is_front = leg.startswith("F")
            is_left = leg.endswith("L")
            sign_pitch = 1.0 if is_front else -1.0
            sign_roll = 1.0 if is_left else -1.0
            base = 3 * i
            dq[base] += sign_roll * roll_cmd * self.roll_hip
            dq[base + 1] += sign_pitch * pitch_cmd * self.pitch_thigh
            dq[base + 1] += sign_roll * roll_cmd * self.roll_thigh
            dq[base + 2] += sign_pitch * pitch_cmd * self.pitch_calf
            dq[base + 2] += sign_roll * roll_cmd * self.roll_calf

        dq = np.clip(dq, -self.max_correction, self.max_correction)
        return AttitudeAssistResult(
            dq_corr=dq,
            freeze_trajectory=freeze,
            active=bool(np.any(np.abs(dq) > 1e-9)),
            roll_err=roll_err,
            pitch_err=pitch_err,
        )
