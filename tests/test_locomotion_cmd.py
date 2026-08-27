"""Tests for keyboard velocity command."""

import numpy as np

from quadruped.config_loader import load_locomotion_config
from quadruped.input.keyboard_cmd import LocomotionKeyboard
from quadruped.input.mujoco_keys import parse_key


class TestLocomotionKeyboard:
    def test_up_sets_forward_velocity(self):
        kb = LocomotionKeyboard(load_locomotion_config())
        kb.on_key(parse_key("UP"), wall_now=1.0)
        v, _ = kb.update(wall_now=1.01, dt=0.05)
        assert v[0] > 0.0

    def test_end_clears(self):
        kb = LocomotionKeyboard(load_locomotion_config())
        kb.on_key(parse_key("UP"), wall_now=1.0)
        kb.on_key(parse_key("END"), wall_now=1.01)
        v, omega = kb.update(wall_now=1.02, dt=0.05)
        assert np.allclose(v, 0.0)
        assert omega == 0.0

    def test_remote_keeps_moving_without_repeat(self):
        cfg = load_locomotion_config()
        cfg["locomotion"]["key_mode"] = "remote"
        kb = LocomotionKeyboard(cfg)
        kb.on_key(parse_key("UP"), wall_now=0.0)
        v1, _ = kb.update(wall_now=0.01, dt=0.002)
        v2, _ = kb.update(wall_now=2.0, dt=0.002)
        assert v1[0] > 0.0
        assert v2[0] > 0.0

    def test_cmd_vx_is_cruise_not_always_max(self):
        cfg = load_locomotion_config()
        cfg["locomotion"]["cmd_vx"] = 0.15
        cfg["locomotion"]["max_vx"] = 0.25
        cfg["locomotion"]["smoothing_tau"] = 0.0
        kb = LocomotionKeyboard(cfg)
        kb.on_key(parse_key("UP"), wall_now=0.0)
        v, _ = kb.update(wall_now=0.01, dt=0.002)
        assert v[0] == 0.15

    def test_key_timeout_zeros_axis_in_repeat_mode(self):
        cfg = load_locomotion_config()
        cfg["locomotion"]["key_mode"] = "repeat"
        cfg["locomotion"]["key_hold_timeout"] = 0.05
        kb = LocomotionKeyboard(cfg)
        kb.on_key(parse_key("UP"), wall_now=0.0)
        v, _ = kb.update(wall_now=0.2, dt=0.05)
        assert abs(v[0]) < 1e-6

    def test_remote_opposite_key_switches(self):
        cfg = load_locomotion_config()
        cfg["locomotion"]["key_mode"] = "remote"
        kb = LocomotionKeyboard(cfg)
        kb.on_key(parse_key("UP"), wall_now=1.0)
        kb.on_key(parse_key("DOWN"), wall_now=1.01)
        v, _ = kb.update(wall_now=1.02, dt=0.002)
        assert v[0] < 0.0
