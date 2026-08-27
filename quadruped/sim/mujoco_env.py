"""MuJoCo simulation: power-off collapse → power-on stand-up → hold."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from quadruped.config_loader import load_gait_config_for_locomotion, load_locomotion_config
from quadruped.controllers.gait_controller import LocomotionController
from quadruped.controllers.low_level import (
    AttitudeStandAssist,
    BalanceController,
    JointController,
    WBCController,
)
from quadruped.controllers.stand import StandController
from quadruped.controllers.stand.stand_up import (
    fuse_pose_vector,
    stand_up_q_des,
    stand_up_total_duration,
    tachi_pose_vector,
)
from quadruped.input.keyboard_cmd import LocomotionKeyboard
from quadruped.input.mujoco_keys import key_matches, parse_key
from quadruped.planners.foot_planner import FootPlanner
from quadruped.planners.gait_scheduler import GaitScheduler
from quadruped.planners.trunk_planner import TrunkPlanner
from quadruped.sim.mujoco_robot import (
    apply_joint_torques,
    build_joint_maps,
    joint_gravity_torques,
    read_foot_positions,
    read_joint_state,
    ensure_mj_forward,
    set_reset_joint_pose,
    zero_joint_torques,
)
from quadruped.sim.foot_debug_viz import draw_foot_planner_debug
from quadruped.sim.mujoco_scene import load_model_with_floor, prepare_urdf, resolve_urdf
from quadruped.sim.sim_timing import SimTiming
from quadruped.sim.viewer_perturb import (
    apply_viewer_perturbation,
    init_default_perturb_target,
    perturb_enabled,
    selected_body_name,
)


class MuJoCoEnv:
    """
    实机对齐流程（Unitree low-level stand-up + IMU 外环）:
      1. 断电: τ=0，重力下瘫软在地
      2. 通电: q_init→Fuse→Tachi 线性插值 + τ_ff + PD（Kp/Kd 同 SDK 教程）
      3. q_des = q_nom(elapsed) + Δq_imu(roll, pitch)；倾角过大时暂停插值（gate）
      4. 保持 Tachi（default_joint_angles）

    Viewer 快捷键（避开 MuJoCo 内置单字母渲染键 W/R/G/C/P…）:
      9 — 电机通/断电（robot.yaml power_key）
      Home — 重置 spawn + 断电
      8 — stand(hold) ↔ 原地踏步 / 行走
      方向键 — 平移；Insert/Delete — 偏航；End — 清零速度
    """

    def __init__(
        self,
        robot_cfg: dict,
        urdf_path: Path | None = None,
        *,
        loco_cfg: dict | None = None,
        gait_cfg: dict | None = None,
    ) -> None:
        self.robot_cfg = robot_cfg
        stand_cfg = robot_cfg.get("stand_joint", {})
        self._auto_power_on_after = float(stand_cfg.get("power_off_duration", 0.0))
        self._stand_up_phase_duration = float(stand_cfg.get("stand_up_phase_duration", 10.0))
        self._q_fuse = fuse_pose_vector(robot_cfg)
        self._q_tachi = tachi_pose_vector(robot_cfg)
        self._gravity_compensation = bool(stand_cfg.get("gravity_compensation", True))
        self._power_key_name = str(stand_cfg.get("power_key", "9"))
        self._power_keycode = parse_key(self._power_key_name)
        self._reset_key_name = str(stand_cfg.get("reset_key", "HOME"))
        self._reset_keycode = parse_key(self._reset_key_name)

        self.urdf_path = prepare_urdf(
            urdf_path or resolve_urdf(robot_cfg["urdf_path"])
        )
        self.stand = StandController.from_robot_config({"robot": robot_cfg})
        self.joint = JointController.from_robot_config({"robot": robot_cfg})
        self.attitude = AttitudeStandAssist(robot_cfg)
        self._stand_up_elapsed = 0.0
        self._assist_frozen = False
        self._model = None
        self._data = None
        self._act_ids: list[int] = []
        self._t = 0.0
        self.lift = 0.0
        self.dt = float(robot_cfg.get("sim_dt", 0.002))

        self.powered = False
        self._stand_up_start: float | None = None
        self._q_at_power_on: np.ndarray | None = None
        self._stand_phase = "collapsed"
        self._auto_power_on_done = False
        self._pending_power_toggle = False
        self._pending_reset = False
        self._clock_needs_reset = True
        self._wall_anchor = 0.0
        self._sim_anchor = 0.0
        self._last_viewer_wall = 0.0
        self._last_print_wall = 0.0
        self._viewer = None
        self._perturb_enabled = perturb_enabled(robot_cfg)
        self.sim_timing = SimTiming.from_robot_config(robot_cfg, sim_dt=self.dt)

        loco_full = loco_cfg if loco_cfg is not None else load_locomotion_config()
        gait_full = gait_cfg if gait_cfg is not None else load_gait_config_for_locomotion(loco_full)
        self._loco_yaml = loco_full.get("locomotion", loco_full)
        self._loco_mode = str(self._loco_yaml.get("mode", "march_in_place"))
        self._loco_enabled = bool(self._loco_yaml.get("enabled", True))
        preview = float(self._loco_yaml.get("trunk", {}).get("preview_horizon", 0.08))
        self._loco_toggle_name = str(self._loco_yaml.get("toggle_key", "8"))
        self._loco_toggle_keycode = parse_key(self._loco_toggle_name)

        self.gait = GaitScheduler.from_config(gait_full)
        height = float(
            self._loco_yaml.get("trunk", {}).get("default_height", robot_cfg.get("default_height", 0.28))
        )
        self.trunk_planner = TrunkPlanner(
            default_height=height,
            gravity=float(robot_cfg.get("gravity", 9.81)),
        )
        self.trunk_planner.preview_horizon = preview
        self.foot_planner = FootPlanner.from_config(gait_full, self.gait)
        self.balance = BalanceController.from_config(loco_full, robot_cfg)
        self.wbc = WBCController.from_config(loco_full)
        self.locomotion = LocomotionController.from_config(
            robot_cfg,
            loco_full,
            balance=self.balance,
            wbc=self.wbc,
            foot_planner=self.foot_planner,
            trunk_planner=self.trunk_planner,
            attitude=self.attitude,
        )
        self.keyboard = LocomotionKeyboard(loco_full)

        self._mode = "stand"
        self._loco_start_t = 0.0
        self._pending_loco_toggle = False
        self._last_wall_for_keys = 0.0
        self._v_cmd_display = np.zeros(3)
        self._omega_cmd_display = 0.0
        dbg = self._loco_yaml.get("debug_viz", {})
        self._foot_debug_viz = bool(dbg.get("enabled", True))
        self._foot_debug_show_path = bool(dbg.get("show_swing_path", True))
        self._foot_debug_show_raibert = bool(dbg.get("show_raibert", True))

    def _refresh_sim_timing(self) -> None:
        self.sim_timing = SimTiming.from_robot_config(self.robot_cfg, sim_dt=self.dt)

    def attach_viewer(self, viewer) -> None:
        """Bind passive viewer for mouse perturbation (call from run_viewer)."""
        self._viewer = viewer
        if not self._perturb_enabled:
            return
        import mujoco

        with viewer.lock():
            body_id = init_default_perturb_target(
                self.model, self.data, viewer.perturb, self.robot_cfg
            )
        if body_id >= 0:
            name = selected_body_name(self.model, viewer.perturb)
            print(f"      扰动目标: {name}  (双击换 body；Ctrl+右键拖=推；Ctrl+左键拖=扭)")

    def detach_viewer(self) -> None:
        self._viewer = None

    def _apply_viewer_perturbation_if_needed(self) -> None:
        if not self._perturb_enabled or self._viewer is None:
            return
        with self._viewer.lock():
            apply_viewer_perturbation(self.model, self.data, self._viewer.perturb)

    def reset_wall_clock(self) -> None:
        self._wall_anchor = time.perf_counter()
        self._sim_anchor = self._t
        self._last_viewer_wall = self._wall_anchor
        self._last_print_wall = self._wall_anchor
        self._clock_needs_reset = False

    def _sync_wall_clock_if_needed(self) -> None:
        if self._clock_needs_reset:
            self.reset_wall_clock()

    def load(self) -> None:
        import mujoco

        self._model, self.lift, _, self.dt = load_model_with_floor(
            self.urdf_path, self.robot_cfg
        )
        self._data = mujoco.MjData(self._model)
        _, _, self._act_ids = build_joint_maps(self._model, self.robot_cfg)
        self._refresh_sim_timing()
        self.reset_collapsed()

    @property
    def model(self):
        if self._model is None:
            raise RuntimeError("Call load() first")
        return self._model

    @property
    def data(self):
        if self._data is None:
            raise RuntimeError("Call load() first")
        return self._data

    def reset_collapsed(self) -> None:
        """Reset spawn: reset_joint_angles + lift, then power off (τ=0, gravity settles pose)."""
        import mujoco

        if self._model is None:
            self.load()
            return

        mujoco.mj_resetData(self._model, self._data)
        self._data.qpos[0:3] = [0.0, 0.0, self.lift]
        self._data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self._data.qvel[:] = 0.0
        set_reset_joint_pose(self._model, self._data, self.robot_cfg)
        mujoco.mj_forward(self._model, self._data)
        self._t = 0.0
        self.power_off(silent=True)
        self._auto_power_on_done = False
        self._clock_needs_reset = True
        self._exit_locomotion(silent=True)

    def _exit_locomotion(self, *, silent: bool = False) -> None:
        if self._mode == "locomote" and not silent:
            print(f"[t={self._t:.2f}s] 退出行走 → stand hold")
        self._mode = "stand"
        self.keyboard.reset()
        self._v_cmd_display[:] = 0.0

    def _enter_locomotion(self) -> bool:
        if not self._loco_enabled:
            return False
        if not self.powered:
            print(f"[t={self._t:.2f}s] 行走: 请先通电")
            return False
        if self._stand_phase not in ("hold",):
            print(f"[t={self._t:.2f}s] 行走: 请等 standUp 完成 (hold)")
            return False
        state = read_joint_state(self.model, self.data, self.robot_cfg, t=self._t)
        height = float(state.base_pos[2])
        self.trunk_planner.default_height = height
        self.trunk_planner.preview_horizon = 0.0 if self._loco_mode == "march_in_place" else float(
            self._loco_yaml.get("trunk", {}).get("preview_horizon", 0.08)
        )
        self.trunk_planner.reset(
            p_com=np.array([state.base_pos[0], state.base_pos[1], height]),
            yaw=float(state.base_rpy[2]),
        )
        self.foot_planner.set_stance_positions(
            read_foot_positions(self.model, self.data, self.robot_cfg),
            base_pos=state.base_pos,
            base_rpy=state.base_rpy,
            yaw=float(state.base_rpy[2]),
        )
        self.balance.reset()
        if self._loco_mode == "march_in_place":
            self.locomotion.begin_march(state)
        self._loco_start_t = self._t
        self.keyboard.reset()
        self._mode = "locomote"
        if self._loco_mode == "march_in_place":
            print(f"[t={self._t:.2f}s] 原地踏步 ON — trot {self.gait.frequency:.1f}Hz，落足=body 锚点；再按 [{self._loco_toggle_name}] 退出")
        else:
            print(f"[t={self._t:.2f}s] 行走模式 ON — 方向键巡航, Insert/Delete 偏航, End 停止")
        return True

    def toggle_locomotion(self) -> None:
        if self._mode == "locomote":
            self._exit_locomotion()
        else:
            self._enter_locomotion()

    def power_on(self) -> None:
        if self.powered:
            return
        state = read_joint_state(self.model, self.data, self.robot_cfg, t=self._t)
        self._q_at_power_on = state.q.copy()
        self._stand_up_start = self._t
        self._stand_up_elapsed = 0.0
        self._assist_frozen = False
        self.attitude.reset()
        self._stand_phase = "fuse"
        self.powered = True
        total = stand_up_total_duration(self.robot_cfg)
        print(
            f"[t={self._t:.2f}s] 电机通电 → standUp "
            f"Fuse→Tachi ({self._stand_up_phase_duration:.0f}s×2 = {total:.0f}s)"
        )

    def power_off(self, *, silent: bool = False) -> None:
        if not self.powered and silent:
            self._stand_up_start = None
            self._q_at_power_on = None
            self._stand_up_elapsed = 0.0
            self._assist_frozen = False
            self.attitude.reset()
            self._stand_phase = "collapsed"
            self._exit_locomotion(silent=True)
            return
        if self.powered and not silent:
            print(f"[t={self._t:.2f}s] 电机断电 → τ=0")
        self.powered = False
        self._stand_up_start = None
        self._q_at_power_on = None
        self._stand_up_elapsed = 0.0
        self._assist_frozen = False
        self.attitude.reset()
        self._stand_phase = "collapsed"
        self._exit_locomotion(silent=True)

    def toggle_power(self) -> None:
        if self.powered:
            self.power_off()
        else:
            self.power_on()

    def _nominal_q_des(self, elapsed: float) -> tuple[np.ndarray, str]:
        if self._q_at_power_on is None or self._stand_up_start is None:
            phase = "hold" if self.powered else "collapsed"
            return self._q_tachi.copy(), phase
        q_des, phase = stand_up_q_des(
            self._q_at_power_on,
            self._q_fuse,
            self._q_tachi,
            elapsed,
            self._stand_up_phase_duration,
        )
        return q_des, phase

    def apply_control(self) -> np.ndarray:
        if not self.powered:
            zero_joint_torques(self.model, self.data)
            return np.zeros(12, dtype=float)

        ensure_mj_forward(self.model, self.data)

        if self._mode == "locomote":
            return self._apply_locomotion_control()

        state = read_joint_state(
            self.model, self.data, self.robot_cfg, t=self._t, include_contact=True
        )
        q_nom, phase = self._nominal_q_des(self._stand_up_elapsed)
        assist = self.attitude.compute(state, phase)
        self._assist_frozen = assist.freeze_trajectory

        if self._stand_up_start is not None and not assist.freeze_trajectory:
            self._stand_up_elapsed += self.dt

        self._stand_phase = phase
        if assist.freeze_trajectory and phase in ("fuse", "tachi"):
            self._stand_phase = f"{phase}|gate"

        if self._gravity_compensation:
            tau_ff = joint_gravity_torques(self.model, self.data, self.robot_cfg)
        else:
            tau_ff = np.zeros(12, dtype=float)
        cmd = self.stand.command(tau_ff)
        cmd.q_des = q_nom + assist.dq_corr
        tau = self.joint.compute(state, cmd)
        apply_joint_torques(self.model, self.data, tau, self._act_ids)
        return tau

    def _apply_locomotion_control(self) -> np.ndarray:
        if self._loco_mode == "march_in_place":
            v_cmd = np.zeros(3, dtype=float)
            omega_cmd = 0.0
        else:
            v_cmd, omega_cmd = self.keyboard.update(time.perf_counter(), self.dt)
        self._v_cmd_display = np.asarray(v_cmd, dtype=float).copy()
        self._omega_cmd_display = float(omega_cmd)
        gait_t = self._t - self._loco_start_t
        self._stand_phase = (
            "march" if self._loco_mode == "march_in_place" else "walk"
        )

        state = read_joint_state(
            self.model,
            self.data,
            self.robot_cfg,
            t=self._t,
            include_contact=True,
        )
        tau = self.locomotion.compute(
            self.model,
            self.data,
            state,
            gait_t=gait_t,
            dt=self.dt,
            v_cmd=v_cmd,
            omega_cmd=omega_cmd,
        )
        apply_joint_torques(self.model, self.data, tau, self._act_ids)
        return tau

    def step(self) -> np.ndarray:
        import mujoco

        tau = self.apply_control()
        self._apply_viewer_perturbation_if_needed()
        mujoco.mj_step(self.model, self.data)
        self._t += self.dt
        return tau

    def _process_input(self) -> None:
        if self._pending_reset:
            self._pending_reset = False
            self.reset_collapsed()
            print(f"[t={self._t:.2f}s] 重置：瘫软 + 断电")
            return
        if self._pending_power_toggle:
            self._pending_power_toggle = False
            self.toggle_power()
        if self._pending_loco_toggle:
            self._pending_loco_toggle = False
            self.toggle_locomotion()

    def _maybe_auto_power_on(self) -> None:
        if self._auto_power_on_after <= 0.0 or self._auto_power_on_done:
            return
        if self._t >= self._auto_power_on_after:
            self.power_on()
            self._auto_power_on_done = True

    def catch_up_control(self, wall_now: float) -> np.ndarray | None:
        """Run 500 Hz control steps until sim time catches wall clock × realtime_scale."""
        timing = self.sim_timing
        tau: np.ndarray | None = None
        steps = 0

        if timing.realtime_scale <= 0.0:
            n_steps = min(timing.max_steps_per_frame, timing.steps_per_viewer_frame)
            while steps < n_steps:
                self._maybe_auto_power_on()
                tau = self.step()
                steps += 1
            return tau

        elapsed_wall = wall_now - self._wall_anchor
        target_sim_t = self._sim_anchor + elapsed_wall * timing.realtime_scale
        while self._t < target_sim_t and steps < timing.max_steps_per_frame:
            self._maybe_auto_power_on()
            tau = self.step()
            steps += 1
        return tau

    def _maybe_print_status(self, wall_now: float) -> None:
        rate = self.sim_timing.print_rate_hz
        if rate <= 0.0:
            return
        interval = 1.0 / rate
        if wall_now - self._last_print_wall < interval:
            return
        wall_elapsed = wall_now - self._wall_anchor
        motor = "ON" if self.powered else "OFF"
        grav = "FF:on" if self._gravity_compensation else "FF:off"
        mode = self._mode if self.powered else "collapsed"
        print(
            f"[sim t={self._t:.2f}s  wall={wall_elapsed:.2f}s  x{self.sim_timing.realtime_scale:g}] "
            f"{motor} {mode}/{self._stand_phase}  {grav}"
        )
        self._last_print_wall = wall_now

    def should_sync_viewer(self, wall_now: float) -> bool:
        rate = self.sim_timing.viewer_rate_hz
        if rate <= 0.0:
            return True
        if wall_now - self._last_viewer_wall >= 1.0 / rate:
            return True
        return False

    def mark_viewer_synced(self, wall_now: float) -> None:
        self._last_viewer_wall = wall_now

    def should_refresh_plot(self, wall_now: float, last_plot_wall: float) -> bool:
        rate = self.sim_timing.plot_rate_hz
        if rate <= 0.0:
            return True
        return wall_now - last_plot_wall >= 1.0 / rate

    def tick(self, wall_now: float | None = None) -> np.ndarray | None:
        """One outer-loop tick: input → catch-up control → optional status print."""
        if wall_now is None:
            wall_now = time.perf_counter()
        self._last_wall_for_keys = wall_now
        self._sync_wall_clock_if_needed()
        self._process_input()
        self._sync_wall_clock_if_needed()
        tau = self.catch_up_control(wall_now)
        self._maybe_print_status(wall_now)
        return tau

    def format_timing_banner(self) -> str:
        t = self.sim_timing
        mode = f"realtime x{t.realtime_scale:g}" if t.realtime_scale > 0 else "fast-forward"
        return (
            f"control {t.control_rate_hz:.0f} Hz (dt={self.dt}s)  "
            f"viewer {t.viewer_rate_hz:g} Hz  print {t.print_rate_hz:g} Hz  {mode}"
        )

    def _viewer_overlay(self, viewer) -> None:
        import mujoco

        status = "ON  (stand)" if self.powered else "OFF (limp, tau=0)"
        if self.powered and self._mode == "locomote":
            if self._loco_mode == "march_in_place":
                status = "ON  (march)"
            else:
                status = "ON  (walk)"
        grav = "on" if self._gravity_compensation else "off"
        phase = self._stand_phase if self.powered else "collapsed"
        if self._mode == "locomote":
            if self._loco_mode == "march_in_place":
                phase = "march trot"
            else:
                phase = f"walk v=({self._v_cmd_display[0]:+.2f},{self._v_cmd_display[1]:+.2f})"
        imu_hint = ""
        if self.powered and self.attitude.enabled:
            gate = "GATE" if self._assist_frozen else "IMU"
            imu_hint = gate
        push_hint = ""
        if self._perturb_enabled and self._viewer is not None:
            with self._viewer.lock():
                target = selected_body_name(self.model, self._viewer.perturb)
            push_hint = f"push:{target}"
        top_right_sub = f"FF:{grav} {imu_hint}"
        if self._mode == "locomote" and self._foot_debug_viz:
            top_right_sub = "Y=touchdown  color=leg  white=ref  black=foot"
        march_hint = (
            "march: trot in place"
            if self._loco_mode == "march_in_place"
            else "arrows=move End=stop"
        )
        viewer.set_texts([
            (None, mujoco.mjtGridPos.mjGRID_TOPLEFT, "Motor", status),
            (None, mujoco.mjtGridPos.mjGRID_TOPRIGHT, f"t={self._t:.2f}s  {phase}", top_right_sub),
            (
                None,
                mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                f"[{self._power_key_name}] power  [{self._reset_key_name}] reset",
                f"[{self._loco_toggle_name}] march/walk  {march_hint}",
            ),
            (
                None,
                mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT,
                "Ctrl+RMB drag: push",
                push_hint or "Ctrl+LMB: rotate",
            ),
        ])
        if self._mode == "locomote" and self._foot_debug_viz and self._model is not None:
            foot_pos = read_foot_positions(self.model, self.data, self.robot_cfg)
            draw_foot_planner_debug(
                viewer,
                self.foot_planner.debug,
                foot_pos,
                enabled=True,
                show_swing_path=self._foot_debug_show_path,
                show_raibert=self._foot_debug_show_raibert,
            )

    def _make_key_callback(self):
        def key_callback(keycode: int) -> None:
            if keycode == self._power_keycode or key_matches(
                keycode, self._power_key_name
            ):
                self._pending_power_toggle = True
            elif keycode == self._reset_keycode or key_matches(
                keycode, self._reset_key_name
            ):
                self._pending_reset = True
            elif keycode == self._loco_toggle_keycode or key_matches(
                keycode, self._loco_toggle_name
            ):
                self._pending_loco_toggle = True
            elif self._mode == "locomote":
                self.keyboard.on_key(keycode, time.perf_counter())

        return key_callback

    def _sleep_until_next_tick(self, wall_now: float) -> None:
        """Avoid busy-spinning the CPU/GPU when sim is caught up with wall clock."""
        timing = self.sim_timing
        if timing.realtime_scale <= 0.0:
            time.sleep(0.001)
            return
        elapsed = wall_now - self._wall_anchor
        target_sim_t = self._sim_anchor + elapsed * timing.realtime_scale
        if self._t + 1e-9 < target_sim_t:
            return
        next_step_wall = self._wall_anchor + (
            (self._t + self.dt - self._sim_anchor) / timing.realtime_scale
        )
        viewer_interval = (
            0.0 if timing.viewer_rate_hz <= 0.0 else 1.0 / timing.viewer_rate_hz
        )
        wake_at = next_step_wall
        if viewer_interval > 0.0:
            wake_at = min(wake_at, self._last_viewer_wall + viewer_interval)
        delay = wake_at - time.perf_counter()
        if delay > 0.0:
            time.sleep(min(delay, 0.05))

    def run_viewer(self) -> None:
        import mujoco.viewer

        self.reset_collapsed()
        self.reset_wall_clock()
        print(
            f"[t=0] 电机断电 (τ=0)，按 [{self._power_key_name}] 通/断电，"
            f"[{self._reset_key_name}] 重置"
        )
        if self._loco_enabled:
            if self._loco_mode == "march_in_place":
                print(
                    f"      [{self._loco_toggle_name}] stand↔原地踏步 (trot 1Hz，落足=stance 锚点)"
                )
            else:
                print(
                    f"      [{self._loco_toggle_name}] stand↔行走  方向键=巡航  End=停"
                )
        print(f"      {self.format_timing_banner()}")
        if self._auto_power_on_after > 0.0:
            print(f"      或等待 {self._auto_power_on_after:.0f}s 自动通电")

        with mujoco.viewer.launch_passive(
            self.model,
            self.data,
            key_callback=self._make_key_callback(),
            show_left_ui=True,
            show_right_ui=True,
        ) as viewer:
            self.attach_viewer(viewer)
            viewer.cam.lookat[0] = 0.0
            viewer.cam.lookat[1] = 0.0
            viewer.cam.lookat[2] = 0.15
            viewer.cam.distance = 2.5

            try:
                while viewer.is_running():
                    wall_now = time.perf_counter()
                    self.tick(wall_now)
                    if self.should_sync_viewer(wall_now):
                        self._viewer_overlay(viewer)
                        viewer.sync()
                        self.mark_viewer_synced(wall_now)
                    else:
                        self._sleep_until_next_tick(wall_now)
            finally:
                self.detach_viewer()
