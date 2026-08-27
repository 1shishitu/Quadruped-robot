"""Tests for viewer perturbation helpers."""

from quadruped.config_loader import load_robot_config
from quadruped.sim.mujoco_env import MuJoCoEnv
from quadruped.sim.viewer_perturb import init_default_perturb_target, perturb_enabled


class TestViewerPerturb:
    def test_enabled_by_default(self):
        cfg = load_robot_config()["robot"]
        assert perturb_enabled(cfg) is True

    def test_init_trunk_selection(self):
        import mujoco

        cfg = load_robot_config()["robot"]
        env = MuJoCoEnv(cfg)
        env.load()
        pert = mujoco.MjvPerturb()
        body_id = init_default_perturb_target(env.model, env.data, pert, cfg)
        assert body_id == env.model.body("/trunk").id
        assert int(pert.select) == body_id
