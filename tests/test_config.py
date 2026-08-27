"""Tests for config loading."""

from quadruped.config_loader import (
    load_gait_config,
    load_gait_config_for_locomotion,
    load_locomotion_config,
    load_mpc_config,
    load_robot_config,
    resolve_gait_config_path,
)


class TestConfigLoader:
    def test_robot_config(self):
        cfg = load_robot_config()
        assert "robot" in cfg
        assert cfg["robot"]["mass"] > 0
        assert len(cfg["robot"]["leg_names"]) == 4

    def test_gait_config(self):
        cfg = load_gait_config("trot")
        assert cfg["gait"]["name"] == "trot"

    def test_gait_config_dir(self):
        path = resolve_gait_config_path("march")
        assert path.parent.name == "gait_config"
        assert path.name == "march.yaml"

    def test_gait_legacy_alias(self):
        cfg = load_gait_config("gait_trot.yaml")
        assert cfg["gait"]["name"] == "trot"

    def test_mpc_config(self):
        cfg = load_mpc_config()
        assert cfg["mpc"]["horizon"] > 0

    def test_locomotion_config(self):
        cfg = load_locomotion_config()
        assert cfg["locomotion"]["max_vx"] > 0
        assert cfg["locomotion"].get("mode") == "march_in_place"
