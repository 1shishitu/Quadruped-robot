"""Configuration loading utilities."""

from __future__ import annotations

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
GAIT_CONFIG_DIR = CONFIG_DIR / "gait_config"
ASSETS_DIR = PROJECT_ROOT / "assets"

# locomotion.yaml gait_config: short name → file under gait_config/
_GAIT_ALIASES: dict[str, str] = {
    "march": "march.yaml",
    "march_in_place": "march.yaml",
    "trot": "trot.yaml",
    "walk": "trot.yaml",
    "fl_lift": "fl_lift.yaml",
    # legacy flat names (config/gait_*.yaml removed)
    "gait_march.yaml": "march.yaml",
    "gait_trot.yaml": "trot.yaml",
}


def load_yaml(path: Path | str) -> dict:
    """Load a YAML file under ``config/`` (or absolute path)."""
    p = Path(path)
    if not p.is_absolute():
        p = CONFIG_DIR / p
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_gait_config_path(name: str) -> Path:
    """
    Resolve gait config name to ``config/gait_config/*.yaml``.

    Accepts: ``march``, ``trot``, ``march.yaml``, ``gait_config/march.yaml``,
    legacy ``gait_march.yaml``.
    """
    raw = str(name).strip()
    if raw.startswith("gait_config/"):
        raw = raw[len("gait_config/") :]
    mapped = _GAIT_ALIASES.get(raw, raw)
    if not mapped.endswith(".yaml"):
        mapped = f"{mapped}.yaml"
    path = GAIT_CONFIG_DIR / mapped
    if not path.is_file():
        # backward: allow path relative to config/ root
        legacy = CONFIG_DIR / raw
        if legacy.is_file():
            return legacy
        raise FileNotFoundError(f"Gait config not found: {name} → {path}")
    return path


def load_gait_config(name: str = "trot") -> dict:
    """Load gait YAML from ``config/gait_config/``."""
    return load_yaml(resolve_gait_config_path(name))


def gait_control_block(gait_cfg: dict) -> dict:
    """Per-gait controller overrides (``control:`` section in gait YAML)."""
    return dict(gait_cfg.get("control") or {})


def load_gait_config_for_locomotion(loco_cfg: dict | None = None) -> dict:
    """Resolve gait YAML from ``locomotion.gait_config`` (default ``march``)."""
    if loco_cfg is None:
        loco_cfg = load_locomotion_config()
    loco = loco_cfg.get("locomotion", loco_cfg)
    fname = str(loco.get("gait_config", "march"))
    return load_gait_config(fname)


def load_robot_config() -> dict:
    return load_yaml("robot.yaml")


def load_mpc_config() -> dict:
    return load_yaml("mpc.yaml")


def load_locomotion_config() -> dict:
    return load_yaml("locomotion.yaml")
