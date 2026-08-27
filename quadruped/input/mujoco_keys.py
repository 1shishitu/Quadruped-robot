"""MuJoCo / GLFW key codes and config parsing.

MuJoCo Simulate reserves many letter keys for visualization toggles, e.g.
W=Wireframe, S=Shadow, A=AutoConnect, D=StaticBody, R=Reflection, G=Fog,
C=ContactPoint, P=ContactSplit, Q=Camera, E=Equality.  Do not use those
for locomotion when the viewer has focus.
"""

from __future__ import annotations

# From mujoco/mjui.h (GLFW key codes used by Simulate)
MUJOCO_KEY = {
    "ESCAPE": 256,
    "ENTER": 257,
    "TAB": 258,
    "BACKSPACE": 259,
    "INSERT": 260,
    "DELETE": 261,
    "RIGHT": 262,
    "LEFT": 263,
    "DOWN": 264,
    "UP": 265,
    "PAGE_UP": 266,
    "PAGE_DOWN": 267,
    "HOME": 268,
    "END": 269,
    "F1": 290,
    "F2": 291,
    "F3": 292,
    "F4": 293,
    "F5": 294,
    "F6": 295,
    "F7": 296,
    "F8": 297,
    "F9": 298,
    "F10": 299,
    "F11": 300,
    "F12": 301,
}

# Aliases for config readability
MUJOCO_KEY.update({
    "ARROW_UP": 265,
    "ARROW_DOWN": 264,
    "ARROW_LEFT": 263,
    "ARROW_RIGHT": 262,
})

# Letter keys consumed by MuJoCo viewer (visualization / rendering flags)
MUJOCO_RESERVED_LETTERS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
)  # essentially all; see README for mapping


def parse_key(name: str) -> int:
    """Parse a config key name to MuJoCo/GLFW keycode."""
    key = str(name).strip()
    if not key:
        raise ValueError("Empty key name")
    upper = key.upper()
    if upper in MUJOCO_KEY:
        return MUJOCO_KEY[upper]
    if len(key) == 1:
        return ord(key.upper())
    raise ValueError(f"Unknown key name: {name!r}")


def key_matches(keycode: int, configured: str) -> bool:
    code = parse_key(configured)
    if code <= 255:
        return keycode in (code, code + 32) if 65 <= code <= 90 else keycode == code
    return keycode == code
