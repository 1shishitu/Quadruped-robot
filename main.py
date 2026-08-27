"""Main entry point."""

from __future__ import annotations

import sys


def main() -> None:
    print("Use: python scripts/view_urdf_mujoco.py", file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
