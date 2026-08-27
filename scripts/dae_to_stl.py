#!/usr/bin/env python3
"""
批量将 DAE mesh 转为 STL，并生成 MuJoCo 可用的 URDF。

用法（项目根目录）:
    pip install trimesh pycollada
    python scripts/dae_to_stl.py
    python scripts/dae_to_stl.py --urdf assets/urdf/go1_description/urdf/go1.urdf

生成:
    meshes/*.STL          与 *.dae 同名
    urdf/go1_mujoco.urdf  mesh 路径改为 ../meshes/xxx.STL

然后将 config/robot.yaml 中 urdf_path 改为:
    assets/urdf/go1_description/urdf/go1_mujoco.urdf
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quadruped.config_loader import PROJECT_ROOT as ROOT, load_robot_config

GAZEBO_BLOCK = re.compile(r"<gazebo\b[^>]*>.*?</gazebo>\s*", re.DOTALL | re.IGNORECASE)
MUJOCO_MAX_FACES = 200_000  # MuJoCo STL 单 mesh 上限


def package_root(urdf_path: Path) -> Path:
    if urdf_path.parent.name == "urdf":
        return urdf_path.parent.parent
    return urdf_path.parent


def collect_dae_files(urdf_path: Path, mesh_dir: Path | None) -> list[Path]:
    mesh_dir = mesh_dir or (package_root(urdf_path) / "meshes")
    if not mesh_dir.is_dir():
        raise FileNotFoundError(f"Mesh directory not found: {mesh_dir}")

    text = urdf_path.read_text(encoding="utf-8")
    names = set(re.findall(r"meshes/([^\"']+\.dae)", text, flags=re.IGNORECASE))
    if names:
        files = [mesh_dir / n for n in sorted(names)]
        missing = [f for f in files if not f.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing DAE: {missing[0]}")
        return files

    return sorted(mesh_dir.glob("*.dae"))


def convert_dae_to_stl(dae_path: Path, stl_path: Path, *, max_faces: int) -> None:
    import trimesh

    loaded = trimesh.load(str(dae_path), force="mesh")
    if isinstance(loaded, trimesh.Scene):
        geoms = list(loaded.geometry.values())
        mesh = trimesh.util.concatenate(geoms) if len(geoms) > 1 else geoms[0]
    else:
        mesh = loaded

    n_faces = len(mesh.faces)
    if n_faces > max_faces:
        try:
            mesh = mesh.simplify_quadric_decimation(face_count=max_faces)
        except TypeError:
            # 旧版 trimesh: 第一个参数是 reduction ratio (0~1)
            reduction = 1.0 - max_faces / n_faces
            mesh = mesh.simplify_quadric_decimation(reduction)
        except ImportError as exc:
            raise RuntimeError(
                f"{dae_path.name}: {n_faces} faces > MuJoCo limit {max_faces}. "
                "Install: pip install fast-simplification"
            ) from exc
        print(f"    simplified: {n_faces} -> {len(mesh.faces)} faces")

    stl_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(stl_path), file_type="stl")


def build_mujoco_urdf(src_urdf: Path, dst_urdf: Path, *, strip_gazebo: bool = True) -> None:
    text = src_urdf.read_text(encoding="utf-8")

    if strip_gazebo:
        text = GAZEBO_BLOCK.sub("", text)

    text = re.sub(
        r'filename="package://[^/]+/meshes/([^"]+)\.dae"',
        r'filename="\1.STL"',
        text,
        flags=re.IGNORECASE,
    )

    if "<mujoco>" not in text:
        text = re.sub(
            r"(<robot\b[^>]*>\s*)",
            r'\1  <mujoco>\n    <compiler meshdir="../meshes" discardvisual="false"/>\n  </mujoco>\n',
            text,
            count=1,
        )

    dst_urdf.parent.mkdir(parents=True, exist_ok=True)
    dst_urdf.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert DAE meshes to STL for MuJoCo")
    parser.add_argument(
        "--urdf",
        type=Path,
        default=None,
        help="Source URDF (default: from config/robot.yaml)",
    )
    parser.add_argument(
        "--meshes",
        type=Path,
        default=None,
        help="Mesh directory (default: <package>/meshes)",
    )
    parser.add_argument(
        "--output-urdf",
        type=Path,
        default=None,
        help="Output URDF (default: <urdf_dir>/<stem>_mujoco.urdf)",
    )
    parser.add_argument(
        "--keep-gazebo",
        action="store_true",
        help="Keep <gazebo> blocks in output URDF",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing STL files",
    )
    parser.add_argument(
        "--max-faces",
        type=int,
        default=180_000,
        help=f"Max faces per STL (MuJoCo limit {MUJOCO_MAX_FACES}, default 180000)",
    )
    args = parser.parse_args()

    try:
        import trimesh  # noqa: F401
    except ImportError:
        print("请先安装: pip install trimesh pycollada", file=sys.stderr)
        return 1

    src_urdf = args.urdf
    if src_urdf is None:
        rel = load_robot_config()["robot"]["urdf_path"]
        src_urdf = (ROOT / rel).resolve()
    else:
        src_urdf = (ROOT / src_urdf if not src_urdf.is_absolute() else src_urdf).resolve()

    if not src_urdf.is_file():
        print(f"URDF not found: {src_urdf}", file=sys.stderr)
        return 1

    # robot.yaml 若已指向 go1_mujoco.urdf，mesh 列表仍从原始 go1.urdf 读取
    mesh_src_urdf = src_urdf
    if "_mujoco" in src_urdf.stem:
        orig = src_urdf.parent / f"{src_urdf.stem.replace('_mujoco', '')}.urdf"
        if orig.is_file():
            mesh_src_urdf = orig

    mesh_dir = args.meshes
    if mesh_dir is not None:
        mesh_dir = (ROOT / mesh_dir if not mesh_dir.is_absolute() else mesh_dir).resolve()

    dst_urdf = args.output_urdf
    if dst_urdf is None:
        dst_urdf = src_urdf.parent / f"{src_urdf.stem}_mujoco.urdf"
    else:
        dst_urdf = (ROOT / dst_urdf if not dst_urdf.is_absolute() else dst_urdf).resolve()

    dae_files = collect_dae_files(mesh_src_urdf, mesh_dir)
    print(f"Converting {len(dae_files)} DAE -> STL in {dae_files[0].parent}")

    for dae in dae_files:
        stl = dae.with_suffix(".STL")
        if stl.exists() and not args.force and stl.stat().st_mtime >= dae.stat().st_mtime:
            print(f"  skip (up to date): {stl.name}")
            continue
        convert_dae_to_stl(dae, stl, max_faces=args.max_faces)
        print(f"  ok: {dae.name} -> {stl.name}")

    build_mujoco_urdf(mesh_src_urdf, dst_urdf, strip_gazebo=not args.keep_gazebo)
    print(f"Wrote URDF: {dst_urdf}")
    print()
    print("下一步: 在 config/robot.yaml 中设置")
    print(f"  urdf_path: {dst_urdf.relative_to(ROOT)}")
    print("然后运行: python scripts/view_urdf_mujoco.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
