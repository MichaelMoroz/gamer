#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import slangpy as spy
from PIL import Image
from PySide6.QtGui import QMatrix4x4, QVector3D

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.galaxy_repro import GalaxyData, load_galaxy


CLASS_TO_ID = {
    "bulge": 0,
    "disk": 1,
    "dust": 2,
    "dust2": 3,
    "dust positive": 4,
    "stars": 5,
}

MAX_COMPONENTS = 64

DEFAULT_SPECTRA = {
    "red": QVector3D(1.0, 0.6, 0.4),
    "yellow": QVector3D(1.0, 0.9, 0.45),
    "blue": QVector3D(0.4, 0.6, 1.0),
    "white": QVector3D(1.0, 1.0, 1.0),
    "cyan": QVector3D(0.3, 0.7, 1.0),
    "purple": QVector3D(1.0, 0.3, 0.8),
}


@dataclass
class RenderConfig:
    camera: QVector3D
    target: QVector3D
    up: QVector3D
    fov: float
    exposure: float
    gamma: float
    saturation: float
    ray_step: float
    size: int


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GPU Slang renderer + GAMER baseline comparison.")
    p.add_argument("--mode", choices=["generate-baseline", "compare", "all"], default="all")

    p.add_argument("--gamer-exe", type=Path, default=Path("publish/win/gamer/Gamer.exe"))
    p.add_argument("--galaxy-dir", type=Path, default=Path("publish/data/galaxies"))
    p.add_argument("--baseline-dir", type=Path, default=Path("artifacts/baseline"))
    p.add_argument("--slang-dir", type=Path, default=Path("artifacts/slang"))

    p.add_argument("--galaxy", action="append", default=[], help="Galaxy stem/filename filter.")

    p.add_argument("--size", type=int, default=64)
    p.add_argument("--camera", nargs=3, type=float, default=[0.5, 0.0, 0.0])
    p.add_argument("--target", nargs=3, type=float, default=[0.0, 0.0, 0.0])
    p.add_argument("--up", nargs=3, type=float, default=[0.0, 1.0, 0.0])
    p.add_argument("--fov", type=float, default=90.0)
    p.add_argument("--exposure", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--saturation", type=float, default=1.0)
    p.add_argument("--ray-step", type=float, default=0.025)

    return p.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> RenderConfig:
    return RenderConfig(
        camera=QVector3D(float(args.camera[0]), float(args.camera[1]), float(args.camera[2])),
        target=QVector3D(float(args.target[0]), float(args.target[1]), float(args.target[2])),
        up=QVector3D(float(args.up[0]), float(args.up[1]), float(args.up[2])),
        fov=float(args.fov),
        exposure=float(args.exposure),
        gamma=float(args.gamma),
        saturation=float(args.saturation),
        ray_step=float(args.ray_step),
        size=int(args.size),
    )


def list_galaxies(galaxy_dir: Path, filters: Sequence[str]) -> List[Path]:
    files = sorted(galaxy_dir.glob("*.gax"))
    if filters:
        wanted = {f.lower() for f in filters}
        files = [f for f in files if f.name.lower() in wanted or f.stem.lower() in wanted]
    if not files:
        raise RuntimeError(f"No .gax files found/matched in {galaxy_dir}")
    return files


def run_gamer_baseline(gamer_exe: Path, gax_file: Path, out_base: Path, cfg: RenderConfig) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(gamer_exe),
        "galaxy",
        "omp",
        str(cfg.camera.x()),
        str(cfg.camera.y()),
        str(cfg.camera.z()),
        str(cfg.target.x()),
        str(cfg.target.y()),
        str(cfg.target.z()),
        str(cfg.up.x()),
        str(cfg.up.y()),
        str(cfg.up.z()),
        str(cfg.fov),
        str(cfg.exposure),
        str(cfg.gamma),
        str(cfg.saturation),
        str(cfg.ray_step),
        str(gax_file),
        str(cfg.size),
        str(out_base),
    ]
    res = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Gamer render failed for {gax_file.name}\n{res.stdout}\n{res.stderr}")


def make_inv_vp(cfg: RenderConfig) -> np.ndarray:
    projection = QMatrix4x4()
    projection.setToIdentity()
    projection.perspective(cfg.fov, 1.0, 1.0, 100.0)

    view = QMatrix4x4()
    view.setToIdentity()
    view.lookAt(cfg.target, cfg.camera, cfg.up)

    inv, ok = (projection * view).inverted()
    if not ok:
        raise RuntimeError("Failed to invert projection*view matrix")
    return np.array(inv.copyDataTo(), dtype=np.float32).reshape((4, 4))


def pack_components(galaxy: GalaxyData):
    comps = [c for c in galaxy.components if c.class_name.lower() in CLASS_TO_ID]
    n = len(comps)
    if n > MAX_COMPONENTS:
        raise RuntimeError(f"Galaxy has {n} components, exceeds MAX_COMPONENTS={MAX_COMPONENTS}")
    class_id = np.zeros((max(n, 1),), dtype=np.int32)
    p0 = np.zeros((max(n, 1), 4), dtype=np.float32)
    p1 = np.zeros((max(n, 1), 4), dtype=np.float32)
    p2 = np.zeros((max(n, 1), 4), dtype=np.float32)
    spec = np.zeros((max(n, 1), 4), dtype=np.float32)

    for i, c in enumerate(comps):
        class_id[i] = CLASS_TO_ID[c.class_name.lower()]
        p0[i, :] = [c.strength, c.arm, c.z0, c.r0]
        p1[i, :] = [c.active, c.delta, c.winding, c.scale]
        p2[i, :] = [c.noise_offset, c.noise_tilt, c.ks, c.inner]
        s = DEFAULT_SPECTRA.get(c.spectrum.lower(), QVector3D(1.0, 1.0, 1.0))
        spec[i, :] = [s.x(), s.y(), s.z(), 0.0]

    return n, class_id, p0, p1, p2, spec


def render_slang(
    device: spy.Device,
    kernel: spy.ComputeKernel,
    galaxy: GalaxyData,
    cfg: RenderConfig,
) -> np.ndarray:
    comp_count, class_id, p0, p1, p2, spec = pack_components(galaxy)
    npx = cfg.size * cfg.size

    class_id_u = np.zeros((MAX_COMPONENTS,), dtype=np.int32)
    p0_u = np.zeros((MAX_COMPONENTS, 4), dtype=np.float32)
    p1_u = np.zeros((MAX_COMPONENTS, 4), dtype=np.float32)
    p2_u = np.zeros((MAX_COMPONENTS, 4), dtype=np.float32)
    spec_u = np.zeros((MAX_COMPONENTS, 4), dtype=np.float32)
    if comp_count > 0:
        class_id_u[:comp_count] = class_id[:comp_count]
        p0_u[:comp_count, :] = p0[:comp_count, :]
        p1_u[:comp_count, :] = p1[:comp_count, :]
        p2_u[:comp_count, :] = p2[:comp_count, :]
        spec_u[:comp_count, :] = spec[:comp_count, :]

    bout = device.create_buffer(
        data=np.zeros((npx,), dtype=np.uint32),
        usage=spy.BufferUsage.unordered_access | spy.BufferUsage.shader_resource,
    )

    gp = galaxy.params
    uniforms = {
        "U": {
            "size": cfg.size,
            "component_count": comp_count,
            "exposure": cfg.exposure,
            "gamma": cfg.gamma,
            "saturation": cfg.saturation,
            "ray_step": cfg.ray_step,
            "camera": [cfg.camera.x(), cfg.camera.y(), cfg.camera.z()],
            "axis": [gp.axis.x(), gp.axis.y(), gp.axis.z()],
            "winding_b": gp.winding_b,
            "winding_n": gp.winding_n,
            "no_arms": gp.no_arms,
            "arm1": gp.arm1,
            "arm2": gp.arm2,
            "arm3": gp.arm3,
            "arm4": gp.arm4,
            "inv_vp": make_inv_vp(cfg),
            "class_id": class_id_u.tolist(),
            "comp_p0": p0_u.tolist(),
            "comp_p1": p1_u.tolist(),
            "comp_p2": p2_u.tolist(),
            "comp_spec": spec_u.tolist(),
        }
    }

    kernel.dispatch(
        thread_count=[cfg.size, cfg.size, 1],
        vars={
            "Uniforms": uniforms,
            "out_pixels": bout,
        },
    )

    raw = bout.to_numpy().view(np.uint32).copy()
    rgb = np.stack(
        [
            ((raw >> 16) & 255).astype(np.uint8),
            ((raw >> 8) & 255).astype(np.uint8),
            (raw & 255).astype(np.uint8),
        ],
        axis=1,
    )
    return rgb.reshape((cfg.size, cfg.size, 3))


def save_png(img: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path)


def load_png(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def create_kernel() -> tuple[spy.Device, spy.ComputeKernel]:
    device = spy.create_device(
        include_paths=[str(Path.cwd()), str((Path.cwd() / "tools" / "shaders"))],
    )
    program = device.load_program("tools/shaders/galaxy_renderer.slang", ["main"])
    kernel = device.create_compute_kernel(program)
    return device, kernel


def generate_baselines(galaxies: Iterable[Path], cfg: RenderConfig, args: argparse.Namespace) -> None:
    for gax in galaxies:
        out_base = args.baseline_dir / gax.stem
        run_gamer_baseline(args.gamer_exe, gax, out_base, cfg)
        expected = out_base.with_suffix(".png")
        if not expected.exists():
            raise RuntimeError(f"Expected baseline output missing: {expected}")
        print(f"[baseline] {gax.name} -> {expected}")


def run_comparisons(galaxies: Iterable[Path], cfg: RenderConfig, args: argparse.Namespace) -> int:
    device, kernel = create_kernel()
    failures = 0

    for gax in galaxies:
        base_png = (args.baseline_dir / gax.stem).with_suffix(".png")
        if not base_png.exists():
            raise RuntimeError(f"Missing baseline image: {base_png}")

        galaxy = load_galaxy(gax)
        img = render_slang(device, kernel, galaxy, cfg)
        out_png = args.slang_dir / f"{gax.stem}.png"
        save_png(img, out_png)

        baseline = load_png(base_png)
        if baseline.shape != img.shape:
            failures += 1
            print(f"[FAIL] {gax.name}: shape mismatch {baseline.shape} vs {img.shape}")
            continue

        if np.array_equal(baseline, img):
            print(f"[PASS] {gax.name}: exact match")
        else:
            failures += 1
            print(f"[FAIL] {gax.name}: pixel mismatch")

    return failures


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    cfg = config_from_args(args)
    galaxies = list_galaxies(args.galaxy_dir, args.galaxy)
    print(f"Found {len(galaxies)} galaxy examples in {args.galaxy_dir}")

    if args.mode in ("generate-baseline", "all"):
        generate_baselines(galaxies, cfg, args)

    failures = 0
    if args.mode in ("compare", "all"):
        failures = run_comparisons(galaxies, cfg, args)

    if failures:
        print(f"Summary: {failures} comparison(s) failed.")
        return 1
    print("Summary: all requested steps completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
