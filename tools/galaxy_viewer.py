#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import slangpy as spy
from PySide6.QtGui import QMatrix4x4, QVector3D

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.galaxy_repro import load_galaxy
from tools.galaxy_repro_slang import CLASS_TO_ID, DEFAULT_SPECTRA, MAX_COMPONENTS


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v), 1e-8)


def _inv_vp(camera: np.ndarray, target: np.ndarray, up: np.ndarray, fov: float) -> np.ndarray:
    proj = QMatrix4x4()
    proj.setToIdentity()
    proj.perspective(float(fov), 1.0, 1.0, 100.0)

    view = QMatrix4x4()
    view.setToIdentity()
    view.lookAt(
        QVector3D(float(target[0]), float(target[1]), float(target[2])),
        QVector3D(float(camera[0]), float(camera[1]), float(camera[2])),
        QVector3D(float(up[0]), float(up[1]), float(up[2])),
    )

    inv, ok = (proj * view).inverted()
    if not ok:
        raise RuntimeError("Failed to invert projection*view")
    return np.array(inv.copyDataTo(), dtype=np.float32).reshape((4, 4))


@dataclass
class PackedGalaxy:
    axis: list[float]
    winding_b: float
    winding_n: float
    no_arms: float
    arm1: float
    arm2: float
    arm3: float
    arm4: float
    component_count: int
    class_id: list[int]
    comp_p0: list[list[float]]
    comp_p1: list[list[float]]
    comp_p2: list[list[float]]
    comp_spec: list[list[float]]


def pack_galaxy(gax_path: Path) -> PackedGalaxy:
    galaxy = load_galaxy(gax_path)
    comps = [c for c in galaxy.components if c.class_name.lower() in CLASS_TO_ID]
    if len(comps) > MAX_COMPONENTS:
        raise RuntimeError(f"{gax_path.name}: {len(comps)} components exceeds MAX_COMPONENTS={MAX_COMPONENTS}")

    class_id = np.zeros((MAX_COMPONENTS,), dtype=np.int32)
    p0 = np.zeros((MAX_COMPONENTS, 4), dtype=np.float32)
    p1 = np.zeros((MAX_COMPONENTS, 4), dtype=np.float32)
    p2 = np.zeros((MAX_COMPONENTS, 4), dtype=np.float32)
    spec = np.zeros((MAX_COMPONENTS, 4), dtype=np.float32)

    for i, c in enumerate(comps):
        class_id[i] = CLASS_TO_ID[c.class_name.lower()]
        p0[i, :] = [c.strength, c.arm, c.z0, c.r0]
        p1[i, :] = [c.active, c.delta, c.winding, c.scale]
        p2[i, :] = [c.noise_offset, c.noise_tilt, c.ks, c.inner]
        s = DEFAULT_SPECTRA.get(c.spectrum.lower(), QVector3D(1.0, 1.0, 1.0))
        spec[i, :] = [s.x(), s.y(), s.z(), 0.0]

    gp = galaxy.params
    return PackedGalaxy(
        axis=[float(gp.axis.x()), float(gp.axis.y()), float(gp.axis.z())],
        winding_b=float(gp.winding_b),
        winding_n=float(gp.winding_n),
        no_arms=float(gp.no_arms),
        arm1=float(gp.arm1),
        arm2=float(gp.arm2),
        arm3=float(gp.arm3),
        arm4=float(gp.arm4),
        component_count=len(comps),
        class_id=class_id.tolist(),
        comp_p0=p0.tolist(),
        comp_p1=p1.tolist(),
        comp_p2=p2.tolist(),
        comp_spec=spec.tolist(),
    )


class GalaxyViewer(spy.AppWindow):
    def __init__(self, app: spy.App, gax_path: Path, width: int = 1024, height: int = 1024) -> None:
        super().__init__(
            app,
            width=width,
            height=height,
            title="GAMER Galaxy Viewer",
            resizable=False,
            enable_vsync=False,
        )
        self.kernel = self.device.create_compute_kernel(
            self.device.load_program("tools/shaders/galaxy_renderer.slang", ["main_rt"])
        )

        self.galaxy_path = gax_path
        self.gallery_files = sorted(Path("publish/data/galaxies").glob("*.gax"))
        self.gallery_index = 0
        for i, p in enumerate(self.gallery_files):
            if p.resolve() == gax_path.resolve():
                self.gallery_index = i
                break
        self.packed = pack_galaxy(gax_path)
        self.size = int(min(width, height))
        self.output = self.device.create_texture(
            format=spy.Format.rgba32_float,
            width=self.size,
            height=self.size,
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
        )

        self.camera_pos = np.array([0.5, 0.0, 0.0], dtype=np.float32)
        self.yaw = -float(np.pi) * 0.5
        self.pitch = 0.0
        self.up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.fov = 90.0
        self.exposure = 1.0
        self.gamma = 1.0
        self.saturation = 1.0
        self.ray_step = 0.025
        self.move_speed = 0.8
        self.look_speed = 0.003

        self.keys: dict[spy.KeyCode, bool] = {}
        self.mouse_left = False
        self.mouse_delta = np.zeros((2,), dtype=np.float32)
        self.scroll_delta = 0.0
        self.move_vel = np.zeros((3,), dtype=np.float32)
        self.rot_vel = np.zeros((2,), dtype=np.float32)
        self._mx: float | None = None
        self._my: float | None = None

        self.last_time = time.perf_counter()
        self.fps_smooth = 60.0
        self.last_error = ""
        self._build_ui()

    def _build_ui(self) -> None:
        panel = spy.ui.Window(self.screen, "Galaxy Viewer", size=spy.float2(420, 280))
        self.fps_text = spy.ui.Text(panel, "FPS: 0.0")
        self.file_text = spy.ui.Text(panel, f"Galaxy: {self.galaxy_path.name}")
        self.error_text = spy.ui.Text(panel, "")
        load_group = spy.ui.Group(panel, "Galaxy")
        spy.ui.Button(load_group, "Load .gax...", callback=self._browse_load_gax)
        spy.ui.Button(load_group, "Prev", callback=self._load_prev_galaxy)
        spy.ui.Button(load_group, "Next", callback=self._load_next_galaxy)
        self.fov_slider = spy.ui.SliderFloat(panel, "FOV", value=float(self.fov), min=25.0, max=100.0)
        self.move_slider = spy.ui.SliderFloat(
            panel, "Move Speed", value=float(self.move_speed), min=0.1, max=20.0, flags=spy.ui.SliderFlags.logarithmic
        )
        self.exposure_slider = spy.ui.SliderFloat(panel, "Exposure", value=float(self.exposure), min=0.1, max=5.0)
        self.gamma_slider = spy.ui.SliderFloat(panel, "Gamma", value=float(self.gamma), min=0.1, max=3.0)
        self.sat_slider = spy.ui.SliderFloat(panel, "Saturation", value=float(self.saturation), min=0.0, max=2.0)
        self.step_slider = spy.ui.SliderFloat(
            panel, "Ray Step", value=float(self.ray_step), min=0.005, max=0.05, flags=spy.ui.SliderFlags.logarithmic
        )
        spy.ui.Text(panel, "Controls: LMB drag=look | WASDQE=move | Wheel=speed")

    def _set_galaxy(self, path: Path) -> None:
        try:
            resolved = resolve_gax_path(path)
            self.packed = pack_galaxy(resolved)
            self.galaxy_path = resolved
            for i, p in enumerate(self.gallery_files):
                if p.resolve() == resolved.resolve():
                    self.gallery_index = i
                    break
            self.file_text.text = f"Galaxy: {self.galaxy_path.name}"
            self.last_error = ""
            self.error_text.text = ""
        except Exception as exc:
            self.last_error = str(exc)
            self.error_text.text = f"Error: {self.last_error}"

    def _browse_load_gax(self) -> None:
        path = spy.platform.open_file_dialog([spy.platform.FileDialogFilter("Galaxy Files", "*.gax")])
        if path:
            self._set_galaxy(Path(path))

    def _load_prev_galaxy(self) -> None:
        if not self.gallery_files:
            return
        self.gallery_index = (self.gallery_index - 1) % len(self.gallery_files)
        self._set_galaxy(self.gallery_files[self.gallery_index])

    def _load_next_galaxy(self) -> None:
        if not self.gallery_files:
            return
        self.gallery_index = (self.gallery_index + 1) % len(self.gallery_files)
        self._set_galaxy(self.gallery_files[self.gallery_index])

    def on_keyboard_event(self, event: spy.KeyboardEvent) -> None:
        if event.type == spy.KeyboardEventType.key_press:
            self.keys[event.key] = True
        elif event.type == spy.KeyboardEventType.key_release:
            self.keys[event.key] = False

    def on_mouse_event(self, event: spy.MouseEvent) -> None:
        if event.type == spy.MouseEventType.button_down and event.button == spy.MouseButton.left:
            self.mouse_left = True
        elif event.type == spy.MouseEventType.button_up and event.button == spy.MouseButton.left:
            self.mouse_left = False
        elif event.type == spy.MouseEventType.move:
            if self._mx is not None and self._my is not None:
                self.mouse_delta += np.array([event.pos.x - self._mx, event.pos.y - self._my], dtype=np.float32)
            self._mx = event.pos.x
            self._my = event.pos.y
        elif event.type == spy.MouseEventType.scroll:
            self.scroll_delta += float(event.scroll.y)

    def _forward(self) -> np.ndarray:
        cy = np.cos(self.yaw)
        sy = np.sin(self.yaw)
        cp = np.cos(self.pitch)
        sp = np.sin(self.pitch)
        return _normalize(np.array([cp * sy, sp, cp * cy], dtype=np.float32))

    def _update_camera(self, dt: float) -> None:
        self.move_speed = float(self.move_slider.value)
        self.fov = float(self.fov_slider.value)
        self.exposure = float(self.exposure_slider.value)
        self.gamma = float(self.gamma_slider.value)
        self.saturation = float(self.sat_slider.value)
        self.ray_step = float(self.step_slider.value)

        if abs(self.scroll_delta) > 1e-5:
            self.move_speed = float(np.clip(self.move_speed * np.power(1.1, self.scroll_delta), 0.1, 20.0))
            self.move_slider.value = self.move_speed
            self.scroll_delta = 0.0

        target_rot = (
            np.array([self.mouse_delta[0], self.mouse_delta[1]], dtype=np.float32) * self.look_speed
            if self.mouse_left
            else np.zeros((2,), dtype=np.float32)
        )
        self.rot_vel += (target_rot - self.rot_vel) * min(1.0, 12.0 * dt)
        self.mouse_delta[:] = 0.0

        self.yaw += float(self.rot_vel[0])
        self.pitch += float(self.rot_vel[1])
        limit = float(np.deg2rad(89.0))
        self.pitch = float(np.clip(self.pitch, -limit, limit))

        forward = self._forward()
        right = _normalize(np.cross(self.up, forward))
        upv = _normalize(np.cross(forward, right))
        move = np.array(
            [
                float(self.keys.get(spy.KeyCode.e, False)) - float(self.keys.get(spy.KeyCode.q, False)),
                float(self.keys.get(spy.KeyCode.d, False)) - float(self.keys.get(spy.KeyCode.a, False)),
                float(self.keys.get(spy.KeyCode.w, False)) - float(self.keys.get(spy.KeyCode.s, False)),
            ],
            dtype=np.float32,
        )
        ml = np.linalg.norm(move)
        target_move = move * (self.move_speed / ml) if ml > 1e-6 else np.zeros((3,), dtype=np.float32)
        self.move_vel += (target_move - self.move_vel) * min(1.0, 10.0 * dt)
        self.camera_pos += (upv * self.move_vel[0] + right * self.move_vel[1] + forward * self.move_vel[2]) * dt

    def render(self, render_context: spy.AppWindow.RenderContext) -> None:
        image = render_context.surface_texture
        encoder = render_context.command_encoder

        now = time.perf_counter()
        dt = max(now - self.last_time, 1e-5)
        self.last_time = now
        self.fps_smooth += (1.0 / dt - self.fps_smooth) * min(dt * 5.0, 1.0)
        self.fps_text.text = f"FPS: {self.fps_smooth:.1f}"
        self.file_text.text = f"Galaxy: {self.galaxy_path.name}"
        self.error_text.text = f"Error: {self.last_error}" if self.last_error else ""

        self._update_camera(dt)
        forward = self._forward()
        target = self.camera_pos + forward

        uniforms = {
            "U": {
                "size": self.size,
                "component_count": self.packed.component_count,
                "exposure": float(self.exposure),
                "gamma": float(self.gamma),
                "saturation": float(self.saturation),
                "ray_step": float(self.ray_step),
                "camera": [float(self.camera_pos[0]), float(self.camera_pos[1]), float(self.camera_pos[2])],
                "axis": self.packed.axis,
                "winding_b": self.packed.winding_b,
                "winding_n": self.packed.winding_n,
                "no_arms": self.packed.no_arms,
                "arm1": self.packed.arm1,
                "arm2": self.packed.arm2,
                "arm3": self.packed.arm3,
                "arm4": self.packed.arm4,
                "inv_vp": _inv_vp(self.camera_pos, target, self.up, self.fov),
                "class_id": self.packed.class_id,
                "comp_p0": self.packed.comp_p0,
                "comp_p1": self.packed.comp_p1,
                "comp_p2": self.packed.comp_p2,
                "comp_spec": self.packed.comp_spec,
            }
        }

        self.kernel.dispatch(
            thread_count=[self.size, self.size, 1],
            command_encoder=encoder,
            vars={
                "Uniforms": uniforms,
                "out_image": self.output,
            },
        )
        encoder.blit(image, self.output)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Realtime SlangPy galaxy viewer.")
    p.add_argument("gax_pos", nargs="?", type=Path, help="Optional positional .gax path (e.g. Spiral.gax).")
    p.add_argument("--gax", type=Path, default=None, help="Path to .gax file.")
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--frames", type=int, default=0, help="Run fixed frame count and exit.")
    return p.parse_args()


def resolve_gax_path(arg_path: Path | None) -> Path:
    if arg_path is None:
        return Path("publish/data/galaxies/Spiral.gax")

    p = Path(arg_path)
    if p.exists():
        return p

    # Common case: user passes "Spiral.gax" from repo root.
    in_gallery = Path("publish/data/galaxies") / p.name
    if in_gallery.exists():
        return in_gallery

    # If extension omitted, try .gax in gallery.
    if p.suffix == "":
        in_gallery_ext = Path("publish/data/galaxies") / f"{p.name}.gax"
        if in_gallery_ext.exists():
            return in_gallery_ext

    return p


def main() -> int:
    args = parse_args()
    gax_arg = args.gax if args.gax is not None else args.gax_pos
    gax_path = resolve_gax_path(gax_arg)
    if not gax_path.exists():
        raise RuntimeError(f"Missing galaxy file: {gax_path}")

    device = spy.create_device(include_paths=[str(Path.cwd()), str((Path.cwd() / "tools" / "shaders"))])
    app = spy.App(device=device)
    viewer = GalaxyViewer(app, gax_path=gax_path, width=int(args.width), height=int(args.height))
    if int(args.frames) > 0:
        for _ in range(int(args.frames)):
            app.run_frame()
        app.terminate()
    else:
        app.run()
    _ = viewer
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
