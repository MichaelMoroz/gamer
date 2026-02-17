#!/usr/bin/env python
from __future__ import annotations

import argparse
import concurrent.futures
import math
import multiprocessing
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image
from PySide6.QtCore import QDataStream, QFile, QIODevice
from PySide6.QtGui import QMatrix4x4, QQuaternion, QVector3D, QVector4D

try:
    from numba import njit
except Exception:  # pragma: no cover - optional acceleration path
    njit = None


GRAD3 = (
    (1, 1, 0),
    (-1, 1, 0),
    (1, -1, 0),
    (-1, -1, 0),
    (1, 0, 1),
    (-1, 0, 1),
    (1, 0, -1),
    (-1, 0, -1),
    (0, 1, 1),
    (0, -1, 1),
    (0, 1, -1),
    (0, -1, -1),
)

PERM = (
    151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225, 140,
    36, 103, 30, 69, 142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148, 247, 120, 234,
    75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32, 57, 177, 33, 88, 237,
    149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175, 74, 165, 71, 134, 139, 48,
    27, 166, 77, 146, 158, 231, 83, 111, 229, 122, 60, 211, 133, 230, 220, 105,
    92, 41, 55, 46, 245, 40, 244, 102, 143, 54, 65, 25, 63, 161, 1, 216, 80, 73,
    209, 76, 132, 187, 208, 89, 18, 169, 200, 196, 135, 130, 116, 188, 159, 86,
    164, 100, 109, 198, 173, 186, 3, 64, 52, 217, 226, 250, 124, 123, 5, 202, 38,
    147, 118, 126, 255, 82, 85, 212, 207, 206, 59, 227, 47, 16, 58, 17, 182, 189,
    28, 42, 223, 183, 170, 213, 119, 248, 152, 2, 44, 154, 163, 70, 221, 153, 101,
    155, 167, 43, 172, 9, 129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232,
    178, 185, 112, 104, 218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144, 12,
    191, 179, 162, 241, 81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31,
    181, 199, 106, 157, 184, 84, 204, 176, 115, 121, 50, 45, 127, 4, 150, 254,
    138, 236, 205, 93, 222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215,
    61, 156, 180,
) * 2

PERM_NP = np.array(PERM, dtype=np.int64)
GRAD3_NP = np.array(GRAD3, dtype=np.int64)


if njit is not None:

    @njit
    def _nb_fastfloor(x: float) -> int:
        return int(x) if x > 0.0 else int(x) - 1


    @njit
    def _nb_dot3(gx: int, gy: int, gz: int, x: float, y: float, z: float) -> float:
        return gx * x + gy * y + gz * z


    @njit
    def _nb_contrib(tval: float, gi: int, x: float, y: float, z: float) -> float:
        if tval < 0.0:
            return 0.0
        tt = tval * tval
        return tt * tt * _nb_dot3(
            GRAD3_NP[gi, 0], GRAD3_NP[gi, 1], GRAD3_NP[gi, 2], x, y, z
        )


    @njit
    def _nb_raw_noise_3d(x: float, y: float, z: float) -> float:
        F3 = 1.0 / 3.0
        s = (x + y + z) * F3
        i = _nb_fastfloor(x + s)
        j = _nb_fastfloor(y + s)
        k = _nb_fastfloor(z + s)

        G3 = 1.0 / 6.0
        t = (i + j + k) * G3
        X0 = i - t
        Y0 = j - t
        Z0 = k - t
        x0 = x - X0
        y0 = y - Y0
        z0 = z - Z0

        if x0 >= y0:
            if y0 >= z0:
                i1, j1, k1, i2, j2, k2 = 1, 0, 0, 1, 1, 0
            elif x0 >= z0:
                i1, j1, k1, i2, j2, k2 = 1, 0, 0, 1, 0, 1
            else:
                i1, j1, k1, i2, j2, k2 = 0, 0, 1, 1, 0, 1
        else:
            if y0 < z0:
                i1, j1, k1, i2, j2, k2 = 0, 0, 1, 0, 1, 1
            elif x0 < z0:
                i1, j1, k1, i2, j2, k2 = 0, 1, 0, 0, 1, 1
            else:
                i1, j1, k1, i2, j2, k2 = 0, 1, 0, 1, 1, 0

        x1 = x0 - i1 + G3
        y1 = y0 - j1 + G3
        z1 = z0 - k1 + G3
        x2 = x0 - i2 + 2.0 * G3
        y2 = y0 - j2 + 2.0 * G3
        z2 = z0 - k2 + 2.0 * G3
        x3 = x0 - 1.0 + 3.0 * G3
        y3 = y0 - 1.0 + 3.0 * G3
        z3 = z0 - 1.0 + 3.0 * G3

        ii = i & 255
        jj = j & 255
        kk = k & 255

        gi0 = PERM_NP[ii + PERM_NP[jj + PERM_NP[kk]]] % 12
        gi1 = PERM_NP[ii + i1 + PERM_NP[jj + j1 + PERM_NP[kk + k1]]] % 12
        gi2 = PERM_NP[ii + i2 + PERM_NP[jj + j2 + PERM_NP[kk + k2]]] % 12
        gi3 = PERM_NP[ii + 1 + PERM_NP[jj + 1 + PERM_NP[kk + 1]]] % 12

        n0 = _nb_contrib(0.6 - x0 * x0 - y0 * y0 - z0 * z0, gi0, x0, y0, z0)
        n1 = _nb_contrib(0.6 - x1 * x1 - y1 * y1 - z1 * z1, gi1, x1, y1, z1)
        n2 = _nb_contrib(0.6 - x2 * x2 - y2 * y2 - z2 * z2, gi2, x2, y2, z2)
        n3 = _nb_contrib(0.6 - x3 * x3 - y3 * y3 - z3 * z3, gi3, x3, y3, z3)

        return 32.0 * (n0 + n1 + n2 + n3)


    @njit
    def _nb_octave_noise_3d(
        octaves: float, persistence: float, scale: float, x: float, y: float, z: float
    ) -> float:
        total = 0.0
        frequency = scale
        amplitude = 1.0
        max_amp = 0.0
        for _ in range(int(octaves)):
            total += _nb_raw_noise_3d(x * frequency, y * frequency, z * frequency) * amplitude
            frequency *= 2.0
            max_amp += amplitude
            amplitude *= persistence
        return total / max_amp


    @njit
    def _nb_ridged_mf(
        px: float,
        py: float,
        pz: float,
        frequency: float,
        octaves: int,
        lacunarity: float,
        offset: float,
        gain: float,
    ) -> float:
        value = 0.0
        weight = 1.0
        w = -0.05
        vx = px
        vy = py
        vz = pz
        freq = frequency
        for _ in range(int(octaves)):
            signal = _nb_raw_noise_3d(vx, vy, vz)
            signal = abs(signal)
            signal = offset - signal
            signal *= signal
            signal *= weight
            weight = signal * gain
            if weight > 1.0:
                weight = 1.0
            if weight < 0.0:
                weight = 0.0
            value += signal * math.pow(freq, w)
            vx *= lacunarity
            vy *= lacunarity
            vz *= lacunarity
            freq *= lacunarity
        return (value * 1.25) - 1.0


DEFAULT_SPECTRA = {
    "red": QVector3D(1.0, 0.6, 0.4),
    "yellow": QVector3D(1.0, 0.9, 0.45),
    "blue": QVector3D(0.4, 0.6, 1.0),
    "white": QVector3D(1.0, 1.0, 1.0),
    "cyan": QVector3D(0.3, 0.7, 1.0),
    "purple": QVector3D(1.0, 0.3, 0.8),
}
DEFAULT_SPECTRUM = QVector3D(1.0, 1.0, 1.0)


@dataclass
class GalaxyParams:
    name: str
    axis: QVector3D
    bulge_dust: float
    bulge_axis: QVector3D
    winding_b: float
    winding_n: float
    no_arms: float
    arm1: float
    arm2: float
    arm3: float
    arm4: float
    inner_twirl: float
    warp_amplitude: float
    warp_scale: float


@dataclass
class ComponentParams:
    class_name: str
    strength: float
    spectrum: str
    arm: float
    z0: float
    r0: float
    active: float
    delta: float
    winding: float
    scale: float
    noise_offset: float
    noise_tilt: float
    ks: float
    inner: float
    name: str


@dataclass
class GalaxyData:
    display_name: str
    params: GalaxyParams
    components: List[ComponentParams]


@dataclass
class RasterPixel:
    I: QVector3D = field(default_factory=lambda: QVector3D(0.0, 0.0, 0.0))
    winding: float = 0.0
    radius: float = 0.0
    z: float = 0.0
    tmp: float = 0.0
    step: float = 0.0
    scale: float = 1.0
    P: QVector3D = field(default_factory=lambda: QVector3D(0.0, 0.0, 0.0))


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


def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    if edge1 == edge0:
        xx = 1.0 if x >= edge1 else 0.0
    else:
        xx = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return xx * xx * (3.0 - 2.0 * xx)


def floor_vec(v: QVector3D) -> QVector3D:
    return QVector3D(max(0.0, v.x()), max(0.0, v.y()), max(0.0, v.z()))


def intersect_sphere(
    o: QVector3D,
    d: QVector3D,
    r: QVector3D,
) -> Tuple[bool, QVector3D, QVector3D, float, float]:
    inv = QVector3D(1.0 / (r.x() * r.x()), 1.0 / (r.y() * r.y()), 1.0 / (r.z() * r.z()))

    r_d = QVector3D(d.x() * inv.x(), d.y() * inv.y(), d.z() * inv.z())
    r_o = QVector3D(o.x() * inv.x(), o.y() * inv.y(), o.z() * inv.z())

    A = QVector3D.dotProduct(d, r_d)
    B = 2.0 * QVector3D.dotProduct(d, r_o)
    C = QVector3D.dotProduct(o, r_o) - 1.0

    S = B * B - 4.0 * A * C
    if S <= 0.0:
        return False, QVector3D(), QVector3D(), 0.0, 0.0

    t0 = (-B - math.sqrt(S)) / (2.0 * A)
    t1 = (-B + math.sqrt(S)) / (2.0 * A)
    isp1 = o + d * t0
    isp2 = o + d * t1
    return True, isp1, isp2, t0, t1


class SimplexNoise:
    USE_NUMBA = njit is not None

    @classmethod
    def warmup(cls) -> None:
        if cls.USE_NUMBA:
            _nb_raw_noise_3d(0.1, 0.2, 0.3)
            _nb_octave_noise_3d(2.0, 0.5, 0.1, 0.1, 0.2, 0.3)
            _nb_ridged_mf(0.1, 0.2, 0.3, 1.0, 2, 2.5, 1.0, 1.0)

    @staticmethod
    def _fastfloor(x: float) -> int:
        return int(x) if x > 0 else int(x) - 1

    @staticmethod
    def _dot2(g: Sequence[int], x: float, y: float) -> float:
        return g[0] * x + g[1] * y

    @staticmethod
    def _dot3(g: Sequence[int], x: float, y: float, z: float) -> float:
        return g[0] * x + g[1] * y + g[2] * z

    @classmethod
    def raw_noise_3d(cls, x: float, y: float, z: float) -> float:
        if cls.USE_NUMBA:
            return float(_nb_raw_noise_3d(x, y, z))
        F3 = 1.0 / 3.0
        s = (x + y + z) * F3
        i = cls._fastfloor(x + s)
        j = cls._fastfloor(y + s)
        k = cls._fastfloor(z + s)

        G3 = 1.0 / 6.0
        t = (i + j + k) * G3
        X0 = i - t
        Y0 = j - t
        Z0 = k - t
        x0 = x - X0
        y0 = y - Y0
        z0 = z - Z0

        if x0 >= y0:
            if y0 >= z0:
                i1, j1, k1, i2, j2, k2 = 1, 0, 0, 1, 1, 0
            elif x0 >= z0:
                i1, j1, k1, i2, j2, k2 = 1, 0, 0, 1, 0, 1
            else:
                i1, j1, k1, i2, j2, k2 = 0, 0, 1, 1, 0, 1
        else:
            if y0 < z0:
                i1, j1, k1, i2, j2, k2 = 0, 0, 1, 0, 1, 1
            elif x0 < z0:
                i1, j1, k1, i2, j2, k2 = 0, 1, 0, 0, 1, 1
            else:
                i1, j1, k1, i2, j2, k2 = 0, 1, 0, 1, 1, 0

        x1 = x0 - i1 + G3
        y1 = y0 - j1 + G3
        z1 = z0 - k1 + G3
        x2 = x0 - i2 + 2.0 * G3
        y2 = y0 - j2 + 2.0 * G3
        z2 = z0 - k2 + 2.0 * G3
        x3 = x0 - 1.0 + 3.0 * G3
        y3 = y0 - 1.0 + 3.0 * G3
        z3 = z0 - 1.0 + 3.0 * G3

        ii = i & 255
        jj = j & 255
        kk = k & 255
        gi0 = PERM[ii + PERM[jj + PERM[kk]]] % 12
        gi1 = PERM[ii + i1 + PERM[jj + j1 + PERM[kk + k1]]] % 12
        gi2 = PERM[ii + i2 + PERM[jj + j2 + PERM[kk + k2]]] % 12
        gi3 = PERM[ii + 1 + PERM[jj + 1 + PERM[kk + 1]]] % 12

        def contrib(tval: float, gx: int, xx: float, yy: float, zz: float) -> float:
            if tval < 0:
                return 0.0
            tt = tval * tval
            return tt * tt * cls._dot3(GRAD3[gx], xx, yy, zz)

        n0 = contrib(0.6 - x0 * x0 - y0 * y0 - z0 * z0, gi0, x0, y0, z0)
        n1 = contrib(0.6 - x1 * x1 - y1 * y1 - z1 * z1, gi1, x1, y1, z1)
        n2 = contrib(0.6 - x2 * x2 - y2 * y2 - z2 * z2, gi2, x2, y2, z2)
        n3 = contrib(0.6 - x3 * x3 - y3 * y3 - z3 * z3, gi3, x3, y3, z3)

        return 32.0 * (n0 + n1 + n2 + n3)

    @classmethod
    def octave_noise_3d(
        cls,
        octaves: float,
        persistence: float,
        scale: float,
        x: float,
        y: float,
        z: float,
    ) -> float:
        if cls.USE_NUMBA:
            return float(_nb_octave_noise_3d(octaves, persistence, scale, x, y, z))
        total = 0.0
        frequency = scale
        amplitude = 1.0
        max_amp = 0.0
        for _ in range(int(octaves)):
            total += cls.raw_noise_3d(x * frequency, y * frequency, z * frequency) * amplitude
            frequency *= 2.0
            max_amp += amplitude
            amplitude *= persistence
        return total / max_amp

    @classmethod
    def get_ridged_mf(
        cls,
        p: QVector3D,
        frequency: float,
        octaves: int,
        lacunarity: float,
        offset: float,
        gain: float,
    ) -> float:
        if cls.USE_NUMBA:
            return float(
                _nb_ridged_mf(
                    p.x(),
                    p.y(),
                    p.z(),
                    frequency,
                    int(octaves),
                    lacunarity,
                    offset,
                    gain,
                )
            )
        value = 0.0
        weight = 1.0
        w = -0.05
        vt = QVector3D(p)
        freq = frequency
        for _ in range(int(octaves)):
            signal = cls.raw_noise_3d(vt.x(), vt.y(), vt.z())
            signal = abs(signal)
            signal = offset - signal
            signal *= signal
            signal *= weight
            weight = signal * gain
            if weight > 1.0:
                weight = 1.0
            if weight < 0.0:
                weight = 0.0
            value += signal * math.pow(freq, w)
            vt *= lacunarity
            freq *= lacunarity
        return (value * 1.25) - 1.0


class GamerCamera:
    def __init__(self, camera: QVector3D, target: QVector3D, up: QVector3D, perspective: float) -> None:
        self.camera = QVector3D(camera)
        self.target = QVector3D(target)
        self.up = QVector3D(up)
        self.perspective = perspective
        self.rot_matrix = QMatrix4x4()
        self.rot_matrix.setToIdentity()
        self.inv_vp = QMatrix4x4()
        self.setup_viewmatrix()

    def setup_viewmatrix(self) -> None:
        projection = QMatrix4x4()
        projection.setToIdentity()
        projection.perspective(self.perspective, 1.0, 1.0, 100.0)

        view = QMatrix4x4()
        view.setToIdentity()
        view.lookAt(
            self.rot_matrix.map(self.target),
            self.rot_matrix.map(self.camera),
            self.rot_matrix.map(self.up),
        )
        inv, ok = (projection * view).inverted()
        if not ok:
            raise RuntimeError("Failed to invert view-projection matrix")
        self.inv_vp = inv

    def coord2ray(self, x: float, y: float, width: float) -> QVector3D:
        xx = x / (width * 0.5) - 1.0
        yy = y / (width * 0.5) - 1.0
        screen = QVector4D(xx, -yy, 1.0, 1.0)
        world = self.inv_vp.map(screen)
        return world.toVector3D().normalized()


class GalaxyInstance:
    def __init__(self, galaxy: GalaxyData) -> None:
        self.galaxy = galaxy
        self.position = QVector3D(0.0, 0.0, 0.0)
        self.orientation = QVector3D(0.0, 1.0, 0.0)
        self.intensity_scale = 1.0
        self.rotmat = QQuaternion.rotationTo(QVector3D(0.0, 1.0, 0.0), self.orientation)


class GalaxyComponent:
    def __init__(self, comp: ComponentParams, params: GalaxyParams, spectrum: QVector3D) -> None:
        self.comp = comp
        self.galaxy_params = params
        self.spectrum = spectrum
        self.current_gi: GalaxyInstance | None = None

    def get_height_modulation(self, height: float) -> float:
        h = abs(height / self.comp.z0)
        if h > 2.0:
            return 0.0
        val = 1.0 / ((math.exp(h) + math.exp(-h)) / 2.0)
        return val * val

    def get_radius(self, p: QVector3D, gi: GalaxyInstance) -> Tuple[float, QVector3D, float]:
        dott = QVector3D.dotProduct(p, gi.orientation)
        P = p - gi.orientation * dott
        return P.length() / self.galaxy_params.axis.x(), P, dott

    def twirl(self, p: QVector3D, twirl: float) -> QVector3D:
        if self.current_gi is None:
            raise RuntimeError("Component used without galaxy instance")
        q = QQuaternion.fromAxisAndAngle(self.current_gi.orientation, twirl * 180.0)
        return q.rotatedVector(p)

    def get_perlin_cloud_noise(self, p: QVector3D, t: float, NN: int, ks: float, pers: float) -> float:
        r = self.twirl(p, t)
        return SimplexNoise.octave_noise_3d(NN, pers, ks * 0.1, r.x(), r.y(), r.z())

    def find_difference(self, t1: float, t2: float) -> float:
        vals = (
            abs(t1 - t2),
            abs(t1 - t2 - 2 * math.pi),
            abs(t1 - t2 + 2 * math.pi),
            abs(t1 - t2 - 4 * math.pi),
            abs(t1 - t2 + 4 * math.pi),
        )
        return min(vals)

    def get_theta(self, p: QVector3D) -> float:
        if self.current_gi is None:
            raise RuntimeError("Component used without galaxy instance")
        quat_rot = self.current_gi.rotmat.rotatedVector(p)
        return math.atan2(quat_rot.x(), quat_rot.z()) + self.comp.delta

    def get_winding(self, rad: float) -> float:
        r = rad + 0.05
        t = (
            math.atan(math.exp(-0.25 / (0.5 * r)) / self.galaxy_params.winding_b)
            * 2.0
            * self.galaxy_params.winding_n
        )
        return t

    def get_arm(self, rad: float, p: QVector3D, disp: float) -> float:
        work_winding = self.get_winding(rad)
        work_theta = -self.get_theta(p)
        v = abs(self.find_difference(work_winding, work_theta + disp)) / math.pi
        return math.pow(1.0 - v, self.comp.arm * 15.0)

    def calculate_arm_value(self, rad: float, P: QVector3D) -> float:
        v1 = self.get_arm(rad, P, self.galaxy_params.arm1)
        if self.galaxy_params.no_arms == 1:
            return v1
        v = max(v1, self.get_arm(rad, P, self.galaxy_params.arm2))
        if self.galaxy_params.no_arms == 2:
            return v
        v = max(v, self.get_arm(rad, P, self.galaxy_params.arm3))
        if self.galaxy_params.no_arms == 3:
            return v
        return max(v, self.get_arm(rad, P, self.galaxy_params.arm4))

    def get_radial_intensity(self, rad: float) -> float:
        r = math.exp(-rad / (self.comp.r0 * 0.5))
        return clamp(r - 0.01, 0.0, 1.0)

    def component_intensity(self, rp: RasterPixel, p: QVector3D, ival: float) -> None:
        rp.I = p
        rp.I.setX(ival)

    def calculate_intensity(self, rp: RasterPixel, p: QVector3D, gi: GalaxyInstance, weight: float) -> float:
        self.current_gi = gi
        arm_val = 1.0
        winding = 0.0
        if rp.z <= 0.01:
            return 0.0

        intensity = self.get_radial_intensity(rp.radius)
        if intensity > 0.1:
            intensity = 0.1

        if intensity <= 0.001:
            return 0.0

        scale = math.pow(smoothstep(0.0, 1.0 * self.comp.inner, rp.radius), 4.0)
        if self.comp.arm != 0:
            arm_val = self.calculate_arm_value(rp.radius, rp.P)
            if self.comp.winding != 0:
                winding = self.get_winding(rp.radius) * self.comp.winding

        rp.winding = winding
        val = self.comp.strength * scale * arm_val * rp.z * intensity * gi.intensity_scale
        if val * weight <= 0.0005:
            return 0.0

        self.component_intensity(rp, p, val * weight)
        return val


class GalaxyComponentBulge(GalaxyComponent):
    def calculate_intensity(self, rp: RasterPixel, p: QVector3D, gi: GalaxyInstance, weight: float) -> float:
        self.current_gi = gi
        self.component_intensity(rp, p, weight)
        return rp.tmp

    def get_height_modulation(self, height: float) -> float:
        return 1.0

    def get_radius(self, p: QVector3D, gi: GalaxyInstance) -> Tuple[float, QVector3D, float]:
        return 0.0, QVector3D(), 0.0

    def component_intensity(self, rp: RasterPixel, p: QVector3D, ival: float) -> None:
        rho_0 = self.comp.strength * ival
        pos = self.current_gi.rotmat.rotatedVector(p)
        rad = (pos.length() + 0.01) * self.comp.r0
        rad += 0.01
        i = rho_0 * (math.pow(rad, -0.855) * math.exp(-math.pow(rad, 1.0 / 4.0)) - 0.05) * self.current_gi.intensity_scale
        if i < 0:
            i = 0.0
        rp.I = rp.I + (self.spectrum * (i * rp.scale))
        rp.tmp = i


class GalaxyComponentDisk(GalaxyComponent):
    def component_intensity(self, rp: RasterPixel, p: QVector3D, ival: float) -> None:
        if ival < 0.0005:
            return
        p2 = abs(self.get_perlin_cloud_noise(p, rp.winding, 10, self.comp.scale, self.comp.ks))
        p2 = max(p2, 0.01)
        p2 = math.pow(p2, self.comp.noise_tilt)
        p2 += self.comp.noise_offset
        if p2 < 0:
            return
        rp.I = rp.I + (self.spectrum * (ival * p2 * rp.scale))


class GalaxyComponentDust(GalaxyComponent):
    def component_intensity(self, rp: RasterPixel, p: QVector3D, ival: float) -> None:
        if ival < 0.0005:
            return
        p2 = self.get_perlin_cloud_noise(p, rp.winding, 9, self.comp.scale, self.comp.ks)
        p2 = max(p2 - self.comp.noise_offset, 0.0)
        p2 = clamp(math.pow(5.0 * p2, self.comp.noise_tilt), -10.0, 10.0)
        s = 0.01
        rp.I.setX(rp.I.x() * math.exp(-p2 * ival * self.spectrum.x() * s))
        rp.I.setY(rp.I.y() * math.exp(-p2 * ival * self.spectrum.y() * s))
        rp.I.setZ(rp.I.z() * math.exp(-p2 * ival * self.spectrum.z() * s))


class GalaxyComponentDust2(GalaxyComponent):
    def component_intensity(self, rp: RasterPixel, p: QVector3D, ival: float) -> None:
        if ival < 0.0005:
            return
        r = self.twirl(p, rp.winding)
        p2 = SimplexNoise.get_ridged_mf(
            r * self.comp.scale,
            self.comp.ks,
            9,
            2.5,
            self.comp.noise_offset,
            self.comp.noise_tilt,
        )
        p2 = max(p2, 0.0)
        s = 0.01
        rp.I.setX(rp.I.x() * math.exp(-p2 * ival * self.spectrum.x() * s))
        rp.I.setY(rp.I.y() * math.exp(-p2 * ival * self.spectrum.y() * s))
        rp.I.setZ(rp.I.z() * math.exp(-p2 * ival * self.spectrum.z() * s))


class GalaxyComponentDustPositive(GalaxyComponent):
    def component_intensity(self, rp: RasterPixel, p: QVector3D, ival: float) -> None:
        if ival < 0.0005:
            return
        r = self.twirl(p, rp.winding)
        p2 = SimplexNoise.get_ridged_mf(
            r * self.comp.scale,
            self.comp.ks,
            9,
            2.5,
            self.comp.noise_offset,
            self.comp.noise_tilt,
        )
        p2 = max(p2, 0.0)
        rp.I = rp.I + self.spectrum * (ival * p2 * rp.scale)


class GalaxyComponentStars(GalaxyComponent):
    def component_intensity(self, rp: RasterPixel, p: QVector3D, ival: float) -> None:
        perlin = abs(
            SimplexNoise.octave_noise_3d(
                10,
                self.comp.ks,
                0.01 * self.comp.scale * 100.0,
                p.x(),
                p.y(),
                p.z(),
            )
        )
        add_noise = 0.0
        if self.comp.noise_offset != 0:
            add_noise = self.comp.noise_offset * self.get_perlin_cloud_noise(p, rp.winding, 4, 2, -2)
            add_noise += 0.5 * self.comp.noise_offset * self.get_perlin_cloud_noise(p, rp.winding * 0.5, 4, 4, -2)
        val = abs(math.pow(perlin + 1.0 + add_noise, self.comp.noise_tilt))
        rp.I = rp.I + self.spectrum * (ival * val * rp.scale)


class GalaxyComponentStarsSmall(GalaxyComponent):
    def component_intensity(self, rp: RasterPixel, p: QVector3D, ival: float) -> None:
        # rand()-based tiny-star sprinkling is intentionally omitted for deterministic output.
        return


COMPONENT_MAP = {
    "bulge": GalaxyComponentBulge,
    "disk": GalaxyComponentDisk,
    "dust": GalaxyComponentDust,
    "dust2": GalaxyComponentDust2,
    "dust positive": GalaxyComponentDustPositive,
    "stars": GalaxyComponentStars,
    "stars small": GalaxyComponentStarsSmall,
}


def load_qvector3d(stream: QDataStream) -> QVector3D:
    v = QVector3D()
    stream >> v
    return v


def load_galaxy(gax_path: Path) -> GalaxyData:
    f = QFile(str(gax_path))
    if not f.open(QIODevice.OpenModeFlag.ReadOnly):
        raise RuntimeError(f"Could not open {gax_path}")
    try:
        stream = QDataStream(f)
        stream.setVersion(QDataStream.Version.Qt_5_6)

        display_name = stream.readQString()

        gp = GalaxyParams(
            name=stream.readQString(),
            axis=load_qvector3d(stream),
            bulge_dust=stream.readDouble(),
            bulge_axis=load_qvector3d(stream),
            winding_b=stream.readDouble(),
            winding_n=stream.readDouble(),
            no_arms=stream.readDouble(),
            arm1=stream.readDouble(),
            arm2=stream.readDouble(),
            arm3=stream.readDouble(),
            arm4=stream.readDouble(),
            inner_twirl=stream.readDouble(),
            warp_amplitude=stream.readDouble(),
            warp_scale=stream.readDouble(),
        )

        comp_count = stream.readInt32()
        components: List[ComponentParams] = []
        for _ in range(comp_count):
            comp = ComponentParams(
                class_name=stream.readQString(),
                strength=stream.readDouble(),
                spectrum=stream.readQString(),
                arm=stream.readDouble(),
                z0=stream.readDouble(),
                r0=stream.readDouble(),
                active=stream.readDouble(),
                delta=stream.readDouble(),
                winding=stream.readDouble(),
                scale=stream.readDouble(),
                noise_offset=stream.readDouble(),
                noise_tilt=stream.readDouble(),
                ks=stream.readDouble(),
                inner=stream.readDouble(),
                name=stream.readQString(),
            )
            components.append(comp)

        return GalaxyData(display_name=display_name, params=gp, components=components)
    finally:
        f.close()


def instantiate_components(galaxy: GalaxyData) -> List[GalaxyComponent]:
    comps: List[GalaxyComponent] = []
    for cp in galaxy.components:
        cls = COMPONENT_MAP.get(cp.class_name.lower())
        if cls is None:
            continue
        spec = DEFAULT_SPECTRA.get(cp.spectrum.lower(), DEFAULT_SPECTRUM)
        comps.append(cls(cp, galaxy.params, QVector3D(spec)))
    return comps


def post_process(val: QVector3D, exposure: float, gamma: float, saturation: float) -> Tuple[int, int, int]:
    v = QVector3D(val)
    v *= float(1.0 / exposure)
    v.setX(float(math.pow(v.x(), gamma)))
    v.setY(float(math.pow(v.y(), gamma)))
    v.setZ(float(math.pow(v.z(), gamma)))

    center = (v.x() + v.y() + v.z()) / 3.0
    tmp = QVector3D(center - v.x(), center - v.y(), center - v.z())
    v.setX(float(center - saturation * tmp.x()))
    v.setY(float(center - saturation * tmp.y()))
    v.setZ(float(center - saturation * tmp.z()))

    c = QVector3D(
        float(clamp(v.x() * 10.0, 0.0, 255.0)),
        float(clamp(v.y() * 10.0, 0.0, 255.0)),
        float(clamp(v.z() * 10.0, 0.0, 255.0)),
    )
    # Equivalent to shadow buffer output: QColor(c.blue(), c.green(), c.red()).
    return int(c.x()), int(c.y()), int(c.z())


def render_python(galaxy: GalaxyData, cfg: RenderConfig) -> np.ndarray:
    camera = GamerCamera(cfg.camera, cfg.target, cfg.up, cfg.fov)
    gi = GalaxyInstance(galaxy)
    components = instantiate_components(galaxy)

    size = cfg.size
    image_linear = np.zeros((size, size, 3), dtype=np.float64)

    for idx in range(size * size):
        i = idx % size
        j = (idx - i) // size
        direction = camera.coord2ray(i, j, size)
        rp = render_pixel(direction, gi, components, camera, cfg)
        image_linear[j, i, 0] = rp.x()
        image_linear[j, i, 1] = rp.y()
        image_linear[j, i, 2] = rp.z()

    out = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            r, g, b = post_process(
                QVector3D(
                    float(image_linear[y, x, 0]),
                    float(image_linear[y, x, 1]),
                    float(image_linear[y, x, 2]),
                ),
                cfg.exposure,
                cfg.gamma,
                cfg.saturation,
            )
            out[y, x, 0] = r
            out[y, x, 1] = g
            out[y, x, 2] = b

    return out


def render_pixel(
    direction: QVector3D,
    gi: GalaxyInstance,
    components: Sequence[GalaxyComponent],
    camera: GamerCamera,
    cfg: RenderConfig,
) -> QVector3D:
    rp = RasterPixel()
    origin = camera.camera - gi.position
    intersects, isp1, isp2, t1, t2 = intersect_sphere(origin, direction, gi.galaxy.params.axis)

    if t2 > 0:
        isp2 = origin

    if t1 > 0 and t2 > 0:
        intersects = False

    if intersects:
        get_intensity(gi, rp, isp1, isp2, components, camera, cfg)

    rp.I *= 0.01 / cfg.ray_step
    return rp.I


def get_intensity(
    gi: GalaxyInstance,
    rp: RasterPixel,
    isp1: QVector3D,
    isp2: QVector3D,
    components: Sequence[GalaxyComponent],
    camera: GamerCamera,
    cfg: RenderConfig,
) -> None:
    origin = QVector3D(isp1)
    length = (isp1 - isp2).length()
    direction = (isp1 - isp2).normalized()
    step = cfg.ray_step
    p = QVector3D(origin)
    rp.scale = step

    cam = camera.camera - gi.position
    rp.step = step

    ll = (isp2 - origin).normalized()
    while QVector3D.dotProduct(p - origin, ll) < length + step:
        step = clamp(math.pow((p - cam).length(), 1.0) * cfg.ray_step, 0.001, 0.01)
        rp.step = step

        for gc in components:
            if gc.comp.active == 1:
                radius, rp.P, rp.z = gc.get_radius(p, gi)
                rp.radius = radius
                rp.z = gc.get_height_modulation(rp.z)
                gc.calculate_intensity(rp, p, gi, step * 200.0)

        p = p - direction * step
        rp.I = floor_vec(rp.I)


def run_gamer_baseline(
    gamer_exe: Path,
    gax_file: Path,
    out_base: Path,
    cfg: RenderConfig,
) -> None:
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
        raise RuntimeError(
            f"Gamer baseline failed for {gax_file.name}\n"
            f"stdout:\n{res.stdout}\n"
            f"stderr:\n{res.stderr}"
        )


def save_png(img: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path)


def load_png(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def compare_images(reference: np.ndarray, candidate: np.ndarray) -> Tuple[bool, int, int, np.ndarray]:
    if reference.shape != candidate.shape:
        raise ValueError(f"Shape mismatch: {reference.shape} vs {candidate.shape}")
    diff = np.abs(reference.astype(np.int16) - candidate.astype(np.int16)).astype(np.uint8)
    mismatch = np.any(diff != 0, axis=2)
    mismatch_count = int(mismatch.sum())
    max_abs = int(diff.max())
    return mismatch_count == 0, mismatch_count, max_abs, diff


def save_diff_heatmap(diff: np.ndarray, out_path: Path) -> None:
    intensity = diff.max(axis=2)
    if int(intensity.max()) > 0:
        scaled = (intensity.astype(np.float32) / float(intensity.max()) * 255.0).astype(np.uint8)
    else:
        scaled = intensity.astype(np.uint8)
    rgb = np.stack([scaled, np.zeros_like(scaled), 255 - scaled], axis=2)
    save_png(rgb, out_path)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reproduce GAMER galaxy renders in Python and compare pixel-exactly.")
    p.add_argument("--mode", choices=["generate-baseline", "compare", "all"], default="all")

    p.add_argument("--gamer-exe", type=Path, default=Path("publish/win/gamer/Gamer.exe"))
    p.add_argument("--galaxy-dir", type=Path, default=Path("publish/data/galaxies"))
    p.add_argument(
        "--galaxy",
        action="append",
        default=[],
        help="Galaxy filter (stem or filename). Can be passed multiple times.",
    )
    p.add_argument("--baseline-dir", type=Path, default=Path("artifacts/baseline"))
    p.add_argument("--python-dir", type=Path, default=Path("artifacts/python"))
    p.add_argument("--diff-dir", type=Path, default=Path("artifacts/diff"))

    p.add_argument("--size", type=int, default=64)
    p.add_argument("--camera", nargs=3, type=float, default=[0.5, 0.0, 0.0])
    p.add_argument("--target", nargs=3, type=float, default=[0.0, 0.0, 0.0])
    p.add_argument("--up", nargs=3, type=float, default=[0.0, 1.0, 0.0])
    p.add_argument("--fov", type=float, default=90.0)
    p.add_argument("--exposure", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--saturation", type=float, default=1.0)
    p.add_argument("--ray-step", type=float, default=0.025)
    p.add_argument("--workers", type=int, default=max(1, min(4, multiprocessing.cpu_count())))

    p.add_argument("--fail-fast", action="store_true")
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
        files = [
            f
            for f in files
            if f.name.lower() in wanted or f.stem.lower() in wanted
        ]
    if not files:
        if filters:
            raise RuntimeError(f"No matching .gax files for filters {list(filters)} in {galaxy_dir}")
        raise RuntimeError(f"No .gax files found in {galaxy_dir}")
    return files


def generate_baselines(galaxies: Iterable[Path], cfg: RenderConfig, args: argparse.Namespace) -> None:
    for gax in galaxies:
        out_base = args.baseline_dir / gax.stem
        run_gamer_baseline(args.gamer_exe, gax, out_base, cfg)
        expected = out_base.with_suffix(".png")
        if not expected.exists():
            raise RuntimeError(f"Expected baseline output missing: {expected}")
        print(f"[baseline] {gax.name} -> {expected}")


def run_comparisons(galaxies: Iterable[Path], cfg: RenderConfig, args: argparse.Namespace) -> int:
    for gax in galaxies:
        base_png = (args.baseline_dir / gax.stem).with_suffix(".png")
        if not base_png.exists():
            raise RuntimeError(
                f"Missing baseline image: {base_png}. "
                f"Run with --mode generate-baseline or --mode all first."
            )

    cfg_payload = {
        "camera": (cfg.camera.x(), cfg.camera.y(), cfg.camera.z()),
        "target": (cfg.target.x(), cfg.target.y(), cfg.target.z()),
        "up": (cfg.up.x(), cfg.up.y(), cfg.up.z()),
        "fov": cfg.fov,
        "exposure": cfg.exposure,
        "gamma": cfg.gamma,
        "saturation": cfg.saturation,
        "ray_step": cfg.ray_step,
        "size": cfg.size,
    }

    failures = 0
    galaxy_list = list(galaxies)
    workers = max(1, int(args.workers))
    if args.fail_fast:
        workers = 1

    if workers == 1 or len(galaxy_list) == 1:
        for gax in galaxy_list:
            result = run_single_comparison(
                str(gax),
                str(args.baseline_dir),
                str(args.python_dir),
                str(args.diff_dir),
                cfg_payload,
            )
            failures += emit_comparison_result(result)
            if args.fail_fast and failures > 0:
                return failures
        return failures

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as exe:
        futures = {
            exe.submit(
                run_single_comparison,
                str(gax),
                str(args.baseline_dir),
                str(args.python_dir),
                str(args.diff_dir),
                cfg_payload,
            ): gax
            for gax in galaxy_list
        }
        for fut in concurrent.futures.as_completed(futures):
            result = fut.result()
            failures += emit_comparison_result(result)
            if args.fail_fast and failures > 0:
                for pending in futures:
                    pending.cancel()
                return failures

    return failures


def emit_comparison_result(result: Dict[str, object]) -> int:
    if bool(result["equal"]):
        print(f"[PASS] {result['name']}: exact match")
        return 0
    print(
        f"[FAIL] {result['name']}: mismatched_pixels={result['mismatch_count']} "
        f"max_abs_diff={result['max_abs']} diff={result['diff_png']}"
    )
    return 1


def run_single_comparison(
    gax_path: str,
    baseline_dir: str,
    python_dir: str,
    diff_dir: str,
    cfg_payload: Dict[str, object],
) -> Dict[str, object]:
    gax = Path(gax_path)
    base_png = (Path(baseline_dir) / gax.stem).with_suffix(".png")
    py_png = Path(python_dir) / f"{gax.stem}.png"
    diff_png = Path(diff_dir) / f"{gax.stem}_diff.png"

    cfg = RenderConfig(
        camera=QVector3D(*cfg_payload["camera"]),
        target=QVector3D(*cfg_payload["target"]),
        up=QVector3D(*cfg_payload["up"]),
        fov=float(cfg_payload["fov"]),
        exposure=float(cfg_payload["exposure"]),
        gamma=float(cfg_payload["gamma"]),
        saturation=float(cfg_payload["saturation"]),
        ray_step=float(cfg_payload["ray_step"]),
        size=int(cfg_payload["size"]),
    )

    galaxy = load_galaxy(gax)
    py_img = render_python(galaxy, cfg)
    save_png(py_img, py_png)

    baseline = load_png(base_png)
    candidate = load_png(py_png)
    equal, mismatch_count, max_abs, diff = compare_images(baseline, candidate)
    if not equal:
        save_diff_heatmap(diff, diff_png)

    return {
        "name": gax.name,
        "equal": equal,
        "mismatch_count": mismatch_count,
        "max_abs": max_abs,
        "diff_png": str(diff_png),
    }


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    cfg = config_from_args(args)
    if args.mode in ("compare", "all"):
        SimplexNoise.warmup()

    galaxies = list_galaxies(args.galaxy_dir, args.galaxy)
    print(f"Found {len(galaxies)} galaxy examples in {args.galaxy_dir}")

    if args.mode in ("generate-baseline", "all"):
        generate_baselines(galaxies, cfg, args)

    failures = 0
    if args.mode in ("compare", "all"):
        failures = run_comparisons(galaxies, cfg, args)

    if failures > 0:
        print(f"Summary: {failures} comparison(s) failed.")
        return 1

    print("Summary: all requested steps completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
