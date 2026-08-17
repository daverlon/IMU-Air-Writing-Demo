"""
Wrist orientation → a 2D writing page.

The wrist holds a virtual pen tip. A "page" (plane + origin) is locked
explicitly by the caller at the moment writing is armed, and never moves
afterwards — so a letter can't inherit drift from earlier motion, and the
page doesn't warp while you write.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ble_client import IMUSample

TIP_LENGTH_M = 0.16  # virtual pen length; scales how far a wrist tilt travels
STROKE_GAIN = 1.15  # slight size boost; speed still uses the raw motion
PITCH_SIGN = -1.0
H_SIGN = -1.0

# Speed is measured on the smoothed point, so ink and speed share the same lag
# and a velocity minimum lands at the right place in the recorded stroke.
POS_SMOOTH = 0.30  # EMA on the page point (lower = calmer, but blunts dips)
SPEED_SMOOTH = 0.55  # less snap so a letter does not drop out mid-stroke


def _quat_from_euler_deg(roll: float, pitch: float, yaw: float) -> np.ndarray:
    r = math.radians(roll) * 0.5
    p = math.radians(pitch) * 0.5
    y = math.radians(yaw) * 0.5
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float64,
    )


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = q
    tx = 2.0 * (qy * v[2] - qz * v[1])
    ty = 2.0 * (qz * v[0] - qx * v[2])
    tz = 2.0 * (qx * v[1] - qy * v[0])
    return np.array(
        [
            v[0] + qw * tx + (qy * tz - qz * ty),
            v[1] + qw * ty + (qz * tx - qx * tz),
            v[2] + qw * tz + (qx * ty - qy * tx),
        ],
        dtype=np.float64,
    )


def _page_basis(forward: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Right/up axes of a page facing `forward`, kept upright against gravity."""
    f = forward / max(np.linalg.norm(forward), 1e-9)
    up = np.array([0.0, 0.0, 1.0]) - f * float(np.array([0.0, 0.0, 1.0]) @ f)
    if np.linalg.norm(up) < 0.25:  # pointing near-vertical: pick any stable up
        fallback = np.array([0.0, 1.0, 0.0])
        up = fallback - f * float(fallback @ f)
    up /= max(np.linalg.norm(up), 1e-9)
    right = np.cross(up, f)
    right /= max(np.linalg.norm(right), 1e-9)
    up = np.cross(f, right)
    up /= max(np.linalg.norm(up), 1e-9)
    return right, up


@dataclass
class Pose:
    x: float  # metres on the page, origin = where the page was locked
    y: float
    speed: float  # metres/second on the page
    t_ms: int


class MotionPipeline:
    def __init__(self):
        self._forward: np.ndarray | None = None
        self._origin: np.ndarray | None = None
        self._tip = np.zeros(3, dtype=np.float64)
        self._pointing = np.array([1.0, 0.0, 0.0])
        self._pos = np.zeros(2, dtype=np.float64)
        self._speed = 0.0
        self._prev_t_ms: int | None = None

    def lock_page(self):
        """Pin the page to the current wrist pose; the pen is now at (0, 0)."""
        self._forward = self._pointing.copy()
        self._origin = self._tip.copy()
        self._pos[:] = 0.0
        self._speed = 0.0

    def update(self, sample: IMUSample) -> Pose:
        q = _quat_from_euler_deg(sample.roll, PITCH_SIGN * sample.pitch, sample.yaw)
        self._pointing = _quat_rotate(q, np.array([1.0, 0.0, 0.0]))
        self._pointing /= max(np.linalg.norm(self._pointing), 1e-9)
        self._tip = _quat_rotate(q, np.array([TIP_LENGTH_M, 0.0, 0.0]))

        if self._forward is None or self._origin is None:
            self.lock_page()

        right, up = _page_basis(self._forward)
        delta = self._tip - self._origin
        target = np.array([H_SIGN * float(delta @ right), float(delta @ up)])

        prev = self._pos.copy()
        self._pos += (target - self._pos) * POS_SMOOTH

        dt = 0.02
        if self._prev_t_ms is not None:
            dt = max(0.004, min(0.2, (sample.timestamp_ms - self._prev_t_ms) / 1000.0))
        self._prev_t_ms = sample.timestamp_ms

        instant = float(np.linalg.norm(self._pos - prev)) / dt
        self._speed += (instant - self._speed) * SPEED_SMOOTH

        return Pose(
            x=float(self._pos[0] * STROKE_GAIN),
            y=float(self._pos[1] * STROKE_GAIN),
            speed=self._speed,
            t_ms=sample.timestamp_ms,
        )
