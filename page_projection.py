#!/usr/bin/env python3
"""
Air-writing plane demo.

Wrist orientation drives a 16 cm hinged tip. A page is locked at the first
pose and the tip is projected onto it. Accel is never integrated.

    python page_projection.py

R  canned loop (synthetic S)
T  live IMU (clears and relocks the page to the current wrist)
D  dark / light
arrows pan, drag orbit, wheel zoom
space pause, Q / Esc quit
"""

from __future__ import annotations

import asyncio
import json
import math
import queue
import string
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pygame

from motion import MotionPipeline
from segmenter import trim_lead


W, H = 1600, 900  # 16:9
HEADER_H = 0
FOOTER_H = 286
VIEW_H = H - HEADER_H - FOOTER_H
STEPS = ("TIP", "LOCK", "PROJECT", "TRIM", "TTA", "COMMIT")
TTA_SHOW = (-15.0, -8.0, 0.0, 8.0, 15.0)
PAD = 18
GAP = 12
TILE = 228

_PALETTES = {
    True: dict(
        BG=(12, 13, 16),
        HEADER_BG=(16, 18, 22),
        FOOTER_BG=(17, 19, 24),
        LINE=(36, 40, 48),
        TEXT=(214, 218, 226),
        DIM=(118, 126, 140),
        ACCENT=(23, 92, 211),
        ACCENT_SOFT=(90, 150, 255),
        INK=(156, 210, 236),
        INK_DIM=(48, 86, 108),
        GOLD=(223, 152, 44),
        GOLD_DIM=(92, 62, 22),
        PLANE=(70, 120, 210),
        RAY=(52, 78, 118),
        PAGE=(236, 238, 242),
        PAGE_INK=(28, 36, 48),
        GOOD=(150, 210, 160),
        GOOD_BRIGHT=(92, 230, 140),
        STEP_DONE=(160, 168, 180),
        STEP_IDLE=(72, 78, 88),
        STEP_ARROW=(58, 62, 70),
        GRID=(24, 26, 32),
        GRID_AXIS=(32, 36, 44),
        GHOST=(58, 52, 44),
        GHOST_DIM=(32, 28, 24),
        HINT=(70, 76, 86),
    ),
    False: dict(
        BG=(236, 238, 242),
        HEADER_BG=(248, 249, 252),
        FOOTER_BG=(244, 246, 249),
        LINE=(206, 210, 218),
        TEXT=(28, 32, 38),
        DIM=(90, 96, 108),
        ACCENT=(23, 92, 211),
        ACCENT_SOFT=(30, 80, 190),
        INK=(20, 90, 160),
        INK_DIM=(150, 186, 214),
        GOLD=(196, 120, 20),
        GOLD_DIM=(214, 176, 120),
        PLANE=(50, 100, 190),
        RAY=(130, 158, 196),
        PAGE=(255, 255, 255),
        PAGE_INK=(28, 36, 48),
        GOOD=(36, 140, 72),
        GOOD_BRIGHT=(16, 150, 64),
        STEP_DONE=(70, 76, 88),
        STEP_IDLE=(168, 174, 184),
        STEP_ARROW=(186, 190, 198),
        GRID=(214, 218, 224),
        GRID_AXIS=(186, 192, 202),
        GHOST=(186, 176, 158),
        GHOST_DIM=(210, 204, 190),
        HINT=(120, 126, 136),
    ),
}


def _apply_palette(dark: bool):
    g = globals()
    g.update(_PALETTES[dark])


_apply_palette(True)

FPS = 60
STROKE_N = 320
PLANE_HALF = 0.12
TIP_LEN = 0.16
CYCLE = 16.0
SWAY_T0 = 5.55
SWAY_T1 = 13.70
RNG_SEED = 7
T_RESTART_S = 0.20
INK_SPEED = 0.14
INK_HOLD = 0.08
INK_PRE_MS = 160
INK_PRE_SPEED = 0.035
LIVE_EVERY_S = 0.10
LIVE_COMMIT_MIN = 0.34
HOLD_LETTER_S = 0.40
PREFIX_HOLD_S = 0.55
PREFIX_LETTERS = "ILCJPT"
UNSURE_MARGIN = 0.05
AGREE_N = 3
COMMIT_RESET_S = 1.5
PULSE_DELAY_S = 0.3


def _mix(a, b, t: float):
    t = max(0.0, min(1.0, t))
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def _page_basis(forward: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Right/up of a page facing `forward`, kept upright against gravity."""
    f = forward / max(float(np.linalg.norm(forward)), 1e-9)
    world_up = np.array([0.0, 0.0, 1.0])
    up = world_up - f * float(world_up @ f)
    if np.linalg.norm(up) < 0.25:
        fallback = np.array([0.0, 1.0, 0.0])
        up = fallback - f * float(fallback @ f)
    up /= max(float(np.linalg.norm(up)), 1e-9)
    right = np.cross(up, f)
    right /= max(float(np.linalg.norm(right)), 1e-9)
    up = np.cross(f, right)
    up /= max(float(np.linalg.norm(up)), 1e-9)
    return right, up


def _rodrigues(v: np.ndarray, axis: np.ndarray, deg: float) -> np.ndarray:
    a = axis / max(float(np.linalg.norm(axis)), 1e-9)
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return v * c + np.cross(a, v) * s + a * float(a @ v) * (1.0 - c)


def _bezier(p0, p1, p2, p3, n: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)[:, None]
    o = 1.0 - t
    return o**3 * p0 + 3 * o**2 * t * p1 + 3 * o * t**2 * p2 + t**3 * p3


def _smooth(x: np.ndarray, k: int = 5) -> np.ndarray:
    k = max(1, k | 1)
    pad = k // 2
    ker = np.ones(k) / k
    if x.ndim == 1:
        return np.convolve(np.pad(x, (pad, pad), mode="edge"), ker, mode="valid")
    return np.stack([_smooth(x[:, i], k) for i in range(x.shape[1])], axis=1)


def make_letter_s() -> np.ndarray:
    """Handwritten S: tight top bowl, open bottom, short lead-in scratch."""
    lead = _bezier(
        np.array([0.050, 0.036]),
        np.array([0.048, 0.050]),
        np.array([0.042, 0.056]),
        np.array([0.034, 0.050]),
        36,
    )
    top = _bezier(
        np.array([0.034, 0.050]),
        np.array([-0.004, 0.070]),
        np.array([-0.048, 0.032]),
        np.array([0.008, 0.002]),
        (STROKE_N - 36) // 2,
    )
    bot = _bezier(
        np.array([0.008, 0.002]),
        np.array([0.062, -0.028]),
        np.array([0.012, -0.078]),
        np.array([-0.044, -0.048]),
        STROKE_N - 36 - (STROKE_N - 36) // 2,
    )
    return np.vstack([lead, top, bot])


def handwriting_index(frac: float, n: int) -> int:
    """Slow plant, rush the bowls, hesitate at the S waist, ease off."""
    t = np.linspace(0.0, 1.0, n)
    speed = 0.28 + 0.95 * np.sin(np.pi * t) ** 1.15
    speed[: max(1, int(0.10 * n))] *= 0.40
    speed[int(0.45 * n) : int(0.56 * n)] *= 0.32
    speed[int(0.88 * n) :] *= 0.48
    cum = np.cumsum(1.0 / np.clip(speed, 0.08, None))
    cum /= cum[-1]
    return int(np.clip(np.searchsorted(cum, frac), 1, n))


def stroke_world(page_xy: np.ndarray, origin: np.ndarray, forward: np.ndarray, rng: np.random.Generator):
    """
    Wrist writing is not a 2D glyph extruded on a sine.
    Tremor, a slow wander, and depth from the hand coming in and peeling off.
    """
    right, up = _page_basis(forward)
    n = len(page_xy)
    t = np.linspace(0.0, 1.0, n)
    jitter = _smooth(rng.normal(0.0, 0.0017, (n, 2)), 4)
    wander = np.stack(
        [
            0.008 * np.sin(2 * np.pi * 0.9 * t + 0.4),
            0.005 * (t - 0.15) * np.sin(np.pi * t) - 0.003 * t**2,
        ],
        axis=1,
    )
    xy = page_xy + jitter + wander
    walk = np.cumsum(rng.normal(0.0, 0.0009, n))
    walk -= walk[0]
    depth = (
        0.007 * np.exp(-(((t - 0.07) / 0.055) ** 2))
        + 0.032 * np.sin(np.pi * t) * (0.55 + 0.45 * np.sin(2.6 * np.pi * t + 0.5))
        + 0.35 * walk
        - 0.012 * np.clip((t - 0.78) / 0.22, 0.0, 1.0)
    )
    depth = _smooth(depth, 5)
    tips = origin + xy[:, 0:1] * right + xy[:, 1:2] * up + depth[:, None] * forward
    wrist0 = origin - TIP_LEN * forward
    wrist = wrist0 + 0.005 * np.stack(
        [
            np.sin(2 * np.pi * 2.1 * t + 0.2),
            np.cos(2 * np.pi * 1.6 * t),
            0.45 * np.sin(2 * np.pi * 1.2 * t + 1.1),
        ],
        axis=1,
    )
    pointing = tips - wrist
    pointing /= np.clip(np.linalg.norm(pointing, axis=1, keepdims=True), 1e-9, None)
    return tips, pointing


CAM_FILE = Path(__file__).resolve().parent / "camera_view.json"


class Camera:
    def __init__(self):
        self.center = np.array([0.0, 0.02, 0.12])
        self.dist = 0.54
        self.elev = 18.0
        self.azim = -48.0
        self.fov = 540.0
        self._eye = None
        self._anchor = None

    def aim_from_hinge(self, origin: np.ndarray, forward: np.ndarray):
        """Face the page: slightly above and in front, whole plane in frame."""
        f = np.asarray(forward, dtype=np.float64)
        n = float(np.linalg.norm(f))
        if n < 1e-9:
            return
        f = f / n
        right, up = _page_basis(f)
        o = np.asarray(origin, dtype=np.float64)
        self.center = o.copy()
        self._eye = o + 0.26 * f + 0.12 * up + 0.05 * right
        self.fov = 540.0
        self._anchor = o.copy()
        self._adopt_eye()

    def _adopt_eye(self):
        if self._eye is None:
            return
        d = self._eye - self.center
        self.dist = max(float(np.linalg.norm(d)), 0.08)
        se = float(d[2] / self.dist)
        self.elev = math.degrees(math.asin(max(-0.99, min(0.99, se))))
        ce = math.cos(math.radians(self.elev))
        if abs(ce) < 1e-6:
            self.azim = 0.0
        else:
            self.azim = math.degrees(
                math.atan2(d[0] / (self.dist * ce), -d[1] / (self.dist * ce))
            )
        self._eye = None

    def slide_to(self, origin: np.ndarray):
        o = np.asarray(origin, dtype=np.float64)
        if self._anchor is None:
            self._anchor = o.copy()
            return
        self.center = self.center + (o - self._anchor)
        self._anchor = o.copy()

    def orbit(self, dazim: float, delev: float):
        self.azim += dazim
        self.elev = max(-85.0, min(85.0, self.elev + delev))

    def pan(self, sx: float, sy: float):
        look = self.center - self.eye()
        look = look / max(float(np.linalg.norm(look)), 1e-9)
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(look, world_up)
        rn = float(np.linalg.norm(right))
        if rn < 1e-6:
            right = np.array([1.0, 0.0, 0.0])
        else:
            right = right / rn
        up = np.cross(right, look)
        self.center = self.center + right * sx + up * sy

    def zoom(self, factor: float):
        self.dist = max(0.08, min(2.4, self.dist * factor))

    def save(self):
        CAM_FILE.write_text(
            json.dumps(
                {
                    "center": [float(x) for x in self.center],
                    "azim": float(self.azim),
                    "elev": float(self.elev),
                    "dist": float(self.dist),
                    "fov": float(self.fov),
                    "anchor": None if self._anchor is None else [float(x) for x in self._anchor],
                },
                indent=2,
            )
        )

    def load(self) -> bool:
        if not CAM_FILE.exists():
            return False
        try:
            data = json.loads(CAM_FILE.read_text())
            self.center = np.array(data["center"], dtype=np.float64)
            self.azim = float(data["azim"])
            self.elev = float(data["elev"])
            self.dist = float(data["dist"])
            self.fov = float(data.get("fov", 540.0))
            a = data.get("anchor")
            self._anchor = None if a is None else np.array(a, dtype=np.float64)
            self._eye = None
            return True
        except (OSError, KeyError, TypeError, ValueError):
            return False

    def eye(self) -> np.ndarray:
        if self._eye is not None:
            return self._eye
        az = math.radians(self.azim)
        el = math.radians(self.elev)
        c, s = math.cos(az), math.sin(az)
        ce, se = math.cos(el), math.sin(el)
        return self.center + np.array(
            [self.dist * s * ce, -self.dist * c * ce, self.dist * se]
        )

    def project(self, p: np.ndarray, rect: pygame.Rect):
        eye = self.eye()
        forward = self.center - eye
        forward = forward / max(float(np.linalg.norm(forward)), 1e-9)
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, world_up)
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1.0, 0.0, 0.0])
        right /= float(np.linalg.norm(right))
        up = np.cross(right, forward)
        rel = p - eye
        x = float(rel @ right)
        y = float(rel @ up)
        z = float(rel @ forward)
        if z < 0.04:
            return None
        sx = rect.centerx + (x / z) * self.fov
        sy = rect.centery - (y / z) * self.fov
        return sx, sy, z


def _basis(forward: np.ndarray, roll: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    right, up = _page_basis(forward)
    if abs(roll) > 1e-6:
        right = _rodrigues(right, forward, roll)
        up = _rodrigues(up, forward, roll)
    return right, up


def project_to_plane(pts: np.ndarray, origin: np.ndarray, forward: np.ndarray, roll: float = 0.0):
    right, up = _basis(forward, roll)
    delta = pts - origin
    u = delta @ right
    v = delta @ up
    on_plane = origin + u[:, None] * right + v[:, None] * up
    return np.stack([u, v], axis=1), on_plane


def _mirror_page_h(pts: np.ndarray, origin: np.ndarray, forward: np.ndarray, roll: float = 0.0) -> np.ndarray:
    """Reflect across the page's vertical axis (flip left/right on the plane)."""
    right, _ = _basis(forward, roll)
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim == 1:
        return pts - 2.0 * float(pts @ right) * right
    u = (pts - origin) @ right
    return pts - 2.0 * u[:, None] * right


class Demo:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Writing plane")
        self.screen = pygame.display.set_mode((W, H))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("helvetica", 15)
        self.font_sm = pygame.font.SysFont("helvetica", 13)
        self.font_title = pygame.font.SysFont("helvetica", 26, bold=True)
        self.font_huge = pygame.font.SysFont("helvetica", 64, bold=True)
        self.font_mono = pygame.font.SysFont("menlo", 13)

        self.origin = np.array([0.0, 0.04, 0.125])
        self.locked_forward = np.array([0.16, -0.78, 0.60])
        self.locked_forward /= float(np.linalg.norm(self.locked_forward))
        rng = np.random.default_rng(RNG_SEED)
        self.world, self.pointing = stroke_world(
            make_letter_s(), self.origin, self.locked_forward, rng
        )
        self.page_full, _ = project_to_plane(self.world, self.origin, self.locked_forward)
        self.classifier = None
        self.pred_track = []
        self._load_classifier()
        self._shot = "--shot" in sys.argv

        self.cam = Camera()
        if not self.cam.load():
            self.cam.aim_from_hinge(self.origin, self.locked_forward)
            self.cam.save()
        elif self.cam._anchor is None:
            self.cam._anchor = self.origin.copy()
        self._drag = None
        self._arrow_dirty = False
        self.t = 0.0
        self.paused = False
        self.manual_yaw = 0.0
        self.running = True
        self.dark = True
        self._cam_hold = None

        self.mode = "demo"
        self.live_status = "T · live IMU"
        self.sample_q: queue.Queue = queue.Queue(maxsize=512)
        self.pipeline = MotionPipeline()
        self.live_world: list[np.ndarray] = []
        self.live_page: list[tuple[float, float, int]] = []
        self.live_pointing: list[np.ndarray] = []
        self.live_ghost: list[tuple[np.ndarray, int]] = []
        self.live_pred = None
        self.live_forward = self.locked_forward.copy()
        self._last_t_at = 0.0
        self._live_t0 = 0.0
        self._ble_started = False
        self._inking = False
        self._pre: list[tuple] = []
        self._live_lock = threading.Lock()
        self._live_req = None
        self._last_live_at = 0.0
        self._still_since = None
        self._pred_gen = 0
        self._still_gen = 0
        self._hold_letter = None
        self._hold_since = None
        self._hold_gen = 0
        self._agree_n = 0
        self._live_committed = False
        self._commit_at = None
        self._aim_pending = False
        self._classify_started = False
        self._tilt = (0.0, 0.0, 0.0)
        self._bar_show: dict[str, float] = {}
        self._start_classify()

    def _load_classifier(self):
        try:
            from classifier import LetterClassifier

            self.classifier = LetterClassifier()
        except Exception as exc:
            print("classifier unavailable:", exc)
            self.classifier = None

    def _start_classify(self):
        if self._classify_started:
            return
        self._classify_started = True

        def classify_thread():
            while self.running:
                with self._live_lock:
                    req, self._live_req = self._live_req, None
                if req is None or self.classifier is None:
                    time.sleep(0.02)
                    continue
                try:
                    self.live_pred = self.classifier.predict_stroke(req)
                    self._pred_gen += 1
                except Exception:
                    pass

        threading.Thread(target=classify_thread, daemon=True).start()

    def _start_ble(self):
        if self._ble_started:
            return
        self._ble_started = True
        self._start_classify()

        def ble_thread():
            from ble_client import GestureBandClient

            async def main():
                def on_sample(sample):
                    try:
                        self.sample_q.put_nowait(sample)
                    except queue.Full:
                        pass

                client = GestureBandClient(on_sample=on_sample)
                while self.running:
                    try:
                        self.live_status = "Scanning for band…"
                        await client.connect()
                        self.live_status = "Live IMU — write a letter"
                        await client.run_forever()
                    except Exception as exc:
                        self.live_status = f"BLE: {exc} · retrying…"
                        await asyncio.sleep(2.0)

            asyncio.run(main())

        threading.Thread(target=ble_thread, daemon=True).start()

    def _start_live(self):
        now = time.time()
        if now - self._last_t_at < T_RESTART_S:
            return
        self._last_t_at = now
        self.mode = "live"
        self.paused = False
        self._start_ble()
        self._reset_live_stroke()
        self.live_status = "Page locked to this wrist pose. Write — the tip will not stay on the plane."

    def _reset_live_stroke(self):
        while True:
            try:
                self.sample_q.get_nowait()
            except queue.Empty:
                break
        self.live_world.clear()
        self.live_page.clear()
        self.live_pointing.clear()
        self.live_ghost.clear()
        self.live_pred = None
        self._inking = False
        self._pre = []
        self._cam_hold = None
        self._still_since = None
        self._still_gen = self._pred_gen
        self._hold_letter = None
        self._hold_since = None
        self._hold_gen = 0
        self._agree_n = 0
        self._live_committed = False
        self._commit_at = None
        self._live_t0 = time.time()
        with self._live_lock:
            self._live_req = None
        self.pipeline.lock_page()
        if self.pipeline._origin is not None:
            self.origin = self.pipeline._origin.copy()
            self.live_forward = self.pipeline._forward.copy()
            self.cam.slide_to(self.origin)
            self._aim_pending = False
        else:
            self._aim_pending = True

    def _back_to_demo(self):
        self.mode = "demo"
        self.t = 0.0
        self.manual_yaw = 0.0
        self._cam_hold = None
        self.origin = np.array([0.0, 0.04, 0.125])
        self.cam.slide_to(self.origin)

    def _drain_live(self):
        for _ in range(64):
            try:
                sample = self.sample_q.get_nowait()
            except queue.Empty:
                break
            pose = self.pipeline.update(sample)
            tip = self.pipeline._tip.copy()
            pointing = self.pipeline._pointing.copy()
            self.live_ghost.append((tip, pose.t_ms))
            ghost_cut = pose.t_ms - 2500
            self.live_ghost = [g for g in self.live_ghost if g[1] >= ghost_cut]
            # No stroke yet and still too slow to ink: keep relocking the page
            # to the current wrist, same as pressing T.
            if not self.live_page and not self._inking and pose.speed < INK_PRE_SPEED:
                self.pipeline.lock_page()
                self._pre = []
                if self.pipeline._origin is not None:
                    self.origin = self.pipeline._origin.copy()
                    self.live_forward = self.pipeline._forward.copy()
                    if self._aim_pending:
                        self.cam.slide_to(self.origin)
                        self._aim_pending = False
                continue
            self._pre.append((pose.x, pose.y, pose.t_ms, pose.speed, tip, pointing))
            cut = pose.t_ms - INK_PRE_MS
            self._pre = [p for p in self._pre if p[2] >= cut]
            was = self._inking
            self._inking = pose.speed >= (INK_HOLD if self._inking else INK_SPEED)
            if self._inking:
                if not was:
                    for x, y, t_ms, spd, tp, pn in self._pre:
                        if spd >= INK_PRE_SPEED:
                            self.live_page.append((x, y, t_ms))
                            self.live_world.append(tp)
                            self.live_pointing.append(pn)
                self.live_page.append((pose.x, pose.y, pose.t_ms))
                self.live_world.append(tip)
                self.live_pointing.append(pointing)
                self._still_since = None
                self._still_gen = self._pred_gen
                self._hold_letter = None
                self._hold_since = None
                self._hold_gen = 0
                self._agree_n = 0
                self._live_committed = False
                self._commit_at = None
            elif self.live_page and pose.speed < INK_HOLD:
                if self._still_since is None:
                    self._still_since = time.time()
                    self._still_gen = self._pred_gen
                    self._hold_letter = None
                    self._hold_since = None
                    self._hold_gen = 0
                    self._agree_n = 0
                    self._last_live_at = 0.0
        now = time.time()
        if self.classifier and len(self.live_page) >= 16 and now - self._last_live_at >= LIVE_EVERY_S:
            self._last_live_at = now
            with self._live_lock:
                self._live_req = list(self.live_page)
        pred = self.live_pred
        unsure = (
            pred is None
            or pred.confidence < LIVE_COMMIT_MIN
            or (
                len(pred.top3) > 1
                and pred.top3[0][1] - pred.top3[1][1] < UNSURE_MARGIN
            )
        )
        if (
            self.live_page
            and self._still_since is not None
            and not self._live_committed
            and pred is not None
            and self._pred_gen > self._still_gen
            and not unsure
        ):
            if pred.letter != self._hold_letter:
                self._hold_letter = pred.letter
                self._hold_since = now
                self._hold_gen = self._pred_gen
                self._agree_n = 1
            elif self._pred_gen != self._hold_gen:
                self._agree_n += 1
                self._hold_gen = self._pred_gen
            need = PREFIX_HOLD_S if pred.letter in PREFIX_LETTERS else HOLD_LETTER_S
            if self._agree_n >= AGREE_N and now - self._hold_since >= need:
                self._live_committed = True
                if self._commit_at is None:
                    self._commit_at = now
        elif self._still_since is not None and not self._live_committed and unsure:
            self._hold_letter = None
            self._hold_since = None
            self._hold_gen = 0
            self._agree_n = 0

    def phase(self, t: float):
        """draw_frac, ray_a, step, do_trim, show_tta, caption."""
        cycle = t % CYCLE
        if cycle < 0.35:
            return 0.0, 0.0, 0, False, False, "Orientation only. 16 cm virtual tip — no accel integration, no ZUPT."
        if cycle < 3.70:
            u = (cycle - 0.35) / 3.35
            return u, 0.15, 0, False, False, "Tip from wrist attitude. Page speed gates ink (hysteresis, not ZUPT)."
        if cycle < 4.40:
            return 1.0, 0.35, 1, False, False, "Page locked at the first pose. Origin and forward never move."
        if cycle < 5.55:
            u = (cycle - 4.40) / 1.15
            return 1.0, u * u * (3 - 2 * u), 2, False, False, "Project: (tip − origin) · right, up. Gravity-upright basis."
        if cycle < SWAY_T1:
            return 1.0, 1.0, 4, True, True, "Search the page — then lock."
        return 1.0, 1.0, 5, True, True, "Commit when the same letter holds. The plane stays locked."

    def _sway_angles(self, cycle: float) -> tuple[float, float, float, bool]:
        """Yaw/pitch/roll (deg) and whether the plane has settled on the true lock."""
        if cycle < SWAY_T0:
            return 0.0, 0.0, 0.0, False
        if cycle >= SWAY_T1:
            return 0.0, 0.0, 0.0, True
        u = (cycle - SWAY_T0) / (SWAY_T1 - SWAY_T0)
        fade_in = min(1.0, u / 0.08)
        if u < 0.68:
            env = fade_in
        else:
            v = (u - 0.68) / 0.32
            env = fade_in * (1.0 - v * v * (3.0 - 2.0 * v))
        yaw = env * 62.0 * math.sin(0.92 * cycle + 0.2)
        pitch = env * 48.0 * math.sin(0.64 * cycle + 1.1)
        roll = env * 72.0 * math.sin(1.08 * cycle + 0.5)
        return yaw, pitch, roll, env < 0.02

    def plane_forward(self, yaw: float, pitch: float, roll: float) -> np.ndarray:
        right, up = _page_basis(self.locked_forward)
        f = _rodrigues(self.locked_forward, up, yaw + self.manual_yaw)
        f = _rodrigues(f, right, pitch)
        return f / max(float(np.linalg.norm(f)), 1e-9)

    def draw(self):
        if self.mode == "live":
            self._draw_live()
            return
        draw_frac, ray_a, step, do_trim, show_tta, caption = self.phase(self.t)
        cycle = self.t % CYCLE
        yaw, pitch, roll, settled = self._sway_angles(cycle)
        self._tilt = (yaw, pitch, roll)
        vis_forward = self.plane_forward(yaw, pitch, roll)
        if draw_frac <= 0.0:
            n = 1
        elif draw_frac >= 1.0:
            n = len(self.world)
        else:
            n = handwriting_index(draw_frac, len(self.world))
        pts = self.world[:n]
        page_xy, on_plane = project_to_plane(pts, self.origin, vis_forward, roll)
        pts = _mirror_page_h(pts, self.origin, vis_forward, roll)
        on_plane = _mirror_page_h(on_plane, self.origin, vis_forward, roll)
        shaft = self.pointing[min(n, len(self.pointing)) - 1]
        shaft = _mirror_page_h(shaft, np.zeros(3), vis_forward, roll)
        raw_pts = [(float(x), float(y), i) for i, (x, y) in enumerate(page_xy)]
        if do_trim and settled and len(raw_pts) >= 8:
            trimmed = trim_lead(raw_pts)
            page_xy = np.array([(p[0], p[1]) for p in trimmed], dtype=np.float64)
        if self.classifier and len(page_xy) >= 16:
            now = time.time()
            if self._shot or now - self._last_live_at >= LIVE_EVERY_S:
                self._last_live_at = now
                req = [(float(x), float(y), i) for i, (x, y) in enumerate(page_xy)]
                if self._shot:
                    try:
                        self.live_pred = self.classifier.predict_stroke(req)
                    except Exception:
                        pass
                else:
                    with self._live_lock:
                        self._live_req = req
        pred = self.live_pred if n >= 24 else None
        ready = settled and pred is not None and pred.confidence >= LIVE_COMMIT_MIN
        committed = ready and cycle >= SWAY_T1 + PULSE_DELAY_S
        if committed:
            caption = "Commit — same letter held. Plane still locked."
        if committed and self._shot:
            pulse = 1.0
        elif committed:
            pulse = 0.55 + 0.45 * math.sin(self.t * math.tau * 1.15)
        else:
            pulse = 0.0

        self.screen.fill(BG)
        view = pygame.Rect(0, HEADER_H, W, VIEW_H)
        pygame.draw.rect(self.screen, BG, view)
        self._scene(view, pts, on_plane[:n], vis_forward, roll, ray_a, n, pulse, shaft)
        self._footer(page_xy, pred, committed, pulse, False)
        pygame.display.flip()

    def _draw_live(self):
        self._drain_live()
        if (
            self._live_committed
            and self._commit_at is not None
            and time.time() - self._commit_at >= COMMIT_RESET_S + PULSE_DELAY_S
        ):
            self._start_live()
        if self.pipeline._origin is not None:
            self.origin = self.pipeline._origin.copy()
            self.live_forward = self.pipeline._forward.copy()
        if self.live_world:
            pts = np.stack(self.live_world)
            _, on_plane = project_to_plane(pts, self.origin, self.live_forward)
            page_xy = np.array([(p[0], p[1]) for p in self.live_page], dtype=np.float64)
            n = len(pts)
            pointing = self.live_pointing[-1]
        else:
            tip = self.pipeline._tip
            pts = tip.reshape(1, 3)
            page_xy = np.zeros((1, 2))
            on_plane = pts
            n = 1
            pointing = self.pipeline._pointing
        pred = self.live_pred
        age = time.time() - (self._commit_at or time.time()) - PULSE_DELAY_S
        committed = self._live_committed and pred is not None and age >= 0.0
        if committed:
            caption = f"Hold still · committed  {pred.letter}"
            step = 5
        elif n > 8:
            caption = "3D tip is off-plane (band/wrist variance). Drop to the locked page → 28×28."
            step = 2
        elif not self.live_page:
            caption = "Aiming — page follows the wrist. A stroke locks it."
            step = 1
        else:
            caption = self.live_status
            step = 1
        pulse = 0.0
        if committed:
            if self._shot:
                pulse = 1.0
            elif age < 0.16:
                pulse = 1.0
            else:
                pulse = 0.50 + 0.50 * math.sin((age - 0.16) * math.tau * 2.2)
        ray_a = 1.0 if n > 4 else 0.35
        self.screen.fill(BG)
        view = pygame.Rect(0, HEADER_H, W, VIEW_H)
        pygame.draw.rect(self.screen, BG, view)
        ghost = np.stack([g[0] for g in self.live_ghost]) if len(self.live_ghost) > 1 else None
        self._scene(
            view, pts, on_plane[:n], self.live_forward, 0.0, ray_a, n, pulse, pointing, ghost,
        )
        self._footer(page_xy[:n] if n else page_xy, pred, committed, pulse, False)
        pygame.display.flip()

    def _scene(self, rect, pts, on_plane, forward, roll, ray_a, n, pulse: float = 0.0, pointing=None, ghost=None):
        clip = self.screen.subsurface(rect)
        self._grid_floor(clip, rect)
        self._plane(clip, rect, forward, roll, 0.35 + 0.45 * ray_a, pulse)
        if ghost is not None and len(ghost) > 1:
            self._polyline(clip, rect, ghost, GHOST, GHOST_DIM, 1)
        if ray_a > 0.02 and len(pts) > 1:
            self._rays(clip, rect, pts, on_plane, ray_a)
        if len(on_plane) > 1 and ray_a > 0.05:
            self._polyline(clip, rect, on_plane, INK, INK_DIM, 2)
        if len(pts) > 1:
            self._polyline(clip, rect, pts, GOLD, GOLD_DIM, 3)
        if len(pts):
            shaft = pointing if pointing is not None else self.pointing[min(n, len(self.pointing)) - 1]
            self._pen(clip, rect, pts[-1], shaft)
        self._axis_gizmo(clip, rect)

    def _grid_floor(self, surf, rect):
        z = 0.0
        for i in range(-4, 5):
            a = np.array([i * 0.06, -0.22, z])
            b = np.array([i * 0.06, 0.28, z])
            c = np.array([-0.24, i * 0.06, z])
            d = np.array([0.24, i * 0.06, z])
            col = GRID if i != 0 else GRID_AXIS
            self._line3(surf, rect, a, b, col, 1)
            self._line3(surf, rect, c, d, col, 1)

    def _plane(self, surf, rect, forward, roll, alpha: float, pulse: float = 0.0):
        right, up = _basis(forward, roll)
        o = self.origin
        h = PLANE_HALF
        corners = [
            o - h * right - h * up,
            o + h * right - h * up,
            o + h * right + h * up,
            o - h * right + h * up,
        ]
        fill = _mix(ACCENT, (36, 160, 72), pulse)
        edge = _mix(PLANE, GOOD_BRIGHT, pulse)
        normal = _mix(ACCENT_SOFT, GOOD_BRIGHT, pulse)
        proj = [self.cam.project(p, rect) for p in corners]
        if all(proj):
            poly = [(p[0] - rect.x, p[1] - rect.y) for p in proj]
            overlay = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
            pygame.draw.polygon(overlay, (*fill, int((46 + 90 * pulse) * alpha)), poly)
            pygame.draw.polygon(overlay, (*edge, int((200 + 50 * pulse) * alpha)), poly, 2 + int(2 * pulse))
            surf.blit(overlay, (0, 0))
        steps = 6
        col = _mix(
            (int(40 + 50 * alpha), int(70 + 40 * alpha), int(130 + 50 * alpha)),
            (int(40 + 40 * alpha), int(160 + 50 * alpha), int(80 + 40 * alpha)),
            pulse,
        )
        for i in range(-steps, steps + 1):
            f = i / steps
            a = o + f * h * right - h * up
            b = o + f * h * right + h * up
            c = o - h * right + f * h * up
            d = o + h * right + f * h * up
            self._line3(surf, rect, a, b, col, 1)
            self._line3(surf, rect, c, d, col, 1)
        tip = o + 0.07 * forward
        self._line3(surf, rect, o, tip, normal, 2)
        p = self.cam.project(tip, rect)
        if p:
            pygame.draw.circle(surf, normal, (int(p[0] - rect.x), int(p[1] - rect.y)), 4)

    def _rays(self, surf, rect, pts, on_plane, a: float):
        step = max(1, len(pts) // 36)
        col = (
            int(90 * a + BG[0] * (1 - a)),
            int(150 * a + BG[1] * (1 - a)),
            int(230 * a + BG[2] * (1 - a)),
        )
        for i in range(0, len(pts), step):
            self._line3(surf, rect, pts[i], on_plane[i], col, 1)

    def _pen(self, surf, rect, tip, forward):
        wrist = tip - TIP_LEN * forward
        self._line3(surf, rect, wrist, tip, GOLD, 3)
        p = self.cam.project(tip, rect)
        w = self.cam.project(wrist, rect)
        if p:
            pygame.draw.circle(surf, GOLD, (int(p[0] - rect.x), int(p[1] - rect.y)), 6)
            pygame.draw.circle(surf, (255, 230, 180), (int(p[0] - rect.x), int(p[1] - rect.y)), 3)
        if w:
            pygame.draw.circle(surf, (90, 96, 108), (int(w[0] - rect.x), int(w[1] - rect.y)), 5)

    def _axis_gizmo(self, surf, rect):
        o = np.array([-0.16, 0.16, 0.01])
        axes = [
            (np.array([0.05, 0, 0]), (180, 80, 80), "x"),
            (np.array([0, 0.05, 0]), (80, 170, 100), "y"),
            (np.array([0, 0, 0.05]), (80, 130, 220), "z"),
        ]
        for vec, col, _ in axes:
            self._line3(surf, rect, o, o + vec, col, 2)

    def _polyline(self, surf, rect, pts, fg, glow, width: int):
        proj = [self.cam.project(p, rect) for p in pts]
        segs = []
        for a, b in zip(proj, proj[1:]):
            if a and b:
                segs.append((a, b, 0.5 * (a[2] + b[2])))
        segs.sort(key=lambda s: -s[2])
        for a, b, _ in segs:
            p1 = (int(a[0] - rect.x), int(a[1] - rect.y))
            p2 = (int(b[0] - rect.x), int(b[1] - rect.y))
            pygame.draw.line(surf, glow, p1, p2, width + 3)
            pygame.draw.line(surf, fg, p1, p2, width)

    def _line3(self, surf, rect, a, b, col, width: int):
        pa, pb = self.cam.project(a, rect), self.cam.project(b, rect)
        if not (pa and pb):
            return
        pygame.draw.line(
            surf,
            col,
            (int(pa[0] - rect.x), int(pa[1] - rect.y)),
            (int(pb[0] - rect.x), int(pb[1] - rect.y)),
            width,
        )

    def _footer(self, page_xy: np.ndarray, pred, committed: bool, pulse: float, show_tta: bool = False):
        y0 = HEADER_H + VIEW_H
        pygame.draw.rect(self.screen, FOOTER_BG, (0, y0, W, FOOTER_H))
        pygame.draw.line(self.screen, LINE, (0, y0), (W, y0))

        top = y0 + 32
        page = pygame.Rect(PAD, top, TILE, TILE)
        raster = pygame.Rect(PAD + TILE + GAP, top, TILE, TILE)
        meta = pygame.Rect(raster.right + GAP, y0 + 10, W - raster.right - GAP - PAD, FOOTER_H - 20)

        self.screen.blit(self.font_sm.render("PAGE", True, DIM), (page.x, y0 + 10))
        in_col = _mix(DIM, GOOD_BRIGHT, pulse) if committed else DIM
        self.screen.blit(
            self.font_sm.render("TTA  ·  ±15°" if show_tta else "MODEL  ·  28×28", True, in_col),
            (raster.x, y0 + 10),
        )

        pygame.draw.rect(self.screen, PAGE, page, border_radius=6)
        pygame.draw.rect(self.screen, LINE, page, 1, border_radius=6)
        self._draw_page_stroke(page, page_xy, PAGE_INK)

        if show_tta and len(page_xy) >= 2:
            self._draw_tta(raster, page_xy, pulse)
        else:
            tile = self._raster_surface(page_xy, TILE, pulse)
            self.screen.blit(tile, raster.topleft)
        border = _mix(LINE, GOOD_BRIGHT, pulse) if committed else LINE
        pygame.draw.rect(self.screen, border, raster.inflate(int(3 * pulse), int(3 * pulse)), 2 + int(2 * pulse))

        status = "COMMIT" if committed else "LIVE"
        scol = _mix(GOOD, GOOD_BRIGHT, pulse) if committed else ACCENT_SOFT
        self.screen.blit(self.font_sm.render(status, True, scol), (meta.x, meta.y + 2))
        if pred is None:
            self.screen.blit(self.font_title.render("—", True, DIM), (meta.x, meta.y + 22))
        else:
            lcol = _mix(TEXT, GOOD_BRIGHT, pulse) if committed else TEXT
            letter = self.font_huge.render(pred.letter, True, lcol)
            self.screen.blit(letter, (meta.x, meta.y + 14))
            self.screen.blit(
                self.font_title.render(f"{pred.confidence * 100:.0f}%", True, scol),
                (meta.x + letter.get_width() + 12, meta.y + 32),
            )
        bar_top = meta.y + 78
        bars = pygame.Rect(meta.x, bar_top, meta.w, page.bottom - bar_top)
        self._draw_pred_bars(bars, pred, committed, pulse)

    def _pred_rows(self, pred) -> list[tuple[str, float]]:
        if pred is None:
            return []
        probs = getattr(pred, "probs", None)
        if probs is not None:
            order = np.argsort(-np.asarray(probs))
            letters = string.ascii_uppercase
            return [(letters[int(i)], float(probs[int(i)])) for i in order[:8]]
        return list(pred.top3)

    def _draw_pred_bars(self, rect: pygame.Rect, pred, committed: bool, pulse: float):
        rows = self._pred_rows(pred)
        target = {L: p for L, p in rows}
        for L in list(self._bar_show):
            if L not in target:
                self._bar_show[L] *= 0.82
                if self._bar_show[L] < 0.008:
                    del self._bar_show[L]
        for L, p in target.items():
            cur = self._bar_show.get(L, 0.0)
            self._bar_show[L] = cur + (p - cur) * 0.28
        if not rows:
            self.screen.blit(self.font_sm.render("waiting for a stroke", True, DIM), (rect.x, rect.y + 8))
            return
        n = min(8, len(rows))
        row_h = rect.h / n
        track_x = rect.x + 22
        track_w = rect.w - 78
        for i, (L, _) in enumerate(rows[:n]):
            p = self._bar_show.get(L, 0.0)
            y = int(rect.y + (i + 1) * row_h - 16)
            top = i == 0
            lcol = _mix(TEXT, GOOD_BRIGHT, pulse) if committed and top else (TEXT if top else DIM)
            self.screen.blit(self.font_sm.render(L, True, lcol), (rect.x, y + 2))
            pygame.draw.rect(self.screen, LINE, (track_x, y + 6, track_w, 10), border_radius=3)
            fill_w = max(0, int(track_w * min(1.0, p)))
            if fill_w > 0:
                fill = _mix(ACCENT_SOFT, GOOD_BRIGHT, pulse) if committed and top else ACCENT_SOFT
                if not top:
                    fill = _mix(LINE, ACCENT_SOFT, 0.55)
                pygame.draw.rect(self.screen, fill, (track_x, y + 6, fill_w, 10), border_radius=3)
            pct = self.font_mono.render(f"{p * 100:4.0f}%", True, lcol if top else DIM)
            self.screen.blit(pct, (track_x + track_w + 8, y + 1))

    def _draw_tta(self, rect: pygame.Rect, xy: np.ndarray, pulse: float):
        pygame.draw.rect(self.screen, BG, rect)
        cell = (rect.w - 16) // 3
        angles = TTA_SHOW
        for i, ang in enumerate(angles):
            r = math.radians(ang)
            c, s = math.cos(r), math.sin(r)
            rot = np.stack([xy[:, 0] * c - xy[:, 1] * s, xy[:, 0] * s + xy[:, 1] * c], axis=1)
            tile = self._raster_surface(rot, cell, pulse * 0.4)
            col = i % 3
            row = 0 if i < 3 else 1
            ox = rect.x + 8 + col * (cell + 4)
            oy = rect.y + 8 + row * (cell + 4)
            if row == 1:
                ox = rect.x + 8 + (i - 3) * (cell + 4) + cell // 2
            self.screen.blit(tile, (ox, oy))

    def _draw_page_stroke(self, rect: pygame.Rect, xy: np.ndarray, ink=PAGE_INK):
        if len(xy) < 2:
            return
        pad = 18
        xs, ys = xy[:, 0], xy[:, 1]
        minx, maxx = float(xs.min()), float(xs.max())
        miny, maxy = float(ys.min()), float(ys.max())
        cx, cy = 0.5 * (minx + maxx), 0.5 * (miny + maxy)
        half_w = max(0.5 * (maxx - minx), 1e-4)
        half_h = max(0.5 * (maxy - miny), 1e-4)
        inner_w = max(rect.w - 2 * pad, 1)
        inner_h = max(rect.h - 2 * pad, 1)
        scale = min(inner_w / (2.0 * half_w), inner_h / (2.0 * half_h))

        def to(p):
            return (
                rect.centerx + (p[0] - cx) * scale,
                rect.centery - (p[1] - cy) * scale,
            )

        pts = [to(p) for p in xy]
        prev = self.screen.get_clip()
        self.screen.set_clip(rect.inflate(-2, -2))
        if len(pts) > 1:
            pygame.draw.lines(self.screen, (210, 216, 224), False, pts, 7)
            pygame.draw.lines(self.screen, ink, False, pts, 3)
        if ink == PAGE_INK:
            pygame.draw.circle(self.screen, ACCENT, (int(pts[-1][0]), int(pts[-1][1])), 5)
        self.screen.set_clip(prev)

    def _raster_surface(self, xy: np.ndarray, px: int, pulse: float = 0.0) -> pygame.Surface:
        hi = 28
        surf = pygame.Surface((hi, hi))
        bg = (
            int(10 + 8 * pulse),
            int(11 + 36 * pulse),
            int(14 + 12 * pulse),
        )
        ink = (
            int(230 - 80 * pulse),
            int(230 + 10 * pulse),
            int(212 - 90 * pulse),
        )
        surf.fill(bg)
        if len(xy) >= 2:
            xs, ys = xy[:, 0], xy[:, 1]
            span = max(float(xs.max() - xs.min()), float(ys.max() - ys.min()), 1e-4)
            cx, cy = float(xs.mean()), float(ys.mean())
            pad = 0.18 * hi
            scale = (hi - 2 * pad) / span
            pts = [
                (hi * 0.5 + (p[0] - cx) * scale, hi * 0.5 - (p[1] - cy) * scale)
                for p in xy
            ]
            if len(pts) > 1:
                pygame.draw.lines(surf, ink, False, pts, 2)
        return pygame.transform.scale(surf, (px, px))

    def handle(self, e):
        if e.type == pygame.QUIT:
            self.running = False
        elif e.type == pygame.KEYDOWN:
            if e.key in (pygame.K_ESCAPE, pygame.K_q):
                self.running = False
            elif e.key == pygame.K_SPACE:
                self.paused = not self.paused
            elif e.key == pygame.K_r:
                self._back_to_demo()
            elif e.key == pygame.K_t:
                self._start_live()
            elif e.key == pygame.K_d:
                self.dark = not self.dark
                _apply_palette(self.dark)
            elif e.key == pygame.K_LEFTBRACKET:
                self.manual_yaw -= 4.0
            elif e.key == pygame.K_RIGHTBRACKET:
                self.manual_yaw += 4.0
        elif e.type == pygame.KEYUP and e.key in (
            pygame.K_LEFT,
            pygame.K_RIGHT,
            pygame.K_UP,
            pygame.K_DOWN,
        ):
            if self._arrow_dirty:
                self.cam.save()
                self._arrow_dirty = False
        elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            if HEADER_H <= e.pos[1] < HEADER_H + VIEW_H:
                self._drag = e.pos
        elif e.type == pygame.MOUSEBUTTONUP and e.button == 1:
            if self._drag is not None:
                self._drag = None
                self.cam.save()
        elif e.type == pygame.MOUSEMOTION and self._drag is not None:
            x, y = e.pos
            self.cam.orbit((x - self._drag[0]) * 0.35, (self._drag[1] - y) * 0.28)
            self._drag = (x, y)
        elif e.type == pygame.MOUSEWHEEL:
            self.cam.zoom(0.90 if e.y > 0 else 1.11)
            self.cam.save()

    def run(self):
        while self.running:
            for e in pygame.event.get():
                self.handle(e)
            dt = self.clock.tick(FPS) / 1000.0
            keys = pygame.key.get_pressed()
            step = 0.22 * dt
            moved = False
            if keys[pygame.K_LEFT]:
                self.cam.pan(-step, 0.0)
                moved = True
            if keys[pygame.K_RIGHT]:
                self.cam.pan(step, 0.0)
                moved = True
            if keys[pygame.K_UP]:
                self.cam.pan(0.0, step)
                moved = True
            if keys[pygame.K_DOWN]:
                self.cam.pan(0.0, -step)
                moved = True
            if moved:
                self._arrow_dirty = True
            if not self.paused:
                self.t += dt
            self.draw()
        if self._arrow_dirty:
            self.cam.save()
        pygame.quit()


def main():
    if "--shot" in sys.argv:
        out = Path(__file__).resolve().parent / "plane_shots"
        out.mkdir(exist_ok=True)
        demo = Demo()
        for t, name in ((0.2, "01_tip"), (2.2, "02_stroke"), (5.0, "03_project"), (9.0, "04_sway"), (14.2, "05_commit")):
            demo.t = t
            demo.draw()
            pygame.image.save(demo.screen, str(out / f"{name}.png"))
        pygame.quit()
        print(f"wrote {out}")
        return 0
    Demo().run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
