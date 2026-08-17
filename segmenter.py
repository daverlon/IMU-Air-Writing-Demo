"""
Cut a continuous air-writing trajectory into sub-segments.

Even when someone writes fluidly, the pen slows almost to a stop at every
stroke ending — between letters, and also inside multi-stroke letters like A
or T. We cut at those velocity minima and let the decoder decide which
neighbouring pieces belong to the same letter.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from motion import Pose

# Cuts are found by prominence, not by an absolute speed: a minimum counts when
# the pen drops to a fraction of its recent peak and then speeds up again. That
# works the same whether the writer is fast or slow, big or small.
STROKE_SPEED = 0.050  # m/s — a burst must reach this to count as drawing at all
DIP_FRACTION = 0.50  # fell to half the current stroke's peak → entering a minimum
FREEZE_SPEED = 0.015  # m/s — hand parked; don't record these points
MIN_SUB_MS = 80  # ignore micro-dips inside a single stroke
MIN_SUB_POINTS = 5
LETTER_GAP_MS = 260  # stopped this long = that letter is finished
WORD_GAP_MS = 1200  # stopped this long = end of the word


def trim_still_tail(
    points: list[tuple[float, float, int]],
) -> list[tuple[float, float, int]]:
    """Drop the drift that follows the last real motion in a letter.

    A letter is only closed once the hand has been still for LETTER_GAP_MS, so
    every captured letter ends with roughly that much slow drift tacked on —
    the hand relaxing, not part of the glyph. It is usually a few percent of
    the path but reached 63% on one letter, and removing it is worth about 4
    points of accuracy.
    """
    if len(points) < 3:
        return points
    last_moving = 0
    for i in range(1, len(points)):
        dt = max((points[i][2] - points[i - 1][2]) / 1000.0, 1e-3)
        dx = points[i][0] - points[i - 1][0]
        dy = points[i][1] - points[i - 1][1]
        if (dx * dx + dy * dy) ** 0.5 / dt >= STROKE_SPEED:
            last_moving = i
    return points[: last_moving + 1] if last_moving else points


LEAD_FRAC = 0.05


def trim_lead(
    points: list[tuple[float, float, int]],
    frac: float = LEAD_FRAC,
) -> list[tuple[float, float, int]]:
    """Drop the first `frac` of path length. Timing/ink are unchanged."""
    if len(points) < 8 or frac <= 0:
        return points
    total = 0.0
    segs: list[float] = []
    for i in range(1, len(points)):
        dx = points[i][0] - points[i - 1][0]
        dy = points[i][1] - points[i - 1][1]
        d = (dx * dx + dy * dy) ** 0.5
        segs.append(d)
        total += d
    if total < 1e-6:
        return points
    cut = total * frac
    acc = 0.0
    for i, d in enumerate(segs):
        acc += d
        if acc >= cut:
            out = points[i + 1 :]
            return out if len(out) >= 4 else points
    return points


@dataclass
class SubSegment:
    points: list[tuple[float, float, int]] = field(default_factory=list)

    @property
    def t0(self) -> int:
        return self.points[0][2]

    @property
    def t1(self) -> int:
        return self.points[-1][2]


class Segmenter:
    def __init__(self):
        self.points: list[tuple[float, float, int]] = []
        self._drawing = False
        self._peak = 0.0
        self._dip_t: int | None = None
        self._dip_speed = float("inf")
        self._last_stroke_ms = 0

    def reset(self):
        self.points = []
        self._drawing = False
        self._peak = 0.0
        self._dip_t = None
        self._dip_speed = float("inf")

    def idle_ms(self, now_ms: int) -> int:
        return now_ms - self._last_stroke_ms

    def push(self, pose: Pose) -> SubSegment | None:
        """Feed one pose; returns a sub-segment when a velocity minimum closes one."""
        speed = pose.speed

        if not self._drawing:
            if speed < STROKE_SPEED:
                return None
            self._drawing = True
            self._peak = speed
            self.points = [(pose.x, pose.y, pose.t_ms)]

        if speed > FREEZE_SPEED:
            self.points.append((pose.x, pose.y, pose.t_ms))
        if speed >= STROKE_SPEED:
            self._last_stroke_ms = pose.t_ms

        if self._dip_t is None:
            self._peak = max(self._peak, speed)
            if self._peak >= STROKE_SPEED and speed <= DIP_FRACTION * self._peak:
                self._dip_t = pose.t_ms
                self._dip_speed = speed
            return None

        if speed < self._dip_speed:
            self._dip_speed = speed
            self._dip_t = pose.t_ms
            return None
        # Close the cut once the pen is clearly drawing again. This is measured
        # against STROKE_SPEED rather than the previous peak, so a slow letter
        # following a fast travel move still closes it.
        if speed < STROKE_SPEED:
            return None

        cut = self._cut_at_dip()
        self._dip_t = None
        self._dip_speed = float("inf")
        self._peak = speed
        return cut

    def flush(self) -> SubSegment | None:
        """Close whatever is buffered — used when the writer stops."""
        sub = SubSegment(self.points) if self._valid(self.points) else None
        self.reset()
        return sub

    def _cut_at_dip(self) -> SubSegment | None:
        head = [p for p in self.points if p[2] <= self._dip_t]
        if not self._valid(head):
            return None
        self.points = [p for p in self.points if p[2] > self._dip_t]
        return SubSegment(head)

    @staticmethod
    def _valid(points: list[tuple[float, float, int]]) -> bool:
        return (
            len(points) >= MIN_SUB_POINTS
            and points[-1][2] - points[0][2] >= MIN_SUB_MS
        )
