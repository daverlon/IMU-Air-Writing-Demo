"""Rasterize a 2D stroke polyline into an EMNIST-like 28×28 grayscale image."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# Missing ink this long, or a jump this far, is a pen-up — do not connect.
GAP_MS = 70
GAP_M = 0.022
# Hand moved this far while the pen was up: old stroke is a different letter.
REPOSITION_M = 0.04


def pen_up(a, b) -> bool:
    dt = abs(int(b[2]) - int(a[2]))
    if dt >= GAP_MS:
        return True
    dx = float(b[0]) - float(a[0])
    dy = float(b[1]) - float(a[1])
    return (dx * dx + dy * dy) ** 0.5 >= GAP_M


def stroke_to_image(
    points: list[tuple[float, float, int]],
    size: int = 28,
    padding: float = 0.18,
    ink: float = 1.0,
) -> np.ndarray:
    """
    Natural screen-style letter: white ink on black, y-up stroke → upright glyph.
    Returns float32 (1, 28, 28) in [0, 1].
    """
    if len(points) < 2:
        return np.zeros((1, size, size), dtype=np.float32)

    xs = np.array([p[0] for p in points], dtype=np.float64)
    ys = np.array([p[1] for p in points], dtype=np.float64)

    min_x, max_x = float(xs.min()), float(xs.max())
    min_y, max_y = float(ys.min()), float(ys.max())
    span_x = max(max_x - min_x, 1e-4)
    span_y = max(max_y - min_y, 1e-4)
    # Keep aspect ratio — don't squash letters into a square of the max span only
    # but letter classifiers expect roughly square; pad the shorter axis.
    span = max(span_x, span_y)
    cx = (min_x + max_x) * 0.5
    cy = (min_y + max_y) * 0.5

    hi = size * 8
    img = Image.new("L", (hi, hi), 0)
    draw = ImageDraw.Draw(img)
    pad = padding * hi
    usable = hi - 2 * pad
    scale = usable / span

    def to_px(x: float, y: float) -> tuple[float, float]:
        px = (hi * 0.5) + (x - cx) * scale
        py = (hi * 0.5) - (y - cy) * scale  # y-up → image y-down
        return px, py

    # Thickness similar to EMNIST pen width after downscale
    width = max(4, int(hi / 14 * ink))
    for i in range(1, len(points)):
        if pen_up(points[i - 1], points[i]):
            continue
        x0, y0 = to_px(points[i - 1][0], points[i - 1][1])
        x1, y1 = to_px(points[i][0], points[i][1])
        draw.line([(x0, y0), (x1, y1)], fill=255, width=width)
        r = width // 2
        draw.ellipse((x0 - r, y0 - r, x0 + r, y0 + r), fill=255)
        draw.ellipse((x1 - r, y1 - r, x1 + r, y1 + r), fill=255)

    img = img.filter(ImageFilter.GaussianBlur(radius=0.8))
    img = img.resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if arr.max() > 0:
        arr = np.clip(arr / arr.max(), 0, 1)
    return arr.reshape(1, size, size)


def path_extent(points: list[tuple[float, float, int]]) -> float:
    if len(points) < 2:
        return 0.0
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return max(max(xs) - min(xs), max(ys) - min(ys))
