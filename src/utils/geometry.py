"""
Shared geometry primitives: Chaikin smoothing, segment intersection.
"""

from __future__ import annotations

import numpy as np


def chaikin(pts: np.ndarray, iterations: int = 2, keep_ends: bool = False) -> np.ndarray:
    """Chaikin corner-cutting subdivision.

    Replaces each edge with two new points at ¼ and ¾ of the edge,
    then iterates.  Converges to a quadratic B-spline.  Safer than
    Catmull-Rom for dense noisy polylines: preserves convexity of
    control points and never self-intersects.

    Args:
        pts: Polyline points with shape ``(N, 2)``.
        iterations: Number of subdivision passes.
        keep_ends: When ``True`` the original first/last points are retained
            (useful for lane boundaries whose endpoints must align with
            junction ports).  When ``False`` each pass yields ``2(N-1)``
            points with the endpoints pulled inward.
    """
    if len(pts) < 3 or iterations < 1:
        return pts
    for _ in range(iterations):
        nxt = [pts[0]] if keep_ends else []
        for i in range(len(pts) - 1):
            p0, p1 = pts[i], pts[i + 1]
            nxt.append(0.75 * p0 + 0.25 * p1)
            nxt.append(0.25 * p0 + 0.75 * p1)
        if keep_ends:
            nxt.append(pts[-1])
        pts = np.array(nxt)
    return pts


def segment_intersection(a, b, c, d, eps: float = 1e-10):
    """Return intersection point of segments ab and cd, or ``None``."""
    x1, y1 = a
    x2, y2 = b
    x3, y3 = c
    x4, y4 = d
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < eps:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
    if -eps < t < 1 + eps and -eps < u < 1 + eps:
        return np.array([x1 + t * (x2 - x1), y1 + t * (y2 - y1)])
    return None
