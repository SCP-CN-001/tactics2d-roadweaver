"""HD Map assembly geometry helpers."""

from __future__ import annotations

import numpy as np

from utils.geometry import chaikin as _chaikin_keep_ends
from utils.geometry import segment_intersection as _segment_intersection


def heading(pts: np.ndarray, at_end: bool = False) -> float:
    """Compute the heading angle of a polyline."""
    if len(pts) < 2:
        return 0.0
    d = pts[-1] - pts[-2] if at_end else pts[1] - pts[0]
    return float(np.arctan2(d[1], d[0]))


def chaikin(pts: np.ndarray, iterations: int = 2) -> np.ndarray:
    """Chaikin corner-cutting subdivision that preserves the endpoints.

    Lane boundaries must keep their first/last points so they align with
    junction ports during port matching.  Delegates to
    :func:`utils.geometry.chaikin` with ``keep_ends=True``.
    """
    return _chaikin_keep_ends(pts, iterations=iterations, keep_ends=True)


def smooth(pts):
    """Smooth a polyline with Chaikin subdivision.

    4 iterations instead of 2: sharp kinks in compressed edge geometries
    otherwise make ``offset_curve`` produce left/right boundaries that cross
    (lane width → 0 — the "hollow" seen near roads).
    """
    if len(pts) < 3:
        return pts
    try:
        return chaikin(pts, iterations=4)
    except Exception:
        return pts


def geom(e, geoms_m, c, ei):
    """Return the geometry of an edge."""
    if e < len(geoms_m) and len(geoms_m[e]) >= 2:
        return np.asarray(geoms_m[e], dtype=np.float64)
    u, v = int(ei[e, 0]), int(ei[e, 1])
    return np.array([c[u], c[v]])


def self_intersection_frac(coords: np.ndarray) -> float | None:
    """Return the arc-length fraction of the first self-intersection, or None.

    Unlike a truncation that returns the truncated polyline, this reports
    *where* the boundary crosses itself so that both lane boundaries can be
    cut at the *same* arc position.  Cutting only one boundary (as the old
    code did) leaves the lane width collapsing to a narrow sliver from the
    cut point onward — the "hollow" seen near intersections.
    """
    n = len(coords)
    if n < 4:
        return None
    seg_len = np.sqrt(((coords[1:] - coords[:-1]) ** 2).sum(axis=1))
    cum = np.concatenate([[0], np.cumsum(seg_len)])
    total = float(cum[-1])
    if total < 1e-12:
        return None
    for i in range(1, n - 2):
        p1, p2 = coords[i], coords[i + 1]
        for j in range(0, i - 1):
            q1, q2 = coords[j], coords[j + 1]
            if _segment_intersection(p1, p2, q1, q2) is not None:
                return float(cum[i]) / total
    return None


def truncate_at_frac(coords: np.ndarray, frac: float) -> np.ndarray:
    """Truncate a polyline at the given arc-length fraction."""
    if frac <= 0.0:
        return coords[:1]
    if frac >= 1.0:
        return coords
    seg_len = np.sqrt(((coords[1:] - coords[:-1]) ** 2).sum(axis=1))
    cum = np.concatenate([[0], np.cumsum(seg_len)])
    total = float(cum[-1])
    if total < 1e-12:
        return coords
    target = frac * total
    for k in range(1, len(coords)):
        if cum[k] >= target:
            t = (target - cum[k - 1]) / (cum[k] - cum[k - 1] + 1e-12)
            pt = coords[k - 1] + t * (coords[k] - coords[k - 1])
            return np.vstack([coords[:k], [pt]])
    return coords


def offset_per_point(pts: np.ndarray, d: float) -> np.ndarray:
    """Per-point perpendicular offset — simple, never self-intersects.

    Offsets each point by *d* along the local perpendicular direction
    (averaged from neighbouring segments for interior points).  The
    result has sharp corners where the centreline turns, but those are
    smoothed by the subsequent Chaikin pass.
    """
    n = len(pts)
    if n < 2:
        return pts
    out = np.empty_like(pts)
    for i in range(n):
        if i == 0:
            dx, dy = pts[1, 0] - pts[0, 0], pts[1, 1] - pts[0, 1]
        elif i == n - 1:
            dx, dy = pts[-1, 0] - pts[-2, 0], pts[-1, 1] - pts[-2, 1]
        else:
            dx, dy = pts[i + 1, 0] - pts[i - 1, 0], pts[i + 1, 1] - pts[i - 1, 1]
        nrm = np.sqrt(dx * dx + dy * dy)
        if nrm < 1e-12:
            out[i] = pts[i]
        else:
            out[i] = pts[i] + np.array([-dy / nrm, dx / nrm]) * d
    return out
