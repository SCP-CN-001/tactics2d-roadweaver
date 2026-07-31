"""HD Map assembly geometry helpers."""

from __future__ import annotations

import numpy as np
from shapely.geometry import LineString as _LineString

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
    """Smooth a polyline with Chaikin subdivision."""
    if len(pts) < 3:
        return pts
    try:
        return chaikin(pts, iterations=2)
    except Exception:
        return pts


def offset(pts: np.ndarray, d: float) -> np.ndarray:
    """Offset centreline using Shapely ``offset_curve``.

    Shapely's implementation inserts circular arcs at tight corners
    instead of sharp angles, producing far fewer self-intersecting
    boundaries than the manual per-segment approach.
    """
    if len(pts) < 2:
        return pts

    result = _LineString(pts).offset_curve(d)
    if result.geom_type == "LineString" and len(result.coords) >= 2:
        return np.array(result.coords, dtype=np.float64)
    return pts


def geom(e, geoms_m, c, ei):
    """Return the geometry of an edge."""
    if e < len(geoms_m) and len(geoms_m[e]) >= 2:
        return np.asarray(geoms_m[e], dtype=np.float64)
    u, v = int(ei[e, 0]), int(ei[e, 1])
    return np.array([c[u], c[v]])


# ---------------------------------------------------------------------------
#  Self-intersection fix  (cut at crossing point)
# ---------------------------------------------------------------------------


def cut_at_self_intersection(coords: np.ndarray) -> np.ndarray:
    """Walk the polyline and truncate before the first crossing segment.

    When a lane boundary self-intersects (a small loop at a tight corner),
    this function removes the loop by cutting off everything from the
    crossing segment onward.  The remaining clean portion is returned.
    """
    n = len(coords)
    if n < 4:
        return coords
    for i in range(1, n - 2):
        p1, p2 = coords[i], coords[i + 1]
        for j in range(0, i - 1):
            q1, q2 = coords[j], coords[j + 1]
            if _segment_intersection(p1, p2, q1, q2) is not None:
                return coords[: i + 1]
    return coords


def offset_per_point(pts: np.ndarray, d: float) -> np.ndarray:
    """Per-point perpendicular offset — simple, never self-intersects.

    Offsets each point by *d* along the local perpendicular direction
    (averaged from neighbouring segments for interior points).  The
    result has sharp corners where the centreline turns, but those are
    smoothed by the subsequent Chaikin pass.  Used as a fallback when
    ``offset`` (Shapely) produces a self-intersection that cannot be
    cleanly cut.
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
