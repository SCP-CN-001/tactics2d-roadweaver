"""
Pathfinding on road field: A*, road snapping, path sampling at intervals.

All coordinates are in [0, 1] normalized space unless specified as pixels.
"""
from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Tuple

import numpy as np


def cost_map_from_road(road_field: np.ndarray,
                       cost_mult: float = 4.0,
                       cost_bias: float = 0.3) -> np.ndarray:
    """Build A* cost map from road probability field.

    Higher road probability = lower cost.  The formula
    ``clip((1 - field) * mult + bias, bias, mult + bias)``
    gives a smooth gradient where road pixels cost ~*bias* and
    off-road pixels cost ~*mult + bias*.
    """
    return np.clip((1.0 - road_field) * cost_mult + cost_bias,
                   cost_bias, cost_mult + cost_bias)


def nearest_road_px(px: int, py: int, road_field: np.ndarray,
                    radius: int = 5, threshold: float = 0.3) -> Tuple[int, int]:
    """Find nearest pixel with road_field > threshold within *radius*."""
    H, W = road_field.shape[:2]
    best_d, best_p = float('inf'), (px, py)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            nx2, ny2 = px + dx, py + dy
            if 0 <= nx2 < W and 0 <= ny2 < H and road_field[ny2, nx2] > threshold:
                d2 = dx * dx + dy * dy
                if d2 < best_d:
                    best_d, best_p = d2, (nx2, ny2)
    return best_p


def astar_grid(sx: int, sy: int, gx: int, gy: int,
               cost_map: np.ndarray, max_steps: int = 10000
               ) -> Optional[np.ndarray]:
    """A* on a grid cost map from (sx, sy) to (gx, gy) (pixel coords).

    Returns an (M, 2) array of normalized (x/W, y/H) waypoints, or None
    if no path is found within *max_steps* visited cells.

    For very short paths (Manhattan < 4) a straight-line interpolation is
    returned without running A*.
    """
    H, W = cost_map.shape[:2]

    if abs(sx - gx) + abs(sy - gy) < 4:
        # Straight line for very short gaps
        return np.linspace([sx / W, sy / H], [gx / W, gy / H],
                           max(3, int(np.hypot(gx - sx, gy - sy))))

    open_set: List[Tuple[float, int, int, int]] = []
    gs: Dict[Tuple[int, int], float] = {}
    cf: Dict[Tuple[int, int], Tuple[int, int]] = {}
    gs[(sx, sy)] = 0.0
    ctr = [0]
    heapq.heappush(open_set, (0.0, 0, sx, sy))

    while open_set and len(gs) < max_steps:
        _, _, cx, cy = heapq.heappop(open_set)
        if (cx, cy) == (gx, gy):
            path = []
            cur = (gx, gy)
            while cur in cf:
                path.append((cur[0] / W, cur[1] / H))
                cur = cf[cur]
            path.append((sx / W, sy / H))
            return np.array(path[::-1], dtype=np.float32)

        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1),
                       (1, 1), (-1, 1), (1, -1), (-1, -1)]:
            nx, ny = cx + dx, cy + dy
            if nx < 0 or nx >= W or ny < 0 or ny >= H:
                continue
            dc = cost_map[ny, nx] * (1.414 if dx != 0 and dy != 0 else 1.0)
            ng = gs[(cx, cy)] + dc
            if (nx, ny) in gs and gs[(nx, ny)] <= ng:
                continue
            gs[(nx, ny)] = ng
            cf[(nx, ny)] = (cx, cy)
            ctr[0] += 1
            h = float(np.hypot(nx - gx, ny - gy)) * 1.5
            heapq.heappush(open_set, (ng + h, ctr[0], nx, ny))

    return None


def astar_connect_path(src_norm: np.ndarray, tgt_norm: np.ndarray,
                       road_field: np.ndarray, cost_map: np.ndarray,
                       W: int, H: int,
                       max_steps: int = 10000) -> Optional[np.ndarray]:
    """A* from *src_norm* to *tgt_norm* with road snapping at both ends.

    Both coordinates are in [0, 1] normalized space.  The source is snapped
    to the nearest road pixel (radius=5, threshold=0.3) before searching.

    Returns an (M, 2) array of normalized waypoints including both endpoints,
    or None if no path is found.
    """
    # Snap source to road
    sx = int(src_norm[0] * W)
    sy = int(src_norm[1] * H)
    rx, ry = nearest_road_px(sx, sy, road_field, 5)
    sx, sy = rx, ry

    gx = int(tgt_norm[0] * W)
    gy = int(tgt_norm[1] * H)

    seg = astar_grid(sx, sy, gx, gy, cost_map, max_steps)
    if seg is None or len(seg) < 3:
        return None

    # Assemble: src → snapped-start → A* path → target
    path_list = [src_norm.ravel()]
    # If snap differs from src, add it
    snapped_start = np.array([rx / W, ry / H], dtype=np.float32)
    if np.linalg.norm(path_list[-1] - snapped_start) > 0.001:
        path_list.append(snapped_start)
    for pt in seg:
        if np.linalg.norm(path_list[-1] - pt.ravel()) > 0.001:
            path_list.append(pt.ravel())
    if np.linalg.norm(path_list[-1] - tgt_norm.ravel()) > 0.001:
        path_list.append(tgt_norm.ravel())

    return np.array(path_list, dtype=np.float32)


def sample_path_at_step(path: np.ndarray, step_sz: float) -> List[np.ndarray]:
    """Walk *path* (M, 2) and pick waypoints at ~*step_sz* intervals.

    Always includes the first and last points.  Returns a list of (2,) arrays.
    """
    idx = [0]
    cum = 0.0
    for k in range(1, len(path)):
        d = float(np.linalg.norm(path[k] - path[k - 1]))
        cum += d
        if cum >= step_sz and k < len(path) - 1:
            idx.append(k)
            cum = 0.0
    if idx[-1] != len(path) - 1:
        idx.append(len(path) - 1)
    return [path[i].reshape(-1) for i in idx]


def line_field_support(a: np.ndarray, b: np.ndarray, field: np.ndarray,
                       n_samples: int = 8, r_px: int = 3) -> float:
    """Average road field value along line a→b within a perpendicular band."""
    H, W = field.shape
    vals = []
    for t in np.linspace(0, 1, n_samples):
        x = int((a[0] + t * (b[0] - a[0])) * W)
        y = int((a[1] + t * (b[1] - a[1])) * H)
        x = np.clip(x, 0, W - 1)
        y = np.clip(y, 0, H - 1)
        y0, y1 = max(0, y - r_px), min(H, y + r_px + 1)
        x0, x1 = max(0, x - r_px), min(W, x + r_px + 1)
        vals.append(float(field[y0:y1, x0:x1].max()))
    return float(np.mean(vals)) if vals else 0.5
