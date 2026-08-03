"""Serialize and visualize tactics2d Map objects."""

from __future__ import annotations

import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely import is_empty


def map_to_file(hd_map: Map, path: str):
    """Pickle a tactics2d Map to a file."""
    with open(path, "wb") as f:
        pickle.dump(hd_map, f)
    print(f"[Map] Saved → {path}")


# ---------------------------------------------------------------------------
#  Quick visualisation  (black / white)
# ---------------------------------------------------------------------------


def quick_vis(hd_map: Map, path: str, dpi: int = 300):
    """Render a tactics2d Map to a PNG image."""
    fig, ax = plt.subplots(figsize=(20, 20))
    lanes = list(hd_map.lanes.values())
    print(f"[Vis] {len(lanes)} lanes")

    road_lanes = [l for l in lanes if l.id_.startswith("e")]
    ra_lanes = [l for l in lanes if l.id_.startswith("ra_")]

    # Road lane boundaries (black)
    for lane in road_lanes:
        try:
            for side in (lane.left_side, lane.right_side):
                if side is not None and not is_empty(side):
                    x, y = side.xy
                    ax.plot(x, y, color="#222", lw=1.5, alpha=0.9)
        except Exception:
            pass

    # Junction shapes (gray fill + border)
    for junc in (hd_map.junctions or {}).values():
        try:
            shape = getattr(junc, "custom_tags", {}).get("shape", [])
            if shape and len(shape) > 2:
                xs, ys = zip(*shape)
                ax.fill(xs, ys, alpha=0.25, color="#888")
                ax.plot(xs + (xs[0],), ys + (ys[0],), color="#555", lw=2.0, alpha=0.7)
        except Exception:
            pass

    # Roundabout areas + ring lanes
    for area in (hd_map.areas or {}).values():
        try:
            if area.geometry and not is_empty(area.geometry):
                x, y = area.geometry.exterior.xy
                ax.fill(x, y, alpha=0.25, color="#aaa")
        except Exception:
            pass
    for lane in ra_lanes:
        try:
            for side in (lane.left_side, lane.right_side):
                if side is not None and not is_empty(side):
                    x, y = side.xy
                    ax.plot(x, y, color="#555", lw=1.0, alpha=0.6)
        except Exception:
            pass

    dead = sum(1 for l in road_lanes if not l.successors)
    ra_dead = sum(1 for l in ra_lanes if not l.successors)
    nj = len(hd_map.junctions or {})
    na = len(hd_map.areas or {})
    ax.set_aspect("equal")
    ax.set_title(
        f"HD Map — {len(road_lanes)} road lanes  {nj} junc  {na} area  "
        f"{dead} road dead  {ra_dead} ra dead",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[Vis] Saved → {path}")
