# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Official tactics2d renderer wrapper for high-DPI HD maps."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

_REPO = Path(__file__).resolve().parent.parent.parent
_T2D = _REPO / "tactics2d"
if str(_T2D) not in sys.path:
    sys.path.insert(0, str(_T2D))

from tactics2d.display.renderers import MatplotlibRenderer  # noqa: E402


def map_to_road_elements(hd):
    """Convert a tactics2d Map to renderer ``road_elements`` (lanes/junctions black)."""
    elems = []
    for lid, lane in hd.lanes.items():
        lt = list(lane.left_side.coords) if lane.left_side is not None else None
        rt = list(lane.right_side.coords) if lane.right_side is not None else None
        if lt and rt and len(lt) >= 2 and len(rt) >= 2:
            poly = list(lt) + list(rt)[::-1]
            if len(poly) >= 3:
                elems.append(
                    {
                        "id": f"lane_{lid}",
                        "shape": "polygon",
                        "type": "road",
                        "color": "black",
                        "geometry": poly,
                    }
                )
        for side, sname in ((lane.left_side, "left"), (lane.right_side, "right")):
            if side is not None and len(side.coords) >= 2:
                elems.append(
                    {
                        "id": f"bound_{lid}_{sname}",
                        "shape": "line",
                        "type": "road_border",
                        "color": "light-gray",
                        "line_width": 1.2,
                        "geometry": list(side.coords),
                    }
                )
    for jid, j in (hd.junctions or {}).items():
        shape = j.custom_tags.get("shape", [])
        if shape and len(shape) > 2:
            elems.append(
                {
                    "id": f"junc_{jid}",
                    "shape": "polygon",
                    "type": "junction",
                    "color": "black",
                    "geometry": list(shape),
                }
            )
    for aid, a in (hd.areas or {}).items():
        if a.geometry is not None:
            try:
                elems.append(
                    {
                        "id": f"area_{aid}",
                        "shape": "polygon",
                        "type": "area",
                        "color": "black",
                        "geometry": list(a.geometry.exterior.coords),
                    }
                )
            except Exception:
                pass
    for rid, rl in (hd.roadlines or {}).items():
        g = rl.geometry
        if g is not None and len(g.coords) >= 2:
            elems.append(
                {
                    "id": f"rl_{rid}",
                    "shape": "line",
                    "type": rl.type_,
                    "color": "white",
                    "geometry": list(g.coords),
                }
            )
    return elems


def render_map(hd, out_path: str, resolution: int = 1800, dpi: int = 150):
    """Render a tactics2d Map with the official renderer and save to *out_path*.

    Lanes and junctions are filled (black) so roads look solid rather than
    hollow outlines — the display-side counterpart to the eval-side
    ``eval/polyline_graph.save_clean_map``.
    """
    elems = map_to_road_elements(hd)
    xs, ys = [], []
    for el in elems:
        for p in el["geometry"]:
            xs.append(p[0])
            ys.append(p[1])
    if not xs:
        print("[render] empty map")
        return
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    renderer = MatplotlibRenderer(
        resolution=(resolution, resolution), xlim=(xmin, xmax), ylim=(ymin, ymax), dpi=dpi
    )
    geometry_data = {
        "metadata": {"sensor_position": [(xmin + xmax) / 2, (ymin + ymax) / 2], "sensor_yaw": 0.0},
        "map_data": {"road_id_to_remove": [], "road_elements": elems},
        "participant_data": {
            "participant_id_to_create": [],
            "participant_id_to_remove": [],
            "participants": [],
        },
    }
    renderer.update(geometry_data)
    renderer.fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(renderer.fig)
    print(f"[render] {len(elems)} elements -> {out_path}")
