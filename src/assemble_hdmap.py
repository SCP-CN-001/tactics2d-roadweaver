"""
Convert RoadWeaver compressed graph → Tactics2D ``Map`` (Lanelet2-style HD map).

Uses tactics2d modules for intersection / roundabout geometry.
Road lanes are smoothed via clamped BSpline then offset per-point.
"""

from __future__ import annotations

import numpy as np
from shapely import is_empty
from shapely.geometry import LineString, Polygon

from tactics2d.map.element import Lane, LaneRelationship, Map
from tactics2d.map.generator.road_segment.intersection import Intersection
from tactics2d.map.generator.road_segment.roundabout import Roundabout

LANE_WIDTH_M = 3.5
_SPEED_BY_RC = {1: 80, 2: 60, 3: 50}
_INTERSECTION_RADIUS = 5.0


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _heading(pts: np.ndarray, at_end: bool = False) -> float:
    if len(pts) < 2:
        return 0.0
    d = pts[-1] - pts[-2] if at_end else pts[1] - pts[0]
    return float(np.arctan2(d[1], d[0]))


def _chaikin(pts: np.ndarray, iterations: int = 2) -> np.ndarray:
    """Chaikin corner-cutting subdivision.

    Safer than Catmull-Rom: always stays within the convex hull of the
    control points and never self-intersects.  2 iterations ≈ smooth curve.
    O(n) per iteration.
    """
    if len(pts) < 3:
        return pts
    for _ in range(iterations):
        if len(pts) < 2:
            break
        nxt = [pts[0]]
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            nxt.append(a + 0.25 * (b - a))
            nxt.append(a + 0.75 * (b - a))
        nxt.append(pts[-1])
        pts = np.array(nxt)
    return pts


def _smooth(pts):
    if len(pts) < 3:
        return pts
    try:
        return _chaikin(pts, iterations=2)
    except Exception:
        return pts


def _offset(pts: np.ndarray, d: float) -> np.ndarray:
    """Offset centreline using Shapely ``offset_curve``.

    Shapely's implementation inserts circular arcs at tight corners
    instead of sharp angles, producing far fewer self-intersecting
    boundaries than the manual per-segment approach.
    """
    if len(pts) < 2:
        return pts
    from shapely.geometry import LineString as _LS

    result = _LS(pts).offset_curve(d)
    if result.geom_type == "LineString" and len(result.coords) >= 2:
        return np.array(result.coords, dtype=np.float64)
    return pts


def _geom(e, geoms_m, c, ei):
    if e < len(geoms_m) and len(geoms_m[e]) >= 2:
        return np.asarray(geoms_m[e], dtype=np.float64)
    u, v = int(ei[e, 0]), int(ei[e, 1])
    return np.array([c[u], c[v]])


# ---------------------------------------------------------------------------
#  Self-intersection fix  (cut at crossing point)
# ---------------------------------------------------------------------------


def _segment_intersection(a, b, c, d):
    """Return intersection point of segments ab and cd, or ``None``."""
    x1, y1 = a
    x2, y2 = b
    x3, y3 = c
    x4, y4 = d
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-12:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
    if 0 < t < 1 and 0 < u < 1:
        return np.array([x1 + t * (x2 - x1), y1 + t * (y2 - y1)])
    return None


def _cut_at_self_intersection(coords: np.ndarray) -> np.ndarray:
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


def _offset_per_point(pts: np.ndarray, d: float) -> np.ndarray:
    """Per-point perpendicular offset — simple, never self-intersects.

    Offsets each point by *d* along the local perpendicular direction
    (averaged from neighbouring segments for interior points).  The
    result has sharp corners where the centreline turns, but those are
    smoothed by the subsequent Chaikin pass.  Used as a fallback when
    ``_offset`` (Shapely) produces a self-intersection that cannot be
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


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------


def graph_to_map(
    coords_int,
    edge_index_int,
    geometries,
    node_types=None,
    lanes_per_dir=None,
    road_class=None,
    density=None,
    map_w=2000.0,
    map_h=2000.0,
    name="roadweaver",
    scenario_type="urban",
) -> Map:
    c = np.asarray(coords_int, dtype=np.float64) * np.array([map_w, map_h])
    ei = np.asarray(edge_index_int, dtype=np.int64)
    E = len(ei)
    if lanes_per_dir is None or len(lanes_per_dir) != E:
        lanes_per_dir = np.full(E, 2, dtype=np.int64)
    if road_class is None or len(road_class) != E:
        road_class = np.full(E, 2, dtype=np.int64)
    if node_types is None:
        node_types = np.zeros(len(c), dtype=np.int64)

    print(f"[Map] {E} edges  area={map_w:.0f}x{map_h:.0f}m")

    geoms_m = [
        (np.asarray(g, dtype=np.float64) * np.array([map_w, map_h]) if len(g) >= 2 else g)
        for g in geometries
    ]

    node_in: dict[int, list[int]] = {n: [] for n in range(len(c))}
    node_out: dict[int, list[int]] = {n: [] for n in range(len(c))}
    for e in range(E):
        u, v = int(ei[e, 0]), int(ei[e, 1])
        node_out[u].append(e)
        node_in[v].append(e)

    all_lanes: dict[str, Lane] = {}
    all_junctions = {}
    all_areas = {}
    id_counter = 0

    # ---- Truncate geometry at junction arm boundaries -----------------
    # Only truncate edges where ONE end is a junction/RA node (spoke edges).
    # Edges where BOTH ends are junction/RA (roundabout ring edges) are
    # internal to the junction and must NOT be truncated.
    geoms_road = list(geoms_m)
    for e in range(E):
        pts = geoms_road[e]
        if len(pts) < 2:
            continue
        u, v = int(ei[e, 0]), int(ei[e, 1])
        u_is_junc = u < len(node_types) and int(node_types[u]) in (1, 3)
        v_is_junc = v < len(node_types) and int(node_types[v]) in (1, 3)
        # RA nodes (type 3): keep full geometry — roundabout outer ring is ~18m
        # from centre, so truncating to 5m would break port matching.
        if u_is_junc and int(node_types[u]) == 3:
            u_is_junc = False
        if v_is_junc and int(node_types[v]) == 3:
            v_is_junc = False

        # Skip edges where BOTH ends are junction/RA (e.g. roundabout ring)
        if u_is_junc and v_is_junc:
            continue

        # Approaching v (edge ends at junction v)
        if v_is_junc:
            centre = c[v]
            h = _heading(pts, at_end=True) + np.pi  # inward
            arm_pt = centre + _INTERSECTION_RADIUS * np.array(
                [np.cos(h + np.pi), np.sin(h + np.pi)]
            )
            for k in range(len(pts) - 1, -1, -1):
                if np.linalg.norm(pts[k] - centre) >= _INTERSECTION_RADIUS:
                    clipped = list(pts[: k + 1])
                    dir_to_centre = centre - clipped[-1]
                    dir_to_centre = dir_to_centre / (np.linalg.norm(dir_to_centre) + 1e-12)
                    clipped[-1] = centre - _INTERSECTION_RADIUS * dir_to_centre
                    geoms_road[e] = np.array(clipped)
                    break

        # Departing from u (edge starts at junction u)
        pts = geoms_road[e]
        if len(pts) >= 2 and u_is_junc:
            centre = c[u]
            h = _heading(pts, at_end=False)  # outward
            arm_pt = centre + _INTERSECTION_RADIUS * np.array([np.cos(h), np.sin(h)])
            for k in range(len(pts)):
                if np.linalg.norm(pts[k] - centre) >= _INTERSECTION_RADIUS:
                    clipped = list(pts[k:])
                    dir_from_centre = clipped[0] - centre
                    dir_from_centre = dir_from_centre / (np.linalg.norm(dir_from_centre) + 1e-12)
                    clipped[0] = centre + _INTERSECTION_RADIUS * dir_from_centre
                    geoms_road[e] = np.array(clipped)
                    break

    # ---- Build road lanes --------------------------------------------
    for e in range(E):
        u, v = int(ei[e, 0]), int(ei[e, 1])
        # Skip edges attached to a dead-end node
        if (u < len(node_types) and int(node_types[u]) == 4) or (
            v < len(node_types) and int(node_types[v]) == 4
        ):
            continue
        n_l = max(1, int(lanes_per_dir[e]))
        speed = _SPEED_BY_RC.get(int(road_class[e]) if e < len(road_class) else 2, 50)
        pts = geoms_road[e]
        if len(pts) < 2:
            continue
        # Smooth
        pts = _smooth(pts)
        if len(pts) < 2:
            continue
        for i in range(n_l * 2):
            off = (i - n_l + 0.5) * LANE_WIDTH_M
            left = _offset(pts, off + LANE_WIDTH_M / 2)
            right = _offset(pts, off - LANE_WIDTH_M / 2)
            # Smooth lane boundaries (2 iterations)
            if len(left) >= 3:
                left = _chaikin(left, iterations=2)
            if len(right) >= 3:
                right = _chaikin(right, iterations=2)
            # Fix self-intersecting boundaries by cutting at crossing point
            if len(left) >= 4 and not LineString(left).is_simple:
                left = _cut_at_self_intersection(left)
            if len(right) >= 4 and not LineString(right).is_simple:
                right = _cut_at_self_intersection(right)
            # Fallback: if cut leaves too few points, use per-point offset
            if len(left) < 3:
                left = _offset_per_point(pts, off + LANE_WIDTH_M / 2)
                if len(left) >= 3:
                    left = _chaikin(left, iterations=2)
            if len(right) < 3:
                right = _offset_per_point(pts, off - LANE_WIDTH_M / 2)
                if len(right) >= 3:
                    right = _chaikin(right, iterations=2)
            # Realign endpoints at junction nodes so port matching succeeds
            u_j = u < len(node_types) and int(node_types[u]) in (1, 3)
            v_j = v < len(node_types) and int(node_types[v]) in (1, 3)
            if u_j and int(node_types[u]) == 3:
                u_j = False
            if v_j and int(node_types[v]) == 3:
                v_j = False
            if v_j and len(pts) >= 2 and len(left) >= 2 and len(right) >= 2:
                _s = pts[-1] - pts[-2]
                _sn = np.linalg.norm(_s)
                if _sn > 1e-6:
                    _perp = np.array([-_s[1] / _sn, _s[0] / _sn])
                    _dl = off + LANE_WIDTH_M / 2
                    _dr = off - LANE_WIDTH_M / 2
                    left[-1] = pts[-1] + _perp * _dl
                    right[-1] = pts[-1] + _perp * _dr
            if u_j and len(pts) >= 2 and len(left) >= 2 and len(right) >= 2:
                _s = pts[1] - pts[0]
                _sn = np.linalg.norm(_s)
                if _sn > 1e-6:
                    _perp = np.array([-_s[1] / _sn, _s[0] / _sn])
                    _dl = off + LANE_WIDTH_M / 2
                    _dr = off - LANE_WIDTH_M / 2
                    left[0] = pts[0] + _perp * _dl
                    right[0] = pts[0] + _perp * _dr
            # Re-check self-intersection after endpoint realignment
            if len(left) >= 4 and not LineString(left).is_simple:
                left = _cut_at_self_intersection(left)
            if len(right) >= 4 and not LineString(right).is_simple:
                right = _cut_at_self_intersection(right)
            forward = i < n_l
            if not forward:
                left = left[::-1]
                right = right[::-1]
            lid = f"e{e}_l{i}"
            all_lanes[lid] = Lane(
                id_=lid,
                left_side=LineString(left) if len(left) >= 2 else None,
                right_side=LineString(right) if len(right) >= 2 else None,
                subtype="road",
                speed_limit=speed,
                speed_limit_unit="km/h",
                location="urban",
            )

    # ---- Intersection / roundabout geometry --------------------------
    pairs: set[tuple[str, str]] = set()

    def _link(a, b):
        if a in all_lanes and b in all_lanes and a != b and (a, b) not in pairs:
            all_lanes[a].add_related_lane(b, LaneRelationship.SUCCESSOR)
            pairs.add((a, b))

    # ---- Roundabout grouping: adjacent RA nodes -> one roundabout ----
    ra_nodes = {n for n, t in enumerate(node_types) if int(t) == 3}
    ra_adj: dict[int, set[int]] = {n: set() for n in ra_nodes}
    for u, v in ei:
        u, v = int(u), int(v)
        if u in ra_adj and v in ra_adj:
            ra_adj[u].add(v)
            ra_adj[v].add(u)

    visited: set[int] = set()
    ra_groups: list[list[int]] = []
    for n in sorted(ra_nodes):
        if n in visited:
            continue
        group: list[int] = []
        stack = [n]
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            group.append(cur)
            stack.extend(ra_adj[cur] - visited)
        ra_groups.append(group)

    for g_idx, group in enumerate(ra_groups):
        centre = np.mean([c[n] for n in group], axis=0)
        arms: list[dict] = []
        for n in group:
            for e in node_in[n] + node_out[n]:
                u, v = int(ei[e, 0]), int(ei[e, 1])
                other = v if u == n else u
                if other in ra_nodes:
                    continue
                pts = _geom(e, geoms_road, c, ei)
                n_l = int(lanes_per_dir[e])
                # Heading toward group centre, not toward individual RA node
                if e in node_in[n]:
                    ep = pts[-1]  # geometry endpoint near roundabout
                else:
                    ep = pts[0]
                dir_to_centre = centre - ep
                h = float(np.arctan2(dir_to_centre[1], dir_to_centre[0]))
                arms.append(
                    {
                        "heading": h,
                        "lane_num": n_l,
                        "radius": _INTERSECTION_RADIUS,
                        "speed_limit": _SPEED_BY_RC.get(
                            int(road_class[e]) if e < len(road_class) else 2, 50
                        ),
                    }
                )
        if len(arms) >= 2:
            ra = Roundabout(
                ring_radius=_INTERSECTION_RADIUS * 3,
                ring_lane_num=1,
                lane_width=LANE_WIDTH_M,
                speed_limit=20.0,
            )
            res = ra.build(centre, arms, id_offset=id_counter)
            for lane in res.lanes:
                nid = f"ra_{lane.id_}"
                all_lanes[nid] = lane
                lane.id_ = nid
            for j in res.junctions:
                all_junctions[f"RJ_{g_idx}"] = j
            for a in res.areas:
                all_areas[f"RA_{g_idx}"] = a
            id_counter = res.id_counter
            for pname, port in res.ports.items():
                is_in = "in" in pname
                pt = np.asarray(port.point)
                ra_ids = [f"ra_{lid}" for lid in port.lane_ids if f"ra_{lid}" in all_lanes]
                if not ra_ids:
                    continue
                for lid in list(all_lanes.keys()):
                    if not lid.startswith("e"):
                        continue
                    try:
                        lane = all_lanes[lid]
                        # Use centreline midpoint for roundabout port matching
                        try:
                            left_end = np.array(list(lane.left_side.coords)[-1 if is_in else 0])
                            right_end = np.array(list(lane.right_side.coords)[-1 if is_in else 0])
                            ep = (left_end + right_end) / 2
                        except:
                            ep = np.array(
                                list((lane.right_side if is_in else lane.left_side).coords)[
                                    -1 if is_in else 0
                                ]
                            )
                        if np.linalg.norm(ep - pt) < 25.0:
                            for rid in ra_ids:
                                (_link if is_in else lambda a, b: _link(b, a))(lid, rid)
                    except Exception:
                        pass

    # ---- Individual intersection nodes (non-RA) -----------------------
    for n in range(len(c)):
        nt = int(node_types[n])
        if nt != 1:
            continue
        centre = c[n]

        arms = []
        for e in node_in[n] + node_out[n]:
            pts = _geom(e, geoms_road, c, ei)
            is_in = e in node_in[n]
            h = _heading(pts, at_end=is_in) + (np.pi if is_in else 0)
            n_l = int(lanes_per_dir[e])
            arms.append(
                {
                    "heading": h,
                    "lane_num": n_l,
                    "radius": _INTERSECTION_RADIUS,
                    "speed_limit": _SPEED_BY_RC.get(
                        int(road_class[e]) if e < len(road_class) else 2, 50
                    ),
                }
            )

        if len(arms) >= 3:  # intersection — any degree-3+ junction
            try:
                inter = Intersection(
                    lane_width=LANE_WIDTH_M, radius=_INTERSECTION_RADIUS, speed_limit=30.0
                )
                res = inter.build(centre, arms, id_offset=id_counter)
                for lane in res.lanes:
                    nid = f"int_{lane.id_}"
                    all_lanes[nid] = lane
                    lane.id_ = nid
                for j in res.junctions:
                    all_junctions[f"J_{n}"] = j
                id_counter = res.id_counter
                for pname, port in res.ports.items():
                    is_in = pname.endswith("_in")
                    pt = np.asarray(port.point)
                    int_ids = [f"int_{lid}" for lid in port.lane_ids if f"int_{lid}" in all_lanes]
                    if not int_ids:
                        continue
                    for lid in list(all_lanes.keys()):
                        if not lid.startswith("e"):
                            continue
                        try:
                            lane = all_lanes[lid]
                            # Use centerline midpoint (not lane boundary) for port matching
                            try:
                                left_end = np.array(list(lane.left_side.coords)[-1 if is_in else 0])
                                right_end = np.array(
                                    list(lane.right_side.coords)[-1 if is_in else 0]
                                )
                                ep = (left_end + right_end) / 2
                            except:
                                ep = np.array(
                                    list((lane.right_side if is_in else lane.left_side).coords)[
                                        -1 if is_in else 0
                                    ]
                                )
                            if np.linalg.norm(ep - pt) < 6.0:
                                for iid in int_ids:
                                    (_link if is_in else lambda a, b: _link(b, a))(lid, iid)
                        except Exception:
                            pass
            except Exception as exc:
                print(f"  [Map] Intersection at node {n} failed: {exc}")

    # ---- Direct road -> road successor (non-junction nodes) ------------
    for n in range(len(c)):
        if int(node_types[n]) in (1, 3):
            continue
    for n in range(len(c)):
        if int(node_types[n]) in (1, 3):
            continue
        for ie in node_in[n]:
            ni = int(lanes_per_dir[ie])
            for oe in node_out[n]:
                if ie == oe:
                    continue
                no = int(lanes_per_dir[oe])
                for k in range(max(ni, no)):
                    mk = min(k, no - 1)
                    ik = min(k, ni - 1)
                    _link(f"e{ie}_l{ik}", f"e{oe}_l{mk}")
                    _link(f"e{oe}_l{mk + no}", f"e{ie}_l{ik + ni}")

    # ---- Assemble Map ------------------------------------------------
    hd_map = Map(name=name, scenario_type=scenario_type)
    for lane in all_lanes.values():
        if lane.left_side is not None and lane.right_side is not None:
            hd_map.add_lane(lane)
    for j in all_junctions.values():
        hd_map.add_junction(j)
    for a in all_areas.values():
        hd_map.add_area(a)
    hd_map.set_boundary((0.0, map_w, 0.0, map_h))

    total_succ = sum(len(l.successors) for l in hd_map.lanes.values())
    dead = sum(1 for l in hd_map.lanes.values() if not l.successors)
    print(
        f"[Map] {len(hd_map.lanes)} lanes  {len(all_junctions)} junc  "
        f"{len(all_areas)} area  {total_succ} succ  {dead} dead"
    )
    return hd_map


# ---------------------------------------------------------------------------
#  Persistence
# ---------------------------------------------------------------------------


def map_to_file(hd_map: Map, path: str):
    import pickle

    with open(path, "wb") as f:
        pickle.dump(hd_map, f)
    print(f"[Map] Saved → {path}")


# ---------------------------------------------------------------------------
#  Quick visualisation  (black / white)
# ---------------------------------------------------------------------------


def quick_vis(hd_map: Map, path: str, dpi: int = 300):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
                    ax.plot(x, y, color="#222", lw=0.6, alpha=0.9)
        except Exception:
            pass

    # Junction shapes (gray fill + border)
    for junc in (hd_map.junctions or {}).values():
        try:
            shape = getattr(junc, "custom_tags", {}).get("shape", [])
            if shape and len(shape) > 2:
                xs, ys = zip(*shape)
                ax.fill(xs, ys, alpha=0.25, color="#888")
                ax.plot(xs + (xs[0],), ys + (ys[0],), color="#555", lw=1.5, alpha=0.7)
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
                    ax.plot(x, y, color="#555", lw=0.5, alpha=0.6)
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
