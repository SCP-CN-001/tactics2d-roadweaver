"""Compressed graph to HD Map converter."""

from __future__ import annotations

import networkx as nx
import numpy as np
from shapely import is_empty
from shapely.geometry import LineString, Polygon

from tactics2d.map.element import Lane, LaneRelationship, Map
from tactics2d.map.generator.road_segment.intersection import Intersection
from tactics2d.map.generator.road_segment.junction_approach import JunctionApproach
from tactics2d.map.generator.road_segment.roundabout import Roundabout
from utils.geometry import segment_intersection as _segment_intersection

from .geometry import (
    chaikin,
    geom,
    heading,
    offset_per_point,
    self_intersection_frac,
    smooth,
    truncate_at_frac,
)

LANE_WIDTH_M = 3.5
_SPEED_BY_RC = {1: 80, 2: 60, 3: 50}
_INTERSECTION_RADIUS = 8.0
_RA_RING_RADIUS = 18.0
_APPROACH_LENGTH = 20.0


def _coarsen(pts: np.ndarray, max_pts: int = 64) -> np.ndarray:
    """Subsample a polyline to at most ``max_pts`` points (keeps both ends)."""
    if len(pts) <= max_pts:
        return pts
    idx = np.unique(np.linspace(0, len(pts) - 1, max_pts).astype(int))
    return pts[idx]


def _repair_hourglass_lane(lane, tol: float = 0.05, max_iter: int = 3) -> bool:
    """Truncate a lane's boundaries at their first crossing, in place.

    Returns True if the lane is valid afterwards (possibly shortened); False if
    it is too degenerate to keep.  Both boundaries are cut at the smaller of the
    two crossing arc fractions, then re-checked (a crossing can reappear after a
    first cut), up to ``max_iter`` times.  The crossing search runs on a
    coarsened copy of the boundaries (they can carry hundreds of points after
    smoothing); the cut itself uses the full-resolution boundaries.
    """
    for _ in range(max_iter):
        lt, rt = lane.left_side, lane.right_side
        if lt is None or rt is None or lt.is_empty or rt.is_empty:
            return True
        lta = np.array(list(lt.coords))
        rta = np.array(list(rt.coords))
        if len(lta) < 2 or len(rta) < 2:
            return False
        poly = Polygon(np.vstack([lta, rta[::-1]]))
        if poly.is_simple and poly.is_valid:
            return True

        ls, rs = LineString(_coarsen(lta)), LineString(_coarsen(rta))
        f_lefts: list[float] = []
        f_rights: list[float] = []
        inter = ls.intersection(rs)
        if not inter.is_empty:
            pts = list(inter.geoms) if inter.geom_type == "MultiPoint" else [inter]
            for p in pts:
                if p.geom_type != "Point":
                    continue
                if ls.length < 1e-12 or rs.length < 1e-12:
                    continue
                f_l = float(ls.project(p) / ls.length)
                f_r = float(rs.project(p) / rs.length)
                if tol < f_l < 1 - tol and tol < f_r < 1 - tol:
                    f_lefts.append(f_l)
                    f_rights.append(f_r)
        # Single-boundary self-intersections (folded road) are cut the same way.
        for arr in (_coarsen(lta), _coarsen(rta)):
            if len(arr) >= 4 and not LineString(arr).is_simple:
                fs = self_intersection_frac(arr)
                if fs is not None and tol < fs < 1 - tol:
                    f_lefts.append(fs)
                    f_rights.append(fs)
        if not f_lefts:
            return False
        f = min(min(f_lefts), min(f_rights))
        lt2 = truncate_at_frac(lta, f)
        rt2 = truncate_at_frac(rta, f)
        if len(lt2) < 2 or len(rt2) < 2:
            return False
        lane.left_side = LineString(lt2)
        lane.right_side = LineString(rt2)
    # Final verification after max_iter cuts.
    lt, rt = lane.left_side, lane.right_side
    lta = np.array(list(lt.coords))
    rta = np.array(list(rt.coords))
    return (
        Polygon(np.vstack([lta, rta[::-1]])).is_simple
        and Polygon(np.vstack([lta, rta[::-1]])).is_valid
    )


def _trim_taper_lane(lane, min_width: float = 1.5) -> None:
    """Trim the near-zero-width tip of a taper lane, in place.

    Junction-approach "added"/"dropped" lanes taper from full width to zero,
    so one end converges to a point ("收束到一点").  The tip narrower than
    ``min_width`` is cut off, leaving a blunt end — the lane keeps its width
    and no longer pinches to a visible point.  Lanes already >= ``min_width``
    at both ends are left untouched.
    """
    lt, rt = lane.left_side, lane.right_side
    if lt is None or rt is None or lt.is_empty or rt.is_empty:
        return
    lta = np.array(list(lt.coords))
    rta = np.array(list(rt.coords))
    n = min(len(lta), len(rta))
    if n < 3:
        return
    w = np.linalg.norm(lta[:n] - rta[:n], axis=1)
    if w[0] >= min_width and w[-1] >= min_width:
        return
    keep = np.where(w >= min_width)[0]
    if len(keep) == 0:
        return
    i0 = int(keep[0])
    i1 = int(keep[-1])
    if w[0] < min_width and w[-1] < min_width:
        # narrow at both ends ("lens" lane) -> keep the wide middle
        lane.left_side = LineString(lta[i0 : i1 + 1])
        lane.right_side = LineString(rta[i0 : i1 + 1])
    elif w[0] < min_width:
        # narrow at the start (road side, "added" lane) -> trim the tip
        lane.left_side = LineString(lta[i0:])
        lane.right_side = LineString(rta[i0:])
    else:
        # narrow at the end (junction side, "dropped" lane) -> trim the tip
        lane.left_side = LineString(lta[: i1 + 1])
        lane.right_side = LineString(rta[: i1 + 1])


def _monotonic_desc(vals):
    """Enforce strictly monotonic decreasing lateral boundary offsets.

    A junction port exposes one connector lane per *turn* (straight plus
    left/right from every other arm), so the port boundary positions along an
    arm cross-section can overlap and reverse order.  Feeding non-monotonic
    offsets into the approach taper makes adjacent lane boundaries cross — the
    hourglass shape.  Clamping to a strictly decreasing fan keeps the approach
    lanes well ordered and crossing-free.
    """
    out = []
    for v in vals:
        out.append(v if not out else min(float(v), out[-1] - 1e-3))
    return out


def _approach_tail(road_pts: np.ndarray, pt: np.ndarray, max_len: float) -> np.ndarray:
    """Return road points within ``max_len`` of the port point ``pt``.

    The approach lane must only bridge a short distance from the junction
    port into the road.  Capping by *distance* (instead of taking a fixed
    number of geometry points) prevents approach lanes from retracing the
    road geometry for hundreds of metres and painting ghost lanes on top of
    the road lanes.  Points are returned in the same order as ``road_pts``.
    """
    road_pts = np.asarray(road_pts, dtype=float)
    if len(road_pts) < 2:
        return road_pts
    d0 = float(np.linalg.norm(road_pts[0] - pt))
    d1 = float(np.linalg.norm(road_pts[-1] - pt))
    start = 0 if d0 <= d1 else len(road_pts) - 1
    step = 1 if start == 0 else -1
    idxs = [start]
    acc = 0.0
    k = start
    while 0 <= k + step < len(road_pts):
        nxt = k + step
        acc += float(np.linalg.norm(road_pts[nxt] - road_pts[k]))
        if acc > max_len:
            break
        idxs.append(nxt)
        k = nxt
    idxs.sort()
    return road_pts[idxs]


def assign_lanes(
    coords_int: np.ndarray,
    edge_index_int: np.ndarray,
    geometries: list[np.ndarray],
    road_class: np.ndarray,
    density: float = 20.0,
) -> np.ndarray:
    """Assign per-direction lane counts (1-4) to each compressed edge.

    Factors: road_class, edge length (in metres), node importance, density.
    """
    G = nx.Graph()
    for i in range(len(coords_int)):
        G.add_node(i)
    for u, v in edge_index_int:
        G.add_edge(int(u), int(v))

    lanes = np.ones(len(edge_index_int), dtype=np.int64)
    for j in range(len(edge_index_int)):
        cls = int(road_class[j])
        u, v = int(edge_index_int[j, 0]), int(edge_index_int[j, 1])

        if j < len(geometries) and len(geometries[j]) >= 2:
            pts = np.asarray(geometries[j])
        else:
            pts = np.array([coords_int[u], coords_int[v]])
        edge_len = sum(np.linalg.norm(pts[k + 1] - pts[k]) for k in range(len(pts) - 1))

        imp = G.degree(u) + G.degree(v)
        base = {1: 3, 2: 2, 3: 1}.get(cls, 1)
        if edge_len > 400:
            base += 1
        elif edge_len < 80:
            base -= 1
        if imp >= 8:
            base += 1
        if density > 25 and base < 3:
            base += 1
        lanes[j] = max(1, min(4, base))

    # Smooth lane counts: adjacent edges should differ by at most 1
    # Build edge adjacency per node
    node_edges: dict[int, list[int]] = {}
    for j in range(len(edge_index_int)):
        u, v = int(edge_index_int[j, 0]), int(edge_index_int[j, 1])
        node_edges.setdefault(u, []).append(j)
        node_edges.setdefault(v, []).append(j)
    for _ in range(3):  # multiple passes for propagation
        for n, e_list in node_edges.items():
            if len(e_list) < 2:
                continue
            n_lanes = [lanes[e] for e in e_list]
            avg = sum(n_lanes) // len(n_lanes)
            for e in e_list:
                if abs(lanes[e] - avg) > 1:
                    lanes[e] = avg + (1 if lanes[e] > avg else -1)
    return lanes


def _respace_approach_lanes(
    all_lanes, geoms_road, ei, node_types, lanes_per_dir, node_in, node_out
):
    """Re-space multi-lane approach cross-sections at junctions to even W spacing.

    The cross-section at each junction end is rebuilt from the road centreline
    and a single lateral axis, so all approach lanes of a road share the same
    inner/outer ordering at both ends (no hourglass twist / inner-outer swap).

    ``perp`` is the left normal of the u->v direction.  Forward lanes (u->v)
    sit on the +perp side with their LEFT outward; backward lanes (v->u) sit
    on the -perp side with their LEFT outward too (driving direction flips).
    The junction end of every approach lane is ``coords[-1]`` (a backward lane
    is stored reversed, so it ends at u).
    """
    W = LANE_WIDTH_M
    for e in range(len(ei)):
        u, v = int(ei[e, 0]), int(ei[e, 1])
        n_l = int(lanes_per_dir[e])
        if n_l < 2:
            continue
        g = geoms_road[e]
        if len(g) < 2:
            continue
        d = g[-1] - g[0]  # u->v direction
        ln = float(np.linalg.norm(d))
        if ln < 1e-9:
            continue
        perp = np.array([-d[1] / ln, d[0] / ln])  # left normal of u->v

        for node, incoming, centreline_pt in (
            (v, True, g[-1]),  # forward approach ends at v
            (u, False, g[0]),  # backward approach ends at u
        ):
            if node >= len(node_types) or int(node_types[node]) != 1:
                continue
            if incoming and e not in node_in[node]:
                continue
            if not incoming and e not in node_out[node]:
                continue
            # Approach lanes INNER -> OUTER.
            if incoming:
                lane_idx = list(range(n_l - 1, -1, -1))  # [n-1, ..., 0]
            else:
                lane_idx = list(range(n_l, 2 * n_l))  # [n, ..., 2n-1]
            for j, i in enumerate(lane_idx):
                lid = f"e{e}_l{i}"
                lane = all_lanes.get(lid)
                if lane is None or lane.left_side is None or lane.right_side is None:
                    break
                if incoming:
                    centre = centreline_pt - (j + 0.5) * W * perp
                    left_pt = centre - 0.5 * W * perp
                    right_pt = centre + 0.5 * W * perp
                else:
                    centre = centreline_pt + (j + 0.5) * W * perp
                    left_pt = centre + 0.5 * W * perp
                    right_pt = centre - 0.5 * W * perp
                _l = np.array(list(lane.left_side.coords))
                _r = np.array(list(lane.right_side.coords))
                _l[-1] = left_pt
                _r[-1] = right_pt
                lane.left_side = LineString(_l)
                lane.right_side = LineString(_r)


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
    """Build a Tactics2D Map from a compressed graph."""
    c = np.asarray(coords_int, dtype=np.float64) * np.array([map_w, map_h])
    ei = np.asarray(edge_index_int, dtype=np.int64)
    n_edges = len(ei)
    if lanes_per_dir is None or len(lanes_per_dir) != n_edges:
        lanes_per_dir = np.full(n_edges, 2, dtype=np.int64)
    if road_class is None or len(road_class) != n_edges:
        road_class = np.full(n_edges, 2, dtype=np.int64)
    if node_types is None:
        node_types = np.zeros(len(c), dtype=np.int64)

    print(f"[Map] {n_edges} edges  area={map_w:.0f}x{map_h:.0f}m")

    geoms_m = [
        (np.asarray(g, dtype=np.float64) * np.array([map_w, map_h]) if len(g) >= 2 else g)
        for g in geometries
    ]

    node_in: dict[int, list[int]] = {n: [] for n in range(len(c))}
    node_out: dict[int, list[int]] = {n: [] for n in range(len(c))}
    for e in range(n_edges):
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
    for e in range(n_edges):
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
            h = heading(pts, at_end=True) + np.pi  # inward
            for k in range(len(pts) - 1, -1, -1):
                if np.linalg.norm(pts[k] - centre) >= _INTERSECTION_RADIUS:
                    clipped = list(pts[: k + 1])
                    dir_to_centre = centre - clipped[-1]
                    dir_to_centre = dir_to_centre / (np.linalg.norm(dir_to_centre) + 1e-12)
                    clipped_end = centre - _INTERSECTION_RADIUS * dir_to_centre
                    if len(clipped) == 1:
                        # straight 2-point edge: keep the far endpoint too,
                        # otherwise the geometry collapses to a single point
                        geoms_road[e] = np.array([clipped[0], clipped_end])
                    else:
                        clipped[-1] = clipped_end
                        geoms_road[e] = np.array(clipped)
                    break

        # Departing from u (edge starts at junction u)
        pts = geoms_road[e]
        if len(pts) >= 2 and u_is_junc:
            centre = c[u]
            h = heading(pts, at_end=False)  # outward
            for k in range(len(pts)):
                if np.linalg.norm(pts[k] - centre) >= _INTERSECTION_RADIUS:
                    clipped = list(pts[k:])
                    dir_from_centre = clipped[0] - centre
                    dir_from_centre = dir_from_centre / (np.linalg.norm(dir_from_centre) + 1e-12)
                    clipped_start = centre + _INTERSECTION_RADIUS * dir_from_centre
                    if len(clipped) == 1:
                        # straight 2-point edge: keep the far endpoint too
                        geoms_road[e] = np.array([clipped_start, clipped[0]])
                    else:
                        clipped[0] = clipped_start
                        geoms_road[e] = np.array(clipped)
                    break

    # ---- Build road lanes (shared-boundary computation) --------------
    # For each edge we compute 2·n_lanes+1 unique offset curves ONCE, then
    # build lanes by pairing adjacent boundaries.  This guarantees adjacent
    # lanes share identical boundary LineStrings — no gaps, no overlaps.
    #
    # Boundary      Offset          Shared by lanes
    # ─────────     ──────          ──────────────
    # b[0]          -(n_l)*W        fwd[0].right
    # b[1]          -(n_l-1)*W      fwd[0].left  = fwd[1].right
    #  ...           ...             ...
    # b[n_l-1]      -W              fwd[n_l-2].left = fwd[n_l-1].right
    # b[n_l]         0 (centreline)  fwd[n_l-1].left = bwd[n_l].left
    # b[n_l+1]      +W              bwd[n_l].right = bwd[n_l+1].left
    #  ...           ...             ...
    # b[2·n_l]      +n_l*W          bwd[2·n_l-1].right
    for e in range(n_edges):
        u, v = int(ei[e, 0]), int(ei[e, 1])
        n_lanes = max(1, int(lanes_per_dir[e]))
        speed = _SPEED_BY_RC.get(int(road_class[e]) if e < len(road_class) else 2, 50)
        pts = geoms_road[e]
        if len(pts) < 2:
            continue
        # Smooth
        pts = smooth(pts)
        if len(pts) < 2:
            continue
        # Drop consecutive (near-)duplicate points: offset_per_point leaves
        # the boundary ON the centreline at zero-length segments, which makes
        # the lane width collapse to 0 (the "hollow").
        if len(pts) > 2:
            _keep = [pts[0]]
            for _p in pts[1:]:
                if np.linalg.norm(_p - _keep[-1]) > 1e-7:
                    _keep.append(_p)
            pts = np.array(_keep)
            if len(pts) < 2:
                continue

        # --- Compute all unique offset boundaries once -----------------
        n_b = 2 * n_lanes + 1  # number of unique boundary curves
        boundaries: list[np.ndarray | None] = [None] * n_b

        for k in range(n_b):
            d = (k - n_lanes) * LANE_WIDTH_M
            b = offset_per_point(pts, d)
            if len(b) < 2:
                boundaries[k] = None
                continue
            boundaries[k] = b

        # --- Find minimum self-intersection fraction across ALL boundaries
        # All boundaries on the same edge must be truncated at the SAME arc
        # position, otherwise boundaries that should be identical diverge and
        # create gaps between adjacent lanes.
        min_cut = 1.0
        for k in range(n_b):
            b = boundaries[k]
            if b is None or len(b) < 4:
                continue
            if not LineString(b).is_simple:
                _f = self_intersection_frac(b)
                if _f is not None:
                    min_cut = min(min_cut, _f)

        if min_cut < 1.0:
            for k in range(n_b):
                if boundaries[k] is not None:
                    boundaries[k] = truncate_at_frac(boundaries[k], min_cut)

        # --- Fallback: if a boundary got too short, recompute ----------
        for k in range(n_b):
            if boundaries[k] is not None and len(boundaries[k]) < 3:
                d = (k - n_lanes) * LANE_WIDTH_M
                boundaries[k] = offset_per_point(pts, d)
                if boundaries[k] is not None and len(boundaries[k]) >= 3:
                    boundaries[k] = chaikin(boundaries[k], iterations=2)
                # Recompute produces a full-length boundary while the others
                # were truncated at min_cut; re-apply the same cut so shared
                # boundary endpoints stay aligned (avoids a width collapse at
                # the truncation point).
                if boundaries[k] is not None and min_cut < 1.0:
                    boundaries[k] = truncate_at_frac(boundaries[k], min_cut)

        # --- Realign endpoints perpendicular to centreline -------------
        # Only the first and last point of each boundary is adjusted;
        # interior points stay as offset_per_point produced them.
        if len(pts) >= 2:
            for k in range(n_b):
                b = boundaries[k]
                if b is None or len(b) < 2:
                    continue
                d = (k - n_lanes) * LANE_WIDTH_M
                # End of centreline → boundary coords[-1]
                _s = pts[-1] - pts[-2]
                _sn = np.linalg.norm(_s)
                if _sn > 1e-6:
                    _perp = np.array([-_s[1] / _sn, _s[0] / _sn])
                    b[-1] = pts[-1] + _perp * d
                # Start of centreline → boundary coords[0]
                _s = pts[1] - pts[0]
                _sn = np.linalg.norm(_s)
                if _sn > 1e-6:
                    _perp = np.array([-_s[1] / _sn, _s[0] / _sn])
                    b[0] = pts[0] + _perp * d

        # --- Re-check self-intersection after realignment --------------
        min_cut2 = 1.0
        for k in range(n_b):
            b = boundaries[k]
            if b is None or len(b) < 4:
                continue
            if not LineString(b).is_simple:
                _f = self_intersection_frac(b)
                if _f is not None:
                    min_cut2 = min(min_cut2, _f)

        if min_cut2 < 1.0:
            for k in range(n_b):
                if boundaries[k] is not None:
                    boundaries[k] = truncate_at_frac(boundaries[k], min_cut2)

        # --- Build lanes from shared boundaries ------------------------
        for i in range(n_lanes * 2):
            forward = i < n_lanes
            if forward:
                # Forward lane i: left=b[i+1], right=b[i]
                b_left = boundaries[i + 1] if i + 1 < n_b else None
                b_right = boundaries[i] if i < n_b else None
                left = b_left
                right = b_right
            else:
                # Backward lane i: left=reversed(b[i]), right=reversed(b[i+1])
                b_left = boundaries[i] if i < n_b else None
                b_right = boundaries[i + 1] if i + 1 < n_b else None
                left = b_left[::-1] if b_left is not None else None
                right = b_right[::-1] if b_right is not None else None

            lid = f"e{e}_l{i}"
            all_lanes[lid] = Lane(
                id_=lid,
                left_side=LineString(left) if left is not None and len(left) >= 2 else None,
                right_side=LineString(right) if right is not None and len(right) >= 2 else None,
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
        _ra_arm_edges: list[int] = []  # edge_id per arm, same order as arms
        _ra_arm_incoming: list[bool] = []  # spoke edge is incoming to its RA node?
        for n in group:
            for e in node_in[n] + node_out[n]:
                u, v = int(ei[e, 0]), int(ei[e, 1])
                other = v if u == n else u
                if other in ra_nodes:
                    continue
                pts = geom(e, geoms_road, c, ei)
                n_lanes = int(lanes_per_dir[e])
                # Heading toward group centre, not toward individual RA node
                if e in node_in[n]:
                    ep = pts[-1]  # geometry endpoint near roundabout
                else:
                    ep = pts[0]
                # Outward heading: from group centre toward the spoke connection.
                # (The roundabout module places sockets at centre + R * heading_outward.)
                h = float(np.arctan2(ep[1] - centre[1], ep[0] - centre[0]))
                arms.append(
                    {
                        "heading": h,
                        "lane_num": n_lanes,
                        "radius": _INTERSECTION_RADIUS,
                        "speed_limit": _SPEED_BY_RC.get(
                            int(road_class[e]) if e < len(road_class) else 2, 50
                        ),
                    }
                )
                _ra_arm_edges.append(e)
                _ra_arm_incoming.append(e in node_in[n])
        if len(arms) >= 2:
            # Ring lane count follows the widest approach arm, so every
            # approach lane gets its own ring lane (pair_num = min(arm, ring)).
            ring_lanes = max(1, max(int(arm["lane_num"]) for arm in arms))
            ra = Roundabout(
                ring_radius=_RA_RING_RADIUS,
                ring_lane_num=ring_lanes,
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
                arm_idx = int(pname.split("_")[1])
                if arm_idx >= len(_ra_arm_edges):
                    continue
                edge_id = _ra_arm_edges[arm_idx]
                n_lanes = int(lanes_per_dir[edge_id])

                pt = np.asarray(port.point)
                h_p = float(port.heading)
                perp = np.array([-np.sin(h_p), np.cos(h_p)])
                ra_ids = [f"ra_{lid}" for lid in port.lane_ids if f"ra_{lid}" in all_lanes]
                if not ra_ids:
                    continue

                port_lanes = []
                for rid in ra_ids:
                    rln = all_lanes.get(rid)
                    if rln is None:
                        continue
                    try:
                        _idx = 0 if is_in else -1
                        _lt = np.array(list(rln.left_side.coords)[_idx])
                        _rt = np.array(list(rln.right_side.coords)[_idx])
                        _mid = (_lt + _rt) / 2
                        _lat = float(np.dot(_mid - pt, perp))
                        port_lanes.append((rid, _lat, _lt, _rt))
                    except Exception:
                        continue
                port_lanes.sort(key=lambda x: x[1])

                candidates = []
                # Lanes approaching n are forward if the edge ends at n, backward if it
                # starts at n; the departing set is the other. Junction-side coords index:
                # approach -> coords[-1], depart -> coords[0] (both fwd & bwd).
                edge_is_incoming = _ra_arm_incoming[arm_idx]
                use_forward = (is_in and edge_is_incoming) or (not is_in and not edge_is_incoming)
                if use_forward:
                    _start, _end, _eidx = 0, n_lanes, (-1 if is_in else 0)
                else:
                    _start, _end, _eidx = n_lanes, 2 * n_lanes, (-1 if is_in else 0)
                for __idx in range(_start, _end):
                    lid = f"e{edge_id}_l{__idx}"
                    if lid not in all_lanes:
                        continue
                    try:
                        lane = all_lanes[lid]
                        le = np.array(list(lane.left_side.coords)[_eidx])
                        re = np.array(list(lane.right_side.coords)[_eidx])
                        ep = (le + re) / 2
                        _lat = float(np.dot(ep - pt, perp))
                        candidates.append((lid, _lat))
                    except Exception:
                        continue
                candidates.sort(key=lambda x: x[1])

                n_port_lanes = len(port_lanes)
                # ---- Build JunctionApproach for roundabout arm ----
                bdy_offsets = []
                for pi, (_, _, _lt, _rt) in enumerate(port_lanes):
                    llat = float(np.dot(_lt - pt, perp))
                    rlat = float(np.dot(_rt - pt, perp))
                    if pi == 0:
                        bdy_offsets.append(llat)
                    bdy_offsets.append(rlat)
                bdy_offsets = _monotonic_desc(bdy_offsets)

                # Road geometry near the roundabout: find the index where
                # distance from centre crosses _trunc_radius, then take a
                # tail of points on the road side to preserve the curve.
                _ra_pts = geom(edge_id, geoms_road, c, ei)
                _trunc_radius = _INTERSECTION_RADIUS + _APPROACH_LENGTH
                _incoming = _ra_arm_incoming[arm_idx]
                _N_TAIL = min(8, len(_ra_pts))
                _cut_idx = None
                if _incoming:
                    for _k in range(len(_ra_pts) - 1, -1, -1):
                        if np.linalg.norm(_ra_pts[_k] - centre) >= _trunc_radius:
                            _cut_idx = _k
                            break
                    if _cut_idx is not None:
                        _tail = _ra_pts[max(0, _cut_idx - _N_TAIL + 1) : _cut_idx + 1]
                    else:
                        _tail = _ra_pts[:_N_TAIL]
                else:
                    for _k in range(len(_ra_pts)):
                        if np.linalg.norm(_ra_pts[_k] - centre) >= _trunc_radius:
                            _cut_idx = _k
                            break
                    if _cut_idx is not None:
                        _tail = _ra_pts[_cut_idx : _cut_idx + _N_TAIL]
                    else:
                        _tail = _ra_pts[-_N_TAIL:]

                # Reverse when travelling away from junction along the road
                if is_in != _incoming:
                    _tail = _tail[::-1]

                if is_in:
                    cl = np.vstack([_tail, pt.reshape(1, 2)])
                    app_start_n, app_end_n = n_lanes, n_port_lanes
                else:
                    cl = np.vstack([pt.reshape(1, 2), _tail])
                    app_start_n, app_end_n = n_port_lanes, n_lanes

                cl = smooth(cl)

                try:
                    app = JunctionApproach(step_size=0.1)
                    app_res = app.build(
                        centerline=cl,
                        start_lane_num=app_start_n,
                        end_lane_num=app_end_n,
                        end_boundary_offsets=np.array(bdy_offsets),
                        lane_width=LANE_WIDTH_M,
                        speed_limit=20.0,
                        id_offset=id_counter,
                    )
                    id_counter = app_res.id_counter

                    for aln in app_res.lanes:
                        alid = f"app_{aln.id_}"
                        all_lanes[alid] = aln
                        aln.id_ = alid

                    entry_ids = [f"app_{lid}" for lid in app_res.ports["entry"].lane_ids]
                    exit_ids = [f"app_{lid}" for lid in app_res.ports["exit"].lane_ids]

                    if is_in:
                        for i in range(min(len(entry_ids), len(candidates))):
                            _link(candidates[i][0], entry_ids[i])
                        for j in range(min(len(exit_ids), n_port_lanes)):
                            _link(exit_ids[j], port_lanes[j][0])
                    else:
                        for j in range(min(len(entry_ids), n_port_lanes)):
                            _link(port_lanes[j][0], entry_ids[j])
                        for i in range(min(len(exit_ids), len(candidates))):
                            _link(exit_ids[i], candidates[i][0])
                except Exception:
                    if is_in:
                        n_conn = len(port_lanes)
                        if n_conn and candidates:
                            for k in range(len(candidates)):
                                lid, _ = candidates[k]
                                rid, _, _, _ = port_lanes[k % n_conn]
                                _link(lid, rid)
                    else:
                        n_depart = len(candidates)
                        if n_depart and port_lanes:
                            for k in range(len(port_lanes)):
                                rid, _, _, _ = port_lanes[k]
                                lid, _ = candidates[k % n_depart]
                                _link(rid, lid)

    # ---- Individual intersection nodes (non-RA) -----------------------
    for n in range(len(c)):
        nt = int(node_types[n])
        if nt != 1:
            continue
        centre = c[n]

        arms = []
        _arm_edges = []  # edge_id per arm, same order as arms
        for e in node_in[n] + node_out[n]:
            pts = geom(e, geoms_road, c, ei)
            is_in = e in node_in[n]
            h = heading(pts, at_end=is_in) + (np.pi if is_in else 0)
            n_lanes = int(lanes_per_dir[e])
            arms.append(
                {
                    "heading": h,
                    "lane_num": n_lanes,
                    "radius": _INTERSECTION_RADIUS,
                    "speed_limit": _SPEED_BY_RC.get(
                        int(road_class[e]) if e < len(road_class) else 2, 50
                    ),
                }
            )
            _arm_edges.append(e)

        if len(arms) >= 3:
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
                    arm_idx = int(pname.split("_")[1])
                    if arm_idx >= len(_arm_edges):
                        continue
                    edge_id = _arm_edges[arm_idx]
                    n_lanes = int(lanes_per_dir[edge_id])

                    pt = np.asarray(port.point)
                    h_p = float(port.heading)
                    perp = np.array([-np.sin(h_p), np.cos(h_p)])
                    int_ids = [f"int_{lid}" for lid in port.lane_ids if f"int_{lid}" in all_lanes]
                    if not int_ids:
                        continue

                    # Collect port connector lane boundary positions at arm side
                    port_lanes = []  # (iid, left_at_boundary, right_at_boundary)
                    for iid in int_ids:
                        iiln = all_lanes.get(iid)
                        if iiln is None:
                            continue
                        try:
                            _idx = 0 if is_in else -1
                            _lt = np.array(list(iiln.left_side.coords)[_idx])
                            _rt = np.array(list(iiln.right_side.coords)[_idx])
                            _mid = (_lt + _rt) / 2
                            _lat = float(np.dot(_mid - pt, perp))
                            port_lanes.append((iid, _lat, _lt, _rt))
                        except Exception:
                            continue
                    port_lanes.sort(key=lambda x: x[1])

                    # Collect road lane boundary positions (only from this edge, right direction)
                    candidates = []
                    # Lanes approaching n are forward if the edge ends at n, backward if it
                    # starts at n; the departing set is the other. Junction-side coords index:
                    # approach -> coords[-1], depart -> coords[0] (both fwd & bwd).
                    edge_is_incoming = edge_id in node_in[n]
                    use_forward = (is_in and edge_is_incoming) or (
                        not is_in and not edge_is_incoming
                    )
                    if use_forward:
                        _start, _end, _eidx = 0, n_lanes, (-1 if is_in else 0)
                    else:
                        _start, _end, _eidx = n_lanes, 2 * n_lanes, (-1 if is_in else 0)
                    for _idx in range(_start, _end):
                        lid = f"e{edge_id}_l{_idx}"
                        if lid not in all_lanes:
                            continue
                        try:
                            lane = all_lanes[lid]
                            le = np.array(list(lane.left_side.coords)[_eidx])
                            re = np.array(list(lane.right_side.coords)[_eidx])
                            ep = (le + re) / 2
                            _lat = float(np.dot(ep - pt, perp))
                            candidates.append((lid, _lat))
                        except Exception:
                            continue
                    candidates.sort(key=lambda x: x[1])

                    n_port_lanes = len(port_lanes)
                    # ---- Build JunctionApproach -----------------
                    bdy_offsets = []
                    for pi, (_, _, _lt, _rt) in enumerate(port_lanes):
                        llat = float(np.dot(_lt - pt, perp))
                        rlat = float(np.dot(_rt - pt, perp))
                        if pi == 0:
                            bdy_offsets.append(llat)
                        bdy_offsets.append(rlat)
                    bdy_offsets = _monotonic_desc(bdy_offsets)

                    # Road geometry near this junction node.  Cap the tail by a
                    # fixed distance from the port (not a fixed point count),
                    # so approach lanes only bridge the junction zone instead of
                    # retracing the road for hundreds of metres.
                    _road_pts = geoms_road[edge_id]
                    _tail = _approach_tail(_road_pts, pt, _APPROACH_LENGTH)
                    # Reverse when going away from junction along the road
                    if is_in != edge_is_incoming:
                        _tail = _tail[::-1]

                    if is_in:
                        cl = np.vstack([_tail, pt.reshape(1, 2)])
                        app_start_n, app_end_n = n_lanes, n_port_lanes
                    else:
                        cl = np.vstack([pt.reshape(1, 2), _tail])
                        app_start_n, app_end_n = n_port_lanes, n_lanes

                    cl = smooth(cl)

                    try:
                        app = JunctionApproach(step_size=0.1)
                        app_res = app.build(
                            centerline=cl,
                            start_lane_num=app_start_n,
                            end_lane_num=app_end_n,
                            end_boundary_offsets=np.array(bdy_offsets),
                            lane_width=LANE_WIDTH_M,
                            speed_limit=30.0,
                            id_offset=id_counter,
                        )
                        id_counter = app_res.id_counter

                        app_lanes = []
                        for aln in app_res.lanes:
                            alid = f"app_{aln.id_}"
                            all_lanes[alid] = aln
                            aln.id_ = alid
                            app_lanes.append(alid)

                        entry_ids = [f"app_{lid}" for lid in app_res.ports["entry"].lane_ids]
                        exit_ids = [f"app_{lid}" for lid in app_res.ports["exit"].lane_ids]

                        if is_in:
                            # road → approach entry → approach exit → connector
                            for i in range(min(len(entry_ids), len(candidates))):
                                _link(candidates[i][0], entry_ids[i])
                            for j in range(min(len(exit_ids), n_port_lanes)):
                                _link(exit_ids[j], port_lanes[j][0])
                        else:
                            # connector → approach entry → approach exit → road
                            for j in range(min(len(entry_ids), n_port_lanes)):
                                _link(port_lanes[j][0], entry_ids[j])
                            for i in range(min(len(exit_ids), len(candidates))):
                                _link(exit_ids[i], candidates[i][0])
                    except Exception:
                        # Fallback: direct link
                        if is_in:
                            if port_lanes and candidates:
                                for k in range(len(candidates)):
                                    lid, _ = candidates[k]
                                    iid = port_lanes[k % n_port_lanes][0]
                                    _link(lid, iid)
                        else:
                            if candidates and port_lanes:
                                for k in range(len(port_lanes)):
                                    iid, _, _, _ = port_lanes[k]
                                    lid, _ = candidates[k % len(candidates)]
                                    _link(iid, lid)
            except Exception as exc:
                print(f"  [Map] Intersection at node {n} failed: {exc}")

    # ---- Direct road -> road successor (non-junction nodes) ------------
    # A degree-2 waypoint is a bend: the approaching lanes of one incident
    # edge continue into the departing lanes of the other, regardless of the
    # graph edge direction (which may be arbitrary after compression).
    for n in range(len(c)):
        if int(node_types[n]) in (1, 3):
            continue
        incident = list(node_in[n]) + list(node_out[n])
        if len(incident) != 2:
            continue
        e1, e2 = incident
        if e1 == e2:
            continue
        for ea, eb in ((e1, e2), (e2, e1)):
            na = int(lanes_per_dir[ea])
            nb = int(lanes_per_dir[eb])
            ea_approach_fwd = ea in node_in[n]  # forward lanes approach n
            eb_depart_fwd = eb not in node_in[n]  # forward lanes depart n
            for k in range(max(na, nb)):
                ia = min(k, na - 1)
                ib = min(k, nb - 1)
                lid_a = f"e{ea}_l{ia if ea_approach_fwd else ia + na}"
                lid_b = f"e{eb}_l{ib if eb_depart_fwd else ib + nb}"
                _link(lid_a, lid_b)

    # ---- Re-space multi-lane approach cross-sections at junctions ----
    # NOTE: _respace_approach_lanes is no longer needed.  Shared-boundary
    # lane building (above) already guarantees constant W spacing and
    # correct inner/outer ordering for both directions.  Calling it here
    # would overwrite shared boundary endpoints with per-lane copies and
    # create gaps between adjacent lanes.
    # _respace_approach_lanes(all_lanes, geoms_road, ei, node_types, lanes_per_dir, node_in, node_out)

    # ---- Remove road lanes disconnected from the junction network ----
    # A road lane is kept if it is reachable (via successor links) from some
    # intersection/roundabout connector, OR if its compressed edge touches a
    # junction node.  The graph-based keep matters: approach/connector lanes
    # may have been dropped earlier by the hourglass/taper cleanup, which would
    # otherwise orphan the whole road and leave a visible gap.  Only genuinely
    # floating edges (no junction endpoint) and redundant roundabout ring edges
    # (both endpoints roundabout) are dropped.
    conn_ids = {lid for lid in all_lanes if lid.startswith(("int", "ra", "app"))}
    if conn_ids:
        _adj: dict[str, set[str]] = {}
        for _lid, _lane in all_lanes.items():
            for _s in _lane.successors:
                _adj.setdefault(_lid, set()).add(_s)
                _adj.setdefault(_s, set()).add(_lid)
        reached: set[str] = set()
        stack = list(conn_ids)
        while stack:
            cur = stack.pop()
            if cur in reached:
                continue
            reached.add(cur)
            for nb in _adj.get(cur, ()):
                if nb not in reached:
                    stack.append(nb)
        junc_nodes = {n for n, t in enumerate(node_types) if int(t) in (1, 3)}
        road_keep: set[str] = set()
        for _e in range(len(ei)):
            _u, _v = int(ei[_e, 0]), int(ei[_e, 1])
            if _u in junc_nodes and _v in junc_nodes:
                if int(node_types[_u]) == 3 and int(node_types[_v]) == 3:
                    continue  # roundabout ring edge
            if _u in junc_nodes or _v in junc_nodes:
                for _i in range(2 * int(lanes_per_dir[_e])):
                    road_keep.add(f"e{_e}_l{_i}")
        isolated = {
            lid
            for lid in all_lanes
            if lid.startswith("e") and lid not in reached and lid not in road_keep
        }
        for lid in isolated:
            del all_lanes[lid]
        if isolated:
            print(f"  [Map] removed {len(isolated)} isolated road lanes")

    # ---- Repair degenerate hourglass lanes (self-intersecting) ----------
    # A lane whose left/right boundaries cross (or whose boundary self-
    # intersects) forms a bowtie/hourglass polygon.  These come from sharp
    # centreline kinks (road lanes), from tapering approach lanes whose lateral
    # offset sweep crosses, and from sharp turn connectors inside junctions.
    # Each crossing lane is truncated at its first crossing so it stays a valid
    # lane (no road gaps); only truly degenerate lanes are dropped.
    repaired_hg = 0
    dropped_hg: dict[str, int] = {}
    for _lid, _lane in list(all_lanes.items()):
        _lt, _rt = _lane.left_side, _lane.right_side
        if _lt is None or _rt is None or _lt.is_empty or _rt.is_empty:
            continue
        try:
            _lta = np.array(list(_lt.coords))
            _rta = np.array(list(_rt.coords))
            if len(_lta) < 2 or len(_rta) < 2:
                continue
            # Only act on the visible hourglass: the two lane sides actually
            # cross each other (or a single side folds back on itself).  Lanes
            # that are merely "invalid" without crossing are kept — dropping
            # them would clean up roads that render fine.
            if not LineString(_lta).crosses(LineString(_rta)) and (
                LineString(_lta).is_simple and LineString(_rta).is_simple
            ):
                continue
            # A crossing lane that cannot be repaired is dropped only when it
            # is visibly wide; thin/near-zero-width crossings are near-invisible
            # and kept so the road network loses as few lanes as possible.
            _max_width = float(np.linalg.norm(_lta[0] - _rta[0]))
            for _k in range(1, min(len(_lta), len(_rta))):
                _max_width = max(_max_width, float(np.linalg.norm(_lta[_k] - _rta[_k])))
            if _repair_hourglass_lane(_lane):
                repaired_hg += 1
                continue
            if _max_width < 1.5:
                continue  # thin crossing — near-invisible, keep it
            all_lanes.pop(_lid)
            _t = _lid.split("_")[0]
            dropped_hg[_t] = dropped_hg.get(_t, 0) + 1
        except Exception:
            continue
    if repaired_hg or dropped_hg:
        print(
            f"  [Map] repaired {repaired_hg} hourglass lanes, "
            f"dropped {sum(dropped_hg.values())} degenerate {dropped_hg}"
        )

    # ---- Trim taper lanes (converge-to-a-point tips) ------------------
    # Junction-approach added/dropped lanes taper to zero width; their tips
    # narrower than a lane's half-width are cut so no road visually pinches
    # to a single point.  3+ point lanes are trimmed; degenerate 2-point
    # lanes (a single segment from full width to a point) are dropped.
    trimmed = 0
    dropped_taper: dict[str, int] = {}
    for _lid, _lane in list(all_lanes.items()):
        _lt, _rt = _lane.left_side, _lane.right_side
        if _lt is None or _rt is None or _lt.is_empty or _rt.is_empty:
            continue
        _lta = np.array(list(_lt.coords))
        _rta = np.array(list(_rt.coords))
        _n = min(len(_lta), len(_rta))
        if _n < 2:
            continue
        _w0 = float(np.linalg.norm(_lta[0] - _rta[0]))
        _w1 = float(np.linalg.norm(_lta[-1] - _rta[-1]))
        if min(_w0, _w1) >= 1.5:
            continue
        # Sub-lane-width slivers that never reach a real lane width and end in
        # a point are phantom — drop them outright.
        _maxw = float(np.linalg.norm(_lta[: min(_n, 32)] - _rta[: min(_n, 32)], axis=1).max())
        for _k in range(min(_n, 32), _n, max(1, _n // 32)):
            _maxw = max(_maxw, float(np.linalg.norm(_lta[_k] - _rta[_k])))
        if _maxw < 1.5:
            all_lanes.pop(_lid)
            _t = _lid.split("_")[0]
            dropped_taper[_t] = dropped_taper.get(_t, 0) + 1
            continue
        if _n < 3:
            all_lanes.pop(_lid)
            _t = _lid.split("_")[0]
            dropped_taper[_t] = dropped_taper.get(_t, 0) + 1
            continue
        _trim_taper_lane(_lane)
        trimmed += 1
    if trimmed or dropped_taper:
        print(
            f"  [Map] trimmed {trimmed} taper tips, "
            f"dropped {sum(dropped_taper.values())} degenerate tapers {dropped_taper}"
        )

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
    dead = {
        p: sum(1 for l in hd_map.lanes.values() if l.id_.startswith(p) and not l.successors)
        for p in ("e", "app", "int", "ra_")
    }
    print(
        f"[Map] {len(hd_map.lanes)} lanes  {len(all_junctions)} junc  "
        f"{len(all_areas)} area  {total_succ} succ  "
        f"{sum(dead.values())} dead "
        f"(e:{dead['e']} app:{dead['app']} int:{dead['int']} ra:{dead['ra_']})"
    )
    return hd_map
