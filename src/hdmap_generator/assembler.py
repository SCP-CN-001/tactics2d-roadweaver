"""Compressed graph to HD Map converter."""

from __future__ import annotations

import networkx as nx
import numpy as np
from shapely import is_empty
from shapely.geometry import LineString, Polygon

from tactics2d.map.element import Lane, LaneRelationship, Map
from tactics2d.map.generator.road_segment.intersection import Intersection
from tactics2d.map.generator.road_segment.roundabout import Roundabout
from utils.geometry import segment_intersection as _segment_intersection

from .geometry import (
    chaikin,
    cut_at_self_intersection,
    geom,
    heading,
    offset,
    offset_per_point,
    smooth,
)

LANE_WIDTH_M = 3.5
_SPEED_BY_RC = {1: 80, 2: 60, 3: 50}
_INTERSECTION_RADIUS = 5.0


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
            h = heading(pts, at_end=False)  # outward
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
        for i in range(n_lanes * 2):
            off = (i - n_lanes + 0.5) * LANE_WIDTH_M
            left = offset(pts, off + LANE_WIDTH_M / 2)
            right = offset(pts, off - LANE_WIDTH_M / 2)
            # Smooth lane boundaries (2 iterations)
            if len(left) >= 3:
                left = chaikin(left, iterations=2)
            if len(right) >= 3:
                right = chaikin(right, iterations=2)
            # Fix self-intersecting boundaries by cutting at crossing point
            if len(left) >= 4 and not LineString(left).is_simple:
                left = cut_at_self_intersection(left)
            if len(right) >= 4 and not LineString(right).is_simple:
                right = cut_at_self_intersection(right)
            # Fallback: if cut leaves too few points, use per-point offset
            if len(left) < 3:
                left = offset_per_point(pts, off + LANE_WIDTH_M / 2)
                if len(left) >= 3:
                    left = chaikin(left, iterations=2)
            if len(right) < 3:
                right = offset_per_point(pts, off - LANE_WIDTH_M / 2)
                if len(right) >= 3:
                    right = chaikin(right, iterations=2)
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
                left = cut_at_self_intersection(left)
            if len(right) >= 4 and not LineString(right).is_simple:
                right = cut_at_self_intersection(right)
            forward = i < n_lanes
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

                if is_in:
                    # Approach: every approaching road lane gets a connector as
                    # successor (a connector may accept several road lanes), so
                    # no road lane is left dangling and successor chains do not
                    # break at the roundabout.
                    n_conn = len(port_lanes)
                    if n_conn and candidates:
                        for k in range(len(candidates)):
                            lid, _ = candidates[k]
                            rid, _lat, _lt, _rt = port_lanes[k % n_conn]
                            _link(lid, rid)
                        for k in range(min(len(candidates), n_conn)):
                            lid, _ = candidates[k]
                            _lt, _rt = port_lanes[k][2], port_lanes[k][3]
                            try:
                                lane = all_lanes[lid]
                                _left = np.array(list(lane.left_side.coords))
                                _right = np.array(list(lane.right_side.coords))
                                _left[-1] = _lt.copy()
                                _right[-1] = _rt.copy()
                                lane.left_side = LineString(_left)
                                lane.right_side = LineString(_right)
                            except Exception:
                                pass
                else:
                    # Depart: every connector ending at this arm gets the depart
                    # road as its successor (a road lane accepts traffic from
                    # several connectors), so no connector is left dangling.
                    n_depart = len(candidates)
                    if n_depart and port_lanes:
                        for k in range(len(port_lanes)):
                            rid, _lat, _lt, _rt = port_lanes[k]
                            lid, _ = candidates[k % n_depart]
                            _link(rid, lid)
                        # Snap each depart lane's start once, from the laterally
                        # best-matching connectors.
                        for k in range(min(n_depart, len(port_lanes))):
                            lid, _ = candidates[k]
                            _lt, _rt = port_lanes[k][2], port_lanes[k][3]
                            try:
                                lane = all_lanes[lid]
                                _left = np.array(list(lane.left_side.coords))
                                _right = np.array(list(lane.right_side.coords))
                                _left[0] = _lt.copy()
                                _right[0] = _rt.copy()
                                lane.left_side = LineString(_left)
                                lane.right_side = LineString(_right)
                            except Exception:
                                pass

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

                    if is_in:
                        # Approach: every approaching road lane gets a connector as
                        # successor (a connector may accept several road lanes), so
                        # no road lane is left dangling and successor chains do not
                        # break at the junction.
                        n_conn = len(port_lanes)
                        if n_conn and candidates:
                            for k in range(len(candidates)):
                                lid, _ = candidates[k]
                                iid, _lat, _lt, _rt = port_lanes[k % n_conn]
                                _link(lid, iid)
                            # Snap road lane junction ends from the laterally
                            # best-matching connectors.
                            for k in range(min(len(candidates), n_conn)):
                                lid, _ = candidates[k]
                                _lt, _rt = port_lanes[k][2], port_lanes[k][3]
                                try:
                                    lane = all_lanes[lid]
                                    _left = np.array(list(lane.left_side.coords))
                                    _right = np.array(list(lane.right_side.coords))
                                    _left[-1] = _lt.copy()
                                    _right[-1] = _rt.copy()
                                    lane.left_side = LineString(_left)
                                    lane.right_side = LineString(_right)
                                except Exception:
                                    pass
                    else:
                        # Depart: every connector ending at this arm gets the depart
                        # road as its successor (a road lane accepts traffic from
                        # several connectors), so no connector is left dangling.
                        n_depart = len(candidates)
                        if n_depart and port_lanes:
                            for k in range(len(port_lanes)):
                                iid, _lat, _lt, _rt = port_lanes[k]
                                lid, _ = candidates[k % n_depart]
                                _link(iid, lid)
                            # Snap each depart lane's start once, from the laterally
                            # best-matching connectors.
                            for k in range(min(n_depart, len(port_lanes))):
                                lid, _ = candidates[k]
                                _lt, _rt = port_lanes[k][2], port_lanes[k][3]
                                try:
                                    lane = all_lanes[lid]
                                    _left = np.array(list(lane.left_side.coords))
                                    _right = np.array(list(lane.right_side.coords))
                                    _left[0] = _lt.copy()
                                    _right[0] = _rt.copy()
                                    lane.left_side = LineString(_left)
                                    lane.right_side = LineString(_right)
                                except Exception:
                                    pass
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

    # ---- Remove road lanes disconnected from the junction network ----
    # A road lane is kept only if it is reachable (via successor links) from
    # some intersection/roundabout connector.  Fully-isolated road segments
    # (edges with no junction on either end, redundant roundabout ring edges)
    # are dropped so the HD map has no floating roads.
    conn_ids = {lid for lid in all_lanes if lid.startswith("int") or lid.startswith("ra")}
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
        isolated = {lid for lid in all_lanes if lid.startswith("e") and lid not in reached}
        for lid in isolated:
            del all_lanes[lid]
        if isolated:
            print(f"  [Map] removed {len(isolated)} isolated road lanes")

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
