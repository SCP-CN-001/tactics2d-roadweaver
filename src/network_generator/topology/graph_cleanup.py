"""Post-growth graph cleanup operations."""

from __future__ import annotations

import networkx as nx
import numpy as np
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

from utils.geometry import segment_intersection as _segment_intersection

from .graph_utils import build_nx

# ── Dead-end pruning ─────────────────────────────────────────────────


def prune_dead_ends(
    coords: np.ndarray, edge_index: np.ndarray, max_chain_m: float, map_size_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """Remove short dead-end chains by walking from degree-1 tips.

    Walks along the chain removing nodes until the cumulative length
    exceeds ``max_chain_m`` or a junction is reached.
    """
    if len(coords) < 5:
        return coords, edge_index
    G = build_nx(coords, edge_index)
    changed = True
    while changed:
        changed = False
        for n in list(G.nodes()):
            if n not in G or G.degree(n) != 1:
                continue
            tip = n
            cum = 0.0
            while tip in G and G.degree(tip) == 1 and cum < max_chain_m / map_size_m:
                nbs = list(G.neighbors(tip))
                if not nbs:
                    break
                nb = nbs[0]
                cum += float(np.linalg.norm(G.nodes[tip]["pos"] - G.nodes[nb]["pos"]))
                if cum < max_chain_m / map_size_m:
                    G.remove_node(tip)
                    changed = True
                    tip = nb
                else:
                    break
    nodes = sorted(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    return (
        np.array([G.nodes[n]["pos"] for n in nodes]),
        np.array([[idx[u], idx[v]] for u, v in G.edges()], dtype=np.int64),
    )


# ── Largest connected component ──────────────────────────────────────


def keep_lcc(coords: np.ndarray, edge_index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Retain only the largest connected component."""
    if len(coords) < 5:
        return coords, edge_index
    G = build_nx(coords, edge_index)
    components = list(nx.connected_components(G))
    largest = max(components, key=len)
    if len(largest) == len(coords):
        return coords, edge_index
    nodes = sorted(largest)
    idx = {n: i for i, n in enumerate(nodes)}
    return (
        np.array([G.nodes[n]["pos"] for n in nodes]),
        np.array([[idx[u], idx[v]] for u, v in G.edges() if u in idx and v in idx], dtype=np.int64),
    )


# ── Sharp-angle cleanup ──────────────────────────────────────────────


def clean_sharp_angles(
    coords: np.ndarray, edge_index: np.ndarray, min_deg: float = 15.0
) -> tuple[np.ndarray, np.ndarray]:
    """At each degree ≥ 3 junction, remove the shorter edge when two
    consecutive edges form an angle smaller than ``min_deg``."""
    if len(edge_index) < 3:
        return coords, edge_index
    changed = True
    c, ei = coords.copy(), edge_index.copy()
    while changed:
        changed = False
        G = build_nx(c, ei)
        to_remove = set()
        for n in range(len(c)):
            if G.degree(n) < 3:
                continue
            nbs = list(G.neighbors(n))
            if len(nbs) < 2:
                continue
            dirs = [
                (
                    float(
                        np.arctan2(
                            G.nodes[nb]["pos"][1] - G.nodes[n]["pos"][1],
                            G.nodes[nb]["pos"][0] - G.nodes[n]["pos"][0],
                        )
                    ),
                    nb,
                )
                for nb in nbs
            ]
            dirs.sort()
            ds = [p[0] for p in dirs]
            ns = [p[1] for p in dirs]
            for i in range(len(ds)):
                diff = abs(ds[(i + 1) % len(ds)] - ds[i])
                if diff > np.pi:
                    diff = 2 * np.pi - diff
                if diff < np.radians(min_deg):
                    l1 = float(np.linalg.norm(G.nodes[n]["pos"] - G.nodes[ns[i]]["pos"]))
                    l2 = float(
                        np.linalg.norm(G.nodes[n]["pos"] - G.nodes[ns[(i + 1) % len(ns)]]["pos"])
                    )
                    shorter = tuple(sorted((n, ns[i] if l1 < l2 else ns[(i + 1) % len(ns)])))
                    to_remove.add(shorter)
        if to_remove:
            keep = [(u, v) for u, v in ei if tuple(sorted([int(u), int(v)])) not in to_remove]
            G2 = build_nx(c, keep)
            orphaned = [n for n in range(len(c)) if G2.degree(n) == 0]
            c = np.delete(c, orphaned, axis=0)
            ei = np.array(
                [
                    [u - sum(1 for o in orphaned if o < u), v - sum(1 for o in orphaned if o < v)]
                    for u, v in keep
                ],
                dtype=np.int64,
            )
            changed = True
    return c, ei


# ── Endpoint snapping ────────────────────────────────────────────────


def snap_endpoints(
    coords: np.ndarray, edge_index: np.ndarray, map_size_m: float, snap_dist_m: float = 50.0
) -> tuple[np.ndarray, np.ndarray]:
    """Snap degree-1 endpoints to the nearest road edge within ``snap_dist_m``."""
    if len(coords) < 5:
        return coords, edge_index
    G = build_nx(coords, edge_index)
    changed = True
    while changed:
        changed = False
        for n in list(G.nodes()):
            if n not in G or G.degree(n) != 1:
                continue
            pn = G.nodes[n]["pos"]
            best_dist = float("inf")
            best_target = None
            for u, v in G.edges():
                if u == n or v == n:
                    continue
                pu = G.nodes[u]["pos"]
                pv = G.nodes[v]["pos"]
                seg = pv - pu
                t = np.dot(pn - pu, seg) / max(np.dot(seg, seg), 1e-10)
                t = np.clip(t, 0, 1)
                proj = pu + t * seg
                d = float(np.linalg.norm(pn - proj))
                if d < best_dist:
                    best_dist = d
                    best_target = (u, v, proj)
            if best_target is not None and best_dist < snap_dist_m / map_size_m:
                u, v, proj = best_target
                nid = max(G.nodes()) + 1
                G.add_node(nid, pos=proj)
                G.remove_edge(u, v)
                G.add_edge(u, nid)
                G.add_edge(nid, v)
                G.add_edge(nid, n)
                changed = True
    nodes = sorted(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    return (
        np.array([G.nodes[n]["pos"] for n in nodes]),
        np.array([[idx[u], idx[v]] for u, v in G.edges()], dtype=np.int64),
    )


# ── Edge crossing fix ────────────────────────────────────────────────


def fix_edge_crossings(coords, edge_index, geometries, map_size_m):
    """Add intersection nodes at every geometric edge crossing.

    One-pass variant: crossings are detected among the *original* edges only
    (spatial-index pruned), then all splits are applied together via
    ``shapely.ops.split``.  This bounds the work to O(E^2) worst case and
    avoids the explosive re-scan of newly split edges.  Mirrors the semantics
    of :func:`fix_growth_crossings`.
    """
    if len(edge_index) < 2:
        return coords, edge_index, geometries

    c = [np.asarray(p, dtype=float) for p in coords]
    ei = [(int(u), int(v)) for u, v in edge_index]
    geoms = [np.asarray(g, dtype=float) for g in geometries]

    def _line(idx):
        g = geoms[idx]
        if len(g) >= 2:
            return LineString(g)
        u, v = ei[idx]
        return LineString([c[u], c[v]])

    lines = [_line(i) for i in range(len(ei))]
    tree = STRtree(lines)

    # Collect crossing points per edge: {edge_idx: set of (x, y) tuples}
    splits: dict[int, set[tuple]] = {}
    added = 0
    for i in range(len(ei)):
        u1, v1 = ei[i]
        g1 = geoms[i]
        for j in tree.query(lines[i]):
            j = int(j)
            if j <= i:
                continue
            u2, v2 = ei[j]
            if len({u1, v1} & {u2, v2}) > 0:
                continue
            g2 = geoms[j]
            for k in range(len(g1) - 1):
                p1, p2 = g1[k], g1[k + 1]
                for l in range(len(g2) - 1):
                    q1, q2 = g2[l], g2[l + 1]
                    intersection_pt = _segment_intersection(p1, p2, q1, q2)
                    if intersection_pt is not None:
                        splits.setdefault(i, set()).add(tuple(np.round(intersection_pt, 6)))
                        splits.setdefault(j, set()).add(tuple(np.round(intersection_pt, 6)))
                        added += 1
                        break  # a segment of g1 can only cross g2 once

    if added == 0:
        return coords, edge_index, geometries

    # Global point -> node id map (shared across the two crossing edges).
    node_of_pt: dict[tuple, int] = {tuple(np.round(p, 6)): int(n) for n, p in enumerate(c)}
    new_nodes: list[np.ndarray] = []

    def _node(p_tuple):
        if p_tuple in node_of_pt:
            return node_of_pt[p_tuple]
        nid = len(c) + len(new_nodes)
        node_of_pt[p_tuple] = nid
        new_nodes.append(np.asarray(p_tuple, dtype=float))
        return nid

    out_c = list(c)
    out_ei: list[tuple[int, int]] = []
    out_geoms: list[np.ndarray] = []

    for idx in range(len(ei)):
        u, v = ei[idx]
        if idx not in splits:
            out_ei.append((u, v))
            out_geoms.append(geoms[idx])
            continue
        g = geoms[idx]
        if len(g) < 2:
            g = np.array([c[u], c[v]])
        # Cumulative arc length along the polyline.
        seg_len = np.sqrt(((g[1:] - g[:-1]) ** 2).sum(axis=1))
        cum = np.concatenate([[0], np.cumsum(seg_len)])
        total = float(cum[-1])
        if total < 1e-12:
            out_ei.append((u, v))
            out_geoms.append(g)
            continue

        # Sort crossing points by arc-length position, drop near-duplicates.
        cuts = sorted(
            (lines[idx].project(Point(p)) / total, np.asarray(p, dtype=float)) for p in splits[idx]
        )
        deduped: list[tuple[float, np.ndarray]] = []
        for f, p in cuts:
            if f < 1e-3 or f > 1.0 - 1e-3:
                # Crossing lands at (or right next to) this edge's own endpoint.
                # Splitting there would create a zero-length sub-edge with a
                # degenerate geometry collapsed to the endpoint node.
                continue
            if not deduped or f - deduped[-1][0] > 1e-4:
                deduped.append((f, p))

        # Cut the polyline at each crossing point by interpolating segments.
        prev_node = u
        prev_pt = np.asarray(c[u], dtype=float)
        prev_cum = 0.0
        for f, p in deduped:
            target_cum = f * total
            sub_pts = [prev_pt.copy()]
            for k in range(1, len(g)):
                if cum[k] <= prev_cum + 1e-12:
                    continue
                if cum[k - 1] >= target_cum - 1e-12:
                    break
                seg_start = max(prev_cum, cum[k - 1])
                seg_end = min(target_cum, cum[k])
                if seg_end <= seg_start:
                    continue
                t = (seg_start - cum[k - 1]) / (cum[k] - cum[k - 1] + 1e-12)
                sub_pts.append(g[k - 1] + t * (g[k] - g[k - 1]))
            sub_pts.append(p)  # endpoint = the (rounded) crossing point
            nid = _node(tuple(np.round(p, 6)))
            out_ei.append((prev_node, nid))
            out_geoms.append(np.array(sub_pts))
            prev_node = nid
            prev_pt = p
            prev_cum = target_cum

        # Final segment to v.  prev_pt already lies at arc-length prev_cum; append
        # every subsequent geometry vertex and force the endpoint to be v exactly.
        sub_pts = [prev_pt.copy()]
        for k in range(1, len(g)):
            if cum[k] > prev_cum + 1e-12:
                sub_pts.append(g[k].copy())
        if np.linalg.norm(np.asarray(sub_pts[-1]) - np.asarray(c[v], dtype=float)) > 1e-9:
            sub_pts.append(np.asarray(c[v], dtype=float))
        if len(sub_pts) >= 2:
            out_ei.append((prev_node, v))
            out_geoms.append(np.array(sub_pts))

    out_c = out_c + new_nodes
    if added:
        print(f"  [FixCrossings] {added} crossing nodes added")
    return np.array(out_c), np.array(out_ei, dtype=np.int64), out_geoms


# ── Parallel road cleanup ────────────────────────────────────────────


def clean_parallel_roads(
    coords, edge_index, geometries, map_size_m, angle_deg: float = 20.0, max_dist_m: float = 30.0
):
    """Remove near-parallel roads at a junction whose endpoints are very close."""
    if len(edge_index) < 3:
        return coords, edge_index, geometries
    G = build_nx(coords, edge_index)
    max_norm = max_dist_m / map_size_m
    to_remove = set()
    for n in range(len(coords)):
        if G.degree(n) < 2:
            continue
        nbs = list(G.neighbors(n))
        dirs = [
            (
                float(
                    np.arctan2(
                        G.nodes[nb]["pos"][1] - G.nodes[n]["pos"][1],
                        G.nodes[nb]["pos"][0] - G.nodes[n]["pos"][0],
                    )
                ),
                nb,
            )
            for nb in nbs
        ]
        dirs.sort()
        for i in range(len(dirs)):
            d1, nb1 = dirs[i]
            d2, nb2 = dirs[(i + 1) % len(dirs)]
            diff = abs(d2 - d1)
            if diff > np.pi:
                diff = 2 * np.pi - diff
            if diff < np.radians(angle_deg):
                dist = float(np.linalg.norm(G.nodes[nb1]["pos"] - G.nodes[nb2]["pos"]))
                if dist < max_norm:
                    l1 = float(np.linalg.norm(G.nodes[n]["pos"] - G.nodes[nb1]["pos"]))
                    l2 = float(np.linalg.norm(G.nodes[n]["pos"] - G.nodes[nb2]["pos"]))
                    to_remove.add(tuple(sorted((n, nb1 if l1 < l2 else nb2))))
    if not to_remove:
        return coords, edge_index, geometries
    keep_idx = [
        j for j, (u, v) in enumerate(edge_index) if tuple(sorted([int(u), int(v)])) not in to_remove
    ]
    keep_ei = edge_index[keep_idx]
    keep_geoms = [geometries[j] for j in keep_idx]
    G2 = build_nx(coords, keep_ei)
    orphaned = [n for n in range(len(coords)) if G2.degree(n) == 0]
    if orphaned:
        c = np.delete(coords, orphaned, axis=0)
        keep_ei = np.array(
            [
                [u - sum(1 for o in orphaned if o < u), v - sum(1 for o in orphaned if o < v)]
                for u, v in keep_ei
            ],
            dtype=np.int64,
        )
        return c, keep_ei, keep_geoms
    return coords.copy(), keep_ei, keep_geoms
