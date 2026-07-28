"""
Post-growth graph cleanup: dead-end pruning, LCC, angle cleanup, endpoint snapping,
edge crossing fixing, and parallel road removal.

All functions operate on normalized [0, 1] coordinates unless noted.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

# ── Building blocks ──────────────────────────────────────────────────


def build_nx(coords: np.ndarray, edge_index: np.ndarray) -> nx.Graph:
    """Build an ``nx.Graph`` with ``pos`` attributes from coords + edges."""
    G = nx.Graph()
    for i in range(len(coords)):
        G.add_node(i, pos=coords[i].copy())
    G.add_edges_from(edge_index)
    return G


# ── Dead-end pruning ─────────────────────────────────────────────────


def prune_dead_ends(
    coords: np.ndarray, edge_index: np.ndarray, max_chain_m: float, map_m: float
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
            while tip in G and G.degree(tip) == 1 and cum < max_chain_m / map_m:
                nbs = list(G.neighbors(tip))
                if not nbs:
                    break
                nb = nbs[0]
                cum += float(np.linalg.norm(G.nodes[tip]["pos"] - G.nodes[nb]["pos"]))
                if cum < max_chain_m / map_m:
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
    coords: np.ndarray, edge_index: np.ndarray, map_m: float, snap_dist_m: float = 50.0
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
            if best_target is not None and best_dist < snap_dist_m / map_m:
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


def _segment_intersection(p1, p2, q1, q2):
    """Return the intersection point of two line segments, or ``None``."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = q1
    x4, y4 = q2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
    if 0 < t < 1 and 0 < u < 1:
        return np.array([x1 + t * (x2 - x1), y1 + t * (y2 - y1)])
    return None


def fix_edge_crossings(coords, edge_index, geometries, map_m):
    """Add intersection nodes at every geometric edge crossing.

    Correctly handles geometry splitting so each new edge's geometry
    traces the full path from its start node to its end node.
    """
    if len(edge_index) < 2:
        return coords, edge_index, geometries
    added = 0
    max_n = len(coords)
    c = [c.copy() for c in coords] if isinstance(coords, np.ndarray) else list(coords)
    ei = [(int(u), int(v)) for u, v in edge_index]
    geoms = list(geometries) if isinstance(geometries, list) else list(geometries)

    i = 0
    while i < len(ei):
        j = i + 1
        while j < len(ei):
            u1, v1 = int(ei[i][0]), int(ei[i][1])
            u2, v2 = int(ei[j][0]), int(ei[j][1])
            if len({u1, v1} & {u2, v2}) > 0:
                j += 1
                continue

            g1 = geoms[i] if i < len(geoms) and len(geoms[i]) >= 2 else np.array([c[u1], c[v1]])
            g2 = geoms[j] if j < len(geoms) and len(geoms[j]) >= 2 else np.array([c[u2], c[v2]])
            found = False
            for k in range(len(g1) - 1):
                p1, p2 = g1[k], g1[k + 1]
                for l in range(len(g2) - 1):
                    q1, q2 = g2[l], g2[l + 1]
                    inter = _segment_intersection(p1, p2, q1, q2)
                    if inter is not None:
                        nid = max_n + added
                        c.append(inter)

                        # Split edge i:  [u1, v1] → [u1, nid] + [nid, v1]
                        ei[i] = (u1, nid)
                        ei.append((nid, v1))
                        geoms[i] = np.vstack([g1[: k + 1], inter.reshape(1, -1)])
                        geoms.append(np.vstack([inter.reshape(1, -1), g1[k + 1 :]]))

                        # Split edge j:  [u2, v2] → [u2, nid] + [nid, v2]
                        ei[j] = (u2, nid)
                        ei.append((nid, v2))
                        geoms[j] = np.vstack([g2[: l + 1], inter.reshape(1, -1)])
                        geoms.append(np.vstack([inter.reshape(1, -1), g2[l + 1 :]]))

                        added += 1
                        found = True
                        break
                if found:
                    break
            if found:
                # Recheck from current i with new crossings
                pass
            j += 1
        i += 1

    return np.array(c), np.array(ei, dtype=np.int64), geoms


# ── Parallel road cleanup ────────────────────────────────────────────


def clean_parallel_roads(
    coords, edge_index, geometries, map_m, angle_deg: float = 20.0, max_dist_m: float = 30.0
):
    """Remove near-parallel roads at a junction whose endpoints are very close."""
    if len(edge_index) < 3:
        return coords, edge_index, geometries
    G = build_nx(coords, edge_index)
    max_norm = max_dist_m / map_m
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


def remove_short_edges(coords, edge_index, geometries, min_len):
    """Remove edges shorter than *min_len* (in normalised [0, 1] space).

    Returns filtered (coords, edge_index, geometries). Orphan nodes are
    removed and indices are remapped.
    """
    keep = [True] * len(edge_index)
    for j, (u, v) in enumerate(edge_index):
        pts = (
            geometries[j]
            if j < len(geometries) and len(geometries[j]) >= 2
            else np.array([coords[int(u)], coords[int(v)]])
        )
        length = sum(np.linalg.norm(pts[k + 1] - pts[k]) for k in range(len(pts) - 1))
        if length < min_len:
            keep[j] = False
    keep_ei = edge_index[keep]
    keep_geoms = [g for g, k in zip(geometries, keep) if k]
    # Remove orphaned nodes
    used = set()
    for u, v in keep_ei:
        used.add(int(u))
        used.add(int(v))
    orphaned = sorted(set(range(len(coords))) - used)
    if not orphaned:
        return coords, keep_ei, keep_geoms
    c = np.delete(coords, orphaned, axis=0)
    keep_ei = np.array(
        [
            [u - sum(1 for o in orphaned if o < u), v - sum(1 for o in orphaned if o < v)]
            for u, v in keep_ei
        ],
        dtype=np.int64,
    )
    return c, keep_ei, keep_geoms
