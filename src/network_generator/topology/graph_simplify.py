"""Degree-2 chain simplification."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .graph_utils import adjacency_from_edges


def simplify_chains(
    coords: np.ndarray,
    edge_index: np.ndarray,
    force_keep: set[int] | None = None,
    angle_threshold_deg: float = 30.0,
    dp_epsilon_norm: float = 0.0,
    max_seg_len_norm: float = 0.04,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, int], dict[tuple[int, int], float]]:
    """Collapse degree-2 chains into single edges.

    When *angle_threshold_deg* > 0, degree-2 nodes where the turning angle
    exceeds this threshold are kept as split points — this preserves curves.

    When *dp_epsilon_norm* > 0, a second pass removes redundant curve points
    whose perpendicular deviation from the straight line between neighbours
    is below *dp_epsilon_norm* (Douglas-Peucker style).  *max_seg_len_norm*
    limits how far apart kept points can be (default 0.04 ≈ 200m on 5km map).

    Returns (simp_coords, simp_edges, simp_types, old2new_map, chain_lengths).
    ``simp_types``: 1 = junction (deg ≥ 3), 4 = endpoint (deg ≤ 1).
    ``chain_lengths``: {(simp_u, simp_v): sum_of_segment_lengths} — the
            polyline length of each simplified edge in normalised space.
    """
    N = len(coords)
    adj = adjacency_from_edges(edge_index, N)

    deg = {i: len(adj[i]) for i in range(N)}

    keep: set[int] = {i for i in range(N) if deg[i] != 2}
    if force_keep:
        keep.update(force_keep)
    if not keep:
        keep = set(range(N))

    klist = sorted(keep)
    old2new = {o: n for n, o in enumerate(klist)}
    simp_coords = [coords[i] for i in klist]  # list for dynamic append
    curve_newids: set[int] = set()  # new indices of curve-split points

    # Build simplified adjacency + track chain lengths
    simp_adj: dict[int, set[int]] = defaultdict(set)
    chain_lengths: dict[tuple[int, int], float] = {}

    def _walk_chain(start: int, first_nb: int):
        """Walk a degree-2 chain, splitting at sharp turns."""
        nonlocal simp_coords, curve_newids
        chain_nodes = [start, first_nb]
        prev, cur = start, first_nb
        while cur not in keep:
            nxt = [n for n in adj[cur] if n != prev]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            chain_nodes.append(cur)

        end = cur
        if end == start or end not in keep:
            return  # chain leads to dead end or self-loop

        seg_start = 0
        seg_len = 0.0
        for i in range(1, len(chain_nodes) - 1):
            p, c, n = chain_nodes[i - 1], chain_nodes[i], chain_nodes[i + 1]
            v1 = coords[c] - coords[p]
            v2 = coords[n] - coords[c]
            seg_len += float(np.linalg.norm(v1))

            dot = np.dot(v1, v2)
            norm = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12
            cos_angle = np.clip(dot / norm, -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_angle))

            if angle > angle_threshold_deg:
                split_node = chain_nodes[i]
                if split_node not in old2new:
                    old2new[split_node] = len(simp_coords)
                    simp_coords.append(coords[split_node])
                    curve_newids.add(old2new[split_node])

                a = old2new[chain_nodes[seg_start]]
                b = old2new[split_node]
                simp_adj[a].add(b)
                simp_adj[b].add(a)
                if a < b:
                    chain_lengths[(a, b)] = seg_len
                else:
                    chain_lengths[(b, a)] = seg_len
                seg_start = i
                seg_len = 0.0

        # Final segment
        seg_len += float(np.linalg.norm(coords[chain_nodes[-1]] - coords[chain_nodes[-2]]))
        a = old2new[chain_nodes[seg_start]]
        b = old2new[end]
        if a != b:
            simp_adj[a].add(b)
            simp_adj[b].add(a)
            if a < b:
                chain_lengths[(a, b)] = seg_len
            else:
                chain_lengths[(b, a)] = seg_len

    for s in keep:
        for nb in adj[s]:
            if nb in keep:
                a, b = old2new[s], old2new[nb]
                if a < b:
                    simp_adj[a].add(b)
                    simp_adj[b].add(a)
                    chain_lengths[(a, b)] = float(np.linalg.norm(coords[s] - coords[nb]))
                continue
            _walk_chain(s, nb)

    # ── Second pass: Douglas-Peucker cleanup ────────────────────────────
    # Walk degree-2 chains in the simplified graph and remove redundant
    # curve points whose perpendicular deviation is below dp_epsilon_norm.
    if dp_epsilon_norm > 0 and len(simp_coords) > 3:
        # Build deg-2 chain nodes (map new-id → [prev, next])
        simp_deg2 = {}
        for a in range(len(simp_coords)):
            bs = [b for b in simp_adj.get(a, set()) if b != a]
            if len(bs) == 2:
                simp_deg2[a] = bs
            elif len(bs) == 1:
                pass  # endpoint, skip
        # Collect continuous deg-2 chains
        visited_dp: set[int] = set()
        to_remove: set[int] = set()
        for start in list(simp_deg2.keys()):
            if start in visited_dp:
                continue
            # Walk forward to find chain endpoints
            chain_dp = [start]
            visited_dp.add(start)
            # Forward
            prev, cur = start, simp_deg2[start][0]
            while cur in simp_deg2 and cur not in visited_dp:
                visited_dp.add(cur)
                chain_dp.append(cur)
                nxt = [n for n in simp_deg2[cur] if n != prev]
                if not nxt:
                    break
                prev, cur = cur, nxt[0]
            # Backward
            prev, cur = start, simp_deg2[start][1]
            while cur in simp_deg2 and cur not in visited_dp:
                visited_dp.add(cur)
                chain_dp.insert(0, cur)
                nxt = [n for n in simp_deg2[cur] if n != prev]
                if not nxt:
                    break
                prev, cur = cur, nxt[0]
            if len(chain_dp) < 2:
                continue
            # Walk chain: keep endpoints always; remove middle points if
            # perpendicular deviation is below epsilon AND segments are short.
            # Uses iterative DP: for each consecutive triple, compute deviation.
            changed = True
            while changed:
                changed = False
                i = 1
                while i < len(chain_dp) - 1:
                    p = chain_dp[i - 1]
                    c = chain_dp[i]
                    n = chain_dp[i + 1]
                    # Perpendicular distance from c to line(p, n)
                    v1 = simp_coords[n] - simp_coords[p]
                    v2 = simp_coords[c] - simp_coords[p]
                    v1_norm = float(np.linalg.norm(v1))
                    if v1_norm < 1e-10:
                        # Degenerate: co-linear points, remove middle
                        to_remove.add(c)
                        chain_dp.pop(i)
                        changed = True
                        continue
                    # Project v2 onto v1
                    t = np.dot(v2, v1) / (v1_norm * v1_norm)
                    t = max(0.0, min(1.0, t))
                    proj = simp_coords[p] + t * v1
                    dev = float(np.linalg.norm(simp_coords[c] - proj))
                    # Segment lengths
                    d1 = float(np.linalg.norm(simp_coords[c] - simp_coords[p]))
                    d2 = float(np.linalg.norm(simp_coords[n] - simp_coords[c]))
                    if dev < dp_epsilon_norm and d1 < max_seg_len_norm and d2 < max_seg_len_norm:
                        to_remove.add(c)
                        chain_dp.pop(i)
                        changed = True
                    else:
                        i += 1
        if to_remove:
            # Rebuild simp_coords, old2new, simp_adj, curve_newids
            keep_ids = sorted(set(range(len(simp_coords))) - to_remove)
            remap = {o: n for n, o in enumerate(keep_ids)}
            simp_coords = np.array([simp_coords[i] for i in keep_ids], dtype=np.float32)
            curve_newids = {remap[i] for i in curve_newids if i not in to_remove}
            new_adj: dict[int, set[int]] = defaultdict(set)
            new_chain_lengths: dict[tuple[int, int], float] = {}
            for a in list(simp_adj.keys()):
                if a in to_remove:
                    continue
                na = remap.get(a)
                if na is None:
                    continue
                for b in list(simp_adj.get(a, set())):
                    if b in to_remove:
                        continue
                    nb = remap.get(b)
                    if nb is not None and na != nb:
                        new_adj[na].add(nb)
                        new_adj[nb].add(na)
                        key = (min(na, nb), max(na, nb))
                        if key not in new_chain_lengths:
                            new_chain_lengths[key] = chain_lengths.get(
                                (min(a, b), max(a, b)),
                                float(np.linalg.norm(simp_coords[na] - simp_coords[nb])),
                            )
            simp_adj = new_adj
            chain_lengths = new_chain_lengths
            old2new = {k: remap[v] for k, v in old2new.items() if v in remap and v not in to_remove}

    simp_coords = np.array(simp_coords, dtype=np.float32)

    simp_edges_list = []
    for a, bset in simp_adj.items():
        for b in bset:
            if a < b:
                simp_edges_list.append([a, b])
    simp_edges = (
        np.array(simp_edges_list, dtype=np.int64).reshape(-1, 2)
        if simp_edges_list
        else np.empty((0, 2), dtype=np.int64)
    )

    simp_types_list = []
    for o in sorted(old2new.keys()):
        nid = old2new[o]
        sd = len(simp_adj.get(nid, []))
        if nid in curve_newids:
            simp_types_list.append(2)  # curve point (deg=2 but sharp turn)
        elif sd >= 3:
            simp_types_list.append(1)  # junction
        elif sd == 2:
            simp_types_list.append(0)  # waypoint (deg=2, straight)
        else:
            simp_types_list.append(4)  # endpoint

    return (
        simp_coords,
        simp_edges,
        np.array(simp_types_list, dtype=np.int64),
        old2new,
        chain_lengths,
    )
