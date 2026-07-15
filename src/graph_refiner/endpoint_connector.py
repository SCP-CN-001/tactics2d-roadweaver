"""
VX — Endpoint Connector (degree-1 only).

Connects cross-component degree-1 endpoints on raw V10 pixel skeleton:

  1. Find degree-1 endpoints, compute connected components
  2. Score cross-component endpoint pairs (distance, direction, field)
  3. A* on road field grid between best pairs (two-phase: guiding path → short segments)
  4. Connect remaining small components (deg1 or closest-pair)
  5. Chain-simplify + spatial merge → clean output graph

This runs on RAW V10 skeleton (keep_all_nodes=True) for dense waypoints.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from utils.pathfinding import (
    astar_connect_path,
    astar_grid,
    cost_map_from_road,
    line_field_support,
    nearest_road_px,
    sample_path_at_step,
)
from utils.graph_ops import (
    adjacency_from_edges,
    components_with_edges,
    endpoint_nodes,
    find_components,
    merge_close_nodes,
    simplify_chains as _simplify_chains_impl,
)


class EndpointConnector:
    """Connects cross-component degree-1 endpoints on RAW V10 skeleton."""

    def __init__(self, map_size_m: float = 2000.0,
                 max_pair_dist_m: float = 600.0):
        self.map_size_m = map_size_m
        self.max_pair_dist = max_pair_dist_m / map_size_m

    # ── Public API ──────────────────────────────────────────────────────

    def run(self, raw_graph: dict,
            road_field: np.ndarray,
            max_connections: int = 20,
            simplify: bool = True,
            connect_remaining: bool = True,
            max_remaining_m: float = 600.0,
            merge_distance: float = 0.008) -> dict:
        """Run endpoint connector on RAW V10 graph.

        Two-pass connection:
          1. Cross-component degree-1↔degree-1 pairing (A*).
          2. Remaining small components connected by deg1 or closest-pair A*.

        Args:
            raw_graph: field_to_graph output (keep_all_nodes=True).
            road_field: (H, W) V10 road probability field.
            max_connections: max cross-component connections (pass 1).
            simplify: collapse degree-2 chains in output.  Default True.
            connect_remaining: connect leftover components.  Default True.
            max_remaining_m: max distance for pass 2 (metres).  Default 600.
            merge_distance: spatial merge radius (normalised).  Default 0.008.

        Returns:
            Graph dict with ``coords``, ``edge_index``, ``node_types``,
            ``closure_edges``.
        """
        coords = np.array(raw_graph["coords"], dtype=np.float32).reshape(-1, 2)
        ei = np.array(raw_graph["edge_index"], dtype=np.int64).reshape(-1, 2)
        N = len(coords)
        H, W = road_field.shape[:2] if road_field is not None else (128, 128)

        adj = adjacency_from_edges(ei, N)

        # ── 1. Components & degree-1 endpoints ──────────────────────────
        comps = find_components(adj, N)
        # comp_of_nid: map node → component id
        comp_of_nid: Dict[int, int] = {}
        for ci, cl in enumerate(comps):
            for n in cl:
                comp_of_nid[n] = ci

        deg1_nids = endpoint_nodes(adj)
        # endpoints per component
        comp_eps: Dict[int, List[int]] = {}
        for nid in deg1_nids:
            comp_eps.setdefault(comp_of_nid[nid], []).append(nid)

        # ── 2. Cross-component candidate scoring ────────────────────────
        candidates = self._score_candidates(coords, adj, comp_of_nid, comps,
                                            comp_eps, road_field)

        # ── 3. Build cost map & helper ───────────────────────────────────
        cost_map = cost_map_from_road(road_field)

        def _road_anchor(nid: int) -> np.ndarray:
            """Snap endpoint neighbour to nearest road pixel."""
            nbrs = adj.get(nid, [])
            if nbrs:
                nb_xy = coords[nbrs[0]]
                px = int(nb_xy[0] * W)
                py = int(nb_xy[1] * H)
                rx, ry = nearest_road_px(px, py, road_field, 4)
                return np.array([rx / W, ry / H], dtype=np.float32)
            return coords[nid]

        # ── 4. Select + connect (Pass 1: deg1↔deg1) ─────────────────────
        used_eps: Set[int] = set()
        result_path_groups: List[Tuple[int, int, np.ndarray]] = []

        for score, ea, eb in candidates:
            if len(result_path_groups) >= max_connections:
                break
            if ea in used_eps or eb in used_eps:
                continue
            if comp_of_nid[ea] == comp_of_nid[eb]:
                continue

            anchor_a = _road_anchor(ea)
            anchor_b = _road_anchor(eb)

            # Full A* (guiding path)
            path_pts = astar_connect_path(anchor_a, anchor_b, road_field,
                                          cost_map, W, H, max_steps=10000)
            if path_pts is None or len(path_pts) < 3:
                continue

            result_path_groups.append((ea, eb, path_pts))
            used_eps.add(ea)
            used_eps.add(eb)

        if not result_path_groups:
            return {
                "coords": coords.copy(),
                "edge_index": ei.copy(),
                "node_types": np.zeros(N, dtype=np.int64),
                "closure_edges": np.empty((0, 2), dtype=np.int64),
                "edge_lengths_m": np.ones(len(ei), dtype=np.float32) * self.map_size_m * 0.05,
                "map_size_m": self.map_size_m,
            }

        # ── 5. Estimate native edge scale ────────────────────────────────
        edge_lengths = [float(np.linalg.norm(coords[int(u)] - coords[int(v)]))
                        for u, v in ei]
        native_step = float(np.median(edge_lengths)) if edge_lengths else 0.003
        native_step = max(0.003, min(native_step, 0.02))

        # ── 6. Build dense graph with two-phase A* insertion ─────────────
        out_c = [coords[i] for i in range(N)]
        out_e: Set[Tuple[int, int]] = set((int(u), int(v)) for u, v in ei)
        closure_pairs: List[Tuple[int, int]] = []

        for ea, eb, guiding_path in result_path_groups:
            self._insert_two_phase_path(ea, eb, guiding_path, out_c, out_e,
                                        closure_pairs, native_step, W, H,
                                        road_field, cost_map)

        # ── 6b. Pass 2: connect remaining components ────────────────────
        if connect_remaining:
            self._connect_remaining_components(
                out_c, out_e, closure_pairs, native_step, W, H,
                road_field, cost_map, max_remaining_m, max_connections)

        # ── 7. Finalize ─────────────────────────────────────────────────
        dense_c = np.array(out_c, dtype=np.float32).reshape(-1, 2)
        dense_e = np.array(list(out_e), dtype=np.int64).reshape(-1, 2) \
                  if out_e else np.empty((0, 2), dtype=np.int64)
        dense_ce = np.array(closure_pairs, dtype=np.int64).reshape(-1, 2) \
                   if closure_pairs else np.empty((0, 2), dtype=np.int64)

        return self._finalize(dense_c, dense_e, dense_ce, N, simplify,
                              merge_distance, self.map_size_m)

    # ── Candidate scoring ───────────────────────────────────────────────

    def _score_candidates(self, coords, adj, comp_of_nid, comps, comp_eps,
                          road_field) -> List[Tuple[float, int, int]]:
        """Score cross-component degree-1↔degree-1 pairs.

        Scoring combines direction alignment and road field support,
        normalised by distance.
        """
        candidates: List[Tuple[float, int, int]] = []
        comp_ids = sorted(set(comp_of_nid.values()))

        def _outward_dir(nid: int) -> np.ndarray:
            nbrs = adj.get(nid, [])
            if not nbrs:
                return np.array([0.0, 0.0])
            vec = coords[nid] - coords[nbrs[0]]
            nrm = np.linalg.norm(vec)
            return vec / nrm if nrm > 1e-8 else np.array([0.0, 0.0])

        for ci in range(len(comp_ids)):
            eps_a = comp_eps.get(comp_ids[ci], [])
            if not eps_a:
                continue
            for cj in range(ci + 1, len(comp_ids)):
                eps_b = comp_eps.get(comp_ids[cj], [])
                if not eps_b:
                    continue
                for ea in eps_a:
                    pa = coords[ea]
                    for eb in eps_b:
                        pb = coords[eb]
                        d = float(np.linalg.norm(pa - pb))
                        if d > self.max_pair_dist or d < 0.005:
                            continue
                        da = _outward_dir(ea)
                        db = _outward_dir(eb)
                        to_vec = pb - pa
                        to_u = to_vec / max(float(np.linalg.norm(to_vec)), 1e-8)
                        align = (max(0.0, float(np.dot(da, to_u)))
                                 + max(0.0, float(np.dot(db, -to_u)))) / 2.0
                        fs = line_field_support(pa, pb, road_field,
                                                 n_samples=12, r_px=3)
                        score = (align * 1.5 + fs * 1.0) / (d * 2000.0 + 10.0)
                        candidates.append((score, ea, eb))

        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates

    # ── Path insertion helpers ──────────────────────────────────────────

    def _insert_two_phase_path(self, ea: int, eb: int,
                               guiding_path: np.ndarray,
                               out_c: List, out_e: Set,
                               closure_pairs: List,
                               native_step: float, W: int, H: int,
                               road_field: np.ndarray,
                               cost_map: np.ndarray) -> None:
        """Two-phase A* insertion for a single closure pair.

        Phase 2a: sample guiding path at native_step → waypoints.
        Phase 2b: short A* per adjacent waypoint pair with node insertion.
        Only one (ea, eb) entry is added to *closure_pairs*.
        """
        wpts = sample_path_at_step(guiding_path, native_step)

        prev_nid = ea
        for wi in range(1, len(wpts)):
            sx = int(wpts[wi - 1][0] * W)
            sy = int(wpts[wi - 1][1] * H)
            gx = int(wpts[wi][0] * W)
            gy = int(wpts[wi][1] * H)
            is_last = (wi == len(wpts) - 1)

            seg = astar_grid(sx, sy, gx, gy, cost_map, max_steps=5000)

            if seg is None or len(seg) < 2:
                target = eb if is_last else len(out_c)
                out_e.add((prev_nid, target))
                if not is_last:
                    out_c.append(wpts[wi].reshape(-1))
                    prev_nid = target
                continue

            # Walk short A* path, placing nodes at native_step intervals.
            # Only intermediates — no segment-level closure entries.
            cur_nid = prev_nid
            cum = 0.0
            for pk in range(1, len(seg)):
                d = float(np.linalg.norm(seg[pk] - seg[pk - 1]))
                cum += d
                if cum >= native_step or pk == len(seg) - 1:
                    if pk == len(seg) - 1 and is_last:
                        out_e.add((cur_nid, eb))
                    else:
                        nid = len(out_c)
                        out_c.append(seg[pk].reshape(-1))
                        out_e.add((cur_nid, nid))
                        cur_nid = nid
                    cum = 0.0
            if not is_last:
                prev_nid = cur_nid

        closure_pairs.append((ea, eb))

    def _simple_astar_insert(self, src_nid: int, tgt_nid: int,
                             out_c: List, out_e: Set,
                             native_step: float, W: int, H: int,
                             road_field: np.ndarray,
                             cost_map: np.ndarray,
                             max_steps: int = 10000) -> bool:
        """Single A* between two nodes; insert intermediate nodes.

        Updates *out_c* and *out_e* in place.  Returns True on success.
        """
        src_coord = out_c[src_nid]
        tgt_coord = out_c[tgt_nid]

        path_pts = astar_connect_path(src_coord, tgt_coord, road_field,
                                      cost_map, W, H, max_steps)
        if path_pts is None or len(path_pts) < 3:
            return False

        prev_rid = src_nid
        cum = 0.0
        for pk in range(1, len(path_pts)):
            d = float(np.linalg.norm(path_pts[pk] - path_pts[pk - 1]))
            cum += d
            if cum >= native_step or pk == len(path_pts) - 1:
                if pk == len(path_pts) - 1:
                    out_e.add((prev_rid, tgt_nid))
                else:
                    nid = len(out_c)
                    out_c.append(path_pts[pk].reshape(-1))
                    out_e.add((prev_rid, nid))
                    prev_rid = nid
                cum = 0.0
        return True

    # ── Pass 2: remaining components ────────────────────────────────────

    def _connect_remaining_components(self, out_c: List, out_e: Set,
                                      closure_pairs: List,
                                      native_step: float, W: int, H: int,
                                      road_field: np.ndarray,
                                      cost_map: np.ndarray,
                                      max_remaining_m: float,
                                      max_connections: int) -> None:
        """Connect small components to the largest graph.

        Two sub-passes:
          a) deg1 endpoints in small components → nearest node in largest.
          b) components with no deg1 (e.g. rings) → closest node pair.
        """
        cur_adj = adjacency_from_edges(
            np.array(list(out_e), dtype=np.int64).reshape(-1, 2),
            len(out_c))

        def _components():
            return find_components(cur_adj, len(out_c))

        max_rem = max_remaining_m / self.map_size_m
        budget = max_connections * 2

        # Pass 2a: deg1 endpoints
        for _ in range(budget):
            cur_comps = _components()
            if len(cur_comps) <= 1:
                return
            largest = max(cur_comps, key=len)

            best_score = -1.0
            best_src = best_tgt = -1
            for comp in cur_comps:
                if comp is largest:
                    continue
                for n in comp:
                    if len(cur_adj.get(n, [])) != 1:
                        continue
                    pn = out_c[n]
                    for m in largest:
                        d = np.linalg.norm(pn - out_c[m])
                        if d > max_rem:
                            continue
                        score = 1.0 / (d + 0.01)
                        if score > best_score:
                            best_score = score
                            best_src, best_tgt = n, m
            if best_src < 0:
                break
            ok = self._simple_astar_insert(best_src, best_tgt, out_c, out_e,
                                           native_step, W, H, road_field,
                                           cost_map)
            if ok:
                # Rebuild adjacency after graph change
                cur_adj = adjacency_from_edges(
                    np.array(list(out_e), dtype=np.int64).reshape(-1, 2),
                    len(out_c))

        # Pass 2b: closest-node-pair for remaining (no deg1)
        for _ in range(budget):
            cur_comps = _components()
            if len(cur_comps) <= 1:
                return
            largest = max(cur_comps, key=len)

            best_d = float('inf')
            best_src = best_tgt = -1
            for comp in cur_comps:
                if comp is largest:
                    continue
                for cn in comp:
                    pcn = out_c[cn]
                    for ln in largest:
                        d = np.linalg.norm(pcn - out_c[ln])
                        if d < best_d:
                            best_d = d
                            best_src, best_tgt = cn, ln
            if best_d > max_rem:
                break
            ok = self._simple_astar_insert(best_src, best_tgt, out_c, out_e,
                                           native_step, W, H, road_field,
                                           cost_map)
            if ok:
                cur_adj = adjacency_from_edges(
                    np.array(list(out_e), dtype=np.int64).reshape(-1, 2),
                    len(out_c))

    # ── Output post-processing ──────────────────────────────────────────

    def _finalize(self, dense_c: np.ndarray, dense_e: np.ndarray,
                  dense_ce: np.ndarray, n_original: int,
                  simplify: bool, merge_distance: float,
                  map_size_m: float) -> dict:
        """Chain-simplify, optionally merge, and return final graph."""
        if not simplify or len(dense_c) == 0:
            return {
                "coords": dense_c,
                "edge_index": dense_e,
                "node_types": np.zeros(len(dense_c), dtype=np.int64),
                "closure_edges": dense_ce,
                "edge_lengths_m": np.ones(len(dense_e), dtype=np.float32) * map_size_m * 0.05,
                "map_size_m": map_size_m,
            }

        # Chain simplification with original closure endpoints force-kept
        force_keep: Set[int] = set()
        for a, b in dense_ce:
            if int(a) < n_original:
                force_keep.add(int(a))
            if int(b) < n_original:
                force_keep.add(int(b))
        simp_c, simp_e, simp_nt, old2new, chain_lengths = _simplify_chains_impl(
            dense_c, dense_e, force_keep)

        # Remap closure_edges
        simp_ce_list: List[Tuple[int, int]] = []
        for a, b in dense_ce:
            sa = old2new.get(int(a))
            sb = old2new.get(int(b))
            if sa is not None and sb is not None and sa != sb:
                simp_ce_list.append((sa, sb))
        simp_ce = np.array(simp_ce_list, dtype=np.int64).reshape(-1, 2) \
                  if simp_ce_list else np.empty((0, 2), dtype=np.int64)

        # Build edge_lengths_m from chain lengths (normalised → metres)
        el_norm = np.ones(len(simp_e), dtype=np.float32) * 0.05
        for ei, (u, v) in enumerate(simp_e):
            key = (int(u), int(v))
            if key in chain_lengths:
                el_norm[ei] = chain_lengths[key]
            elif (int(v), int(u)) in chain_lengths:
                el_norm[ei] = chain_lengths[(int(v), int(u))]
        edge_lengths_m = el_norm * map_size_m

        # Spatial merge — may change edge set, so recompute lengths after
        if merge_distance > 0 and len(simp_c) > 0:
            simp_c, simp_e, simp_nt, simp_ce, merge_map = merge_close_nodes(
                simp_c, simp_e, simp_nt, merge_distance, closure_edges=simp_ce)
            edge_lengths_m = np.ones(len(simp_e), dtype=np.float32) * map_size_m * 0.05
            for ei, (u, v) in enumerate(simp_e):
                pu, pv = simp_c[int(u)], simp_c[int(v)]
                edge_lengths_m[ei] = float(np.linalg.norm(pu - pv)) * map_size_m

        return {
            "coords": simp_c,
            "edge_index": simp_e,
            "node_types": simp_nt,
            "closure_edges": simp_ce,
            "edge_lengths_m": edge_lengths_m,
            "map_size_m": map_size_m,
        }
