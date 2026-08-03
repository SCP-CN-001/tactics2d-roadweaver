# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Route coverage metrics implementation."""

from __future__ import annotations

import networkx as nx
import numpy as np


def compute_route_coverage(G: nx.Graph, n_pairs: int = 100, seed: int = 42) -> dict:
    """Random OD sampling → reachable ratio + avg shortest path length.

    Args:
        G: Road network graph.
        n_pairs: Number of random origin-destination pairs to test.
        seed: Random seed for reproducibility.

    Returns:
        dict with keys: reachable_ratio, avg_route_length, n_pairs_tested, n_reachable
    """
    if G.number_of_nodes() < 2:
        return {
            "reachable_ratio": 0.0,
            "avg_route_length": 0.0,
            "n_pairs_tested": 0,
            "n_reachable": 0,
        }

    nodes = list(G.nodes())
    rng = np.random.default_rng(seed)
    reachable = 0
    lengths = []

    for _ in range(n_pairs):
        u, v = rng.choice(nodes, size=2, replace=False)
        try:
            path = nx.shortest_path(G, source=u, target=v)
            reachable += 1
            lengths.append(len(path) - 1)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass

    return {
        "reachable_ratio": reachable / n_pairs,
        "avg_route_length": float(np.mean(lengths)) if lengths else 0.0,
        "n_pairs_tested": n_pairs,
        "n_reachable": reachable,
    }


def compute_all_pairs_reachability(G: nx.Graph) -> dict:
    """Percentage of ALL node pairs connected by a valid path.

    Unlike :func:`compute_route_coverage` (random OD sampling), this computes
    the exact fraction over every unordered node pair.  Feasible for the
    10–80 node graphs in this evaluation (max ~3200 pairs).

    Returns:
        dict with keys: reachable_ratio (all-pairs), n_pairs, n_reachable
    """
    n = G.number_of_nodes()
    if n < 2:
        return {"reachable_ratio": 0.0, "n_pairs": 0, "n_reachable": 0}

    nodes = list(G.nodes())
    reachable = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1
            try:
                if nx.has_path(G, nodes[i], nodes[j]):
                    reachable += 1
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                pass

    return {
        "reachable_ratio": reachable / total if total else 0.0,
        "n_pairs": total,
        "n_reachable": reachable,
    }
