# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""OSM reference statistics implementation."""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np

from .topology import contract_degree2_nodes

_REF_DEG_CACHE: float | None = None


def load_osm_reference_degree(data_path: str | Path | None = None, use_cache: bool = True) -> float:
    """Load average node degree from OSM training data after degree-2 contraction.

    The skeleton graphs are contracted (degree-2 road midpoints removed) so that
    the reference operates at the same intersection-level granularity as
    MetaDrive / RoadGen / HDMapGen evaluation graphs.

    The first call computes the value and caches it; subsequent calls return the
    cached value instantly.  The cache also persists to a JSON sidecar file.

    Falls back to Boeing (2019) global average of 2.84 if data unavailable.
    """
    global _REF_DEG_CACHE

    if use_cache and _REF_DEG_CACHE is not None:
        return _REF_DEG_CACHE

    # Try persistent cache file
    repo_root = Path(__file__).resolve().parent.parent.parent
    cache_path = repo_root / "runtimes" / ".osm_ref_degree_contracted.json"
    if use_cache and cache_path.exists():
        try:
            _REF_DEG_CACHE = json.loads(cache_path.read_text())["ref_deg"]
            return _REF_DEG_CACHE
        except Exception:
            pass

    if data_path is None:
        data_path = repo_root / "data" / "urban_prior" / "5km" / "splits" / "train.parquet"

    # Compute from scratch — subsample to first 1000 graphs for speed;
    # 1000 is enough for a stable average (std < 0.02 across OSM samples).
    try:
        import pandas as pd

        df = pd.read_parquet(data_path)
        degs = []
        n_processed = 0
        for g in df["skeleton_graph_json"]:
            if n_processed >= 1000:
                break
            graph = json.loads(g)
            nodes = graph.get("nodes", [])
            edges = graph.get("edges", [])
            if not nodes:
                continue

            # Build graph from skeleton JSON
            G = nx.Graph()
            for n in nodes:
                nid = n.get("id")
                if nid is not None:
                    G.add_node(nid)
            for e in edges:
                u, v = e.get("source"), e.get("target")
                if u is not None and v is not None:
                    G.add_edge(u, v)

            # Contract degree-2 nodes to match intersection-level granularity
            G = contract_degree2_nodes(G)

            if G.number_of_nodes() > 0:
                degs.append(sum(d for _, d in G.degree()) / G.number_of_nodes())
            n_processed += 1

        result = float(np.mean(degs)) if degs else 2.84

        # Persist cache
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps({"ref_deg": result}))
        except Exception:
            pass

        _REF_DEG_CACHE = result
        return result
    except Exception:
        return 2.84
