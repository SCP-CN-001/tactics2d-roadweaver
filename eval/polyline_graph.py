"""
Polyline → binary mask → skeleton graph extraction.

Shared pipeline for all baselines (RoadGen, MetaDrive, HDMapGen).
Renders road polylines as a binary mask at high resolution, skeletonizes,
and extracts a clean intersection-level graph with coordinates.

**No short-branch pruning** is applied — all topological features are
preserved so evaluation metrics reflect the generated road network faithfully.

Degree-2 contraction (collapsing intermediate skeleton waypoints) **is**
applied by default so the output is at the same intersection-level granularity
as MetaDrive/HDMapGen.  To inspect the raw skeleton instead, pass
``contract_degree2=False``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np

# matplotlib imported lazily in save_vis() to avoid backend issues

# ── Ensure src/ is on the path for raster_to_graph ──────────────────────
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from network_generator.topology.raster_to_graph import field_to_graph  # noqa: E402

# ── Degree-2 contraction (same logic as eval/metrics.py, local copy
#    to avoid circular imports) ───────────────────────────────────────────


def _contract_degree2_nodes(G_raw: nx.Graph) -> nx.Graph:
    """Contract all degree-2 skeleton waypoints to intersection-level nodes.

    Every skeleton pixel becomes a graph node in the raw output from
    ``field_to_graph``.  This collapses contiguous chains of degree-2
    pixels so the resulting graph has nodes only at junctions (deg != 2)
    and dead-ends (deg == 1).

    This is **not** pruning — dead-end branches are preserved, only
    intermediate waypoint pixels are removed.
    """
    G = nx.Graph()

    # Keep only nodes with degree != 2
    keep = {node for node, deg in G_raw.degree() if deg != 2}

    for node in keep:
        G.add_node(node, **{k: v for k, v in G_raw.nodes[node].items()})

    # Walk through degree-2 chains
    visited_edges: set[tuple[int, int]] = set()

    for start in keep:
        for neighbor in G_raw.neighbors(start):
            edge_key = tuple(sorted((start, neighbor)))
            if edge_key in visited_edges:
                continue

            # Walk through degree-2 nodes
            prev, curr = start, neighbor
            while curr in G_raw and G_raw.degree(curr) == 2:
                nbrs = [n for n in G_raw.neighbors(curr) if n != prev]
                if not nbrs:
                    break
                prev, curr = curr, nbrs[0]

            end = curr

            if start != end:
                edge_key = tuple(sorted((start, end)))
                if edge_key not in visited_edges:
                    G.add_edge(start, end)
                    visited_edges.add(edge_key)

    return G


def _merge_nearby_nodes(G: nx.Graph, distance: float) -> nx.Graph:
    """Merge graph nodes that are within *distance* of each other.

    Skeletonisation of dense road networks can produce multiple nearby
    junction pixels at the same physical intersection.  This function
    clusters those pixels and replaces each cluster with a single node
    at the centroid of its members.

    This is **not** topology pruning — all edges are preserved and
    dead-end branches are kept.
    """
    if distance <= 0 or G.number_of_nodes() < 2:
        return G

    coords = np.array([G.nodes[n].get("coords", np.zeros(2)) for n in G.nodes()])
    n = len(coords)

    # Build proximity graph — edge if distance < threshold
    prox = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(coords[i] - coords[j])
            if d < distance:
                prox[i].add(j)
                prox[j].add(i)

    # Find connected components in the proximity graph (clusters)
    visited = set()
    clusters: list[list[int]] = []
    for i in range(n):
        if i in visited:
            continue
        cluster = []
        stack = [i]
        while stack:
            v = stack.pop()
            if v in visited:
                continue
            visited.add(v)
            cluster.append(v)
            for nb in prox[v]:
                if nb not in visited:
                    stack.append(nb)
        clusters.append(cluster)

    # Single-node clusters stay as-is
    singletons = [c for c in clusters if len(c) == 1]
    multi = [c for c in clusters if len(c) > 1]

    if not multi:
        return G

    node_ids = list(G.nodes())
    old_to_new: dict[int, int] = {}
    new_G = nx.Graph()

    # Keep singletons
    for c in singletons:
        old = node_ids[c[0]]
        old_to_new[old] = old
        new_G.add_node(old, **{k: v for k, v in G.nodes[old].items()})

    # Merge multi-node clusters
    next_id = max(node_ids) + 1 if node_ids else 0
    for cluster in multi:
        cluster_ids = [node_ids[i] for i in cluster]
        centroid = np.mean([G.nodes[nid]["coords"] for nid in cluster_ids], axis=0)
        new_id = next_id
        next_id += 1
        old_to_new.update({oid: new_id for oid in cluster_ids})
        new_G.add_node(new_id, coords=centroid)

    # Rewire edges
    for u, v in G.edges():
        nu, nv = old_to_new.get(u, u), old_to_new.get(v, v)
        if nu != nv:
            new_G.add_edge(nu, nv)

    return new_G


# ═══════════════════════════════════════════════════════════════════════════
#  Rendering
# ═══════════════════════════════════════════════════════════════════════════


def polylines_to_binary_mask(
    polylines: list[np.ndarray],
    resolution: int = 1024,
    padding_ratio: float = 0.05,
    line_width: int = 1,
) -> tuple[np.ndarray, dict[str, float]]:
    """Render road polylines as a binary mask.

    Steps:
      1. Compute bounding box of all polylines.
      2. Add *padding_ratio* padding (fraction of extent).
      3. Draw each polyline segment point-by-point into a
         ``resolution × resolution`` mask.

    Args:
        polylines: List of ``(N_i, 2)`` arrays in **any** coordinate space
                   (meters, RoadGen units, etc.).
        resolution: Output mask size (``resolution × resolution``).
        padding_ratio: Fraction of the data extent to add as padding around
                       the bounding box — prevents edge clipping.

    Returns:
        mask: ``(resolution, resolution)`` uint8 binary mask (0/255).
        norm: Dict with ``x_min``, ``y_min``, ``size`` — used later to
              denormalize skeleton coordinates back to the original space.
    """
    if not polylines:
        return np.zeros((resolution, resolution), dtype=np.uint8), {
            "x_min": 0.0,
            "y_min": 0.0,
            "size": 1.0,
        }

    # 1. Compute bounding box
    all_pts = np.vstack(polylines)
    x_min, y_min = all_pts.min(axis=0)
    x_max, y_max = all_pts.max(axis=0)

    raw_size = max(x_max - x_min, y_max - y_min, 1.0)
    pad = raw_size * padding_ratio
    x_min -= pad
    y_min -= pad
    size = max(x_max - x_min, y_max - y_min, 1.0) + 2 * pad

    # 2. Render each polyline segment
    mask = np.zeros((resolution, resolution), dtype=np.uint8)
    for pts in polylines:
        pts = np.asarray(pts, dtype=np.float64)
        if len(pts) < 2:
            continue

        # Normalise to pixel coords
        pts_norm = (pts - np.array([x_min, y_min])) / size
        pts_pix = (pts_norm * (resolution - 1)).astype(np.int32)
        np.clip(pts_pix, 0, resolution - 1, out=pts_pix)

        for i in range(len(pts_pix) - 1):
            x1, y1 = int(pts_pix[i, 0]), int(pts_pix[i, 1])
            x2, y2 = int(pts_pix[i + 1, 0]), int(pts_pix[i + 1, 1])
            dx, dy = x2 - x1, y2 - y1
            steps = max(abs(dx), abs(dy), 1) * 2
            for t in np.linspace(0.0, 1.0, int(steps) + 2):
                cx = int(round(x1 + t * dx))
                cy = int(round(y1 + t * dy))
                if 0 <= cx < resolution and 0 <= cy < resolution:
                    if line_width <= 1:
                        mask[cy, cx] = 255
                    else:
                        half = line_width // 2
                        y0 = max(0, cy - half)
                        y1_m = min(resolution, cy + half + 1)
                        x0 = max(0, cx - half)
                        x1_m = min(resolution, cx + half + 1)
                        mask[y0:y1_m, x0:x1_m] = 255

    return mask, {"x_min": float(x_min), "y_min": float(y_min), "size": float(size)}


# ═══════════════════════════════════════════════════════════════════════════
#  Graph extraction (render → skeletonise → graph)
# ═══════════════════════════════════════════════════════════════════════════


def polylines_to_graph(
    polylines: list[np.ndarray],
    resolution: int = 1024,
    cleanup: bool = False,
    contract_degree2: bool = True,
    merge_distance: float = 0.0,
) -> nx.Graph:
    """Convert road polylines to a topology graph with coordinates.

    Full pipeline::

        polylines → binary mask → skeletonise → skeleton graph
            → (optional) degree-2 contraction → intersection-level graph

    **No short-branch pruning is applied** — all topological features are
    preserved so evaluation metrics reflect the generated road network.

    Args:
        polylines: List of ``(N_i, 2)`` arrays in any coordinate space.
        resolution: Raster resolution for rendering **and** skeletonisation.
                    Higher values capture finer detail (default 1024).
        cleanup: If ``True``, apply morphological opening/closing **before**
                 skeletonisation.  This can remove tiny artifacts but may also
                 remove genuine thin roads — use ``False`` for evaluation.
        contract_degree2: If ``True`` (default), collapse degree-2 skeleton
                          waypoint chains into single edges so the output is
                          at intersection-level granularity (comparable with
                          MetaDrive / HDMapGen baselines).
        merge_distance: If > 0, merge any two graph nodes whose Euclidean
                        distance is below this threshold.  This clusters
                        nearby skeleton junction pixels that represent the
                        same physical intersection.  Use a small value
                        (e.g. 5 % of the map extent) to clean up dense
                        road networks without removing dead-end branches.

    Returns:
        ``nx.Graph`` where every node has attribute ``coords``
        (``np.ndarray`` of shape ``(2,)`` in the **original** coordinate space).
    """
    mask, norm = polylines_to_binary_mask(polylines, resolution=resolution, line_width=1)

    if mask.sum() == 0:
        return nx.Graph()

    # field_to_graph requires road_prob as the first positional arg;
    # we pass a dummy array since binary_center takes precedence.
    dummy_prob = np.zeros((resolution, resolution), dtype=np.float32)
    result = field_to_graph(
        dummy_prob,
        binary_center=mask,
        resolution=resolution,
        prune_short_branches=False,
        cleanup=cleanup,
    )

    # Build raw skeleton graph from field_to_graph output
    G = nx.Graph()
    coords_arr: np.ndarray = result.get("coords", np.zeros((0, 2)))
    edges_arr: np.ndarray = result.get("edge_index", np.zeros((0, 2), dtype=np.int64))

    for i in range(coords_arr.shape[0]):
        x_orig = float(coords_arr[i, 0]) * norm["size"] + norm["x_min"]
        y_orig = float(coords_arr[i, 1]) * norm["size"] + norm["y_min"]
        G.add_node(i, coords=np.array([x_orig, y_orig]))

    for e in range(edges_arr.shape[0]):
        G.add_edge(int(edges_arr[e, 0]), int(edges_arr[e, 1]))

    # Optional: contract degree-2 waypoints to intersection-level granularity
    if contract_degree2 and G.number_of_nodes() > 0:
        G = _contract_degree2_nodes(G)

    # Optional: merge nearby junction pixels from skeletonisation
    if merge_distance > 0.0 and G.number_of_nodes() > 0:
        G = _merge_nearby_nodes(G, merge_distance)

    return G


# ═══════════════════════════════════════════════════════════════════════════
#  Visualisation: mask + graph side-by-side
# ═══════════════════════════════════════════════════════════════════════════


def save_vis(
    polylines: list[np.ndarray], G: nx.Graph, filepath: str, resolution: int = 1024, title: str = ""
):
    """Render the binary mask (left) and the extracted graph (right) side by side.

    This shows the exact same pipeline the eval uses — the mask is rendered from
    *polylines*, then the graph is overlaid on the right panel in the same
    coordinate space.

    Args:
        polylines: Input road polylines (same as passed to ``polylines_to_graph``).
        G: Graph returned by ``polylines_to_graph``.
        filepath: Where to save the PNG.
        resolution: Mask resolution used during extraction.
        title: Optional title prefix.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mask, norm = polylines_to_binary_mask(polylines, resolution=resolution, line_width=1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # ── Left: binary mask ──
    ax = axes[0]
    ax.imshow(mask, cmap="gray_r", origin="lower", interpolation="nearest")
    ax.set_title(f"{title}Binary mask ({resolution}×{resolution})", fontsize=11)
    ax.axis("off")

    # ── Right: graph overlaid on mask ──
    ax = axes[1]
    ax.imshow(mask, cmap="gray_r", origin="lower", alpha=0.25, interpolation="nearest")

    if G and G.number_of_nodes() > 0:
        # Convert graph coordinates to pixel space
        coords = np.array([G.nodes[n]["coords"] for n in G.nodes()])
        pix = (
            (coords - np.array([norm["x_min"], norm["y_min"]])) / norm["size"] * (resolution - 1)
        ).astype(int)
        nid_map = {n: i for i, n in enumerate(G.nodes())}

        for u, v in G.edges():
            if u in nid_map and v in nid_map:
                x1, y1 = pix[nid_map[u]]
                x2, y2 = pix[nid_map[v]]
                ax.plot([x1, x2], [y1, y2], color="#e74c3c", lw=2.5, alpha=0.85)

        ax.scatter(
            pix[:, 0], pix[:, 1], c="#2ecc71", s=45, zorder=5, edgecolors="white", linewidths=0.8
        )

        info = (
            f"{G.number_of_nodes()}N  {G.number_of_edges()}E  "
            f"deg1={sum(1 for _,d in G.degree() if d==1)}  "
            f"deg3+={sum(1 for _,d in G.degree() if d>=3)}"
        )
    else:
        info = "Empty graph"

    ax.set_title(f"{title}Graph — {info}", fontsize=11)
    ax.axis("off")

    plt.tight_layout()
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [vis] Saved {filepath}")
