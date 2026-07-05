"""Graph comparison metrics for roundtrip evaluation."""
from typing import Dict
import numpy as np
import networkx as nx

def _edges_to_pairs(edge_index):
    """Convert edge_index to list of (i,j) tuples.
    Accepts both (E,2) and (2,E) formats."""
    if edge_index is None or len(edge_index) == 0:
        return []
    e = np.asarray(edge_index)
    if e.shape[0] == 2 and e.shape[1] != 2:
        # (2, E) format — transpose
        e = e.T
    # Now e is (E, 2)
    return [(int(e[i, 0]), int(e[i, 1])) for i in range(e.shape[0])]



def compute_roundtrip_metrics(gt_graph: Dict, recovered_graph: Dict) -> Dict:
    """
    Compare GT graph vs recovered graph (from field rendering + vectorization).
    Accepts both (E,2) and (2,E) edge_index formats.
    """
    gt_coords = gt_graph["coords"]
    gt_edges_pairs = _edges_to_pairs(gt_graph.get("edge_index", []))
    rec_coords = recovered_graph["coords"]
    rec_edge_pairs = _edges_to_pairs(recovered_graph.get("edge_index", []))

    metrics = {}

    # Node count error
    metrics["node_count_error"] = abs(len(rec_coords) - len(gt_coords))

    # Edge count error
    metrics["edge_count_error"] = abs(len(rec_edge_pairs) - len(gt_edges_pairs))

    # Connected components (recovered)
    if rec_edge_pairs:
        G = nx.Graph()
        G.add_nodes_from(range(len(rec_coords)))
        G.add_edges_from(rec_edge_pairs)
        metrics["connected_components"] = nx.number_connected_components(G)
    else:
        metrics["connected_components"] = len(rec_coords)

    # Degree distribution comparison
    if gt_edges_pairs:
        G_gt = nx.Graph()
        G_gt.add_nodes_from(range(len(gt_coords)))
        G_gt.add_edges_from(gt_edges_pairs)
        gt_deg = [d for _, d in G_gt.degree()]
        gt_deg_hist, _ = np.histogram(gt_deg, bins=range(0, 10), density=True)
    else:
        gt_deg_hist = np.ones(9) / 9

    if rec_edge_pairs:
        G_rec = nx.Graph()
        G_rec.add_nodes_from(range(len(rec_coords)))
        G_rec.add_edges_from(rec_edge_pairs)
        rec_deg = [d for _, d in G_rec.degree()]
        rec_deg_hist, _ = np.histogram(rec_deg, bins=range(0, 10), density=True)
    else:
        rec_deg_hist = np.ones(9) / 9

    metrics["degree_l1"] = np.abs(gt_deg_hist - rec_deg_hist).sum().item()

    # Junction count
    if gt_edges_pairs:
        gt_deg_a = np.zeros(len(gt_coords), dtype=np.int32)
        for i, j in gt_edges_pairs:
            gt_deg_a[i] += 1
            gt_deg_a[j] += 1
        n_gt_junc = int((gt_deg_a >= 3).sum())
    else:
        n_gt_junc = 0

    if rec_edge_pairs:
        rec_deg_a = np.zeros(len(rec_coords), dtype=np.int32)
        for i, j in rec_edge_pairs:
            rec_deg_a[i] += 1
            rec_deg_a[j] += 1
        n_rec_junc = int((rec_deg_a >= 3).sum())
    else:
        n_rec_junc = 0

    metrics["gt_junction_count"] = n_gt_junc
    metrics["recovered_junction_count"] = n_rec_junc

    # Overall roundtrip score (composite, lower is better)
    score = (
        metrics["node_count_error"] / max(len(gt_coords), 1)
        + metrics["edge_count_error"] / max(len(gt_edges_pairs), 1)
        + metrics["degree_l1"]
    )
    metrics["roundtrip_score"] = score

    return metrics



