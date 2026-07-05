"""Graph pruning and field cleanup utilities."""
from typing import Dict, List
import numpy as np
from scipy.ndimage import binary_opening, binary_closing
from skimage.morphology import remove_small_objects, square

def prune_graph(graph: Dict, min_edge_len: float = 0.005, min_degree1_chain: int = 2) -> Dict:
    """
    Post-process a skeleton graph: prune short dead-end branches.

    Args:
        graph: dict with 'coords', 'edge_index', 'node_types'
        min_edge_len: minimum edge length in [0,1] space
        min_degree1_chain: minimum number of degree-1 nodes before pruning
    Returns:
        pruned graph with same structure
    """
    coords = graph['coords']
    edges = graph['edge_index']
    types = graph['node_types']
    N = len(coords)

    if N < 3 or len(edges) < 2:
        return graph

    # Build adjacency
    adj = {i: set() for i in range(N)}
    for (i, j) in edges:
        if i < N and j < N:
            adj[i].add(j)
            adj[j].add(i)

    # Iteratively prune short dead-end chains
    changed = True
    while changed:
        changed = False
        to_remove = set()
        for i in range(N):
            if i in to_remove:
                continue
            if len(adj[i]) == 1:  # Dead end
                # Walk the chain
                chain = [i]
                prev = i
                cur = list(adj[i])[0]
                chain.append(cur)
                while len(adj[cur]) == 2 and cur not in to_remove:
                    nbrs = [n for n in adj[cur] if n != prev]
                    if len(nbrs) != 1:
                        break
                    prev = cur
                    cur = nbrs[0]
                    chain.append(cur)
                # Check chain length
                chain_len = 0
                for k in range(len(chain) - 1):
                    chain_len += np.linalg.norm(coords[chain[k]] - coords[chain[k + 1]])
                if chain_len < min_edge_len and len(chain) >= min_degree1_chain:
                    for n in chain:
                        if n not in to_remove:
                            to_remove.add(n)
                            for nb in adj[n]:
                                if nb in adj:
                                    adj[nb].discard(n)
                            adj[n] = set()
                    changed = True

    if not to_remove:
        return graph

    # Rebuild
    keep = [i for i in range(N) if i not in to_remove]
    id_map = {old: new for new, old in enumerate(keep)}

    new_coords = coords[keep]
    new_types = types[keep]
    new_edges = []
    for (i, j) in edges:
        if i not in to_remove and j not in to_remove:
            new_edges.append([id_map[i], id_map[j]])

    return {
        "coords": np.array(new_coords, dtype=np.float32),
        "edge_index": np.array(new_edges, dtype=np.int64) if new_edges else np.zeros((0, 2), dtype=np.int64),
        "node_types": np.array(new_types, dtype=np.int64),
    }



def cleanup_field(road_binary: np.ndarray, opening_radius: int = 2,
                  closing_radius: int = 2, min_obj_size: int = 64) -> np.ndarray:
    """
    Morphological cleanup for generated road fields.

    Steps:
      1. Opening (erosion + dilation) — removes speckle noise
      2. Closing (dilation + erosion) — fills small gaps
      3. Remove small connected components

    This prevents skeletonization from turning minor field noise
    into hundreds of graph nodes.
    """
    from scipy.ndimage import binary_opening, binary_closing
    from skimage.morphology import remove_small_objects, square

    # Opening: remove isolated noise pixels
    if opening_radius > 0:
        road_binary = binary_opening(road_binary, structure=square(opening_radius * 2 + 1))

    # Closing: fill small gaps in roads
    if closing_radius > 0:
        road_binary = binary_closing(road_binary, structure=square(closing_radius * 2 + 1))

    # Remove small objects
    if min_obj_size > 0:
        road_binary = remove_small_objects(road_binary, min_size=min_obj_size)

    return road_binary




