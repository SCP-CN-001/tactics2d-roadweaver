"""Skeleton graph to raster field converter implementation."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt


def graph_to_raster(
    coords: np.ndarray,
    edge_index: np.ndarray,
    resolution: int = 256,
    sigma_dist: float = 0.02,
    sigma_junction: float = 0.02,
    sigma_endpoint: float = 0.015,
    binary_centerline: bool = True,
) -> dict[str, np.ndarray]:
    """
    Render a skeleton graph as a multi-channel raster field.

    Returns dict of (H, W) channels:
        road_prob:      binary centerline (if binary_centerline=True) or soft distance
        sin_2theta:     sin(2θ) of road tangent direction
        cos_2theta:     cos(2θ) of road tangent direction
        junction_hm:    Gaussian heatmap at degree≥3 nodes
        endpoint_hm:    Gaussian heatmap at degree=1 nodes
        binary_center:  binary 1-pixel centerline mask (uint8 0/255)
        binary_thick:   dilated binary mask for robust vectorization
        soft_distance:  soft distance field (auxiliary)
    """
    N = coords.shape[0]
    H = W = resolution

    pix = np.clip(coords * resolution, 0, resolution - 1).astype(np.float32)

    road_mask = np.zeros((H, W), dtype=np.uint8)
    orient_sin = np.zeros((H, W), dtype=np.float32)
    orient_cos = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.int32)

    for e in range(edge_index.shape[0]):
        i, j = int(edge_index[e, 0]), int(edge_index[e, 1])
        if i >= N or j >= N:
            continue
        x1, y1 = pix[i]
        x2, y2 = pix[j]
        dx = x2 - x1
        dy = y2 - y1
        length = max(np.sqrt(dx**2 + dy**2), 1.0)
        tx = dx / length
        ty = dy / length
        angle = np.arctan2(ty, tx)
        s2 = np.sin(2 * angle)
        c2 = np.cos(2 * angle)

        num_steps = max(int(length * 2), 2)
        for t in np.linspace(0, 1, num_steps):
            cx = int(round(x1 + t * dx))
            cy = int(round(y1 + t * dy))
            if 0 <= cx < W and 0 <= cy < H:
                road_mask[cy, cx] = 1
                orient_sin[cy, cx] += s2
                orient_cos[cy, cx] += c2
                count_map[cy, cx] += 1

    mask_pixels = count_map > 0
    orient_sin[mask_pixels] /= count_map[mask_pixels]
    orient_cos[mask_pixels] /= count_map[mask_pixels]

    binary_center = road_mask.copy().astype(np.uint8) * 255
    binary_thick = binary_dilation(binary_center > 0, iterations=2).astype(np.uint8) * 255

    dist = distance_transform_edt(1 - road_mask)
    dist_normalized = dist / resolution
    soft_distance = np.exp(-dist_normalized / sigma_dist)

    if binary_centerline:
        road_prob = np.exp(-dist_normalized / 0.002)
    else:
        road_prob = soft_distance.copy()

    junction_hm = np.zeros((H, W), dtype=np.float32)
    endpoint_hm = np.zeros((H, W), dtype=np.float32)

    degrees = np.zeros(N, dtype=np.int32)
    for e in range(edge_index.shape[0]):
        i, j = int(edge_index[e, 0]), int(edge_index[e, 1])
        if i < N and j < N:
            degrees[i] += 1
            degrees[j] += 1

    for n in range(N):
        cx, cy = pix[n]
        cx_i, cy_i = int(round(cx)), int(round(cy))
        if not (0 <= cx_i < W and 0 <= cy_i < H):
            continue
        sigma = sigma_junction * resolution if degrees[n] >= 3 else sigma_endpoint * resolution
        sigma = max(sigma, 1.0)
        radius = int(round(sigma * 3))
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                px, py = cx_i + dx, cy_i + dy
                if 0 <= px < W and 0 <= py < H:
                    val = np.exp(-(dx**2 + dy**2) / (2 * sigma**2))
                    if degrees[n] >= 3:
                        junction_hm[py, px] = max(junction_hm[py, px], val)
                    elif degrees[n] == 1:
                        endpoint_hm[py, px] = max(endpoint_hm[py, px], val)

    return {
        "road_prob": road_prob.astype(np.float32),
        "sin_2theta": orient_sin.astype(np.float32),
        "cos_2theta": orient_cos.astype(np.float32),
        "junction_hm": junction_hm.astype(np.float32),
        "endpoint_hm": endpoint_hm.astype(np.float32),
        "binary_center": binary_center.astype(np.uint8),
        "binary_thick": binary_thick.astype(np.uint8),
        "soft_distance": soft_distance.astype(np.float32),
    }
