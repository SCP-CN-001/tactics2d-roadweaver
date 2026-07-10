"""
GraphTensorField — local direction field from H-Graph edge tangents.

Constructs a queryable tensor field by sampling edge tangents from the
H-Graph and propagating them via Gaussian-weighted neighbours.

Each query returns (major_axis, minor_axis, anisotropy) at a given point,
where major_axis is the dominant road direction and minor_axis is orthogonal.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import LineString


def normalize(v: np.ndarray) -> np.ndarray:
    nrm = np.linalg.norm(v)
    if nrm < 1e-9:
        return np.zeros_like(v)
    return v / nrm


def sample_linestring_tangents(
    line: LineString,
    step_m: float,
    eps: float = 2.0,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Sample (position, tangent) pairs along a LineString at ~step_m intervals."""
    length = line.length
    if length < 1:
        return []
    count = max(2, int(np.ceil(length / step_m)) + 1)
    samples: List[Tuple[np.ndarray, np.ndarray]] = []
    for d in np.linspace(0, length, count):
        pt = np.asarray(line.interpolate(d).coords[0])
        d0 = max(0.0, d - eps)
        d1 = min(length, d + eps)
        p0 = np.asarray(line.interpolate(d0).coords[0])
        p1 = np.asarray(line.interpolate(d1).coords[0])
        t = normalize(p1 - p0)
        if np.linalg.norm(t) > 0:
            samples.append((pt, t))
    return samples


class GraphTensorField:
    """Gaussian-weighted tensor field from graph edge tangents."""

    def __init__(
        self,
        positions: np.ndarray,
        tangents: np.ndarray,
        radius_m: float = 220.0,
        sigma_m: float = 90.0,
    ):
        self.positions = positions
        self.tangents = tangents
        self.radius_m = radius_m
        self.sigma_m = sigma_m
        self.tree = cKDTree(positions)

    @classmethod
    def from_h_graph(
        cls,
        coords_m: np.ndarray,
        edge_index: np.ndarray,
        step_m: float = 20.0,
        radius_m: float = 220.0,
        sigma_m: float = 90.0,
    ) -> "GraphTensorField":
        """Build tensor field from H-Graph (metre coordinates).

        Args:
            coords_m: (N, 2) node positions in metres.
            edge_index: (E, 2) edge indices.
            step_m: sampling interval along edges.
            radius_m: neighbour search radius.
            sigma_m: Gaussian falloff.

        Returns:
            GraphTensorField instance.
        """
        positions = []
        tangents = []
        for u, v in edge_index:
            u, v = int(u), int(v)
            p_u, p_v = coords_m[u], coords_m[v]
            ev = p_v - p_u
            length = float(np.linalg.norm(ev))
            if length < 1:
                continue
            line = LineString([p_u, p_v])
            samples = sample_linestring_tangents(line, step_m)
            for pt, t in samples:
                positions.append(pt)
                tangents.append(t)
        if not positions:
            # Fallback: use edge directions directly
            for u, v in edge_index:
                u, v = int(u), int(v)
                ev = coords_m[v] - coords_m[u]
                length = float(np.linalg.norm(ev))
                if length < 1:
                    continue
                t = ev / length
                mid = (coords_m[u] + coords_m[v]) / 2
                positions.append(mid)
                tangents.append(t)
        return cls(
            positions=np.array(positions, dtype=np.float64),
            tangents=np.array(tangents, dtype=np.float64),
            radius_m=radius_m,
            sigma_m=sigma_m,
        )

    def axes(
        self,
        position: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Query major axis, minor axis, and anisotropy at *position*.

        Returns:
            (major_axis, minor_axis, anisotropy) where:
            - major_axis: unit vector of dominant direction
            - minor_axis: orthogonal unit vector
            - anisotropy: float in [0, 1] (1 = perfectly aligned)
        """
        indices = self.tree.query_ball_point(position, self.radius_m)
        if not indices:
            return (
                np.array([1.0, 0.0]),
                np.array([0.0, 1.0]),
                0.0,
            )
        local_pos = self.positions[indices]
        local_tan = self.tangents[indices]
        dists = np.linalg.norm(local_pos - position[None, :], axis=1)
        weights = np.exp(-(dists ** 2) / (2.0 * self.sigma_m ** 2))

        tensor = np.zeros((2, 2), dtype=np.float64)
        for t, w in zip(local_tan, weights):
            tensor += w * np.outer(t, t)
        tensor /= max(weights.sum(), 1e-9)

        eigenvalues, eigenvectors = np.linalg.eigh(tensor)
        order = np.argsort(eigenvalues)[::-1]
        major = normalize(eigenvectors[:, order[0]])
        minor = normalize(eigenvectors[:, order[1]])
        e0, e1 = eigenvalues[order[0]], eigenvalues[order[1]]
        aniso = float((e0 - e1) / max(e0 + e1, 1e-9))
        return major, minor, aniso

    def align_axis(self, axis: np.ndarray, ref: np.ndarray) -> np.ndarray:
        """Flip *axis* to point in the same half-plane as *ref*."""
        if np.dot(axis, ref) < 0:
            return -axis
        return axis

    def choose_direction(
        self,
        position: np.ndarray,
        previous_direction: np.ndarray,
    ) -> np.ndarray:
        """Pick the tensor axis (major or minor) closest to *previous_direction*."""
        major, minor, _ = self.axes(position)
        major = self.align_axis(major, previous_direction)
        minor = self.align_axis(minor, previous_direction)
        if abs(np.dot(major, previous_direction)) >= abs(np.dot(minor, previous_direction)):
            return major
        return minor
