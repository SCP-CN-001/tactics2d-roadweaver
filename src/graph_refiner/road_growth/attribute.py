"""
L-Level: attribute assignment.
road_class mapping: 1=arterial, 2=collector, 3=local.
"""
from __future__ import annotations

import numpy as np

from .config import GrowthConfig


def assign_attributes(graph: dict, config: GrowthConfig) -> dict:
    """Add per-edge attributes: lanes, bidirectional, speed, surface."""
    rc = graph.get("road_class", np.ones(len(graph["edge_index"]), dtype=np.int64))
    E = len(graph["edge_index"])

    lanes = np.zeros(E, dtype=np.int64)
    bidirectional = np.ones(E, dtype=bool)
    speed = np.zeros(E, dtype=np.int64)
    surface = np.full(E, "paved", dtype=object)

    for ei in range(E):
        cls = int(rc[ei])
        if cls == 1:
            lanes[ei] = config.arterial_lanes_per_dir
            speed[ei] = 60
        elif cls == 2:
            lanes[ei] = config.collector_lanes_per_dir
            speed[ei] = 40
        else:
            lanes[ei] = config.local_lanes_per_dir
            speed[ei] = 20
        bidirectional[ei] = True

    graph["lanes_per_dir"] = lanes
    graph["bidirectional"] = bidirectional
    graph["speed_limit_kmh"] = speed
    graph["surface"] = surface
    return graph
