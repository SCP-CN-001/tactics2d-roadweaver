# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Metrics module."""

from .geometry import (
    compute_all_geometric_metrics,
    compute_chamfer_distance,
    compute_chamfer_leave_one_out,
    compute_edge_length_distribution,
    compute_edge_smoothness,
    compute_endpoint_alignment,
    compute_node_angle_distribution,
    compute_self_intersection_rate,
    compute_subnode_uniformity,
)
from .io import (
    append_csv_row,
    load_csv_keys,
    load_csv_rows,
    print_results_table,
    save_binned_summary,
    save_results,
    save_system_info,
)
from .reference import load_osm_reference_degree
from .route import compute_all_pairs_reachability, compute_route_coverage
from .system import get_resource_stats, monitor_resources
from .topology import (
    METADRIVE_NODE_PATTERN,
    classify_scale,
    compute_cycle_ratio,
    compute_topological_metrics,
    contract_degree2_nodes,
    extract_intersection_graph,
    is_metadrive_road_node,
    merge_close_nodes,
)

__all__ = [
    "METADRIVE_NODE_PATTERN",
    "append_csv_row",
    "classify_scale",
    "compute_all_geometric_metrics",
    "compute_all_pairs_reachability",
    "compute_chamfer_distance",
    "compute_chamfer_leave_one_out",
    "compute_cycle_ratio",
    "compute_edge_length_distribution",
    "compute_edge_smoothness",
    "compute_endpoint_alignment",
    "compute_node_angle_distribution",
    "compute_route_coverage",
    "compute_self_intersection_rate",
    "compute_subnode_uniformity",
    "compute_topological_metrics",
    "contract_degree2_nodes",
    "extract_intersection_graph",
    "get_resource_stats",
    "is_metadrive_road_node",
    "load_csv_keys",
    "load_csv_rows",
    "load_osm_reference_degree",
    "merge_close_nodes",
    "monitor_resources",
    "print_results_table",
    "save_binned_summary",
    "save_results",
    "save_system_info",
]
