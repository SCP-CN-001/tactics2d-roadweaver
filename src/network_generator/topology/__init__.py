"""Topology module."""

from .connector import EndpointConnector
from .graph_intersection import detect_roundabouts
from .graph_merge import merge_close_nodes
from .graph_simplify import simplify_chains
from .graph_to_raster import graph_to_raster
from .graph_utils import (
    NT_CURVE,
    NT_ENDPOINT,
    NT_JUNCTION,
    NT_ROUNDABOUT,
    NT_WAYPOINT,
    adjacency_from_edges,
    endpoint_nodes,
    find_components,
)
from .raster_to_graph import field_to_graph

__all__ = [
    "field_to_graph",
    "graph_to_raster",
    "simplify_chains",
    "merge_close_nodes",
    "detect_roundabouts",
    "NT_WAYPOINT",
    "NT_JUNCTION",
    "NT_CURVE",
    "NT_ROUNDABOUT",
    "NT_ENDPOINT",
    "adjacency_from_edges",
    "endpoint_nodes",
    "find_components",
    "EndpointConnector",
]
