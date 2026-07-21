"""
Road network topology — graph extraction, cleanup, and connection.

Components:
  raster_to_graph:  Road field → skeleton graph.
  graph_to_raster:  Skeleton graph → multi-channel raster field.
  connector:        Cross-component endpoint connection.
  graph_ops:        Chain simplification, node merging, roundabout detection.
  pathfinding:      A* pathfinding on road field cost maps.
"""

from .graph_to_raster import graph_to_raster
from .raster_to_graph import field_to_graph

__all__ = ["field_to_graph", "graph_to_raster"]
