"""
RoadWeaver — hierarchical road network generation.

Pipeline:
  backbone.Generator  →  code map  →  VQVAE.decode  →  road field
  topology.raster_to_graph          →  skeleton graph
  topology.connector + graph_ops    →  cleaned graph
  growth.grow                       →  detail-infilled road graph
"""
