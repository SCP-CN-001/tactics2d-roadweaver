"""
Data generator — OSM road network preprocessing for dataset construction.

Pipeline:
  osm_graph:           OSM road graph construction and analysis.
  block_analysis:      Urban block (parcel) extraction.
  prior_extractor:     Structural prior computation.
  data_generator:      Build flat Parquet dataset from priors + style vectors.
  crhd_generator:      CRHD image rendering from OSM data.
  filter_split:        Quality filtering and train/val splitting.
"""
