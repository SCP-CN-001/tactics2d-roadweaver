# Tactics2D-RoadWeaver

Official implementation of **RoadWeaver: Large-Scale Lane-Level HD Map Generation from Scratch for Autonomous Driving Simulation**.

## Overview

![Release once accepted]()

Tactics2D-RoadWeaver is a framework for generating diverse, large-scale, and simulation-ready lane-level HD maps from scratch.

Tactics2D-RoadWeaver adopts a coarse-to-fine generation pipeline that progressively constructs a road network from global topology to lane-level geometry. It supports controllable road morphology and map scale, while maintaining road-network connectivity and lane-level geometric consistency.

## Architecture

RoadWeaver consists of three main stages:

1. **Global Road Skeleton Generation**
   - A VQ-VAE learns a discrete codebook of global road skeletons.
   - A conditional masked Transformer generates road skeleton tokens conditioned on road style and structural priors.

2. **Skeleton-Guided Road Graph Expansion**
   - The generated skeleton is expanded into a dense road graph using a structure-tensor-guided procedural growth strategy.
   - Graph refinement and reconnection are applied to improve connectivity and local road coverage.

3. **Lane-Level HD Map Construction** (dependency: Tactics2D)
   - The refined road graph is converted into lane-level geometry.
   - Lane boundaries, directional lanes, junction connections, and lane-level topology are constructed and repaired to produce the final HD map.

## Environment Setup

- Python 3.9
- Runs on CPU; training auto-detects and uses GPU when available.

```bash
conda create -n road-weaver python=3.9
conda activate road-weaver
# Install dependencies
pip install -r requirements.txt
git submodule update tactics2d
cd tactics2d
pip install -e .
```

## Quick Start

To generate HD maps with the pretrained models:

```bash
python scripts/generate_maps.py --n-samples 6
```

This requires trained VQ-VAE and Transformer checkpoints (release links [TODO]); train them from scratch via the [Complete Pipeline](#complete-pipeline) below, or pass `--vq-ckpt` / `--model-ckpt` to point at your own checkpoints. Outputs are written to `analysis/e2e/` (`e2e_grid.png`, `hdmap_grid.png`, `hdmap_{i}.png`).

If you only need map generation, skip the training and evaluation steps below.

## Complete Pipeline

This section describes the complete from-scratch pipeline. Data artifacts are built up step by step as the pipeline advances — each stage produces the inputs the next stage consumes. Stages 1–3 below correspond to the three stages in [Architecture](#architecture); stages 2 and 3 are purely algorithmic and require no training.

### 1. Stage 1: Road-Skeleton Generation (learned backbone)

The learned backbone is trained on data that is itself built up step by step within this stage. First, construct the road-skeleton training data from real-world OpenStreetMap (OSM) road networks:

#### 1.1 Training data: from OSM to skeleton splits

##### 1. Download OSM road networks

Downloads OSM roads via the Overpass API for every city in the grid directory (adaptive tiling + rate-limit backoff), and caches the verified road networks as GeoJSON in `data/osm/`.

```bash
python scripts/download_osm.py --input data/grids/classified_grids_of_cities
```

| Option | Default | Meaning |
|---|---|---|
| `--input DIR` | — | Grid shapefiles directory; per-city `.shp` subdirectories define the city list and bounding box |
| `--cities "A,B"` | — | Comma-separated city names instead (geocoded via osmnx). Exactly one of `--input` / `--cities` is required |
| `--osm-dir DIR` | `data/osm` | Verified GeoJSON output directory |

Full options: `python scripts/download_osm.py --help`.

##### 2. Render CRHD images

One image per grid cell with a 2 km context; these CRHD images are also the input to the style predictor in [1.2](#12-road-style-predictor):

```bash
python scripts/generate_crhd.py \
    --input data/grids/classified_grids_of_cities \
    --osm-dir data/osm \
    --output data/crhd_2km \
    --image-size 600 \
    --context-size 1000
```

| Option | Default | Meaning |
|---|---|---|
| `--osm-dir DIR` | *(required)* | OSM GeoJSON directory |
| `--output DIR` | *(required)* | CRHD output directory (also writes `manifest.json`) |
| `--context-size M` | `1000` | Context radius in metres around each cell centroid; `0` clips to the exact cell |
| `--city-wide` | off | Instead render one CRHD per city, centred on the city bounding box |

Full options: `python scripts/generate_crhd.py --help`.

##### 3. Build the urban-prior parquet

Extracts structural priors and the road-skeleton graph for every CRHD cell from its OSM road network:

```bash
python scripts/build_parquet.py \
    --crhd-root data/crhd_2km \
    --graph-root data/osm \
    --output data/urban_prior/urban_prior_2km.parquet
```

| Option | Default | Meaning |
|---|---|---|
| `--crhd-root DIR` | `data/crhd_2km` | CRHD directory containing the PNGs and `manifest.json` |
| `--context-size-m M` | `2000.0` | Context radius in metres for skeleton extraction |
| `--max-samples N` | `-1` | Limit the number of samples (`-1` = all) |

Full options: `python scripts/build_parquet.py --help`.

##### 4. Filter and split

Filter and split into train/val/test parquets (quality + percentile filters, 80/10/10 split):

```bash
python scripts/filter_split.py \
    --parquet data/urban_prior/urban_prior_2km.parquet \
    --output-dir data/urban_prior/2km/splits
```

| Option | Default | Meaning |
|---|---|---|
| `--style-predictions JSON` | — | Optional style predictions (from `predict_style.py`) to merge and filter on |
| `--confidence-threshold F` | `0.7` | Minimum style confidence (only with `--style-predictions`) |
| `--seed N` | `42` | Random seed for the split |

Full options: `python scripts/filter_split.py --help`.

A pre-built urban-prior parquet is available at [TODO: parquet link].

These splits train the VQ-VAE in [1.3](#13-vq-vae). The Transformer additionally needs style-augmented splits — built in [1.4](#14-transformer).

#### 1.2 Road Style Predictor

Input: CRHD images (from [1.1](#11-training-data-from-osm-to-skeleton-splits)); output: a 6-dimensional style vector consumed by the Transformer in [1.4](#14-transformer).

To enable controllable generation of road networks with different urban styles, RoadWeaver uses a road-style predictor adapted from:

> *Global urban road network patterns: Unveiling multiscale planning paradigms of 144 cities with a novel deep learning approach*

The original implementation is available at [ualsg/Global-road-network-patterns](https://github.com/ualsg/Global-road-network-patterns). We reimplement the road-style predictor in PyTorch for integration with the RoadWeaver pipeline.

Given a rasterized road-network representation, the predictor outputs a 6-dimensional probability vector corresponding to six road-network patterns:

- gridiron
- linear
- no pattern
- organic
- radial
- tributary

##### Architecture

A ResNet-34 backbone (timm) followed by a two-layer MLP head
(`Linear(512→256) → ReLU → Linear(256→style_dim)`) and softmax, producing the
6-dim style vector (`style_dim` is configurable).

The predictor is shipped as a **pretrained component** used via `scripts/predict_style.py` (no training script is included). Its checkpoint is available at [TODO: checkpoint link]. If you want to download the OSM data and generate the CRHD images yourself, the [road-style labels](https://figshare.com/articles/dataset/Global_Multiscale_Road_Network_Patterns/19375103/3) provided by [ualsg/Global-road-network-patterns](https://github.com/ualsg/Global-road-network-patterns) are also required — download them and place them in `data/grids/classified_grids_of_cities/` (the `mid_cls` field is used for training).

#### 1.3 VQ-VAE

The VQ-VAE trains on the skeleton splits built in [1.1](#11-training-data-from-osm-to-skeleton-splits).

RoadWeaver uses a **VQ-VAE** to learn a discrete latent representation of global road skeletons. Rather than directly encoding the road graph, each road network is rasterized into a six-channel road-field tensor containing:

- road probability
- road orientation (2 channels)
- junction heatmap
- endpoint heatmap
- distance field

The VQ-VAE encodes the road field into a grid of discrete latent tokens. The resulting codebook provides a compact representation of global road topology and is subsequently used by the conditional masked Transformer for road-skeleton generation.

##### VQ-VAE Training

The VQ-VAE is trained on the skeleton graphs stored in the split parquets: each `skeleton_graph_json` row is rasterized on the fly into the 6-channel road-field tensor described above (CRHD PNGs are not read at training time).

This step requires `data/urban_prior/2km/splits/train.parquet` and `val.parquet` (produced in step 4 above).

```bash
python scripts/train_vq_vae.py --config scripts/config_vq_vae.yaml
```

Our pretrained VQ-VAE checkpoint is available at [TODO: checkpoint link].

#### 1.4 Transformer

The conditional masked Transformer generates the road-skeleton code map conditioned
on road style and structural priors, using the VQ-VAE codebook as the tokenizer.

Two prerequisites:

1. **Style-augmented splits**: the training parquets must carry `style_vector_0..5` columns. Produce them by running `scripts/predict_style.py` on the CRHD images (from [1.1](#11-training-data-from-osm-to-skeleton-splits)), then re-splitting with style filtering:

   ```bash
   python scripts/filter_split.py \
       --parquet data/urban_prior/urban_prior_2km.parquet \
       --output-dir data/urban_prior/2km/splits_style \
       --style-predictions data/crhd_2km_style_predictions.json \
       --confidence-threshold 0.7
   ```

   `scripts/config_transformer_style.yaml` points its `data_dir` at `data/urban_prior/2km/splits_style`.

2. **A trained VQ-VAE checkpoint** (from [1.3](#13-vq-vae)): set `vq_checkpoint` in `scripts/config_transformer_style.yaml` to your trained `best.pth` (the default `runtimes/vq_vae_2km/best.pth` is a placeholder).

```bash
python scripts/train_transformer.py [--config scripts/config_transformer_style.yaml]
```

The pretrained Transformer checkpoint is available at [TODO: checkpoint link].

### 2. Stage 2: Road Graph Expansion (algorithmic)

The generated road skeleton is expanded into a dense road graph by procedural growth guided by a structure tensor field (`src/network_generator/growth/`):

- A **structure tensor field** is built from the skeleton edges; at each location it encodes the dominant road direction and anisotropy.
- **G1 growth** grows collector roads stepwise from the skeleton highways, each step along a direction weighted by the tensor field, inertia, and turning constraints, snapping to nearby roads.
- **A\* closure** connects degree-1 G1 endpoints to the existing road field via shortest-path search.
- **G2 growth** fills remaining empty areas with local roads aligned to the tensor directions.
- The grown graph is cleaned (pruning, largest-connected-component, angle fix, endpoint snap) and compressed into an intersection-level graph with road-class labels.

No training is required for this stage.

### 3. Stage 3: Lane-Level HD Map Construction (algorithmic)

The refined road graph is converted into a lane-level HD map using the Tactics2D library (`src/hdmap_generator/`):

- Lane boundaries are offset from the road centerline so adjacent lanes share the same boundary (no gaps or overlaps).
- Intersection geometry — including roundabouts and junction approaches — is constructed from the compressed intersection graph.
- Lane-level topology is repaired so routing succeeds across the map.

This stage depends on Tactics2D (see [Environment Setup](#environment-setup)).

### 4. Generation

#### End-to-end generation

```bash
conda activate road-weaver

# Generate N samples: skeleton → branch → compressed graph → HD Map
python scripts/generate_maps.py --n-samples 6

# Output:
#   analysis/e2e/e2e_grid.png      — skeleton + compressed graph grid
#   analysis/e2e/hdmap_grid.png    — HD Map comparison grid
#   analysis/e2e/hdmap_{i}.png     — individual HD Maps
```

### 5. Visualization

Inspecting the inference outputs of the trained pipeline — VQ reconstruction,
skeleton code-map generation, and the final HD map:

All visualisation scripts follow the same output rule: one subfolder per
sample/scene, one PNG per panel, plus a `combined.png` overview.

```bash
conda activate road-weaver

# 1) Pipeline intermediates (8 panels per map) → analysis/pipeline/map_{idx}_{w}x{h}/
python scripts/visualize_generation.py pipeline [--n 12]

# 2) Condition sweeps (style / density / diversity) → analysis/sweep/{vq}+{tfm}/
python scripts/visualize_generation.py sweep [--run-full]

# 3) VQ-VAE reconstruction → analysis/vq_recon/recon_{i}/ + grid.png
python scripts/visualize_generation.py vq

# 4) Closed-loop routing demo (tactics2d Router + kinematic bicycle) → analysis/router_closed_loop/
python scripts/visualize_router_closed_loop.py
```

The `pipeline` mode renders the production `run_pipeline(return_intermediates=True)`
states — one subfolder per map with 8 independent panels plus a combined figure
(6 map sizes × 2 conditions by default):

| Panel | Content |
|-------|---------|
| 0 | VQ-decoded road field (field only) |
| 1 | VQ Field (raw skeleton overlay) |
| 2 | Skeleton (simplify_chains) |
| 3 | Scaled + cleaned (merge_close_nodes) |
| 4 | Grown graph (G1 + A* + G2) |
| 5 | Cleaned growth (prune + LCC + angle fix + endpoint snap) |
| 6 | Compressed intersection graph (edges colored by road_class, nodes by type) |
| 7 | HD Map (tactics2d official renderer) |

## Citation

If you use this code, please cite:

```bibtex

```

(TODO: update with the paper reference.)
