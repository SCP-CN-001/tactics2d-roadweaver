# Tactics2D-RoadWeaver

## Overview

This module provides a complete pipeline for:
1. **CRHD generation** — Convert OSM road network data into Color Road Hierarchy Diagrams
2. **Style encoding** — A PyTorch-based encoder that maps CRHD images to continuous style vectors
3. **Training & inference** — Train the encoder on labeled CRHD data and run batch predictions

### Position in the pipeline

```
OSM road network / shapefile → [CRHD generator] → CRHD image
                                                      ↓
CRHD image → [style encoder] → urban style vector → [hierarchical road generator]
```

The **CRHD generator** (`scripts/generate_crhd.py`) produces CRHD images from OSM data
or from classified grid shapefiles (e.g., the `original_grids` dataset).

The **style encoder** (`src/style_predictor/`) takes CRHD images as input and outputs
a 6-dimensional style vector representing road network pattern probabilities.

The **hierarchical road generator** (future work) will use the style vector
as a conditional input to generate road networks.

---

## Project Structure

```
src/
├── __init__.py
├── data_generator/               # OSM → urban prior extraction (lib; CLI moved to scripts/)
├── hdmap_generator/              # Compressed graph → tactics2d HD Map (assembler, geometry, io)
├── network_generator/            # Two-phase road generation
│   ├── backbone/                 # VQ-VAE + Masked Transformer (learned)
│   ├── growth/                   # Algorithmic growth (G1 + G2)
│   ├── topology/                 # Graph ops (simplify, merge, classify, cleanup)
│   └── pipeline.py               # run_pipeline entry
├── style_predictor/              # Style encoder package
│   ├── dataset.py                # CRHD dataset (soft/hard label support)
│   └── encoder.py                # StyleEncoder (ResNet34 backbone + style head)
└── utils/
    ├── geometry.py               # Shared geometry primitives
    ├── patterns.py               # Road-network pattern vocabulary
    ├── render.py                 # tactics2d official renderer wrapper
    └── visualization.py          # Road graph / field visualization
```

---

## Model Architecture

```
CRHD image (224×224×3)
    ↓
ResNet-34 backbone (timm)
    ↓  global average pooling
512-dim feature vector
    ↓
Style head: Linear(512→256) → BN → ReLU → Dropout → Linear(256→style_dim)
    ↓
Softmax
    ↓
Style vector (sum = 1.0)
```

### Style Vector (default 6-dim)

The 6 pattern probabilities correspond to:
1. Gridiron
2. Linear
3. No pattern
4. Organic
5. Radial
6. Tributary

The `style_dim` is configurable. For future use cases it can be replaced with
continuous metrics:
- `grid_score`, `radial_score`, `organic_score`
- `hierarchy_score`, `density_score`, `curvature_score`

---

## CRHD Generation

### From a single lat/lon point

```python
from scripts.generate_crhd import generate_centroid_crhd

success = generate_centroid_crhd(
    lon=103.8198, lat=1.3521,          # Singapore
    output_path="outputs/singapore.png",
    dist=1000,                          # 1 km radius
    network_type="drive",
    figsize=(6, 6), dpi=100
)
```

### From classified grid shapefiles (e.g., original_grids dataset)

```bash
python scripts/generate_crhd.py \
    --input ./data/original_grids/classified_grids_of_cities \
    --output ./data/crhd \
    --image-size 512 \
    --limit 20
```

This reads city shapefiles, extracts grid cells with road pattern labels
(`mid_cls` field), generates a CRHD for each cell's centroid, and writes
a manifest JSON with image paths and labels.

---

## Prediction

### Command

```bash
# Single image (random init, no checkpoint)
python scripts/predict_style.py \
    --input outputs/singapore.png \
    --output outputs/predictions.json

# Directory of images with checkpoint
python scripts/predict_style.py \
    --input data/crhd \
    --checkpoint checkpoints/style_predictor/best.pth \
    --output outputs/predictions.json

# Without checkpoint (tests pipeline only)
python scripts/predict_style.py \
    --input data/crhd \
    --checkpoint none \
    --output outputs/predictions.json
```

### Prediction Input

- One or more CRHD images (.png, .jpg)
- Or a directory of images

### Prediction Output

Single image:
```json
{
    "image_path": "outputs/singapore.png",
    "style_vector": [0.12, 0.33, 0.08, 0.20, 0.17, 0.10],
    "style_dim": 6,
    "top_pattern": 2,
    "top_pattern_name": "No pattern",
    "confidence": 0.33,
    "checkpoint_loaded": false
}
```

Directory: saved as a JSON list of the above objects.

---

## Training

### Command

```bash
# VQ-VAE: road field → discrete code map → field reconstruction
python scripts/train_vq_vae.py [--config scripts/config_vq_vae.yaml]

# Conditional masked Transformer: code map generation conditioned on style/density
python scripts/train_transformer.py [--config scripts/config_transformer_style.yaml]
```

---

## RoadWeaver Generation Pipeline

### End-to-end generation

```bash
conda activate road-weaver

# Generate N samples: skeleton → branch → compressed graph → HD Map
python scripts/generate_maps.py --n-samples 6

# Output:
#   analysis/e2e/e2e_grid.png      — skeleton + compressed graph grid
#   analysis/e2e/hdmap_grid.png    — HD Map comparison grid
#   analysis/e2e/hdmap_{i}.png     — individual HD Maps
```

### Visualization

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

### Key files

| File | Purpose |
|------|---------|
| `src/network_generator/pipeline.py` | Two-phase pipeline (generate_skeleton + generate_branch + build_map) |
| `src/hdmap_generator/` | Compressed graph to tactics2d Map (lane offset, intersection geometry, topology repair) |
| `src/utils/visualization.py` | Shared visualization utilities (panels, recon grid) |
| `src/utils/render.py` | tactics2d official renderer wrapper (`render_map`) |
| `src/network_generator/backbone/` | VQ-VAE + Masked Transformer (learned backbone) |
| `src/network_generator/growth/` | Algorithmic road growth (G1 + G2) |
| `src/network_generator/topology/` | Graph topology ops (simplify, merge, classify, cleanup) |
| `scripts/generate_maps.py` | CLI entry point for batch generation |
| `scripts/visualize_generation.py` | pipeline / sweep / vq 三模式可视化（subcommand） |
| `scripts/visualize_router_closed_loop.py` | closed-loop routing / driving demo |
| `scripts/build_parquet.py` | Urban prior dataset builder (Parquet output) |
| `scripts/generate_crhd.py` | CRHD image rendering from OSM networks |
| `scripts/filter_split.py` | Urban prior dataset filtering and splitting |
| `scripts/predict_style.py` | Predict city style vectors from CRHD images |

---

### Training Input

A JSON manifest file:

```json
[
    {
        "image_path": "data/crhd/city_001.png",
        "label": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    },
    {
        "image_path": "data/crhd/city_002.png",
        "label": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    },
    {
        "image_path": "data/crhd/city_003.png",
        "label": [0.7, 0.1, 0.1, 0.05, 0.03, 0.02]
    }
]
```

Both hard labels (one-hot) and soft labels (probability vectors) are supported.

### Training Output

```
checkpoints/
├── config.json                        # Saved config
├── best.pth                           # Best model (lowest val loss)
├── latest.pth                         # Latest checkpoint
└── train_log.json                     # Per-epoch train/val loss
```

---

## Loss Functions

- `soft_label` (default): MSE loss between predicted and target probability
  vectors. Supports both soft labels and one-hot labels.
- `hard_label`: CrossEntropy loss. Labels must be class indices (integer).

---

## Original Grid Dataset

The directory `data/original_grids/` (originally `19375103`, renamed for clarity)
contains shapefiles of ~144 cities with 79,836 grid cells classified by road
network pattern type (`mid_cls` field).
Label distribution:

| Pattern    | Count   | Percentage |
|------------|---------|------------|
| Tributary  | 21,625  | 27.1%      |
| No pattern | 17,929  | 22.5%      |
| Organic    | 13,475  | 16.9%      |
| Gridiron   | 9,665   | 12.1%      |
| Linear     | 9,336   | 11.7%      |
| Radial     | 7,806   | 9.8%       |

The generator script (`python scripts/generate_crhd.py`) maps
`mid_cls` values (`Gridiron`, `Linear`, `Nopattern`, `Organic`, `Radial`,
`Tributary`) to one-hot labels for training.

---

## Original Model Comparison

The original TensorFlow/Keras model (`Global-road-network-patterns/ResNet-34-6class-aug5.h5`)
cannot be loaded by the current environment (TF 2.20 / Keras 3.x). The `.h5` file
is a weights-only file from TF 2.7-era Keras 2.x with no model config embedded.

The PyTorch training pipeline is not blocked by this — training and inference
work fully with the new PyTorch implementation.

---

## Compatibility

- Python 3.9
- CPU and Mac compatible (all operations supported on CPU)

---

## Baseline Evaluation

### Overview

Three road network generation baselines are evaluated for scalability (10–80 intersection-level nodes, 5-node bins, ≥5 maps/bin):

| Baseline | Type | Method | Graph Extraction | Run Command |
|----------|------|--------|-----------------|-------------|
| MetaDrive | Rule-based | Block assembly | Skeleton (1024², merge_d=15) | `python eval/metadrive.py --all` |
| HDMapGen | Data-driven (GRAN) | Autoregressive graph gen | Legacy (degree-2 contract, ×32 meter scale) | `python eval/hdmapgen.py --all --max-maps 0` |
| RoadGen | Rule-based | Widget (component) assembly | Skeleton (1024², merge_d=3, no cleanup) | `python eval/roadgen.py --all --silent` |

All baselines share the same metric module [`eval/metrics.py`](eval/metrics.py):

**Topological**: LCC, dead-end ratio, avg degree, Δd̄ (vs. OSM ref. 2.84)
**Route**: Random OD reachable ratio, avg shortest path length
**Geometric**: Chamfer LOO (normalised [0,1]²), endpoint alignment, edge smoothness/turning angle, self-intersection rate, edge length distribution, subnode uniformity, junction angle distribution

**System**: CPU peak (%), memory peak (MB), GPU memory peak (MB) — tracked via background thread during map generation ([`monitor_resources()`](eval/metrics.py)).

### Quick Start

```bash
conda activate road-weaver

# Run all three baselines (recommended — parallel tmux sessions)
PY=/home/rowena/miniconda3/envs/road-weaver/bin/python
WD=/home/rowena/Documents/tactics2d-roadweaver

tmux new-session -d -s eval-md -n metadrive "cd $WD && $PY eval/metadrive.py --all 2>&1; exec bash"
tmux new-session -d -s eval-hd -n hdmapgen "cd $WD && $PY eval/hdmapgen.py --all --max-maps 0 2>&1; exec bash"
tmux new-session -d -s eval-rg -n roadgen "cd $WD && $PY eval/roadgen.py --all --silent 2>&1; exec bash"

# Monitor progress
tmux attach -t eval-md
tmux attach -t eval-hd
tmux attach -t eval-rg
```

### Per-Baseline Details

#### MetaDrive (`eval/metadrive.py`)

- **Map generation**: `MetaDriveEnv` with `map_config ∈ [7, 10, 15, 20, 25, 30, 35, 40, 45, 50]`; 200 random seeds per config.
- **Graph extraction**: Skeleton method — render lane polylines to 1024×1024 binary mask, skeletonise, extract graph with `merge_distance=15.0`.
- **Stall detection**: If 30 consecutive seeds produce maps that are too small (<10 nodes) or fall into already-full bins, skip to next map_config.
- **Resource monitoring**: CPU/memory sampled every 0.3s during map generation; peak values recorded per map.

#### HDMapGen (`eval/hdmapgen.py`)

- **Model**: GRANMixtureBernoulli (nuplan checkpoint, 9–69 nodes, 10 per size = 610 pre-generated graphs).
- **Graph extraction**: Legacy — build from adjacency matrix, assign node coordinates (×32 to recover meters via `_HDMAPGEN_METER_SCALE`), contract degree-2 nodes.
- **Limitation**: Model architecture caps at 70 nodes → bins 75/80 have no data.

#### RoadGen (`eval/roadgen.py`)

- **Map generation**: CountBasedAlgorithm with widget_number ∈ [6, 10, 14, 18, 22, 26, 30, 34, 38]; 30 attempts per size.
- **Graph extraction**: Skeleton method — render widget lane polylines to 1024×1024 mask, skeletonise, extract graph with `merge_distance=3.0`. All skeleton branches preserved (no pruning).
- **Performance**: 3–100s per map (geometric constraint solving is bottleneck). Use `--silent` to suppress verbose widget output.

### Output Files

All results are written to `runtimes/{baseline}_eval/`:

| File | Content |
|------|---------|
| `all_metrics.csv` | Per-map metrics, one row per generated map (appended incrementally) |
| `vis/` | Mask + extracted graph visualisations (≤5 per scale) |

**Resume behaviour**: If a run is interrupted, re-running the same command appends to the existing CSV and skips previously completed (config, seed) pairs via `load_csv_keys()`.

### Visualization

All metrics tables are documented in [`docs/baseline-eval.md`](docs/baseline-eval.md).
- No TensorFlow dependency required
- All code is in `./src`; original `Global-road-network-patterns/` is not modified.
