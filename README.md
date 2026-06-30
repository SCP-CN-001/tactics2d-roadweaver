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

The **CRHD generator** (`utils/crhd_generator.py`) produces CRHD images from OSM data
or from classified grid shapefiles (e.g., the `original_grids` dataset).

The **style encoder** (`predict_city_style/`) takes CRHD images as input and outputs
a 6-dimensional style vector representing road network pattern probabilities.

The **hierarchical road generator** (future work) will use the style vector
as a conditional input to generate road networks.

---

## Project Structure

```
src/
├── README.md                           # This file
├── predict_city_style/                 # Style encoder package
│   ├── __init__.py
│   ├── model.py                        # StyleEncoder (ResNet34 backbone + style head)
│   ├── dataset.py                      # CRHD dataset with soft/hard label support
│   ├── train.py                        # CLI: training script
│   ├── predict.py                      # CLI: batch prediction script
│   └── utils.py                        # Image I/O, label mapping, constants
└── utils/
    ├── __init__.py
    └── crhd_generator.py               # Core CRHD generation logic (OSM network → CRHD image)
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
from src.utils.crhd_generator import generate_centroid_crhd

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
python -m src.utils.crhd_generator \
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
python -m src.predict_city_style.predict \
    --input outputs/singapore.png \
    --output outputs/predictions.json \
    --style-dim 6

# Directory of images with checkpoint
python -m src.predict_city_style.predict \
    --input data/crhd \
    --checkpoint checkpoints/style_encoder/best.pth \
    --output outputs/predictions.json \
    --style-dim 6

# Without checkpoint (tests pipeline only)
python -m src.predict_city_style.predict \
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
python -m src.predict_city_style.train \
    --manifest data/crhd/manifest.json \
    --output-dir checkpoints \
    --style-dim 6 \
    --backbone resnet34 \
    --epochs 100 \
    --batch-size 8 \
    --lr 1e-4 \
    --device cpu \
    --loss-type soft_label
```

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

The generator script (`python -m src.utils.crhd_generator`) maps
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
- No TensorFlow dependency required
- All code is in `./src`; original `Global-road-network-patterns/` is not modified.
