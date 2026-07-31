"""Road-network pattern vocabulary."""

from __future__ import annotations

# Style pattern names (order matches the style-encoder output indices).
PATTERN_NAMES = ["Gridiron", "Linear", "No pattern", "Organic", "Radial", "Tributary"]

# Dimensionality of the style vector (one softmax logit per pattern).
STYLE_DIM = 6
