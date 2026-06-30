"""Generic I/O utilities: JSON save/load, image file discovery."""

import json
import os
from typing import List, Union


def save_json(data, path: str):
    """Save data as pretty-printed JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def find_images(path: Union[str, List[str]],
                exts=('.png', '.jpg', '.jpeg')) -> List[str]:
    """Find image files from a path (file, dir, or list)."""
    if isinstance(path, list):
        return sorted(p for p in path if p.lower().endswith(exts))
    if os.path.isfile(path):
        return [path]
    if os.path.isdir(path):
        return sorted(
            os.path.join(path, f) for f in os.listdir(path)
            if f.lower().endswith(exts)
        )
    raise FileNotFoundError(f'Path not found: {path}')
