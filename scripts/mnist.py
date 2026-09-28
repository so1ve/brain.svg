"""MNIST pixels normalized to the same centered 8×8 representation as the canvas."""

import gzip
import hashlib
from pathlib import Path
import urllib.request

import numpy as np

BASE = "https://storage.googleapis.com/cvdf-datasets/mnist/"


def _normalize(image):
    weights = image.astype(np.float64)
    ys, xs = np.nonzero(image > 24)
    if not len(xs):
        return np.zeros(64, dtype=np.float32)
    yy, xx = np.indices(image.shape, dtype=np.float64)
    xx += 0.5
    yy += 0.5
    mass = weights.sum()
    cx, cy = (xx * weights).sum() / mass, (yy * weights).sum() / mass
    variance = (((yy - cy) ** 2) * weights).sum()
    shear = (
        np.clip(((xx - cx) * (yy - cy) * weights).sum() / variance, -2, 2)
        if variance > 0
        else 0
    )
    corrected_x = xs + 0.5 - shear * (ys + 0.5 - cy)
    half_width = 0.5 * (1 + abs(shear))
    left, right = corrected_x.min() - half_width, corrected_x.max() + half_width
    top, bottom = ys.min(), ys.max() + 1
    scale = 28 / max(right - left, bottom - top)
    margin_x, margin_y = (32 - (right - left) * scale) / 2, (32 - (bottom - top) * scale) / 2
    origin_x, origin_y = margin_x - left * scale, margin_y - top * scale
    origin_x += np.clip(16 - (origin_x + cx * scale), -margin_x, margin_x)
    origin_y += np.clip(16 - (origin_y + cy * scale), -margin_y, margin_y)
    sy, sx = np.indices((64, 64), dtype=np.float64)
    source_y = ((sy + 0.5) / 2 - origin_y) / scale
    source_x = ((sx + 0.5) / 2 - origin_x) / scale + shear * (source_y - cy)
    ix = np.floor(source_x - 0.5).astype(int)
    iy = np.floor(source_y - 0.5).astype(int)
    fx, fy = source_x - 0.5 - ix, source_y - 0.5 - iy

    def alpha(x, y):
        valid = (x >= 0) & (x < image.shape[1]) & (y >= 0) & (y < image.shape[0])
        return np.where(
            valid,
            weights[np.clip(y, 0, image.shape[0] - 1), np.clip(x, 0, image.shape[1] - 1)],
            0,
        )

    sampled = (
        (alpha(ix, iy) * (1 - fx) + alpha(ix + 1, iy) * fx) * (1 - fy)
        + (alpha(ix, iy + 1) * (1 - fx) + alpha(ix + 1, iy + 1) * fx) * fy
    )
    blocks = sampled.reshape(8, 8, 8, 8).mean(axis=(1, 3))
    return (np.floor(blocks + 0.5) / 255).reshape(64).astype(np.float32)


def load_mnist():
    cache = Path(".cache/digits/mnist")
    cache.mkdir(parents=True, exist_ok=True)
    hashes = {}
    splits = []
    for split in ["train", "t10k"]:
        arrays = []
        for kind, magic, offset, shape in [
            ("images-idx3", 2051, 16, (-1, 28, 28)),
            ("labels-idx1", 2049, 8, (-1,)),
        ]:
            name = f"{split}-{kind}-ubyte.gz"
            filename = cache / name
            if not filename.exists():
                with urllib.request.urlopen(BASE + name, timeout=60) as response:
                    filename.write_bytes(response.read())
            compressed = filename.read_bytes()
            hashes[name] = hashlib.sha256(compressed).hexdigest()
            raw = gzip.decompress(compressed)
            if int.from_bytes(raw[:4], "big") != magic:
                raise ValueError(f"Invalid MNIST IDX header: {filename}")
            arrays.append(np.frombuffer(raw, dtype=np.uint8, offset=offset).reshape(shape))
        images, labels = arrays
        labels = labels.astype(np.int64)
        normalized = cache / f"{split}-deskew-v2.npy"
        if not normalized.exists():
            print(f"Normalizing {len(images)} MNIST {split} images", flush=True)
            np.save(normalized, np.stack([_normalize(image) for image in images]))
        inputs = np.load(normalized)
        if len(inputs) != len(labels):
            raise ValueError(f"MNIST cache does not match its labels: {normalized}")
        splits.append((inputs, labels))
    return splits, hashes
