"""
data_preprocessing.py

Convert Next-Day Wildfire Spread TFRecord files into NumPy arrays:
    X_train.npy, Y_train.npy, X_val.npy, Y_val.npy

Expected location:
    ndws/data/next_day_wildfire_spread_train_*.tfrecord
    ndws/data/next_day_wildfire_spread_eval_*.tfrecord
"""

import os
import glob
import numpy as np
import tensorflow as tf
from tqdm import tqdm

# 12 feature channels used as model input
FEATURE_KEYS = [
    "NDVI",
    "tmmn",
    "tmmx",
    "sph",
    "pr",
    "pdsi",
    "erc",
    "vs",
    "th",
    "elevation",
    "population",
    "PrevFireMask",
]

LABEL_KEY = "FireMask"

def parse_example(serialized_example):
    """
    Parse one TFRecord example using tf.train.Example, not tf.io.parse_single_example,
    to keep it simple and explicit.
    """
    example = tf.train.Example()
    example.ParseFromString(serialized_example.numpy())

    feat = example.features.feature

    # Each feature is a flat array of length 4096 = 64*64
    x_channels = []
    for key in FEATURE_KEYS:
        values = np.array(feat[key].float_list.value, dtype=np.float32)
        if values.size != 4096:
            raise ValueError(f"Feature {key} has size {values.size}, expected 4096.")
        x_channels.append(values.reshape(64, 64))

    # Stack into (64, 64, 12)
    X = np.stack(x_channels, axis=-1)

    # Label: FireMask
    fire_vals = np.array(feat[LABEL_KEY].float_list.value, dtype=np.float32)
    if fire_vals.size != 4096:
        raise ValueError(f"Label FireMask has size {fire_vals.size}, expected 4096.")
    Y = fire_vals.reshape(64, 64, 1)  # (64,64,1)

    return X, Y


def convert_split(pattern: str, split_name: str, data_dir: str = "data"):
    """
    Convert a set of TFRecord files matching `pattern` in data_dir into X/Y NumPy arrays.

    pattern: e.g., "next_day_wildfire_spread_train_*.tfrecord"
    split_name: "train" or "val"
    """
    search_pattern = os.path.join(data_dir, pattern)
    tfrecord_paths = sorted(glob.glob(search_pattern))

    if not tfrecord_paths:
        raise FileNotFoundError(f"No TFRecord files found for pattern: {search_pattern}")

    print(f"Converting {len(tfrecord_paths)} files for {split_name}...")

    xs = []
    ys = []

    for path in tqdm(tfrecord_paths):
        dataset = tf.data.TFRecordDataset(path)
        for raw in dataset:
            X, Y = parse_example(raw)
            xs.append(X)
            ys.append(Y)

    X = np.stack(xs, axis=0)  # (N, 64, 64, 12)
    Y = np.stack(ys, axis=0)  # (N, 64, 64, 1)

    out_x = os.path.join(data_dir, f"X_{split_name}.npy")
    out_y = os.path.join(data_dir, f"Y_{split_name}.npy")
    np.save(out_x, X)
    np.save(out_y, Y)

    print(f"Saved {split_name}: X={X.shape}, Y={Y.shape}")
    return X.shape, Y.shape


def main():
    data_dir = "data"

    os.makedirs(data_dir, exist_ok=True)

    # Train split
    convert_split("next_day_wildfire_spread_train_*.tfrecord", "train", data_dir=data_dir)

    # Validation split
    convert_split("next_day_wildfire_spread_eval_*.tfrecord", "val", data_dir=data_dir)


if __name__ == "__main__":
    main()
