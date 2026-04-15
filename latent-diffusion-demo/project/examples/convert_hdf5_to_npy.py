#!/usr/bin/env python3

"""Convert the preprocessing output into the .npy files expected by tld.train."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def load_dataset(file_path: Path, dataset_name: str) -> np.ndarray:
    with h5py.File(file_path, "r") as handle:
        return handle[dataset_name][:]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing the HDF5 preprocessing output.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write .npy files to. Defaults to --input-dir.",
    )
    parser.add_argument(
        "--val-count",
        type=int,
        default=8,
        help="How many text embeddings to keep in val_emb.npy for training previews.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    latents = load_dataset(input_dir / "image_latents.hdf5", "image_latents")
    text_emb = load_dataset(input_dir / "text_encodings.hdf5", "text_encodings")

    if len(text_emb) < args.val_count:
        raise ValueError(
            f"Expected at least {args.val_count} text embeddings for val_emb.npy, found {len(text_emb)}."
        )

    np.save(output_dir / "latents.npy", latents)
    np.save(output_dir / "text_emb.npy", text_emb)
    np.save(output_dir / "val_emb.npy", text_emb[: args.val_count])

    print(f"Saved {len(latents)} latents to {output_dir / 'latents.npy'}")
    print(f"Saved {len(text_emb)} text embeddings to {output_dir / 'text_emb.npy'}")
    print(f"Saved {args.val_count} validation embeddings to {output_dir / 'val_emb.npy'}")


if __name__ == "__main__":
    main()
