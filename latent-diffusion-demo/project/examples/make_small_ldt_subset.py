#!/usr/bin/env python3

"""Create a row-sliced subset from the preprocessed small_ldt features."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing mj_latents.npy/mj_text_emb.npy/val_encs.npy.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write the subset files into.")
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.5,
        help="Fraction of rows to keep from the start of the dataset.",
    )
    parser.add_argument(
        "--layout",
        choices=["mj"],
        default="mj",
        help="Currently only the mj layout is supported.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.fraction <= 1.0:
        raise ValueError("--fraction must be in (0, 1].")

    latent_path = args.input_dir / "mj_latents.npy"
    text_path = args.input_dir / "mj_text_emb.npy"
    val_path = args.input_dir / "val_encs.npy"

    if not latent_path.exists() or not text_path.exists() or not val_path.exists():
        raise FileNotFoundError("Expected mj_latents.npy, mj_text_emb.npy, and val_encs.npy in --input-dir.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    latents = np.load(latent_path, mmap_mode="r")
    text_emb = np.load(text_path, mmap_mode="r")
    if len(latents) != len(text_emb):
        raise ValueError("Latent/text row counts do not match.")

    keep_rows = max(1, int(len(latents) * args.fraction))
    np.save(args.output_dir / "mj_latents.npy", latents[:keep_rows])
    np.save(args.output_dir / "mj_text_emb.npy", text_emb[:keep_rows])
    shutil.copy2(val_path, args.output_dir / "val_encs.npy")

    print(
        {
            "input_rows": int(len(latents)),
            "kept_rows": int(keep_rows),
            "fraction": args.fraction,
            "output_dir": str(args.output_dir),
        }
    )


if __name__ == "__main__":
    main()
