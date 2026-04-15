#!/usr/bin/env python3

"""Download preprocessed features from apapiu/small_ldt via huggingface_hub."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_ROOT = PROJECT_ROOT.parent


FILE_GROUPS = {
    "mj": [
        "mj_latents.npy",
        "mj_text_emb.npy",
        "val_encs.npy",
    ],
    "image256": [
        "image_latents256.npy",
        "orig_text_encodings256.npy",
        "val_encs.npy",
    ],
    "checkpoints": [
        "model_state_dict.pth",
        "state_dict_378000.pth",
    ],
    "all-data": [
        "*.npy",
    ],
    "all": [
        "*.npy",
        "*.pth",
        "README.md",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        default="apapiu/small_ldt",
        help="Hugging Face repo to download from.",
    )
    parser.add_argument(
        "--group",
        choices=sorted(FILE_GROUPS),
        default="mj",
        help="Which published file set to download.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEMO_ROOT / "data" / "small_ldt",
        help="Where to place the downloaded files.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face token. Defaults to HF_TOKEN/HUGGING_FACE_HUB_TOKEN if set.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Concurrent download workers for snapshot_download.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    allow_patterns = FILE_GROUPS[args.group]
    print(f"Downloading from repo: {args.repo_id}")
    print(f"Output directory: {args.output_dir.resolve()}")
    print(f"Selected group: {args.group}")
    print(f"Patterns: {allow_patterns}")

    snapshot_download(
        repo_id=args.repo_id,
        repo_type="model",
        local_dir=str(args.output_dir),
        allow_patterns=allow_patterns,
        token=args.token,
        max_workers=args.max_workers,
    )

    print("\nDownloaded files:")
    for path in sorted(p for p in args.output_dir.rglob("*") if p.is_file()):
        print(path.resolve())


if __name__ == "__main__":
    main()
