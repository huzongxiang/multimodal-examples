#!/usr/bin/env python3

"""Sample a curated prompt set from a trained checkpoint and save a review gallery."""

from __future__ import annotations

import argparse
import json
import math
import os
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from tld.configs import DenoiserConfig, DenoiserLoad, LTDConfig
from tld.diffusion import DiffusionTransformer


DEFAULT_PROMPTS = [
    "woman portrait",
    "cyberpunk city",
    "living room",
    "fantasy character",
    "skincare product",
    "sloth sticker",
    "ocean waves",
    "logo symbol",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to best_model.pth or another training checkpoint.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save prompt samples into.")
    parser.add_argument("--num-imgs", type=int, default=4, help="How many images to sample per prompt.")
    parser.add_argument("--class-guidance", type=float, default=5.0, help="Classifier-free guidance scale.")
    parser.add_argument("--n-iter", type=int, default=20, help="Number of diffusion sampling steps.")
    parser.add_argument("--seed-base", type=int, default=101, help="Base seed; each prompt gets an incremented seed.")
    return parser.parse_args()


def load_font(size: int, bold: bool = False):
    candidates = [
        Path(r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf"),
        Path(r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def prompt_slug(prompt: str, limit: int = 64) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in prompt)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")[:limit] or "sample"


def make_gallery(sample_paths: list[Path], prompts: list[str], out_path: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in sample_paths]
    if not images:
        return

    caption_font = load_font(22, bold=False)
    title_font = load_font(34, bold=True)
    cols = 2
    rows = math.ceil(len(images) / cols)
    tile_width, tile_height = images[0].size
    gutter = 28
    caption_height = 84
    title_height = 86
    canvas_width = cols * tile_width + (cols + 1) * gutter
    canvas_height = title_height + rows * (tile_height + caption_height) + (rows + 1) * gutter

    canvas = Image.new("RGB", (canvas_width, canvas_height), "#f6f7fb")
    draw = ImageDraw.Draw(canvas)
    draw.text((gutter, 22), "Best Checkpoint Prompt Samples", font=title_font, fill="#1f2937")

    for idx, (img, prompt) in enumerate(zip(images, prompts)):
        row = idx // cols
        col = idx % cols
        x = gutter + col * (tile_width + gutter)
        y = title_height + gutter + row * (tile_height + caption_height)
        canvas.paste(img, (x, y))
        wrapped = textwrap.fill(prompt, width=42)
        draw.multiline_text((x + 8, y + tile_height + 10), wrapped, font=caption_font, fill="#4b5563", spacing=6)

    canvas.save(out_path)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg = LTDConfig(
        denoiser_cfg=DenoiserConfig(
            image_size=32,
            patch_size=2,
            embed_dim=768,
            n_layers=12,
            n_channels=4,
            text_emb_size=768,
            noise_embed_dims=256,
            dropout=0,
            mlp_multiplier=4,
        ),
        denoiser_load=DenoiserLoad(local_filename=str(args.checkpoint)),
    )

    diffuser = DiffusionTransformer(cfg)

    manifest = []
    saved_paths: list[Path] = []
    for idx, prompt in enumerate(DEFAULT_PROMPTS):
        seed = args.seed_base + idx
        image = diffuser.generate_image_from_text(
            prompt=prompt,
            class_guidance=args.class_guidance,
            seed=seed,
            num_imgs=args.num_imgs,
            img_size=32,
            n_iter=args.n_iter,
        )
        filename = f"{idx + 1:02d}_{prompt_slug(prompt)}.png"
        out_path = args.output_dir / filename
        image.save(out_path)
        saved_paths.append(out_path)
        manifest.append(
            {
                "prompt": prompt,
                "seed": seed,
                "num_imgs": args.num_imgs,
                "class_guidance": args.class_guidance,
                "n_iter": args.n_iter,
                "file": filename,
            }
        )

    (args.output_dir / "prompts.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    make_gallery(saved_paths, DEFAULT_PROMPTS, args.output_dir / "prompt_gallery.png")
    print(f"Saved {len(saved_paths)} prompt grids to {args.output_dir}")


if __name__ == "__main__":
    main()
