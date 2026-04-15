from __future__ import annotations

import argparse

import torch
from PIL import Image

from .model import MiniVLM


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run inference with a saved MiniVLM checkpoint.")
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--image-path", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device)

    model = MiniVLM.from_pretrained(args.checkpoint_dir)
    model.to(device)

    image = Image.open(args.image_path).convert("RGB")
    pixel_values = model.image_processor(image, return_tensors="pt").pixel_values.to(device)
    output = model.generate(
        pixel_values=pixel_values,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=args.do_sample,
    )
    print(output)


if __name__ == "__main__":
    main()
