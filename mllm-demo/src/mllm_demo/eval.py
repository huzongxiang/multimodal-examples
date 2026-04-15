from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .eval_utils import evaluate_records, save_metrics, save_predictions
from .model import MiniVLM


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a MiniVLM checkpoint on a JSONL dataset.")
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--eval-jsonl", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_dir / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model = MiniVLM.from_pretrained(checkpoint_dir)
    model.to(device)

    metrics, predictions = evaluate_records(
        model=model,
        jsonl_path=args.eval_jsonl,
        device=device,
        max_new_tokens=args.max_new_tokens,
        limit=args.limit or None,
    )
    save_metrics(output_dir / "eval_metrics.json", metrics)
    save_predictions(output_dir / "predictions.jsonl", predictions)

    print(f"Saved metrics to {output_dir / 'eval_metrics.json'}")
    print(f"Saved predictions to {output_dir / 'predictions.jsonl'}")


if __name__ == "__main__":
    main()
