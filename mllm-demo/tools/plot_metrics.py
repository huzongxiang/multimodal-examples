from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MPL_DIR = ROOT / ".mplconfig"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))
lock_file = MPL_DIR / "fontlist-v390.json.matplotlib-lock"
if lock_file.exists():
    try:
        lock_file.unlink()
    except OSError:
        pass

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot train loss from metrics.jsonl.")
    parser.add_argument("--metrics-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    args = parser.parse_args()

    metrics_path = Path(args.metrics_path)
    steps: list[int] = []
    losses: list[float] = []
    epoch_x: list[int] = []
    epoch_losses: list[float] = []

    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("event") == "train_step":
                steps.append(record["global_step"])
                losses.append(record["loss"])
            elif record.get("event") == "train_epoch":
                epoch_x.append(record["epoch"])
                epoch_losses.append(record["average_loss"])

    plt.figure(figsize=(10, 5))
    if steps:
        plt.plot(steps, losses, label="step_loss", alpha=0.5)
    if epoch_x:
        plt.plot(epoch_x, epoch_losses, label="epoch_avg_loss", marker="o", linewidth=2)
    plt.xlabel("Step / Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
