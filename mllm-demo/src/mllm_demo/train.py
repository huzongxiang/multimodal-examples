from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw, ImageFont

from .data import JsonlVisionLanguageDataset
from .eval_utils import evaluate_records, save_metrics, save_predictions
from .model import MiniVLM, ModelConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a local reproduction of the VLM from Scratch stages.")
    parser.add_argument("--train-jsonl", type=str, required=True, help="Path to the JSONL dataset.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save the final model.")
    parser.add_argument("--checkpoint-path", type=str, default="", help="Optional resumable epoch checkpoint path.")
    parser.add_argument("--init-from", type=str, default="", help="Optional previous stage checkpoint directory.")
    parser.add_argument("--vision-model", type=str, default="google/vit-large-patch16-224")
    parser.add_argument("--lm-model", type=str, default="HuggingFaceTB/SmolLM-360M")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-freeze-vision", action="store_true")
    parser.add_argument("--eval-jsonl", type=str, default="", help="Optional eval dataset path.")
    parser.add_argument("--metrics-path", type=str, default="", help="Optional JSONL metrics log path.")
    parser.add_argument("--sample-count", type=int, default=3, help="How many eval predictions to store per epoch.")
    parser.add_argument("--early-stop-patience", type=int, default=0, help="Stop if the monitored metric does not improve for N epochs. 0 disables early stopping.")
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0, help="Minimum monitored metric improvement to reset early stopping patience.")
    return parser


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_epoch_checkpoint(
    checkpoint_path: Path,
    epoch: int,
    losses: list[float],
    model: MiniVLM,
    optimizer: torch.optim.Optimizer,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "losses": losses,
            "projector_state_dict": model.projector.state_dict(),
            "language_model_state_dict": model.language_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_path,
    )


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def write_run_config(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_run_summary(
    output_dir: Path,
    losses: list[float],
    global_step: int,
    best_metric: float | None,
    best_epoch: int | None,
    early_stop_epoch: int | None = None,
    final_model_path: str = "",
) -> None:
    payload = {
        "completed_epochs": len(losses),
        "global_step": global_step,
        "final_epoch_loss": losses[-1] if losses else None,
        "best_epoch_loss": min(losses) if losses else None,
        "best_monitor_value": best_metric,
        "best_epoch": best_epoch,
        "early_stop_epoch": early_stop_epoch,
        "losses_path": str(output_dir / "losses.json"),
        "metrics_path": str(output_dir / "metrics.jsonl"),
        "loss_curve_path": str(output_dir / "loss_curve.png"),
        "final_model_path": final_model_path,
    }
    write_run_config(output_dir / "run_summary.json", payload)


def plot_loss_curves(metrics_path: Path, output_path: Path) -> None:
    step_x: list[int] = []
    step_losses: list[float] = []
    epoch_x: list[int] = []
    epoch_losses: list[float] = []

    if not metrics_path.exists():
        return

    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("event") == "train_step":
                step_x.append(record["global_step"])
                step_losses.append(record["loss"])
            elif record.get("event") == "train_epoch":
                epoch_x.append(record["epoch"])
                epoch_losses.append(record["average_loss"])

    if not step_x and not epoch_x:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    render_loss_plot_with_pil(
        step_x=step_x,
        step_losses=step_losses,
        epoch_x=epoch_x,
        epoch_losses=epoch_losses,
        output_path=output_path,
    )


def render_loss_plot_with_pil(
    step_x: list[int],
    step_losses: list[float],
    epoch_x: list[int],
    epoch_losses: list[float],
    output_path: Path,
) -> None:
    values = step_losses + epoch_losses
    if not values:
        return

    width, height = 1200, 700
    margin = 70
    left, top, right, bottom = margin, margin, width - margin, height - margin
    min_loss = min(values)
    max_loss = max(values)
    span = max(max_loss - min_loss, 1e-6)

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.line((left, top, left, bottom), fill="black", width=2)
    draw.line((left, bottom, right, bottom), fill="black", width=2)

    for idx in range(6):
        y = top + (bottom - top) * idx / 5
        draw.line((left, y, right, y), fill=(220, 220, 220), width=1)

    if step_losses:
        points = []
        for idx, value in enumerate(step_losses):
            x = left + (right - left) * idx / max(len(step_losses) - 1, 1)
            y = bottom - (bottom - top) * (value - min_loss) / span
            points.append((x, y))
        draw.line(points, fill=(80, 140, 220), width=2)

    if epoch_losses:
        points = []
        for idx, value in enumerate(epoch_losses):
            x = left + (right - left) * idx / max(len(epoch_losses) - 1, 1)
            y = bottom - (bottom - top) * (value - min_loss) / span
            points.append((x, y))
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(220, 80, 80), outline=(220, 80, 80))
        if len(points) > 1:
            draw.line(points, fill=(220, 80, 80), width=3)

    try:
        title_font = ImageFont.truetype("arial.ttf", 24)
        body_font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        title_font = ImageFont.load_default()
        body_font = ImageFont.load_default()

    draw.text((left, 20), "Training Loss", fill="black", font=title_font)
    draw.text((left, height - 50), "Step / Epoch progression", fill="black", font=body_font)
    draw.text((20, top), f"max {max_loss:.3f}", fill="black", font=body_font)
    draw.text((20, bottom - 20), f"min {min_loss:.3f}", fill="black", font=body_font)
    draw.rectangle((right - 240, top + 10, right - 20, top + 80), outline="black", width=1)
    draw.line((right - 220, top + 30, right - 170, top + 30), fill=(80, 140, 220), width=2)
    draw.text((right - 160, top + 20), "step loss", fill="black", font=body_font)
    draw.line((right - 220, top + 60, right - 170, top + 60), fill=(220, 80, 80), width=3)
    draw.text((right - 160, top + 50), "epoch avg loss", fill="black", font=body_font)
    image.save(output_path)


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)

    config = ModelConfig(
        vision_model_name=args.vision_model,
        lm_model_name=args.lm_model,
        freeze_vision=not args.no_freeze_vision,
    )
    model = MiniVLM(config)

    if args.init_from:
        init_path = Path(args.init_from)
        checkpoint_file = init_path / "mini_vlm_full.pt"
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Init checkpoint not found: {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file, map_location="cpu")
        try:
            model.projector.load_state_dict(checkpoint["projector_state_dict"])
            model.language_model.load_state_dict(checkpoint["language_model_state_dict"])
            print(f"Loaded initial weights from {checkpoint_file}")
        except Exception as exc:
            print(f"Could not load initial weights from {checkpoint_file}: {exc}")
            print("Continuing from base pretrained models instead.")

    dataset = JsonlVisionLanguageDataset(
        args.train_jsonl,
        image_processor=model.image_processor,
        tokenizer=model.tokenizer,
        max_length=args.max_length,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    device = torch.device(args.device)
    model.to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(args.metrics_path) if args.metrics_path else output_dir / "metrics.jsonl"
    start_epoch = 0
    losses: list[float] = []
    global_step = 0
    best_metric: float | None = None
    best_epoch: int = 0
    stale_epochs = 0

    if checkpoint_path and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        start_epoch = checkpoint.get("epoch", 0)
        losses = checkpoint.get("losses", [])
        model.projector.load_state_dict(checkpoint["projector_state_dict"])
        model.language_model.load_state_dict(checkpoint["language_model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)
        print(f"Resumed from epoch {start_epoch}")

    print(f"Loaded {len(dataset)} samples from {args.train_jsonl}")
    print(f"Training on device: {device}")
    print(f"Vision model: {args.vision_model}")
    print(f"Language model: {args.lm_model}")
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    append_jsonl(
        metrics_path,
        {
            "event": "run_started",
            "timestamp": now_iso(),
            "train_jsonl": args.train_jsonl,
            "eval_jsonl": args.eval_jsonl,
            "output_dir": str(output_dir),
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else "",
            "init_from": args.init_from,
            "vision_model": args.vision_model,
            "lm_model": args.lm_model,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "max_length": args.max_length,
            "seed": args.seed,
            "device": str(device),
            "dataset_size": len(dataset),
            "early_stop_patience": args.early_stop_patience,
            "early_stop_min_delta": args.early_stop_min_delta,
        },
    )
    write_run_config(
        output_dir / "run_config.json",
        {
            "timestamp": now_iso(),
            "train_jsonl": args.train_jsonl,
            "eval_jsonl": args.eval_jsonl,
            "output_dir": str(output_dir),
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else "",
            "init_from": args.init_from,
            "vision_model": args.vision_model,
            "lm_model": args.lm_model,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "max_length": args.max_length,
            "seed": args.seed,
            "device": str(device),
            "dataset_size": len(dataset),
            "freeze_vision": not args.no_freeze_vision,
        },
    )

    model.train()
    model.vision_encoder.eval()

    try:
        for epoch in range(start_epoch, args.epochs):
            epoch_loss = 0.0
            for step, batch in enumerate(dataloader, start=1):
                global_step += 1
                pixel_values = batch["pixel_values"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                print(
                    f"epoch={epoch + 1}/{args.epochs} "
                    f"step={step}/{len(dataloader)} "
                    f"loss={loss.item():.4f}"
                )
                append_jsonl(
                    metrics_path,
                    {
                        "event": "train_step",
                        "timestamp": now_iso(),
                        "epoch": epoch + 1,
                        "step": step,
                        "global_step": global_step,
                        "loss": round(loss.item(), 6),
                        "lr": optimizer.param_groups[0]["lr"],
                    },
                )

            avg_loss = epoch_loss / max(len(dataloader), 1)
            losses.append(avg_loss)
            print(f"Epoch {epoch + 1} average_loss={avg_loss:.4f}")
            epoch_record = {
                "event": "train_epoch",
                "timestamp": now_iso(),
                "epoch": epoch + 1,
                "average_loss": round(avg_loss, 6),
            }

            if checkpoint_path:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                save_epoch_checkpoint(checkpoint_path, epoch + 1, losses, model, optimizer)
                print(f"Checkpoint saved to {checkpoint_path}")
                epoch_record["checkpoint_path"] = str(checkpoint_path)

            if args.eval_jsonl:
                summary, predictions = evaluate_records(
                    model=model,
                    jsonl_path=args.eval_jsonl,
                    device=device,
                    max_new_tokens=args.max_length,
                )
                eval_metrics_path = output_dir / f"eval_epoch_{epoch + 1:03d}.json"
                eval_predictions_path = output_dir / f"eval_predictions_epoch_{epoch + 1:03d}.jsonl"
                save_metrics(eval_metrics_path, summary)
                save_predictions(eval_predictions_path, predictions[: args.sample_count])
                epoch_record["eval_metrics_path"] = str(eval_metrics_path)
                epoch_record["eval_predictions_path"] = str(eval_predictions_path)
                epoch_record["eval_summary"] = summary["tasks"]

            monitored_metric = avg_loss
            monitor_name = "train_average_loss"
            improved = False
            if best_metric is None or monitored_metric < (best_metric - args.early_stop_min_delta):
                best_metric = monitored_metric
                best_epoch = epoch + 1
                stale_epochs = 0
                improved = True
            else:
                stale_epochs += 1

            epoch_record["monitor_name"] = monitor_name
            epoch_record["monitor_value"] = monitored_metric
            epoch_record["best_monitor_value"] = best_metric
            epoch_record["best_epoch"] = best_epoch
            epoch_record["stale_epochs"] = stale_epochs
            epoch_record["improved"] = improved
            append_jsonl(metrics_path, epoch_record)
            with (output_dir / "losses.json").open("w", encoding="utf-8") as handle:
                json.dump({"losses": losses}, handle, indent=2)
            plot_loss_curves(metrics_path, output_dir / "loss_curve.png")
            write_run_summary(
                output_dir=output_dir,
                losses=losses,
                global_step=global_step,
                best_metric=best_metric,
                best_epoch=best_epoch,
            )

            if args.early_stop_patience > 0 and stale_epochs >= args.early_stop_patience:
                append_jsonl(
                    metrics_path,
                    {
                        "event": "early_stop",
                        "timestamp": now_iso(),
                        "epoch": epoch + 1,
                        "monitor_name": monitor_name,
                        "monitor_value": monitored_metric,
                        "best_monitor_value": best_metric,
                        "best_epoch": best_epoch,
                        "stale_epochs": stale_epochs,
                    },
                )
                print(
                    f"Early stopping triggered at epoch {epoch + 1} "
                    f"(best_epoch={best_epoch}, best_{monitor_name}={best_metric:.6f})"
                )
                break

    except KeyboardInterrupt:
        print("Training interrupted, saving checkpoint.")
        if checkpoint_path:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            save_epoch_checkpoint(checkpoint_path, len(losses), losses, model, optimizer)
            print(f"Checkpoint saved to {checkpoint_path}")
        append_jsonl(
            metrics_path,
            {
                "event": "run_interrupted",
                "timestamp": now_iso(),
                "completed_epochs": len(losses),
                "global_step": global_step,
            },
        )
        with (output_dir / "losses.json").open("w", encoding="utf-8") as handle:
            json.dump({"losses": losses}, handle, indent=2)
        plot_loss_curves(metrics_path, output_dir / "loss_curve.png")
        write_run_summary(
            output_dir=output_dir,
            losses=losses,
            global_step=global_step,
            best_metric=best_metric,
            best_epoch=best_epoch,
        )
        raise

    model.save_pretrained(output_dir)
    with (output_dir / "losses.json").open("w", encoding="utf-8") as handle:
        json.dump({"losses": losses}, handle, indent=2)
    plot_loss_curves(metrics_path, output_dir / "loss_curve.png")
    early_stop_epoch = len(losses) if args.early_stop_patience > 0 and len(losses) < args.epochs else None
    write_run_summary(
        output_dir=output_dir,
        losses=losses,
        global_step=global_step,
        best_metric=best_metric,
        best_epoch=best_epoch,
        early_stop_epoch=early_stop_epoch,
        final_model_path=str(output_dir / "mini_vlm_full.pt"),
    )
    append_jsonl(
        metrics_path,
        {
            "event": "run_finished",
            "timestamp": now_iso(),
            "completed_epochs": len(losses),
            "global_step": global_step,
            "losses_path": str(output_dir / "losses.json"),
            "loss_curve_path": str(output_dir / "loss_curve.png"),
            "final_model_path": str(output_dir / "mini_vlm_full.pt"),
        },
    )
    print(f"Saved final model to {output_dir}")


if __name__ == "__main__":
    main()
