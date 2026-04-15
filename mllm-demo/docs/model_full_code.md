# 模型完整代码

## 说明

本文档汇总当前实验复现中最核心的完整源码，便于后续写作时直接引用。需要说明的是：

- 当前项目在结构上并不是“每个阶段一套不同模型”；
- 真正的核心模型主体只有一套，即 `MiniVLM`；
- 各阶段的变化主要来自模型规模配置、数据格式和训练脚本参数，而不是模型拓扑发生根本变化。

因此，这里的“完整模型代码”包含：

- `model.py`：模型主体
- `data.py`：统一数据组织
- `train.py`：完整训练逻辑
- `eval_utils.py`：评估逻辑
- `eval.py`：批量评估入口
- `infer.py`：单图推理入口

## 1. `model.py`

文件路径：

- `src\mllm_demo\model.py`

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, ViTImageProcessor, ViTModel


@dataclass
class ModelConfig:
    vision_model_name: str
    lm_model_name: str
    freeze_vision: bool = True


class VisionProjector(nn.Module):
    def __init__(self, vision_dim: int, language_dim: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(vision_dim, language_dim),
            nn.GELU(),
            nn.LayerNorm(language_dim),
            nn.Linear(language_dim, language_dim),
        )

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        return self.projection(vision_features)


class MiniVLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.vision_encoder = ViTModel.from_pretrained(config.vision_model_name)
        self.language_model = AutoModelForCausalLM.from_pretrained(config.lm_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(config.lm_model_name)
        self.image_processor = ViTImageProcessor.from_pretrained(config.vision_model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.vision_dim = self.vision_encoder.config.hidden_size
        self.language_dim = self.language_model.config.hidden_size
        self.projector = VisionProjector(self.vision_dim, self.language_dim)

        if config.freeze_vision:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.config.freeze_vision:
            with torch.no_grad():
                vision_outputs = self.vision_encoder(pixel_values=pixel_values)
        else:
            vision_outputs = self.vision_encoder(pixel_values=pixel_values)

        image_features = vision_outputs.last_hidden_state
        return self.projector(image_features)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ):
        batch_size = pixel_values.shape[0]
        image_embeds = self.encode_image(pixel_values)
        num_image_tokens = image_embeds.shape[1]

        embedding_layer = self.language_model.get_input_embeddings()
        lm_dtype = embedding_layer.weight.dtype
        image_embeds = image_embeds.to(dtype=lm_dtype)
        text_embeds = embedding_layer(input_ids).to(dtype=lm_dtype)
        combined_embeds = torch.cat([image_embeds, text_embeds], dim=1)

        image_attention = torch.ones(
            (batch_size, num_image_tokens),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        combined_attention = torch.cat([image_attention, attention_mask], dim=1)

        if labels is not None:
            image_labels = torch.full(
                (batch_size, num_image_tokens),
                fill_value=-100,
                dtype=labels.dtype,
                device=labels.device,
            )
            combined_labels = torch.cat([image_labels, labels], dim=1)
        else:
            combined_labels = None

        return self.language_model(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attention,
            labels=combined_labels,
            return_dict=True,
        )

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        prompt: str = "",
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        do_sample: bool = True,
    ) -> str:
        self.eval()
        image_embeds = self.encode_image(pixel_values)
        embedding_layer = self.language_model.get_input_embeddings()
        lm_dtype = embedding_layer.weight.dtype
        image_embeds = image_embeds.to(dtype=lm_dtype)

        if prompt:
            prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(pixel_values.device)
        else:
            bos_token_id = self.tokenizer.bos_token_id
            if bos_token_id is None:
                bos_token_id = self.tokenizer.eos_token_id
            prompt_ids = torch.tensor([[bos_token_id]], device=pixel_values.device)

        generated_ids = prompt_ids.clone()

        for _ in range(max_new_tokens):
            current_embeds = embedding_layer(generated_ids).to(dtype=lm_dtype)
            full_embeds = torch.cat([image_embeds, current_embeds], dim=1)
            outputs = self.language_model(inputs_embeds=full_embeds)
            next_token_logits = outputs.logits[:, -1, :]

            if do_sample:
                probs = F.softmax(next_token_logits / max(temperature, 1e-5), dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)

            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            if next_token.item() == self.tokenizer.eos_token_id:
                break

        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    def save_pretrained(self, output_dir: str | Path) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "config": {
                "vision_model_name": self.config.vision_model_name,
                "lm_model_name": self.config.lm_model_name,
                "vision_dim": self.vision_dim,
                "language_dim": self.language_dim,
                "freeze_vision": self.config.freeze_vision,
            },
            "projector_state_dict": self.projector.state_dict(),
            "language_model_state_dict": self.language_model.state_dict(),
        }

        torch.save(self.projector.state_dict(), output_path / "projector.pt")
        torch.save(checkpoint, output_path / "mini_vlm_full.pt")
        self.tokenizer.save_pretrained(output_path / "tokenizer")
        self.image_processor.save_pretrained(output_path / "image_processor")

        with (output_path / "config.json").open("w", encoding="utf-8") as handle:
            json.dump(checkpoint["config"], handle, indent=2)

    @classmethod
    def from_pretrained(cls, checkpoint_dir: str | Path):
        checkpoint_path = Path(checkpoint_dir)
        checkpoint = torch.load(checkpoint_path / "mini_vlm_full.pt", map_location="cpu")
        config = ModelConfig(
            vision_model_name=checkpoint["config"]["vision_model_name"],
            lm_model_name=checkpoint["config"]["lm_model_name"],
            freeze_vision=checkpoint["config"].get("freeze_vision", True),
        )

        model = cls(config)
        model.projector.load_state_dict(checkpoint["projector_state_dict"])
        model.language_model.load_state_dict(checkpoint["language_model_state_dict"])
        model.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path / "tokenizer")
        model.image_processor = ViTImageProcessor.from_pretrained(checkpoint_path / "image_processor")

        if model.tokenizer.pad_token is None:
            model.tokenizer.pad_token = model.tokenizer.eos_token

        return model
```

## 2. `data.py`

文件路径：

- `src\mllm_demo\data.py`

```python
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset


def load_jsonl_records(jsonl_path: str | Path) -> list[dict[str, Any]]:
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def select_prompt(record: dict[str, Any], deterministic: bool = False) -> str:
    prompt_options = record.get("prompt_options")
    if prompt_options:
        return prompt_options[0] if deterministic else random.choice(prompt_options)
    return record.get("prompt", "")


def select_target(record: dict[str, Any], deterministic: bool = False) -> str:
    if "captions" in record:
        captions = record["captions"]
        return captions[0] if deterministic else random.choice(captions)
    return record["target"]


class JsonlVisionLanguageDataset(Dataset):
    """JSONL dataset that mirrors the original notebook task formatting."""

    def __init__(
        self,
        jsonl_path: str | Path,
        image_processor,
        tokenizer,
        max_length: int = 256,
    ):
        self.jsonl_path = Path(jsonl_path)
        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.jsonl_path}")

        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.records = load_jsonl_records(self.jsonl_path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image_path = Path(record["image"])
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        pixel_values = self.image_processor(image, return_tensors="pt").pixel_values.squeeze(0)

        prompt = select_prompt(record, deterministic=False)
        target = select_target(record, deterministic=False)
        append_eos = record.get("append_eos", bool(prompt))
        mask_prompt_loss = record.get("mask_prompt_loss", bool(prompt))

        text = f"{prompt} {target}".strip() if prompt else target
        if append_eos and self.tokenizer.eos_token is not None:
            text = f"{text}{self.tokenizer.eos_token}"

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        labels = input_ids.clone()
        if mask_prompt_loss and prompt:
            prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
            prompt_len = min(len(prompt_tokens), labels.numel())
            labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "task": record.get("task", "caption"),
            "prompt": prompt,
            "target": target,
            "image_path": str(image_path),
        }
```

## 3. `train.py`

文件路径：

- `src\mllm_demo\train.py`

```python
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
```

## 4. `eval_utils.py`

文件路径：

- `src\mllm_demo\eval_utils.py`

```python
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from .data import load_jsonl_records, select_prompt


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def strip_prompt_prefix(prediction: str, prompt: str) -> str:
    prediction = prediction.strip()
    prompt = prompt.strip()
    if prompt and prediction.startswith(prompt):
        return prediction[len(prompt) :].strip()
    return prediction


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)
    overlap = sum((pred_counts & ref_counts).values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def best_caption_scores(prediction: str, references: list[str]) -> tuple[float, float]:
    best_exact = 0.0
    best_f1 = 0.0
    normalized_prediction = normalize_text(prediction)
    for reference in references:
        exact = 1.0 if normalized_prediction == normalize_text(reference) else 0.0
        f1 = token_f1(prediction, reference)
        best_exact = max(best_exact, exact)
        best_f1 = max(best_f1, f1)
    return best_exact, best_f1


def safe_json_loads(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def bbox_iou(box_a: list[float], box_b: list[float]) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    a_x2 = ax + aw
    a_y2 = ay + ah
    b_x2 = bx + bw
    b_y2 = by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(a_x2, b_x2)
    inter_y2 = min(a_y2, b_y2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    union = aw * ah + bw * bh - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def detection_metrics(prediction: str, target: str) -> dict[str, float]:
    pred_json = safe_json_loads(prediction)
    target_json = safe_json_loads(target)

    if pred_json is None or target_json is None:
        return {
            "json_parse_success": 0.0,
            "object_count_accuracy": 0.0,
            "label_accuracy": 0.0,
            "bbox_iou": 0.0,
        }

    pred_objects = pred_json.get("objects", [])
    target_objects = target_json.get("objects", [])
    paired = min(len(pred_objects), len(target_objects))
    if paired == 0:
        return {
            "json_parse_success": 1.0,
            "object_count_accuracy": 1.0 if len(pred_objects) == len(target_objects) else 0.0,
            "label_accuracy": 0.0,
            "bbox_iou": 0.0,
        }

    label_hits = 0.0
    total_iou = 0.0
    for pred_obj, target_obj in zip(pred_objects[:paired], target_objects[:paired], strict=True):
        if normalize_text(str(pred_obj.get("label", ""))) == normalize_text(str(target_obj.get("label", ""))):
            label_hits += 1.0
        pred_bbox = pred_obj.get("bbox", [0.0, 0.0, 0.0, 0.0])
        target_bbox = target_obj.get("bbox", [0.0, 0.0, 0.0, 0.0])
        if len(pred_bbox) == 4 and len(target_bbox) == 4:
            total_iou += bbox_iou(pred_bbox, target_bbox)

    return {
        "json_parse_success": 1.0,
        "object_count_accuracy": 1.0 if len(pred_objects) == len(target_objects) else 0.0,
        "label_accuracy": label_hits / paired,
        "bbox_iou": total_iou / paired,
    }


def evaluate_records(
    model,
    jsonl_path: str | Path,
    device: torch.device,
    max_new_tokens: int = 64,
    limit: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records = load_jsonl_records(jsonl_path)
    if limit is not None:
        records = records[:limit]

    task_totals: dict[str, Counter] = defaultdict(Counter)
    predictions: list[dict[str, Any]] = []

    for index, record in enumerate(records):
        image = Image.open(record["image"]).convert("RGB")
        pixel_values = model.image_processor(image, return_tensors="pt").pixel_values.to(device)
        prompt = select_prompt(record, deterministic=True)
        prediction = model.generate(
            pixel_values=pixel_values,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
        )
        content_prediction = strip_prompt_prefix(prediction, prompt)

        task = record.get("task", "caption")
        row: dict[str, Any] = {
            "index": index,
            "task": task,
            "image": record["image"],
            "prompt": prompt,
            "prediction": prediction,
            "prediction_content": content_prediction,
        }

        task_totals[task]["count"] += 1

        if task == "caption":
            references = record.get("captions", [record.get("target", "")])
            exact, f1 = best_caption_scores(content_prediction, references)
            task_totals[task]["exact_match"] += exact
            task_totals[task]["token_f1"] += f1
            row["references"] = references
            row["metrics"] = {"exact_match": exact, "token_f1": f1}
        elif task == "vqa":
            target = record["target"]
            exact = 1.0 if normalize_text(content_prediction) == normalize_text(target) else 0.0
            f1 = token_f1(content_prediction, target)
            task_totals[task]["exact_match"] += exact
            task_totals[task]["token_f1"] += f1
            row["target"] = target
            row["metrics"] = {"exact_match": exact, "token_f1": f1}
        elif task == "object_detection":
            metrics = detection_metrics(content_prediction, record["target"])
            for key, value in metrics.items():
                task_totals[task][key] += value
            row["target"] = record["target"]
            row["metrics"] = metrics
        else:
            target = record.get("target", "")
            exact = 1.0 if normalize_text(content_prediction) == normalize_text(target) else 0.0
            task_totals[task]["exact_match"] += exact
            row["target"] = target
            row["metrics"] = {"exact_match": exact}

        predictions.append(row)

    summary: dict[str, Any] = {"dataset": str(jsonl_path), "num_samples": len(records), "tasks": {}}
    for task, counter in task_totals.items():
        count = max(counter["count"], 1)
        task_summary = {"count": counter["count"]}
        for key, value in counter.items():
            if key != "count":
                task_summary[key] = value / count
        summary["tasks"][task] = task_summary

    return summary, predictions


def save_predictions(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_metrics(path: str | Path, metrics: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
```

## 5. `eval.py`

文件路径：

- `src\mllm_demo\eval.py`

```python
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
```

## 6. `infer.py`

文件路径：

- `src\mllm_demo\infer.py`

```python
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
```

## 7. 如何理解“所有模型代码”

如果从实验组织角度看，有 `Part 1 / Part 2 / Part 3 / Part 4` 多个阶段；  
但如果从代码实现角度看，真正的核心模型代码只有这一套：

- `MiniVLM`
- `VisionProjector`
- 统一 JSONL 数据组织
- 统一训练循环
- 统一推理与评估逻辑

各阶段差异主要体现在：

- 使用的视觉编码器大小不同；
- 使用的语言模型大小不同；
- 数据集不同；
- prompt 和目标文本格式不同；
- 训练脚本参数不同。

因此，后续写作时最稳的表述是：

**本实验不是为每个阶段分别设计独立模型，而是在统一最小多模态骨架上，通过任务文本化和阶段化训练实现能力扩展。**
