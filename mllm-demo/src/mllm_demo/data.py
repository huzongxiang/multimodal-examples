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
