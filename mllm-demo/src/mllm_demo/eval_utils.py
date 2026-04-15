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
