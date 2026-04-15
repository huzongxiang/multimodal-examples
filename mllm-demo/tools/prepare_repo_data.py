from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from datasets import concatenate_datasets, load_dataset


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "repo"

OD_PROMPTS = [
    "What objects are in this image? Output as JSON.",
    "Detect all objects in this image and return JSON with bounding boxes.",
    "List the objects with their locations in JSON format.",
    "Find all objects and output their bounding boxes as JSON.",
]

MULTITASK_OD_PROMPTS = [
    "What objects are in this image? Output as JSON.",
    "Detect all objects and return JSON with bounding boxes.",
    "List objects with locations in JSON format.",
]


def save_image(image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path)


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def create_od_json(objects, width, height, category_names) -> str:
    result = {"objects": []}
    for bbox, cat_id in zip(objects["bbox"], objects["category"], strict=True):
        x, y, w, h = bbox
        norm_bbox = [
            round(x / width, 3),
            round(y / height, 3),
            round(w / width, 3),
            round(h / height, 3),
        ]
        result["objects"].append({"label": category_names[cat_id], "bbox": norm_bbox})
    return json.dumps(result)


def most_common_answer(answers) -> str:
    counts = Counter(answer["answer"] for answer in answers)
    return counts.most_common(1)[0][0]


def convert_caption_split(dataset, output_dir: Path, limit: int | None = None) -> Path:
    if limit is not None:
        dataset = dataset.shuffle(seed=42).select(range(limit))
    image_dir = output_dir / "images"
    records: list[dict] = []

    for idx, item in enumerate(dataset):
        image_path = image_dir / f"{idx:05d}.jpg"
        save_image(item["image"], image_path)
        records.append(
            {
                "image": str(image_path.resolve()),
                "task": "caption",
                "captions": [item[f"caption_{i}"] for i in range(5)],
                "append_eos": False,
                "mask_prompt_loss": False,
            }
        )

    output_jsonl = output_dir / "data.jsonl"
    write_jsonl(output_jsonl, records)
    return output_jsonl


def prepare_part1_caption() -> tuple[Path, Path, Path]:
    train_dataset = load_dataset("jxie/flickr8k", split="train")
    val_dataset = load_dataset("jxie/flickr8k", split="validation")
    test_dataset = load_dataset("jxie/flickr8k", split="test")
    output_dir = DATA_ROOT / "part1_flickr8k"
    train_jsonl = convert_caption_split(train_dataset, output_dir / "train", limit=2000)
    val_jsonl = convert_caption_split(val_dataset, output_dir / "validation")
    test_jsonl = convert_caption_split(test_dataset, output_dir / "test")
    return train_jsonl, val_jsonl, test_jsonl


def convert_od_split(dataset, output_dir: Path) -> Path:
    category_names = dataset.features["objects"]["category"].feature.names
    image_dir = output_dir / "images"
    records: list[dict] = []

    for idx, item in enumerate(dataset):
        image_path = image_dir / f"{idx:05d}.jpg"
        save_image(item["image"], image_path)
        records.append(
            {
                "image": str(image_path.resolve()),
                "task": "object_detection",
                "prompt_options": OD_PROMPTS,
                "target": create_od_json(item["objects"], item["width"], item["height"], category_names),
                "append_eos": True,
                "mask_prompt_loss": True,
            }
        )

    output_jsonl = output_dir / "data.jsonl"
    write_jsonl(output_jsonl, records)
    return output_jsonl


def prepare_part2_od() -> tuple[Path, Path]:
    od_train = load_dataset("Francesco/animals-ij5d2", split="train")
    od_val = load_dataset("Francesco/animals-ij5d2", split="validation")
    output_dir = DATA_ROOT / "part2_animals_od"
    train_jsonl = convert_od_split(concatenate_datasets([od_train, od_val]), output_dir / "train")
    test_jsonl = convert_od_split(load_dataset("Francesco/animals-ij5d2", split="test"), output_dir / "test")
    return train_jsonl, test_jsonl


def prepare_part3_vqa() -> tuple[Path, Path]:
    stream = load_dataset("lmms-lab/VQAv2", split="validation", streaming=True)
    output_dir = DATA_ROOT / "part3_vqav2"
    image_dir = output_dir / "images"
    train_records: list[dict] = []
    test_records: list[dict] = []

    for idx, item in enumerate(stream):
        if idx >= 4000:
            break
        image_path = image_dir / f"{idx:05d}.jpg"
        save_image(item["image"], image_path)
        record = {
            "image": str(image_path.resolve()),
            "task": "vqa",
            "prompt": f"Question: {item['question']} Answer:",
            "target": most_common_answer(item["answers"]),
            "append_eos": True,
            "mask_prompt_loss": True,
        }
        if idx < 3600:
            train_records.append(record)
        else:
            test_records.append(record)

    train_jsonl = output_dir / "train.jsonl"
    test_jsonl = output_dir / "test.jsonl"
    write_jsonl(train_jsonl, train_records)
    write_jsonl(test_jsonl, test_records)
    return train_jsonl, test_jsonl


def prepare_part4_multitask() -> Path:
    caption_dataset = load_dataset("jxie/flickr8k", split="train").shuffle(seed=42).select(range(1000))
    od_train = load_dataset("Francesco/animals-ij5d2", split="train")
    od_val = load_dataset("Francesco/animals-ij5d2", split="validation")
    od_dataset = concatenate_datasets([od_train, od_val]).select(range(400))
    category_names = od_dataset.features["objects"]["category"].feature.names

    vqa_stream = load_dataset("lmms-lab/VQAv2", split="validation", streaming=True)
    vqa_samples = []
    for idx, item in enumerate(vqa_stream):
        if idx >= 1000:
            break
        vqa_samples.append(item)

    output_dir = DATA_ROOT / "part4_multitask"
    image_dir = output_dir / "images"
    records: list[dict] = []

    for idx, item in enumerate(caption_dataset):
        image_path = image_dir / "caption" / f"{idx:05d}.jpg"
        save_image(item["image"], image_path)
        records.append(
            {
                "image": str(image_path.resolve()),
                "task": "caption",
                "prompt": "Describe this image.",
                "captions": [item[f"caption_{i}"] for i in range(5)],
                "append_eos": True,
                "mask_prompt_loss": True,
            }
        )

    for idx, item in enumerate(od_dataset):
        image_path = image_dir / "od" / f"{idx:05d}.jpg"
        save_image(item["image"], image_path)
        records.append(
            {
                "image": str(image_path.resolve()),
                "task": "object_detection",
                "prompt_options": MULTITASK_OD_PROMPTS,
                "target": create_od_json(item["objects"], item["width"], item["height"], category_names),
                "append_eos": True,
                "mask_prompt_loss": True,
            }
        )

    for idx, item in enumerate(vqa_samples):
        image_path = image_dir / "vqa" / f"{idx:05d}.jpg"
        save_image(item["image"], image_path)
        records.append(
            {
                "image": str(image_path.resolve()),
                "task": "vqa",
                "prompt": f"Question: {item['question']} Answer:",
                "target": most_common_answer(item["answers"]),
                "append_eos": True,
                "mask_prompt_loss": True,
            }
        )

    output_jsonl = output_dir / "train.jsonl"
    write_jsonl(output_jsonl, records)
    return output_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare local JSONL data using the same sources as VLM from Scratch.")
    parser.add_argument(
        "--stage",
        choices=["part1", "part2", "part3", "part4", "all"],
        default="all",
        help="Which project stage dataset to prepare.",
    )
    args = parser.parse_args()

    outputs: list[str] = []

    if args.stage in {"part1", "all"}:
        train_path, val_path, test_path = prepare_part1_caption()
        outputs.append(f"part1_train={train_path}")
        outputs.append(f"part1_validation={val_path}")
        outputs.append(f"part1_test={test_path}")
    if args.stage in {"part2", "all"}:
        train_path, test_path = prepare_part2_od()
        outputs.append(f"part2_train={train_path}")
        outputs.append(f"part2_test={test_path}")
    if args.stage in {"part3", "all"}:
        train_path, test_path = prepare_part3_vqa()
        outputs.append(f"part3_train={train_path}")
        outputs.append(f"part3_test={test_path}")
    if args.stage in {"part4", "all"}:
        outputs.append(f"part4={prepare_part4_multitask()}")

    for line in outputs:
        print(line)


if __name__ == "__main__":
    main()
