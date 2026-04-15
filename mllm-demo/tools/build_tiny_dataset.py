from __future__ import annotations

import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "tiny"
IMAGES_DIR = DATA_DIR / "images"

SOURCE_IMAGES = {
    "lion.jpg": Path(
        r"D:\codex\multimodal-examples\.conda-envs\tld-demo\Lib\site-packages\gradio\media_assets\images\lion.jpg"
    ),
    "bus.png": Path(
        r"D:\codex\multimodal-examples\.conda-envs\tld-demo\Lib\site-packages\gradio\media_assets\images\bus.png"
    ),
    "tower.jpg": Path(
        r"D:\codex\multimodal-examples\.conda-envs\tld-demo\Lib\site-packages\gradio\media_assets\images\tower.jpg"
    ),
    "cheetah1.jpg": Path(
        r"D:\codex\multimodal-examples\.conda-envs\tld-demo\Lib\site-packages\gradio\test_data\cheetah1-copy.jpg"
    ),
    "cheetah2.jpg": Path(
        r"D:\codex\multimodal-examples\.conda-envs\tld-demo\Lib\site-packages\gradio\test_data\cheetah2.jpg"
    ),
}


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_caption_records() -> list[dict]:
    return [
        {
            "image": str((IMAGES_DIR / "lion.jpg").resolve()),
            "task": "caption",
            "prompt": "Describe the image in one sentence.",
            "target": "A lion is lying on the grass.",
        },
        {
            "image": str((IMAGES_DIR / "bus.png").resolve()),
            "task": "caption",
            "prompt": "Describe the image in one sentence.",
            "target": "A blue bus is shown from the side.",
        },
        {
            "image": str((IMAGES_DIR / "tower.jpg").resolve()),
            "task": "caption",
            "prompt": "Describe the image in one sentence.",
            "target": "A tall tower rises above the city.",
        },
        {
            "image": str((IMAGES_DIR / "cheetah1.jpg").resolve()),
            "task": "caption",
            "prompt": "Describe the image in one sentence.",
            "target": "A cheetah is standing in dry grass.",
        },
        {
            "image": str((IMAGES_DIR / "cheetah2.jpg").resolve()),
            "task": "caption",
            "prompt": "Describe the image in one sentence.",
            "target": "A cheetah is walking across the field.",
        },
    ]


def build_detection_records() -> list[dict]:
    return [
        {
            "image": str((IMAGES_DIR / "lion.jpg").resolve()),
            "task": "detection",
            "prompt": "Detect the main object and return JSON with normalized bbox coordinates.",
            "target": '{"objects":[{"label":"lion","bbox":[0.08,0.18,0.95,0.95]}]}',
        },
        {
            "image": str((IMAGES_DIR / "bus.png").resolve()),
            "task": "detection",
            "prompt": "Detect the main object and return JSON with normalized bbox coordinates.",
            "target": '{"objects":[{"label":"bus","bbox":[0.10,0.22,0.92,0.84]}]}',
        },
        {
            "image": str((IMAGES_DIR / "cheetah1.jpg").resolve()),
            "task": "detection",
            "prompt": "Detect the main object and return JSON with normalized bbox coordinates.",
            "target": '{"objects":[{"label":"cheetah","bbox":[0.05,0.14,0.95,0.93]}]}',
        },
    ]


def build_vqa_records() -> list[dict]:
    return [
        {
            "image": str((IMAGES_DIR / "lion.jpg").resolve()),
            "task": "vqa",
            "prompt": "Question: What animal is in the image?\nAnswer:",
            "target": "lion",
        },
        {
            "image": str((IMAGES_DIR / "bus.png").resolve()),
            "task": "vqa",
            "prompt": "Question: What vehicle is shown?\nAnswer:",
            "target": "bus",
        },
        {
            "image": str((IMAGES_DIR / "tower.jpg").resolve()),
            "task": "vqa",
            "prompt": "Question: What structure is shown in the image?\nAnswer:",
            "target": "tower",
        },
        {
            "image": str((IMAGES_DIR / "cheetah2.jpg").resolve()),
            "task": "vqa",
            "prompt": "Question: What animal is walking in the field?\nAnswer:",
            "target": "cheetah",
        },
    ]


def main() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    for filename, source_path in SOURCE_IMAGES.items():
        if not source_path.exists():
            raise FileNotFoundError(f"Missing source image: {source_path}")
        shutil.copy2(source_path, IMAGES_DIR / filename)

    caption_records = build_caption_records()
    detection_records = build_detection_records()
    vqa_records = build_vqa_records()
    multitask_records = caption_records + detection_records + vqa_records

    write_jsonl(DATA_DIR / "caption.jsonl", caption_records)
    write_jsonl(DATA_DIR / "detection.jsonl", detection_records)
    write_jsonl(DATA_DIR / "vqa.jsonl", vqa_records)
    write_jsonl(DATA_DIR / "multitask.jsonl", multitask_records)

    print(f"Wrote dataset to {DATA_DIR}")
    print(f"caption={len(caption_records)} detection={len(detection_records)} vqa={len(vqa_records)}")


if __name__ == "__main__":
    main()
