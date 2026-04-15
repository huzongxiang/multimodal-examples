from colpali_engine.models import ColPali
from colpali_engine.models.paligemma.colpali.processing_colpali import ColPaliProcessor
from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor
from colpali_engine.utils.torch_utils import ListDataset
from torch.utils.data import DataLoader
import torch
from typing import List, cast
from pathlib import Path

from tqdm import tqdm
from PIL import Image
import os
from dotenv import load_dotenv

import spaces

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _resolve_device(requested_device: str | None = None) -> str:
    if requested_device:
        return requested_device

    env_device = os.getenv("COLPALI_DEVICE")
    if env_device:
        return env_device

    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def _resolve_torch_dtype(device: str):
    if device == "cuda":
        return torch.bfloat16

    if device == "mps":
        return torch.float16

    return torch.float32


def _resolve_model_name(default_model_name: str) -> str:
    configured_path = os.getenv("COLPALI_MODEL_PATH")
    if configured_path:
        model_path = Path(configured_path)
        if not model_path.is_absolute():
            model_path = BASE_DIR / model_path

        if model_path.exists():
            return str(model_path)

        print(f"Configured COLPALI_MODEL_PATH does not exist: {model_path}")

    default_local_path = BASE_DIR / "models" / "vidore__colpali-v1.2"
    if default_local_path.exists():
        return str(default_local_path)

    return os.getenv("COLPALI_MODEL_NAME", default_model_name)

class ColpaliManager:

    
    def __init__(self, device = None, model_name = "vidore/colpali-v1.2"):
        self.device = _resolve_device(device)
        self.model_name = _resolve_model_name(model_name)
        self.torch_dtype = _resolve_torch_dtype(self.device)
        self.model = None
        self.processor = None

        print(
            f"Initializing ColpaliManager with device={self.device}, "
            f"model={self.model_name}, torch_dtype={self.torch_dtype}"
        )

    def _ensure_model_loaded(self):
        if self.model is not None and self.processor is not None:
            return

        self.model = ColPali.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
            device_map=self.device,
        ).eval()

        self.processor = cast(
            ColPaliProcessor,
            ColPaliProcessor.from_pretrained(self.model_name),
        )

    @spaces.GPU
    def get_images(self, paths: list[str]) -> List[Image.Image]:
        return [Image.open(path) for path in paths]

    @spaces.GPU
    def process_images(self, image_paths:list[str], batch_size=5):
        self._ensure_model_loaded()

        print(f"Processing {len(image_paths)} image_paths")
        
        images = self.get_images(image_paths)

        dataloader = DataLoader(
            dataset=ListDataset[str](images),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda x: self.processor.process_images(x),
        )

        ds: List[torch.Tensor] = []
        for batch_doc in tqdm(dataloader):
            with torch.no_grad():
                batch_doc = {k: v.to(self.model.device) for k, v in batch_doc.items()}
                embeddings_doc = self.model(**batch_doc)
            ds.extend(list(torch.unbind(embeddings_doc.to(self.device))))
                
        ds_np = [d.float().cpu().numpy() for d in ds]

        return ds_np
    

    @spaces.GPU
    def process_text(self, texts: list[str]):
        self._ensure_model_loaded()
        print(f"Processing {len(texts)} texts")

        dataloader = DataLoader(
            dataset=ListDataset[str](texts),
            batch_size=1,
            shuffle=False,
            collate_fn=lambda x: self.processor.process_queries(x),
        )

        qs: List[torch.Tensor] = []
        for batch_query in dataloader:
            with torch.no_grad():
                batch_query = {k: v.to(self.model.device) for k, v in batch_query.items()}
                embeddings_query = self.model(**batch_query)

            qs.extend(list(torch.unbind(embeddings_query.to(self.device))))

        qs_np = [q.float().cpu().numpy() for q in qs]

        return qs_np
    
