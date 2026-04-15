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
