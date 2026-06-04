from __future__ import annotations

import torch
from torch import Tensor, nn


class FrozenTextEncoder(nn.Module):
    """Frozen pretrained text encoder used by standalone generation."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        *,
        encoder_type: str = "clip",
        max_length: int | None = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.encoder_type = encoder_type.lower()
        self.max_length = max_length

        if self.encoder_type == "clip":
            try:
                from transformers import CLIPTextModel, CLIPTokenizer
            except ImportError as exc:
                raise ImportError("text_encoder.type=clip requires `pip install transformers`") from exc
            self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
            self.encoder = CLIPTextModel.from_pretrained(model_name)
            self.out_dim = int(self.encoder.config.hidden_size)
            if self.max_length is None:
                self.max_length = int(self.tokenizer.model_max_length)
        elif self.encoder_type == "t5":
            try:
                from transformers import T5EncoderModel, T5TokenizerFast
            except ImportError as exc:
                raise ImportError("text_encoder.type=t5 requires `pip install transformers sentencepiece`") from exc
            self.tokenizer = T5TokenizerFast.from_pretrained(model_name)
            self.encoder = T5EncoderModel.from_pretrained(model_name)
            self.out_dim = int(self.encoder.config.d_model)
            if self.max_length is None:
                self.max_length = 128
        else:
            raise ValueError(f"unknown text encoder type: {encoder_type}")

        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad_(False)

    def forward(self, prompts: list[str], device: torch.device) -> Tensor:
        tokens = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}
        with torch.no_grad():
            outputs = self.encoder(**tokens)

        if self.encoder_type == "clip":
            return outputs.pooler_output

        hidden = outputs.last_hidden_state
        mask = tokens["attention_mask"].to(hidden.dtype)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (hidden * mask[:, :, None]).sum(dim=1) / denom
