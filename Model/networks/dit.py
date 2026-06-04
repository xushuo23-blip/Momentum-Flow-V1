from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1.0 + scale[:, None, :]) + shift[:, None, :]


class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: Tensor, tau: Tensor) -> Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / max(half - 1, 1)
        )
        emb_t = t[:, None] * freqs[None, :]
        emb_tau = tau.log()[:, None] * freqs[None, :]
        emb = torch.cat([emb_t.sin(), emb_t.cos(), emb_tau.sin(), emb_tau.cos()], dim=-1)
        return self.mlp(emb)


class DiTBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.ada(cond).chunk(6, dim=-1)
        h = modulate(self.norm1(x), shift_msa, scale_msa)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate_msa[:, None, :] * h
        h = self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x + gate_mlp[:, None, :] * h


class TinyDiT(nn.Module):
    """Small DiT for image-shaped kinetic flow targets.

    Input channels are 6 by default: concat(x_t, v_t). Output channels are 3.
    """

    def __init__(
        self,
        *,
        image_size: int = 32,
        in_channels: int = 6,
        out_channels: int = 3,
        patch_size: int = 2,
        dim: int = 384,
        depth: int = 8,
        heads: int = 6,
        dropout: float = 0.0,
        num_prompts: int = 0,
        text_embed_dim: int = 0,
    ):
        super().__init__()
        assert image_size % patch_size == 0
        self.image_size = image_size
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.num_prompts = num_prompts
        self.text_embed_dim = text_embed_dim
        patch_dim = in_channels * patch_size * patch_size
        out_patch_dim = out_channels * patch_size * patch_size
        num_patches = (image_size // patch_size) ** 2

        self.patch_embed = nn.Linear(patch_dim, dim)
        self.pos = nn.Parameter(torch.zeros(1, num_patches, dim))
        self.time_embed = TimestepEmbedding(dim)
        self.prompt_embed = nn.Embedding(num_prompts, dim) if num_prompts > 0 else None
        self.text_proj = nn.Linear(text_embed_dim, dim) if text_embed_dim > 0 else None
        self.blocks = nn.ModuleList([DiTBlock(dim, heads, dropout=dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 2))
        self.out = nn.Linear(dim, out_patch_dim)
        nn.init.normal_(self.pos, std=0.02)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def patchify(self, x: Tensor) -> Tensor:
        p = self.patch_size
        b, c, h, w = x.shape
        x = x.reshape(b, c, h // p, p, w // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5)
        return x.reshape(b, (h // p) * (w // p), c * p * p)

    def unpatchify(self, x: Tensor) -> Tensor:
        p = self.patch_size
        b, n, d = x.shape
        h = w = int(math.sqrt(n))
        c = self.out_channels
        x = x.reshape(b, h, w, c, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5)
        return x.reshape(b, c, h * p, w * p)

    def forward(
        self,
        x_t: Tensor,
        v_t: Tensor,
        t: Tensor,
        tau: Tensor,
        prompt_id: Tensor | None = None,
        text_embed: Tensor | None = None,
    ) -> Tensor:
        x = torch.cat([x_t, v_t], dim=1)
        tokens = self.patch_embed(self.patchify(x)) + self.pos
        cond = self.time_embed(t, tau)
        if self.prompt_embed is not None:
            if prompt_id is None:
                raise ValueError("prompt_id is required when num_prompts > 0")
            cond = cond + self.prompt_embed(prompt_id)
        if self.text_proj is not None:
            if text_embed is None:
                raise ValueError("text_embed is required when text_embed_dim > 0")
            cond = cond + self.text_proj(text_embed)
        for block in self.blocks:
            tokens = block(tokens, cond)
        shift, scale = self.final_ada(cond).chunk(2, dim=-1)
        tokens = modulate(self.norm(tokens), shift, scale)
        return self.unpatchify(self.out(tokens))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
