from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from Model.networks import TinyDiT
from Sampling.reverse import sample_reverse_kinetic
from Sampling.text_encoder import FrozenTextEncoder


@dataclass
class GeneratorBundle:
    r_net: nn.Module
    score_net: nn.Module
    text_encoder: FrozenTextEncoder | None
    config: dict
    device: torch.device


def expand_prompts(prompts: list[str], batch_size: int) -> list[str]:
    if len(prompts) == 1:
        return prompts * batch_size
    if len(prompts) != batch_size:
        raise ValueError("--prompt must be supplied once or exactly batch-size times")
    return prompts


def load_generator(
    checkpoint_path: str,
    *,
    config: dict | None = None,
    device: torch.device | None = None,
) -> GeneratorBundle:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = config or ckpt.get("config")
    if cfg is None:
        raise ValueError("checkpoint does not contain config; pass the training config explicitly")
    model_cfg = dict(cfg["model"])

    text_encoder = None
    text_cfg = cfg.get("text_encoder", {})
    if text_cfg.get("enabled", False):
        text_encoder = FrozenTextEncoder(
            model_name=text_cfg.get("model_name", "openai/clip-vit-base-patch32"),
            encoder_type=text_cfg.get("type", "clip"),
            max_length=text_cfg.get("max_length"),
        ).to(device)
        text_encoder.eval()
        if model_cfg.get("text_embed_dim", 0) <= 0:
            model_cfg["text_embed_dim"] = text_encoder.out_dim

    r_net = TinyDiT(**model_cfg).to(device)
    score_net = TinyDiT(**model_cfg).to(device)
    r_net.load_state_dict(ckpt["r_net"])
    score_net.load_state_dict(ckpt["score_net"])
    r_net.eval()
    score_net.eval()

    return GeneratorBundle(r_net=r_net, score_net=score_net, text_encoder=text_encoder, config=cfg, device=device)


@torch.no_grad()
def encode_generation_prompts(bundle: GeneratorBundle, prompts: list[str], batch_size: int) -> Tensor | None:
    if bundle.text_encoder is None:
        return None
    prompts = expand_prompts(prompts, batch_size)
    return bundle.text_encoder(prompts, bundle.device)


@torch.no_grad()
def generate_samples(
    bundle: GeneratorBundle,
    *,
    prompts: list[str],
    batch_size: int = 16,
    steps: int = 100,
    tau: float | None = None,
    eta: float = 0.0,
    prompt_id: int | None = None,
) -> Tensor:
    model_cfg = dict(bundle.config["model"])
    kinetic_cfg = bundle.config["kinetic"]

    prompt_id_tensor = None
    if model_cfg.get("num_prompts", 0) > 0:
        if prompt_id is None:
            raise ValueError("prompt_id is required for prompt-id conditioned checkpoints")
        prompt_id_tensor = torch.full((batch_size,), prompt_id, dtype=torch.long, device=bundle.device)

    text_embed = None
    if bundle.text_encoder is not None:
        if not prompts:
            raise ValueError("prompts are required for text-conditioned checkpoints")
        text_embed = encode_generation_prompts(bundle, prompts, batch_size)

    tau = tau if tau is not None else float(kinetic_cfg.get("tau_max", kinetic_cfg.get("tau_min", 0.05)))
    return sample_reverse_kinetic(
        bundle.r_net,
        bundle.score_net,
        batch_size=batch_size,
        image_size=model_cfg["image_size"],
        channels=model_cfg.get("out_channels", 3),
        steps=steps,
        tau=tau,
        eta=eta,
        lambda_const=kinetic_cfg.get("lambda_const", 2.0),
        rho=kinetic_cfg.get("rho", 2.0),
        num_quad=kinetic_cfg.get("num_quad", 128),
        prompt_id=prompt_id_tensor,
        text_embed=text_embed,
        device=bundle.device,
    )


@torch.no_grad()
def generate_samples_from_checkpoint(
    checkpoint_path: str,
    *,
    config: dict | None = None,
    prompts: list[str],
    batch_size: int = 16,
    steps: int = 100,
    tau: float | None = None,
    eta: float = 0.0,
    prompt_id: int | None = None,
    device: torch.device | None = None,
) -> Tensor:
    bundle = load_generator(checkpoint_path, config=config, device=device)
    return generate_samples(
        bundle,
        prompts=prompts,
        batch_size=batch_size,
        steps=steps,
        tau=tau,
        eta=eta,
        prompt_id=prompt_id,
    )
