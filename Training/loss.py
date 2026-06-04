from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn

from Training.state_sampling import append_dims, sample_conditional_state


def endpoint_direction_label(x0: Tensor, eps: Tensor) -> Tensor:
    return eps - x0


def velocity_score_label(batch: dict[str, Tensor]) -> Tensor:
    x_t = batch["x_t"]
    v_t = batch["v_t"]
    mu_x = batch["mu_x"]
    mu_v = batch["mu_v"]
    s_b = append_dims(batch["S"], x_t.ndim)
    c_b = append_dims(batch["C"], x_t.ndim)
    q_b = append_dims(batch["Q"], x_t.ndim)
    det = (s_b * q_b - c_b.square()).clamp_min(1e-12)
    return (c_b * (x_t - mu_x) - s_b * (v_t - mu_v)) / det


def endpoint_direction_loss(pred_r: Tensor, target_r: Tensor) -> Tensor:
    return F.mse_loss(pred_r, target_r)


def velocity_score_loss(pred_score: Tensor, target_score: Tensor) -> Tensor:
    return F.mse_loss(pred_score, target_score)


def compute_training_losses(
    r_net: nn.Module,
    score_net: nn.Module,
    x0: Tensor,
    *,
    prompt_id: Tensor | None = None,
    text_embed: Tensor | None = None,
    tau_min: float = 0.02,
    tau_max: float = 0.2,
    lambda_const: float = 2.0,
    rho: float = 2.0,
    num_quad: int = 128,
) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    """Compute both training losses on the same sampled state.

    Returns (loss_r, loss_s, logs) where loss_r and loss_s are unscaled
    scalar tensors ready for independent backward() calls.

    loss_r only backpropagates through r_net; loss_s only through score_net.
    The two networks share zero parameters, so their gradients are isolated.
    """
    batch = sample_conditional_state(
        x0,
        tau_min=tau_min,
        tau_max=tau_max,
        lambda_const=lambda_const,
        rho=rho,
        num_quad=num_quad,
    )
    pred_r = r_net(batch["x_t"], batch["v_t"], batch["t"], batch["tau"], prompt_id=prompt_id, text_embed=text_embed)
    pred_score = score_net(
        batch["x_t"], batch["v_t"], batch["t"], batch["tau"], prompt_id=prompt_id, text_embed=text_embed
    )

    target_r = batch["r"]
    target_score = velocity_score_label(batch)

    loss_r = endpoint_direction_loss(pred_r, target_r)
    loss_s = velocity_score_loss(pred_score, target_score)

    logs = {
        "loss": (loss_r + loss_s).detach(),
        "loss_r": loss_r.detach(),
        "loss_s": loss_s.detach(),
    }
    return loss_r, loss_s, logs
