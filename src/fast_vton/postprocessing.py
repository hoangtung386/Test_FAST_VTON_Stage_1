"""Postprocessing: turn model latents / tensors back into PIL images."""

from __future__ import annotations

import torch
from PIL import Image


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a ``(1, 3, H, W)`` float tensor in ``[0, 1]`` to a PIL RGB image.

    Args:
        tensor: Model output tensor on any device.

    Returns:
        The same image as a PIL ``RGB`` image.
    """
    array = tensor[0].permute(1, 2, 0).detach().cpu().float().clamp(0, 1).numpy()
    return Image.fromarray((array * 255).astype("uint8"))


def decode_latent(
    pred: torch.Tensor,
    vae: torch.nn.Module,
    scheduler: torch.nn.Module,
) -> torch.Tensor:
    """Optionally threshold / clip the predicted ``x0`` and decode it through the VAE.

    This mirrors the final stage of the training pipeline: clamp the predicted
    denoised latent according to the scheduler's sample rule, then decode to pixels.

    Args:
        pred: Predicted ``x0`` latent (float32, ``(1, 4, H/8, W/8)``).
        vae: The bundle VAE (kept in fp32 for decoding).
        scheduler: DDPM scheduler providing ``thresholding`` / ``clip_sample`` config.

    Returns:
        A ``(1, 3, H, W)`` tensor in ``[0, 1]``.
    """
    if scheduler.config.thresholding:
        pred = scheduler._threshold_sample(pred)
    elif scheduler.config.clip_sample:
        pred = pred.clamp(
            -scheduler.config.clip_sample_range,
            scheduler.config.clip_sample_range,
        )
    image = vae.decode((pred / vae.config.scaling_factor).to(vae.dtype)).sample.float()
    return (image + 1) / 2
