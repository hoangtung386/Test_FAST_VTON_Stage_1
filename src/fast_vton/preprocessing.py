"""Input preprocessing for one-step virtual try-on.

Responsibilities:

* Convert a PIL image into the normalised pixel tensor the VAE expects.
* Build the *agnostic* person view (garment area painted grey) using human parsing.
  The segmentation model is loaded lazily and cached per :class:`AgnosticBuilder`
  instance, so repeated calls -- e.g. a Gradio session -- do not reload it.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_tensor

from fast_vton.vendors import build_agnostic_mask

logger = logging.getLogger(__name__)


@torch.no_grad()
def image_to_tensor(image: Image.Image, width: int, height: int, device: str | torch.device) -> torch.Tensor:
    """Resize ``image`` to ``(width, height)`` and scale pixels from ``[0, 1]`` to ``[-1, 1]``.

    Args:
        image: Source photograph.
        width: Target width in pixels.
        height: Target height in pixels.
        device: Device the returned tensor is placed on.

    Returns:
        A ``(1, 3, height, width)`` float tensor on ``device``.
    """
    resized = image.convert("RGB").resize((width, height), Image.BILINEAR)
    return to_tensor(resized).unsqueeze(0).to(device) * 2 - 1


class AgnosticBuilder:
    """Lazily-loaded human-parsing model that paints the garment area grey.

    The segmentation backbone (``mattmdjaga/segformer_b2_clothes``) is fetched on the
    first call and reused afterwards, which avoids re-downloading / re-building the
    model every time :meth:`__call__` runs.
    """

    def __init__(
        self,
        repo: str,
        clothing_ids: tuple[int, ...],
        device: str | torch.device = "cuda",
    ) -> None:
        self.repo = repo
        self.clothing_ids = tuple(clothing_ids)
        self.device = torch.device(device)
        self._processor = None
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import (
            AutoImageProcessor as SegProcessor,
        )
        from transformers import (
            SegformerForSemanticSegmentation,
        )

        logger.info("loading human-parsing model from %s", self.repo)
        self._processor = SegProcessor.from_pretrained(self.repo)
        self._model = (
            SegformerForSemanticSegmentation.from_pretrained(self.repo)
            .to(self.device)
            .eval()
        )

    @torch.no_grad()
    def __call__(self, person_image: Image.Image) -> Image.Image:
        """Return ``person_image`` with the clothing region painted mid-grey.

        Args:
            person_image: Photograph of the person.

        Returns:
            A new PIL image where every pixel classified as clothing is set to
            ``(128, 128, 128)``.
        """
        self._ensure_loaded()
        assert self._processor is not None and self._model is not None
        inputs = self._processor(images=person_image, return_tensors="pt").to(self.device)
        logits = self._model(**inputs).logits
        seg = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

        width, height = person_image.size
        if seg.shape[0] != height or seg.shape[1] != width:
            seg = np.array(Image.fromarray(seg).resize((width, height), Image.NEAREST))

        cloth_mask = np.isin(seg, self.clothing_ids)
        array = np.array(person_image.convert("RGB")).copy()
        array[cloth_mask] = (128, 128, 128)
        return Image.fromarray(array)

    def clear(self) -> None:
        """Drop the cached model to free VRAM (call after a batch / on shutdown)."""
        self._model = None
        self._processor = None


def build_mask_latent(
    person_image: Image.Image,
    agnostic_image: Image.Image,
    size: tuple[int, int],
    diff_threshold: int,
    morph_kernel: int,
    device: str | torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build the inpainting mask and downsample it to latent resolution.

    Args:
        person_image: Original person photograph.
        agnostic_image: Person photograph with the garment area removed.
        size: ``(width, height)`` the mask is computed at.
        diff_threshold: Per-channel intensity delta to flag a removed pixel.
        morph_kernel: Morphological structuring-element diameter.
        device: Target device.
        dtype: Target dtype.

    Returns:
        A ``(1, 1, height // 8, width // 8)`` mask tensor.
    """
    mask = build_agnostic_mask(
        person_image, agnostic_image, size, diff_threshold, morph_kernel
    )
    mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float().to(device)
    return F.interpolate(
        mask_tensor, (size[1] // 8, size[0] // 8), mode="nearest"
    ).to(dtype)
