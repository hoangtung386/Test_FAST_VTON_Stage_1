"""One-step virtual try-on pipeline for the Fast-VTON Stage 1 bundle.

The pipeline owns every piece of mutable state needed for inference: the loaded bundle,
the DDPM scheduler (only used to read ``alpha_t`` / ``sigma_t``), the garment / CLIP
image processors, and a lazily-built human-parsing model for auto-agnostic generation.

The public surface is intentionally small:

* :meth:`FastVTONPipeline.try_on` runs the forward pass on a person / agnostic / garment
  triple and returns a ``(1, 3, H, W)`` tensor in ``[0, 1]``.
* :meth:`FastVTONPipeline.build_agnostic` paints the garment area grey using human
  parsing.

Everything else (tensor conversion, mask building, VAE decode) lives in
:mod:`fast_vton.preprocessing` and :mod:`fast_vton.postprocessing`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from diffusers import DDPMScheduler
from PIL import Image
from transformers import AutoImageProcessor, CLIPImageProcessor

from fast_vton.config import Config, config_from_bundle
from fast_vton.postprocessing import decode_latent
from fast_vton.preprocessing import AgnosticBuilder, build_mask_latent, image_to_tensor
from fast_vton.vendors import (
    INVERSION_TIMESTEP,
    LoadedBundle,
    load_bundle,
    pad_to_square,
)

logger = logging.getLogger(__name__)

_DTYPES: dict[str, torch.dtype] = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


class FastVTONPipeline:
    """Load a Fast-VTON bundle and run one-step virtual try-on."""

    def __init__(
        self,
        config: Config | None = None,
        bundle_path: str | Path | None = None,
        device: str | None = None,
    ) -> None:
        """Build the pipeline.

        Args:
            config: Explicit configuration. If omitted, defaults are used and
                ``bundle_path`` (or ``models/Fast_VTON_full.pt``) selects the bundle.
            bundle_path: Optional full path to the ``.pt`` bundle. Takes precedence over
                ``config.bundle_path`` when supplied.
            device: Torch device string; overrides ``config.device`` when supplied.

        Raises:
            FileNotFoundError: If the resolved bundle file does not exist.
        """
        self.config = config or Config()
        if bundle_path is not None:
            self.config = config_from_bundle(bundle_path)
        self.device = torch.device(device or self.config.device)

        bundle_file = self.config.bundle_path
        if not bundle_file.exists():
            raise FileNotFoundError(
                f"Không tìm thấy bundle: {bundle_file}. "
                f"Hãy đặt file Fast_VTON_full.pt vào thư mục models/."
            )

        self.bundle: LoadedBundle = load_bundle(bundle_file, device=self.device)
        manifest = self.bundle.manifest
        self.height: int = manifest.height
        self.width: int = manifest.width
        self.garment_resolution: int = manifest.garment_resolution
        self.dtype: torch.dtype = _DTYPES[manifest.dtype]

        self._init_scheduler()
        self._init_processors()
        self._init_null_embedding()

        self._agnostic_builder: AgnosticBuilder | None = None
        if self.device.type == "cpu":
            logger.warning("CUDA không khả dụng, chạy trên CPU (rất chậm).")

    # ------------------------------------------------------------------ init helpers
    def _init_scheduler(self) -> None:
        self.scheduler = DDPMScheduler.from_pretrained(
            self.config.scheduler_repo, subfolder="scheduler"
        )
        num_train = self.scheduler.config.num_train_timesteps
        alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)
        t = torch.tensor([num_train - 1], device=self.device)
        self.alpha_t = (alphas_cumprod[t] ** 0.5).view(-1, 1, 1, 1)
        self.sigma_t = ((1 - alphas_cumprod[t]) ** 0.5).view(-1, 1, 1, 1)
        self.timestep = torch.full((1,), num_train - 1, dtype=torch.int64, device=self.device)
        self.inv_timestep = torch.full(
            (1,), INVERSION_TIMESTEP, dtype=torch.int64, device=self.device
        )

    def _init_processors(self) -> None:
        self.garment_processor = AutoImageProcessor.from_pretrained(self.config.dinov2_repo)
        self.garment_processor.do_center_crop = False
        self.garment_processor.size = {"shortest_edge": self.garment_resolution}
        self.clip_processor = CLIPImageProcessor()

    def _init_null_embedding(self) -> None:
        if self.bundle.null_embedding is not None:
            self.null_embedding = self.bundle.null_embedding.to(self.device, self.dtype)
        else:
            self.null_embedding = torch.zeros(
                (1, 77, 1024), device=self.device, dtype=self.dtype
            )

    # -------------------------------------------------------------- agnostic handling
    def build_agnostic(self, person_image: Image.Image) -> Image.Image:
        """Return ``person_image`` with the clothing region painted grey.

        The human-parsing model is created on first use and reused afterwards.
        """
        if self._agnostic_builder is None:
            self._agnostic_builder = AgnosticBuilder(
                self.config.segformer_repo, self.config.seg_clothing_ids, self.device
            )
        return self._agnostic_builder(person_image)

    def clear_agnostic_model(self) -> None:
        """Free the cached human-parsing model to reclaim VRAM."""
        if self._agnostic_builder is not None:
            self._agnostic_builder.clear()

    # ------------------------------------------------------------------- inference
    @torch.no_grad()
    def encode_latent(self, pixel_tensor: torch.Tensor) -> torch.Tensor:
        """Encode a ``[-1, 1]`` pixel tensor to a scaled VAE latent."""
        vae = self.bundle.vae
        latents = vae.encode(pixel_tensor.to(vae.dtype)).latent_dist.sample()
        return latents * vae.config.scaling_factor

    @torch.no_grad()
    def try_on(self, person_image, agnostic_image, garment_image) -> torch.Tensor:
        """Run one-step virtual try-on and return the result as a ``[0, 1]`` tensor.

        Args:
            person_image: PIL image of the person.
            agnostic_image: PIL image of the person with the garment area removed.
            garment_image: PIL image of the garment to try on.

        Returns:
            A ``(1, 3, height, width)`` float tensor in ``[0, 1]``.
        """
        agnostic = image_to_tensor(agnostic_image, self.width, self.height, self.device)

        mask_latent = build_mask_latent(
            person_image,
            agnostic_image,
            (self.width, self.height),
            self.config.mask_diff_threshold,
            self.config.mask_morph_kernel,
            self.device,
            self.dtype,
        )

        z_agnostic = self.encode_latent(agnostic)
        null = self.null_embedding.expand(z_agnostic.shape[0], -1, -1)
        inverted = self.bundle.inversion_unet(
            z_agnostic.to(self.dtype), self.inv_timestep, null
        ).sample.to(torch.float32)

        g_pixel = self.garment_processor(
            images=[pad_to_square(garment_image)], return_tensors="pt"
        ).pixel_values.to(self.device)
        g_feat = self.bundle.garment_encoder.backbone(g_pixel).last_hidden_state
        prompt_tokens = self.bundle.garment_encoder(cached_features=g_feat)

        clip_embeds = self.bundle.image_encoder(
            self.clip_processor(images=[agnostic_image], return_tensors="pt")
            .pixel_values.to(self.device, torch.float32)
        ).image_embeds
        ip_tokens = self.bundle.image_proj_model(clip_embeds)

        noisy = self.alpha_t * z_agnostic + self.sigma_t * inverted
        sample = torch.cat(
            [noisy.to(self.dtype), z_agnostic.to(self.dtype), mask_latent], dim=1
        )
        condition = torch.cat([prompt_tokens, ip_tokens], dim=1)
        model_pred = self.bundle.unet(sample, self.timestep, condition).sample.to(
            torch.float32
        )

        if model_pred.shape[1] == noisy.shape[1] * 2:
            model_pred, _ = torch.split(model_pred, noisy.shape[1], dim=1)
        pred = (noisy - self.sigma_t * model_pred) / self.alpha_t

        return decode_latent(pred, self.bundle.vae, self.scheduler).clamp(0, 1)
