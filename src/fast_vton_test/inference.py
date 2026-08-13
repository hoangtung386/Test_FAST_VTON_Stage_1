import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import DDPMScheduler
from torchvision.transforms.functional import to_tensor
from transformers import AutoImageProcessor, CLIPImageProcessor

from src.constants import INVERSION_TIMESTEP
from src.vton import load_bundle
from src.vton.garment_encoder import GarmentEncoder, pad_to_square
from src.vton.masking import build_agnostic_mask

SD21_BASE_REPO = "Manojb/stable-diffusion-2-1-base"
DEFAULT_BUNDLE = "models/Fast_VTON_full.pt"


class FastVTONInference:
    def __init__(self, bundle_path=None, device="cuda", scheduler_repo=SD21_BASE_REPO):
        bundle_path = bundle_path or DEFAULT_BUNDLE
        self.device = torch.device(device)
        self.bundle = load_bundle(bundle_path, device=self.device)
        manifest = self.bundle.manifest
        self.height = manifest.height
        self.width = manifest.width
        self.garment_resolution = manifest.garment_resolution
        self.dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }[manifest.dtype]

        self.scheduler = DDPMScheduler.from_pretrained(scheduler_repo, subfolder="scheduler")
        num_train = self.scheduler.config.num_train_timesteps
        alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)
        t = torch.tensor([num_train - 1], device=self.device)
        self.alpha_t = (alphas_cumprod[t] ** 0.5).view(-1, 1, 1, 1)
        self.sigma_t = ((1 - alphas_cumprod[t]) ** 0.5).view(-1, 1, 1, 1)
        self.timestep = torch.full((1,), num_train - 1, dtype=torch.int64, device=self.device)
        self.inv_timestep = torch.full((1,), INVERSION_TIMESTEP, dtype=torch.int64, device=self.device)

        self.garment_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
        self.garment_processor.do_center_crop = False
        self.garment_processor.size = {"shortest_edge": self.garment_resolution}
        self.clip_processor = CLIPImageProcessor()

        if self.bundle.null_embedding is not None:
            self.null_embedding = self.bundle.null_embedding.to(self.device, self.dtype)
        else:
            self.null_embedding = torch.zeros((1, 77, 1024), device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def _to_tensor(self, image):
        resized = image.convert("RGB").resize((self.width, self.height), Image.BILINEAR)
        return to_tensor(resized).unsqueeze(0).to(self.device) * 2 - 1

    @torch.no_grad()
    def encode_latent(self, pixel_tensor):
        vae = self.bundle.vae
        latents = vae.encode(pixel_tensor.to(vae.dtype)).latent_dist.sample()
        return latents * vae.config.scaling_factor

    @torch.no_grad()
    def try_on(self, person_image, agnostic_image, garment_image):
        size = (self.width, self.height)
        person = self._to_tensor(person_image)
        agnostic = self._to_tensor(agnostic_image)

        mask = build_agnostic_mask(person_image, agnostic_image, size, 12, 9)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float().to(self.device)
        mask_latent = F.interpolate(
            mask_tensor, (self.height // 8, self.width // 8), mode="nearest"
        ).to(self.dtype)

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
        model_pred = self.bundle.unet(sample, self.timestep, condition).sample.to(torch.float32)

        if model_pred.shape[1] == noisy.shape[1] * 2:
            model_pred, _ = torch.split(model_pred, noisy.shape[1], dim=1)
        pred = (noisy - self.sigma_t * model_pred) / self.alpha_t

        if self.scheduler.config.thresholding:
            pred = self.scheduler._threshold_sample(pred)
        elif self.scheduler.config.clip_sample:
            pred = pred.clamp(
                -self.scheduler.config.clip_sample_range,
                self.scheduler.config.clip_sample_range,
            )

        vae = self.bundle.vae
        image = vae.decode((pred / vae.config.scaling_factor).to(vae.dtype)).sample.float()
        image = (image + 1) / 2
        return image.clamp(0, 1)

    @torch.no_grad()
    def build_agnostic(self, person_image):
        from transformers import (
            AutoImageProcessor as SegProcessor,
            SegformerForSemanticSegmentation,
        )

        proc = SegProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
        model = (
            SegformerForSemanticSegmentation.from_pretrained(
                "mattmdjaga/segformer_b2_clothes"
            )
            .to(self.device)
            .eval()
        )
        inputs = proc(images=person_image, return_tensors="pt").to(self.device)
        logits = model(**inputs).logits
        seg = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        w, h = person_image.size
        if seg.shape[0] != h or seg.shape[1] != w:
            seg = np.array(Image.fromarray(seg).resize((w, h), Image.NEAREST))
        clothing_ids = np.array([4, 5, 6, 7])
        cloth_mask = np.isin(seg, clothing_ids)
        arr = np.array(person_image.convert("RGB")).copy()
        arr[cloth_mask] = (128, 128, 128)
        return Image.fromarray(arr)
