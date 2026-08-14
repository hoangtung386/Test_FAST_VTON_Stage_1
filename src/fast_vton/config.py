"""Central, typed configuration for the try-on inference harness.

All magic numbers that were previously scattered through the inference code
(scheduler repository, segmentation label ids, mask morphology sizes, default bundle
location) now live here so they can be overridden from a single YAML file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

try:  # PyYAML is optional; the dataclass defaults are enough for most runs.
    import yaml
except ImportError:  # pragma: no cover - exercised only without PyYAML installed.
    yaml = None


@dataclass
class Config:
    """Runtime configuration for :class:`fast_vton.pipeline.FastVTONPipeline`.

    Attributes:
        model_dir: Directory that holds the ``.pt`` bundle.
        bundle_name: File name of the bundle inside ``model_dir``.
        scheduler_repo: Hugging Face repo used to fetch the DDPM scheduler config
            (only the config is needed, to read ``alpha_t`` / ``sigma_t``).
        dinov2_repo: Garment encoder backbone repository.
        segformer_repo: Human-parsing repository used to auto-build the agnostic view.
        seg_clothing_ids: Segmentation class ids treated as "clothing" when painting
            the agnostic view grey (upper / skirt / pants / dress for
            ``mattmdjaga/segformer_b2_clothes``).
        mask_diff_threshold: Per-channel intensity delta above which a pixel counts as
            removed when building the inpainting mask from person/agnostic.
        mask_morph_kernel: Diameter of the morphological structuring element used to
            de-speckle and close the inpainting mask.
        device: Torch device string (``"cuda"`` or ``"cpu"``).
    """

    model_dir: Path = Path("models")
    bundle_name: str = "Fast_VTON_full.pt"
    scheduler_repo: str = "Manojb/stable-diffusion-2-1-base"
    dinov2_repo: str = "facebook/dinov2-large"
    segformer_repo: str = "mattmdjaga/segformer_b2_clothes"
    seg_clothing_ids: tuple[int, ...] = (4, 5, 6, 7)
    mask_diff_threshold: int = 12
    mask_morph_kernel: int = 9
    device: str = "cuda"

    @property
    def bundle_path(self) -> Path:
        """Absolute path to the ``.pt`` bundle."""
        return self.model_dir / self.bundle_name

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Build a configuration from a YAML file, overriding the dataclass defaults.

        Raises:
            RuntimeError: If PyYAML is not installed.
        """
        if yaml is None:
            raise RuntimeError(
                "PyYAML is required to load a config file. Install it with "
                "`pip install pyyaml` or use the default Config()."
            )
        with open(path, encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if "model_dir" in data:
            data["model_dir"] = Path(data["model_dir"])
        if "seg_clothing_ids" in data:
            data["seg_clothing_ids"] = tuple(data["seg_clothing_ids"])
        return cls(**data)


def config_from_bundle(bundle: str | Path) -> Config:
    """Build a :class:`Config` whose bundle points at ``bundle``.

    Useful when the caller supplies a full path to the ``.pt`` file instead of relying
    on the ``model_dir`` / ``bundle_name`` split.
    """
    path = Path(bundle)
    return Config(model_dir=path.parent, bundle_name=path.name)
