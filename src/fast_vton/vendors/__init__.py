"""Isolation layer for the external Fast-VTON (``swiftedit``) package.

The inference code depends on a handful of symbols defined in Fast-VTON: the bundle
loader and the garment / masking helpers. They are re-exported here so the rest of this
project never imports ``src`` directly, which keeps the coupling in one file.

To make this project fully self-contained (no ``../Fast-VTON`` checkout required), copy
the corresponding modules from ``../Fast-VTON/src`` into this package and replace the
imports below with local ones. The public surface (names exported here) would stay
identical, so no caller has to change.
"""

from __future__ import annotations

from src.constants import INVERSION_TIMESTEP
from src.vton import LoadedBundle, load_bundle
from src.vton.garment_encoder import GarmentEncoder, pad_to_square
from src.vton.masking import build_agnostic_mask

__all__ = [
    "GarmentEncoder",
    "INVERSION_TIMESTEP",
    "LoadedBundle",
    "build_agnostic_mask",
    "load_bundle",
    "pad_to_square",
]
