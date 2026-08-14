"""Virtual try-on inference for ``Fast_VTON_full.pt`` (Stage 1 bundle).

This package is a self-contained test harness: given the ``.pt`` bundle exported by
Fast-VTON, it loads the network and runs one-step virtual try-on through a Gradio demo
or a CLI. Heavy model classes are imported from the external Fast-VTON package via the
:mod:`fast_vton.vendors` isolation layer.
"""

from fast_vton.pipeline import FastVTONPipeline

__all__ = ["FastVTONPipeline"]
__version__ = "0.1.0"
