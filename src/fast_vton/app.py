"""Gradio demo for one-step virtual try-on."""

from __future__ import annotations

import gradio as gr

from fast_vton.config import Config, config_from_bundle
from fast_vton.pipeline import FastVTONPipeline
from fast_vton.postprocessing import tensor_to_pil

_PREDICTORS: dict[str, FastVTONPipeline] = {}


def get_predictor(bundle_path: str | None) -> FastVTONPipeline:
    """Return a cached :class:`FastVTONPipeline` for ``bundle_path`` (lazily built)."""
    key = str(bundle_path) if bundle_path else str(Config().bundle_path)
    if key not in _PREDICTORS:
        config = config_from_bundle(bundle_path) if bundle_path else Config()
        _PREDICTORS[key] = FastVTONPipeline(config)
    return _PREDICTORS[key]


def run(person, garment, agnostic, bundle_path, auto_agnostic):
    if person is None or garment is None:
        raise gr.Error("Cần cung cấp ảnh người mẫu và ảnh quần áo")

    predictor = get_predictor(bundle_path)

    if agnostic is not None and not auto_agnostic:
        agnostic_image = agnostic
    else:
        agnostic_image = predictor.build_agnostic(person)

    result = predictor.try_on(person, agnostic_image, garment)
    return tensor_to_pil(result), agnostic_image


def build_demo():
    default_bundle = str(Config().bundle_path)
    with gr.Blocks(title="Fast-VTON Try-On") as demo:
        gr.Markdown("# Fast-VTON — Virtual Try-On (Fast_VTON_full.pt)")
        bundle_box = gr.Textbox(label="Đường dẫn bundle (Fast_VTON_full.pt)", value=default_bundle)
        with gr.Row():
            person_in = gr.Image(label="Ảnh người mẫu", type="pil")
            garment_in = gr.Image(label="Ảnh quần áo", type="pil")
            agnostic_in = gr.Image(label="Ảnh agnostic (tùy chọn)", type="pil")
        auto_box = gr.Checkbox(
            label="Tự động tạo agnostic từ ảnh người (dùng human parsing)", value=True
        )
        run_btn = gr.Button("Thử đồ")
        with gr.Row():
            out_img = gr.Image(label="Kết quả", type="pil")
            agn_out = gr.Image(label="Agnostic đã tạo", type="pil")

        run_btn.click(
            run,
            [person_in, garment_in, agnostic_in, bundle_box, auto_box],
            [out_img, agn_out],
        )
    return demo


def main():
    build_demo().launch(share=True)


if __name__ == "__main__":
    main()
