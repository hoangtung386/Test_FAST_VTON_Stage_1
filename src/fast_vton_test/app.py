import gradio as gr
from PIL import Image

from fast_vton_test.inference import DEFAULT_BUNDLE, FastVTONInference

_PREDICTORS = {}


def get_predictor(bundle_path):
    path = bundle_path or DEFAULT_BUNDLE
    if path not in _PREDICTORS:
        _PREDICTORS[path] = FastVTONInference(path, device="cuda")
    return _PREDICTORS[path]


def run(person, garment, agnostic, bundle_path, auto_agnostic):
    if person is None or garment is None:
        raise gr.Error("Cần cung cấp ảnh người mẫu và ảnh quần áo")

    predictor = get_predictor(bundle_path)

    if agnostic is not None and not auto_agnostic:
        agnostic_image = agnostic
    else:
        agnostic_image = predictor.build_agnostic(person)

    result = predictor.try_on(person, agnostic_image, garment)
    result_image = result[0].permute(1, 2, 0).cpu().numpy()
    result_image = (result_image * 255).astype("uint8")

    return Image.fromarray(result_image), agnostic_image


def build_demo():
    with gr.Blocks(title="Fast-VTON Try-On") as demo:
        gr.Markdown("# Fast-VTON — Virtual Try-On (Fast_VTON_full.pt)")
        bundle_box = gr.Textbox(
            label="Đường dẫn bundle (Fast_VTON_full.pt)", value=DEFAULT_BUNDLE
        )
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
