# Fast-VTON Test — Inference cho `Fast_VTON_full.pt`

Dự án test (virtual try-on) cho model **Fast_VTON_full.pt** đã huấn luyện xong Stage 1.
Model là một file bundle tự đóng gói (không cần `weights/` khi inference, nhưng vẫn cần
thư viện **Fast-VTON** để tái tạo kiến trúc mạng từ bundle):

- Input: ảnh người mẫu + ảnh quần áo (và tùy chọn ảnh agnostic).
- Nếu không có ảnh agnostic, code **tự tạo** bằng human parsing (`mattmdjaga/segformer_b2_clothes`).
- Output: ảnh người mẫu đã mặc quần áo mới (1 bước diffusion).

## Cấu trúc thư mục

```
Test_model_Fast_VTON/
├── README.md
├── pyproject.toml            # cấu hình dự án + entry points (duy nhất)
├── configs/
│   └── default.yaml          # tập trung các hằng số (repo, mask, device...)
├── models/                   # 📦 ĐẶT Fast_VTON_full.pt VÀO ĐÂY
│   └── .gitkeep
├── data/                     # ảnh test đầu vào (tùy chọn)
├── outputs/                  # ảnh kết quả sinh ra
├── src/
│   └── fast_vton/
│       ├── __init__.py
│       ├── config.py         # Config (dataclass) + load YAML
│       ├── vendors/          # lớp cách ly dependency Fast-VTON (src.*)
│       ├── bundle.py         # re-export load_bundle
│       ├── preprocessing.py  # image_to_tensor, AgnosticBuilder (cache segformer)
│       ├── postprocessing.py # tensor_to_pil, decode_latent
│       ├── pipeline.py       # FastVTONPipeline: try_on + build_agnostic
│       ├── app.py            # Gradio demo
│       └── cli.py            # chạy thử trên 1 cặp ảnh qua CLI
├── notebooks/
│   └── test_fast_vton.ipynb  # test trên Kaggle P100
└── tests/
    ├── test_smoke.py         # import + config + preprocess (không cần GPU)
    ├── test_preprocessing.py # masking / agnostic (mock)
    └── test_pipeline.py      # @pytest.mark.slow inference thật (cần GPU)
```

## Chuẩn bị

1. Đặt file model vào `models/`:
   ```bash
   cp /path/to/Fast_VTON_full.pt models/
   ```

2. Cài Fast-VTON (cung cấp các class mạng: `src.vton`, `src.models`, ...). Dự án import
   chúng qua lớp cách ly `src/fast_vton/vendors`, nên **phải** cài Fast-VTON trước:

   ```bash
   pip install -e ../Fast-VTON     # package "swiftedit" (import name: src)
   pip install -e .                # dự án test
   ```

   > Muốn tự đóng gói hoàn toàn (không cần Fast-VTON): copy các module tương ứng từ
   > `../Fast-VTON/src` vào `src/fast_vton/vendors` và sửa import trong file đó.

3. (Tuỳ chọn) tuỳ chỉnh `configs/default.yaml` — ví dụ đổi `device`, `mask_*` hoặc
   `seg_clothing_ids`. Khi chạy, pipeline dùng `Config()` (mặc định) hoặc bạn truyền
   đường dẫn bundle trực tiếp qua `--bundle` / textbox.

> Lưu ý: `torch==2.2.1`, `diffusers==0.22.0`, `transformers==4.37.2` là version đã pin
> theo Fast-VTON. **Không cài `peft`** (xung đột với diffusers 0.22).

## Chạy

### Gradio (khuyên dùng)
```bash
python -m fast_vton.app
# hoặc: fast-vton-app
```
Mở link hiện ra, up ảnh người + ảnh quần áo, bấm **Thử đồ**. Ảnh agnostic sẽ tự sinh
(trừ khi bạn up sẵn và tắt "Tự động tạo agnostic").

### CLI (test nhanh 1 cặp ảnh)
```bash
python -m fast_vton.cli \
    --person data/nguoi.jpg \
    --garment data/ao.jpg \
    --output outputs/result.png
# Tự tạo agnostic là mặc định. Để dùng ảnh agnostic có sẵn:
python -m fast_vton.cli \
    --person data/nguoi.jpg --garment data/ao.jpg \
    --agnostic data/agnostic.jpg --no-auto-agnostic \
    --output outputs/result.png
```

### Test tự động
```bash
pip install -e ".[dev]"
pytest                      # unit test (không cần GPU)
pytest -m slow              # test inference thật (cần GPU + models/Fast_VTON_full.pt)
```

## Cách inference hoạt động (1 bước)

1. Resize person/agnostic về `384×512` (đọc từ bundle manifest).
2. Tạo mask từ hiệu `|person − agnostic|` (vùng quần áo).
3. Encode agnostic → `z_agnostic` (VAE).
4. Inversion network dự đoán `inverted_noise` từ `z_agnostic` + null-embedding (timestep 500).
5. Garment → DINOv2 → token đưa vào **prompt branch**; agnostic → CLIP → token đưa vào **IP branch**.
6. `noisy = α·z_agnostic + σ·inverted_noise`, ghép thành tensor 9 kênh cùng mask, chạy UNet 1 bước.
7. Decode latent → ảnh kết quả.

Đây là đúng pipeline Stage 1 lúc train, nên kết quả faithful với checkpoint.

## Lưu ý

- **Auto-agnostic** tô xám vùng quần áo (upper / skirt / pants / dress) rồi lấy hiệu ảnh.
  Kết quả đẹp nhất khi ảnh người có nền tương đối sạch. Nếu mask sai vùng, hãy up ảnh
  agnostic thủ công.
- Bundle dùng VAE fp32 để decode; VRAM ~3.2 GB (fp16) trên 24 GB là thoải mái.
- Scheduler (`Manojb/stable-diffusion-2-1-base`) được tải từ Hub để lấy `α_t/σ_t` — cần internet.
- Human-parsing (`mattmdjaga/segformer_b2_clothes`) chỉ tải 1 lần và được cache trong
  suốt phiên chạy (trong Gradio cũng như CLI).
