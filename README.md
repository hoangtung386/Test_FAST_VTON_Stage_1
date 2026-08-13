# Fast-VTON Test — Inference cho `Fast_VTON_full.pt`

Dự án test (virtual try-on) cho model **Fast_VTON_full.pt** đã huấn luyện xong Stage 1.
Model là một file bundle tự đóng gói (không cần `weights/` hay tải Hugging Face khi inference):
chỉ cần `Fast_VTON_full.pt` là chạy được.

- Input: ảnh người mẫu + ảnh quần áo (và tùy chọn ảnh agnostic).
- Nếu không có ảnh agnostic, code **tự tạo** bằng human parsing (`mattmdjaga/segformer_b2_clothes`).
- Output: ảnh người mẫu đã mặc quần áo mới (1 bước diffusion).

## Cấu trúc thư mục

```
Test_model_Fast_VTON/
├── README.md
├── pyproject.toml            # cấu hình dự án + entry points
├── requirements.txt          # cài nhanh các dependency
├── models/                   # 📦 ĐẶT Fast_VTON_full.pt VÀO ĐÂY
│   └── .gitkeep
├── data/                     # ảnh test đầu vào (tùy chọn)
├── outputs/                  # ảnh kết quả sinh ra
├── src/
│   └── fast_vton_test/
│       ├── __init__.py
│       ├── inference.py      # FastVTONInference: load bundle + try_on + build_agnostic
│       ├── app.py            # Gradio demo
│       └── cli.py            # chạy thử trên 1 cặp ảnh qua CLI
└── tests/
    └── test_smoke.py         # smoke test (import + inference có GPU)
```

## Chuẩn bị

1. Đặt file model vào `models/`:
   ```bash
   cp /path/to/Fast_VTON_full.pt models/
   ```

2. Cài dependency. Dự án import package `src` của **Fast-VTON**, nên cần cài Fast-VTON
   (chứa `src.vton.load_bundle`, `src.vton.masking`, ...).

   **Trên máy local** (Fast-VTON nằm ở `../Fast-VTON`):
   ```bash
   pip install -e .
   # hoặc: pip install -r requirements.txt && pip install -e ../Fast-VTON
   ```

   **Trên Colab / server khác**:
   ```bash
   !git clone https://github.com/hoangtung386/Fast-VTON.git /content/Fast-VTON
   !pip install -e /content/Fast-VTON
   !pip install -r requirements.txt
   # rồi sửa dòng dependency "swiftedit" trong pyproject thành path "/content/Fast-VTON"
   # (hoặc bỏ qua pyproject, chỉ cài như 2 lệnh trên)
   ```

> Lưu ý: `torch==2.2.1`, `diffusers==0.22.0`, `transformers==4.37.2` là version đã pin
> theo Fast-VTON. **Không cài `peft`** (xung đột với diffusers 0.22).

## Chạy

### Gradio (khuyên dùng)
```bash
python -m fast_vton_test.app
# hoặc: fast-vton-app
```
Mở link hiện ra, up ảnh người + ảnh quần áo, bấm **Thử đồ**. Ảnh agnostic sẽ tự sinh
(trừ khi bạn up sẵn và tắt "Tự động tạo agnostic").

### CLI (test nhanh 1 cặp ảnh)
```bash
python -m fast_vton_test.cli \
    --person data/nguoi.jpg \
    --garment data/ao.jpg \
    --auto-agnostic \
    --output outputs/result.png
# Nếu đã có sẵn ảnh agnostic (cấu trúc VITON-HD):
python -m fast_vton_test.cli \
    --person data/nguoi.jpg --garment data/ao.jpg \
    --agnostic data/agnostic.jpg --output outputs/result.png
```

### Test tự động
```bash
pip install -e ".[dev]"
pytest                      # chỉ test import
pytest -m slow              # test inference thật (cần GPU + models/Fast_VTON_full.pt)
```

## Cách inference hoạt động (1 bước)

1. Resize person/agnostic về `384×512`.
2. Tạo mask từ hiệu `|person − agnostic|` (vùng quần áo).
3. Encode agnostic → `z_agnostic` (VAE).
4. Inversion network dự đoán `inverted_noise` từ `z_agnostic` + null-embedding (timestep 500).
5. Garment → DINOv2 → token đưa vào **prompt branch**; agnostic → CLIP → token đưa vào **IP branch**.
6. `noisy = α·z_agnostic + σ·inverted_noise`, ghép thành tensor 9 kênh cùng mask, chạy UNet 1 bước.
7. Decode latent → ảnh kết quả.

Đây là đúng pipeline Stage 1 lúc train (xem `src/vton/trainer.py::compute_loss` và
`src/models/generator.py::forward_train`), nên kết quả faithful với checkpoint.

## Lưu ý

- **Auto-agnostic** tô xám vùng quần áo (upper / skirt / pants / dress) rồi lấy hiệu ảnh.
  Kết quả đẹp nhất khi ảnh người có nền tương đối sạch. Nếu mask sai vùng, hãy up ảnh
  agnostic thủ công.
- Bundle dùng VAE fp32 để decode; VRAM ~3.2 GB (fp16) trên 24 GB là thoải mái.
- Scheduler (`Manojb/stable-diffusion-2-1-base`) được tải từ Hub để lấy `α_t/σ_t` — cần internet.
