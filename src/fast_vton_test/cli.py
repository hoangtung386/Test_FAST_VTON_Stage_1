import argparse
from pathlib import Path

import torch
from PIL import Image

from fast_vton_test.inference import DEFAULT_BUNDLE, FastVTONInference


def main():
    parser = argparse.ArgumentParser(description="Test Fast_VTON_full.pt trên ảnh đơn")
    parser.add_argument("--bundle", type=str, default=DEFAULT_BUNDLE)
    parser.add_argument("--person", type=str, required=True, help="ảnh người mẫu")
    parser.add_argument("--garment", type=str, required=True, help="ảnh quần áo")
    parser.add_argument("--agnostic", type=str, default=None, help="ảnh agnostic (tùy chọn)")
    parser.add_argument("--auto-agnostic", action="store_true", help="tự tạo agnostic")
    parser.add_argument("--output", type=str, default="outputs/result.png")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    predictor = FastVTONInference(args.bundle, device=args.device)

    person = Image.open(args.person).convert("RGB")
    garment = Image.open(args.garment).convert("RGB")
    if args.agnostic is not None and not args.auto_agnostic:
        agnostic = Image.open(args.agnostic).convert("RGB")
    else:
        agnostic = predictor.build_agnostic(person)

    with torch.no_grad():
        result = predictor.try_on(person, agnostic, garment)
    result_image = result[0].permute(1, 2, 0).cpu().numpy()
    result_image = (result_image * 255).astype("uint8")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result_image).save(args.output)
    print(f"đã ghi kết quả vào {args.output}")


if __name__ == "__main__":
    main()
