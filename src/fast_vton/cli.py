"""Command-line entry point for one-step virtual try-on on a single image pair."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from fast_vton.config import Config
from fast_vton.pipeline import FastVTONPipeline
from fast_vton.postprocessing import tensor_to_pil


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test Fast_VTON_full.pt trên ảnh đơn")
    parser.add_argument("--bundle", type=str, default=str(Config().bundle_path))
    parser.add_argument("--person", type=str, required=True, help="ảnh người mẫu")
    parser.add_argument("--garment", type=str, required=True, help="ảnh quần áo")
    parser.add_argument("--agnostic", type=str, default=None, help="ảnh agnostic (tùy chọn)")
    parser.add_argument(
        "--no-auto-agnostic",
        action="store_true",
        help="không tự tạo agnostic; bắt buộc phải truyền --agnostic",
    )
    parser.add_argument("--output", type=str, default="outputs/result.png")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    predictor = FastVTONPipeline(bundle_path=args.bundle, device=args.device)

    person = Image.open(args.person).convert("RGB")
    garment = Image.open(args.garment).convert("RGB")

    if args.agnostic is not None and args.no_auto_agnostic:
        agnostic = Image.open(args.agnostic).convert("RGB")
    else:
        agnostic = predictor.build_agnostic(person)

    with torch.no_grad():
        result = predictor.try_on(person, agnostic, garment)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tensor_to_pil(result).save(output_path)
    print(f"đã ghi kết quả vào {output_path}")


if __name__ == "__main__":
    main()
