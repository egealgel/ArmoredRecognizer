"""
Train YOLOv11 on the merged armored vehicle dataset.

Usage:
  python train.py                          # default settings
  python train.py --model yolo11m.pt       # medium model
  python train.py --epochs 150 --batch 16
  python train.py --resume runs/train/exp/weights/last.pt
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

PROJECT_DIR = Path(__file__).parent


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",   default="yolo11s.pt",            help="Pretrained weights or model yaml")
    ap.add_argument("--data",    default=str(PROJECT_DIR / "configs/dataset.yaml"))
    ap.add_argument("--epochs",  type=int,   default=100)
    ap.add_argument("--imgsz",   type=int,   default=416)
    ap.add_argument("--batch",   type=int,   default=16)
    ap.add_argument("--workers", type=int,   default=8)
    ap.add_argument("--device",  default="mps",                   help="cuda device, mps, or cpu")
    ap.add_argument("--project", default=str(PROJECT_DIR / "runs/train"))
    ap.add_argument("--name",    default="armored")
    ap.add_argument("--resume",  default=None,                     help="Path to last.pt to resume training")
    return ap.parse_args()


def main():
    args = parse_args()

    if args.resume:
        model = YOLO(args.resume)
        model.train(resume=True)
        return

    model = YOLO(args.model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=args.project,
        name=args.name,
        # Augmentation
        mosaic=1.0,
        flipud=0.1,
        fliplr=0.5,
        scale=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        # Logging
        plots=True,
        save=True,
        save_period=10,
        val=True,
    )

    print(f"\nBest weights saved to: {Path(args.project) / args.name / 'weights' / 'best.pt'}")
    print("Next step: python infer_video.py --weights runs/train/armored/weights/best.pt --source <video_or_0>")


if __name__ == "__main__":
    main()
