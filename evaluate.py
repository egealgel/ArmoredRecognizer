"""
Evaluate a trained model on the test split and print per-class metrics.

Usage:
  python evaluate.py --weights runs/train/armored/weights/best.pt
  python evaluate.py --weights best.pt --data data/merged/dataset.yaml --split test
"""

import argparse
from ultralytics import YOLO


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data",    default="configs/dataset.yaml")
    ap.add_argument("--split",   default="test", choices=["train", "val", "test"])
    ap.add_argument("--imgsz",   type=int, default=640)
    ap.add_argument("--batch",   type=int, default=16)
    ap.add_argument("--conf",    type=float, default=0.001)
    ap.add_argument("--iou",     type=float, default=0.6)
    ap.add_argument("--device",  default="")
    return ap.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)

    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        plots=True,
        save_json=True,
    )

    print("\n--- Results ---")
    print(f"mAP@0.5      : {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95 : {metrics.box.map:.4f}")
    print(f"Precision    : {metrics.box.mp:.4f}")
    print(f"Recall       : {metrics.box.mr:.4f}")


if __name__ == "__main__":
    main()
