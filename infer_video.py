"""
Real-time armored vehicle detection on video or webcam.

Usage:
  python infer_video.py --weights runs/train/armored/weights/best.pt --source 0
  python infer_video.py --weights best.pt --source video.mp4 --save
  python infer_video.py --weights best.pt --source video.mp4 --save --output output.mp4
"""

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


# Distinct BGR colors per class index (cycles if more than 10 classes)
PALETTE = [
    (0, 255, 0),    (0, 0, 255),   (255, 0, 0),   (0, 255, 255),
    (255, 0, 255),  (255, 255, 0), (128, 0, 255),  (0, 128, 255),
    (0, 255, 128),  (128, 255, 0),
]


def draw_detections(frame, result, conf_thresh: float):
    for box in result.boxes:
        conf = float(box.conf[0])
        if conf < conf_thresh:
            continue

        cls = int(box.cls[0])
        label = result.names[cls]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = PALETTE[cls % len(PALETTE)]

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True,       help="Path to best.pt")
    ap.add_argument("--source",  default="0",         help="Video file path or webcam index")
    ap.add_argument("--imgsz",   type=int, default=640)
    ap.add_argument("--conf",    type=float, default=0.35)
    ap.add_argument("--iou",     type=float, default=0.45)
    ap.add_argument("--device",  default="",          help="cuda device or cpu")
    ap.add_argument("--save",    action="store_true", help="Save output video")
    ap.add_argument("--output",  default="output.mp4")
    ap.add_argument("--no-display", action="store_true", help="Disable live window (useful on headless servers)")
    return ap.parse_args()


def main():
    args = parse_args()

    model = YOLO(args.weights)
    if args.device:
        model.to(args.device)

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {args.source}")

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.save:
        writer = cv2.VideoWriter(
            args.output,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps_src, (w, h),
        )
        print(f"Saving to: {args.output}")

    fps_timer = time.time()
    frame_count = 0

    print("Press Q to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(
            frame,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            verbose=False,
        )
        frame = results[0].plot()  # draws boxes + class names + confidence

        # FPS overlay
        frame_count += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            fps_timer = time.time()
        else:
            fps = frame_count / max(elapsed, 1e-6)

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        if writer:
            writer.write(frame)

        if not args.no_display:
            cv2.imshow("Armored Vehicle Recognizer", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
