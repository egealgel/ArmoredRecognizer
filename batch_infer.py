"""
Run inference on all videos in tests/ and save annotated outputs.
"""
import time
import cv2
from pathlib import Path
from ultralytics import YOLO

WEIGHTS = "runs/train/armored-7/weights/best.pt"
TESTS   = Path("tests")
CONF    = 0.25

model = YOLO(WEIGHTS)
videos = sorted(TESTS.glob("*.mp4"))
# Skip already-processed outputs
videos = [v for v in videos if "output" not in v.name]

for video in videos:
    out_path = TESTS / f"{video.stem}_output.mp4"
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (w, h),
    )

    print(f"\nProcessing {video.name} ({total} frames)...")
    t0 = time.time()
    frame_i = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame, imgsz=640, conf=CONF, verbose=False)
        annotated = results[0].plot()
        writer.write(annotated)
        frame_i += 1
        if frame_i % 30 == 0:
            pct = frame_i / total * 100
            elapsed = time.time() - t0
            eta = (elapsed / frame_i) * (total - frame_i)
            print(f"  {pct:.0f}%  ETA {eta:.0f}s")

    cap.release()
    writer.release()
    print(f"  Saved -> {out_path.name}")

print("\nAll done. Open tests/ to view the output videos.")
