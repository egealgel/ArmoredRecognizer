import cv2
from ultralytics import YOLO

model = YOLO('runs/train/armored-7/weights/best.pt')

for v in ['tank1', 'tank2', 'tank3', 'tank4', 'tank5', 'tank6']:
    cap = cv2.VideoCapture(f'tests/{v}.mp4')
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f'{v}: could not read')
        continue
    r = model.predict(frame, conf=0.1, verbose=False)
    dets = [(r[0].names[int(b.cls[0])], f'{float(b.conf[0]):.2f}') for b in r[0].boxes]
    print(f'{v}: {dets}')
