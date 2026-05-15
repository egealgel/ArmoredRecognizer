

https://github.com/user-attachments/assets/893b3abe-55dd-407d-bc53-ad7948deb940

# Armored Vehicle Recognizer
Real-time military armored vehicle detection using **YOLOv11** trained on 16,000+ annotated images.

Two Engineering Vehicles correctly detected. 

Note: The model is trained on Eastern/Russian military vehicle imagery. NATO vehicles (Leopard 2, M1 Abrams, LAV) are not represented in the training dataset and may be misclassified.


## Results

| Metric | Score |
|---|---|
| mAP@0.5 | **98.2%** |
| mAP@0.5:0.95 | **78.1%** |
| Precision | **97.7%** |
| Recall | **96.2%** |

### Per-Class Performance

| Class | mAP@0.5 |
|---|---|
| TANK | 98.4% |
| IFV | 99.4% |
| APC | 99.1% |
| EV | 99.5% |
| AH | 95.3% |
| TH | 99.4% |
| AAP | 96.6% |
| TA | 99.5% |
| AA | 99.5% |
| TART | 98.2% |
| SPART | 95.2% |

## Classes

The model detects 11 military vehicle and aircraft categories:

| ID | Class | Full Name |
|---|---|---|
| 0 | TANK | Main Battle Tank |
| 1 | IFV | Infantry Fighting Vehicle |
| 2 | APC | Armored Personnel Carrier |
| 3 | EV | Engineering Vehicle |
| 4 | AH | Assault Helicopter |
| 5 | TH | Transport Helicopter |
| 6 | AAP | Assault Airplane |
| 7 | TA | Transport Airplane |
| 8 | AA | Anti-Aircraft Vehicle |
| 9 | TART | Towed Artillery |
| 10 | SPART | Self-Propelled Artillery |

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/ArmoredRecognizer.git
cd ArmoredRecognizer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Run on a video file
```bash
python infer_video.py --weights runs/train/armored-7/weights/best.pt --source your_video.mp4 --save
```

### Run on webcam
```bash
python infer_video.py --weights runs/train/armored-7/weights/best.pt --source 0
```

### Evaluate on test set
```bash
python evaluate.py --weights runs/train/armored-7/weights/best.pt
```

## Training

```bash
# Prepare dataset (YOLO or COCO format)
python data/merge_datasets.py --datasets data/raw/dataset1 data/raw/dataset2 --output data/merged

# Train
python train.py --model yolo11s.pt --epochs 100 --device mps
```

## Model

- Architecture: YOLOv11s
- Input size: 416×416
- Training: 100 epochs on Apple M4 (MPS)
- Dataset: 11,768 train / 1,680 val / 3,361 test images
