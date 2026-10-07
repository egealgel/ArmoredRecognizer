

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


## Model

- Architecture: YOLOv11s
- Input size: 416×416
- Training: 100 epochs on Apple M4 (MPS)
- Dataset: 11,768 train / 1,680 val / 3,361 test images
- https://www.kaggle.com/datasets/nzigulic/military-equipment
