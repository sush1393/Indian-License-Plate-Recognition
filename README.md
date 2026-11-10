# 🚗 Indian License Plate Recognition System

End-to-end **Indian License Plate Recognition (ALPR)** system using **YOLOv5-LSE** (detection) and **CRNN + CTC** (recognition). This project is an implementation of the IEEE Access paper *"License Plate Recognition System Based on Improved YOLOv5 and GRU"* by Shi & Zhao (2023), adapted for **Indian license plates**.

---

## 📊 Results

| Metric | Paper (Shi & Zhao) | This Project |
| :--- | :--- | :--- |
| **Dataset Size** | 12,500 images | 1,500 images |
| **Detection mAP@0.5** | 97.1% | **99.5%** ✅ |
| **L-SE Improvement** | +2% | **+0.39%** |
| **Recognition Accuracy** | 98.98% | **~76%** |
| **Inference Time (CPU)** | 70 ms | **16 ms** ✅ |

- **Detection:** 99.5% mAP@0.5 with YOLOv5-LSE.
- **Recognition:** ~76% character accuracy (limited by small dataset).
- **Speed:** Runs on CPU in ~16 ms per image.

---

## 🏗️ Architecture

The system uses a **two-stage end-to-end pipeline**:
Input Image (Car)
↓
[Stage 1] YOLOv5-LSE Detection → Crops the plate
↓
[Stage 2] CRNN + CTC Recognition → Reads the characters
↓
Output Text (e.g., "UP32FH2653")


### Stage 1: YOLOv5-LSE (Detection)
- Base model: **YOLOv5s** (7M parameters).
- Custom improvement: Added **3 L-SE (Location-aware Squeeze and Excitation)** attention blocks in the Neck (PAN) downsampling path.
- Why L-SE? Standard SE pools globally and loses spatial information. L-SE pools in **both X and Y directions**, preserving location information critical for small plates.

### Stage 2: CRNN + CTC (Recognition)
- **CNN:** 7-layer feature extractor.
- **BiGRU:** 2-layer bidirectional GRU for sequence modeling.
- **CTC:** Connectionist Temporal Classification for segmentation-free decoding.
- **Charset:** `0-9`, `A-Z` (36 characters) + CTC blank = **37 classes**.

---

## 📁 Project Structure
Indian-License-Plate-Recognition/
├── code/
│ ├── demo_inference.py # End-to-end pipeline
│ ├── resume_setup.py # CRNN loader + inference
│ └── crnn_model.py # CRNN architecture
├── yolov5/models/
│ ├── lse.py # L-SE attention block
│ ├── yolov5s-lse.yaml # Custom YOLOv5 config
│ └── yolo.py # Modified YOLO parser
├── results/ # Graphs and evaluation outputs
│ ├── LSE_vs_Baseline.png
│ ├── crnn_confusion_matrix.png
│ ├── YOLOv5_training_curves.png
│ └── annotated_car.jpg
├── requirements.txt
├── PROJECT_REPORT.md
└── README.md

---

## ⚙️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/sush1393/Indian-License-Plate-Recognition.git
cd Indian-License-Plate-Recognition
2. Install dependencies
bash
pip install -r requirements.txt
3. Prepare model weights
⚠️ Note: Trained model weights (.pt, .pth) are not included in this repository due to size limits. You need to either:

Option A (Recommended): Train your own models using the code in this repository and place the weights here:

text
weights/
├── yolov5_LSE_pure_state.pt   # YOLOv5-LSE trained weights
└── crnn_tiny_plate.pth        # CRNN trained weights
Option B: Contact the authors for the trained weights.

4. Run the demo
bash
python code/demo_inference.py --image car.jpg
The annotated image will be saved in results/annotated_car.jpg.

🧪 Sample Results
Image	Detected Plate	Confidence
Nissan Terrano	UP32FH2653	90% ✅
Skoda Rapid	MH20EE0943	83% ✅
Maruti Swift	KL10AV6342	84% ✅
Custom Image	AP31BL8656	91% ✅
⚠️ Limitations
Small dataset: 1,500 images compared to the paper's 12,500. This limits recognition accuracy to ~76%.

Two-line plates: Fails on plates with two rows (e.g., commercial vehicles).

Non-Indian plates: The CRNN only knows English characters (0-9, A-Z). It fails on plates with Urdu or other scripts.

Pretrained CRNN: Not fine-tuned on Indian fonts, so it sometimes confuses similar characters (e.g., 0 vs O, 1 vs I).

🚀 Future Work
Retrain CRNN from scratch on 10,000+ Indian plate images.

Add data augmentation (blur, rotation, lighting) for robustness.

Support multi-line plates.

Deploy as a REST API for real-time video stream processing.

Quantize models (INT8) for embedded deployment.

📄 References
Shi H and Zhao D (2023) License plate recognition system based on improved YOLOv5 and GRU. IEEE Access, 11, 1-13.

Redmon J, Divvala S, Girshick R and Farhadi A (2016) You only look once: Unified, real-time object detection. Proceedings of IEEE CVPR, 779-788.

Jocher G (2020) YOLOv5 by Ultralytics. GitHub Repository.

Graves A, Fernández S, Gomez F and Schmidhuber J (2006) Connectionist temporal classification. Proceedings of the 23rd ICML, 369-376.

Hu J, Shen L and Sun G (2018) Squeeze-and-Excitation networks. Proceedings of IEEE CVPR, 7132-7141.

👩‍🎓 Author
Sushma Boreddy

Course: CSE397 - Mini Project (M.C.A)
Guide: Dr. Karthikeyan M.P., Assistant Professor-III
Institution: SASTRA Deemed to be University, Thanjavur

📜 License
This project is submitted as part of the CSE397 Mini Project at SASTRA Deemed to be University.

---

### Step 3: Commit the file

1. Scroll down to the bottom of the page.
2. In the **Commit message** box, type: `Added README with project details`
3. Click the green **"Commit new file"** button.

---

### ✅ Done!
