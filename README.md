# 🌿 PlantGuard AI — Plant Disease Detection System

> An AI-powered web application that detects plant diseases from leaf images using a MobileNetV2 deep learning model, providing instant diagnosis, severity assessment, treatment advice, and downloadable PDF reports.

**Developed by:**
[Deepika Singh](https://www.linkedin.com/in/deepika-singh-9b6b98388/?isSelfProfile=false) &nbsp;|&nbsp;
[Hritik Mishra](https://www.linkedin.com/in/hritik-mishra-23aa42347/?isSelfProfile=false) &nbsp;|&nbsp;
[Kartik Gupta](https://www.linkedin.com/in/kartik-gupta-2851863b1/?isSelfProfile=false) &nbsp;|&nbsp;
[Rakshit Kumar](https://www.linkedin.com/in/rakshit-kumar-842124386/?isSelfProfile=false) &nbsp;|&nbsp;
[Saurav Kumar](https://www.linkedin.com/in/saurav-kumar-11ba41320/?isSelfProfile=false)

---

## ✨ Features

| Feature | Description |
|---|---|
| 🤖 **AI Disease Detection** | MobileNetV2 transfer learning model trained on 38 plant disease classes |
| 🎯 **Confidence Scoring** | Shows prediction confidence; flags unknown/unclear images |
| 📄 **PDF Diagnostic Report** | Professional downloadable report with diagnosis, treatment & prevention |
| 🌿 **38 Disease Classes** | Covers Apple, Grape, Tomato, Potato, Corn, Peach, Pepper and more |
| 🎨 **Modern UI** | Dark-themed, drag-and-drop leaf image upload interface |
| 📊 **Severity Levels** | None / Moderate / High / Critical classification |

---

## 🗂️ Project Structure

```
plant-disease/
├── app.py                  ← Flask backend — routes, prediction, PDF generation
├── train_model.py          ← Full MobileNetV2 training script
├── train_fast.py           ← Faster training variant
├── diagnose_model.py       ← Model diagnostic utilities
├── merge_dataset.py        ← Dataset merging helper
├── requirements.txt        ← Python dependencies
├── templates/
│   └── index.html          ← Frontend UI
├── model/
│   ├── class_names.json    ← Class index → disease name mapping
│   ├── plant_model.keras   ← Trained model weights (not committed — see below)
│   └── plant_model.h5      ← Trained model weights (not committed — see below)
└── dataset/                ← Training images (not committed — see below)
```

> ⚠️ **Model weights** (`*.keras`, `*.h5`) and the **dataset** are excluded from this repo due to size. See [Training](#-training-your-own-model) to retrain, or download pre-trained weights separately.

---

## ⚙️ Setup & Run

### Prerequisites
- Python 3.9+
- pip

### Step 1 — Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/plant-disease.git
cd plant-disease
```

### Step 2 — Create & activate virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / Mac
python -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

> **Additional dependencies needed:**
> ```bash
> pip install tensorflow opencv-python
> ```

### Step 4 — Add model weights
Place your trained model file in the `model/` directory:
```
model/plant_model.keras   (or plant_model.h5)
model/class_names.json
```

### Step 5 — Run the app
```bash
python app.py
```

### Step 6 — Open in browser
```
http://127.0.0.1:5000
```

---

## 🧠 Training Your Own Model

To train the MobileNetV2 model on the [PlantVillage dataset](https://www.kaggle.com/datasets/emmarex/plantdisease):

1. Download and place dataset images in `dataset/<ClassName>/` folders
2. Run the training script:
```bash
python train_model.py
```

This saves `model/plant_model.keras` and `model/class_names.json` automatically.

---

## 🌱 Supported Plant & Disease Classes

38 classes across 14 plant species including:

- 🍎 Apple (Scab, Black Rot, Cedar Rust, Healthy)
- 🍇 Grape (Black Rot, Esca, Leaf Blight, Healthy)
- 🍅 Tomato (9 disease classes + Healthy)
- 🌽 Corn / Maize (Gray Leaf Spot, Common Rust, Northern Blight, Healthy)
- 🥔 Potato (Early Blight, Late Blight, Healthy)
- 🍑 Peach, 🫑 Pepper, 🍓 Strawberry, 🫐 Blueberry, and more

---

## 📦 Dependencies

```
flask>=3.0.0
tensorflow
opencv-python
reportlab>=4.0.0
numpy
```

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease) — training data
- [TensorFlow / Keras](https://www.tensorflow.org/) — model framework
- [MobileNetV2](https://arxiv.org/abs/1801.04381) — base architecture
- [ReportLab](https://www.reportlab.com/) — PDF generation
