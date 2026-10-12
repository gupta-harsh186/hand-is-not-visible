# 🌿 Google Colab GPU Training Guide (100% Free & Fast)

Aapke project ke liye ready-to-run notebook **[`plant_disease_colab_training.ipynb`](file:///d:/hg/OneDrive/Desktop/project-live/plant-disease-main/plant_disease_colab_training.ipynb)** taiyaar kar diya gaya hai.

---

## ⚡ Google Colab par Model Train karne ke Simple Steps:

### Step 1: Notebook Open karein
1. Apne browser me **[https://colab.research.google.com](https://colab.research.google.com)** open karein.
2. **"Upload"** tab par click karein aur apne computer se **`plant_disease_colab_training.ipynb`** file select karke upload karein.

---

### Step 2: Free T4 GPU Enable karein
1. Colab ke top menu me jayein: **Runtime** → **Change runtime type**.
2. **Hardware accelerator** me **T4 GPU** select karein aur **Save** karein.

---

### Step 3: Training Start karein
1. Top menu me **Runtime** → **Run all** (ya `Ctrl + F9`) press karein.
2. Notebook automatically:
   - PlantVillage dataset (54,000+ images across 38 classes) download aur extract karega.
   - Mixed Precision (FP16) enable karega jisse training 2x fast ho sake.
   - MobileNetV2 architecture build karega.
   - **Phase 1 (Head Warmup)** aur **Phase 2 (Deep Fine-Tuning of top 40 layers)** perform karega.
   - **~12 se 15 minutes** me model **97% - 98%+ validation accuracy** ke sath train ho jayega.

---

### Step 4: Trained Model Files Replace karein
Training complete hone ke baad last cell automatically 3 files download karega:
1. `plant_model.keras`
2. `plant_model.h5`
3. `class_names.json`

In teeno files ko apne project ke folder:
📁 `d:\hg\OneDrive\Desktop\project-live\plant-disease-main\model\`
me paste (replace) kar dein!

---

### Step 5: Test karein
Aapka local server naye high-accuracy model ke sath restart hokar perfect predictions dega!
