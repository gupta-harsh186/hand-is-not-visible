"""Quick diagnostic: test model predictions on actual dataset images"""
import tensorflow as tf
import numpy as np
import cv2
import json
import os

# Load model
if os.path.exists("model/plant_model.keras"):
    model = tf.keras.models.load_model("model/plant_model.keras")
else:
    model = tf.keras.models.load_model("model/plant_model.h5")

# Load class names from JSON
with open("model/class_names.json") as f:
    class_map = json.load(f)
class_names = [class_map[str(i)] for i in range(len(class_map))]

print(f"Model output shape: {model.output_shape}")
print(f"Total classes: {len(class_names)}")
print("=" * 70)

# Test with images from different classes
test_classes = [
    "Grape___Black_rot",
    "Grape___healthy",
    "Tomato___Early_blight",
    "Potato___Late_blight",
    "Apple___Apple_scab",
]

for cls in test_classes:
    folder = os.path.join("dataset", cls)
    if not os.path.exists(folder):
        print(f"[SKIP] {cls} - folder not found")
        continue
    
    # Get first JPG image (not masked)
    imgs = [f for f in os.listdir(folder) if f.endswith('.JPG') or f.endswith('.jpg')]
    if not imgs:
        imgs = [f for f in os.listdir(folder) if f.endswith('.png') or f.endswith('.jpeg')]
    if not imgs:
        print(f"[SKIP] {cls} - no images found")
        continue
    
    img_path = os.path.join(folder, imgs[0])
    
    # Test with BGR (current app.py behavior - NO conversion)
    img_bgr = cv2.imread(img_path)
    img_bgr = cv2.resize(img_bgr, (224, 224))
    img_bgr = img_bgr / 255.0
    img_bgr = np.reshape(img_bgr, (1, 224, 224, 3))
    
    pred_bgr = model.predict(img_bgr, verbose=0)
    idx_bgr = int(np.argmax(pred_bgr))
    conf_bgr = float(np.max(pred_bgr))
    
    # Test with RGB (with BGR->RGB conversion)
    img_rgb = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (224, 224))
    img_rgb = img_rgb / 255.0
    img_rgb = np.reshape(img_rgb, (1, 224, 224, 3))
    
    pred_rgb = model.predict(img_rgb, verbose=0)
    idx_rgb = int(np.argmax(pred_rgb))
    conf_rgb = float(np.max(pred_rgb))
    
    print(f"\nExpected: {cls}")
    print(f"  BGR → {class_names[idx_bgr]} ({conf_bgr*100:.1f}%)  {'✅' if class_names[idx_bgr] == cls else '❌'}")
    print(f"  RGB → {class_names[idx_rgb]} ({conf_rgb*100:.1f}%)  {'✅' if class_names[idx_rgb] == cls else '❌'}")

print("\n" + "=" * 70)
print("If RGB predictions are correct, add BGR->RGB conversion to app.py")
print("If BOTH are wrong, the model needs retraining")
