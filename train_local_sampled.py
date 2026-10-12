"""
Local Fast Training for Plant Disease Detection
Trained on sampled dataset (~1,900 images across 38 classes)
Uses pre-trained MobileNetV2 with transfer learning for fast convergence on CPU.
"""
import os
import sys
import json
import time

# Fix encoding for Windows console
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 6
DATASET_DIR = "dataset_sampled"
MODEL_SAVE_PATH = "model/plant_model.keras"
MODEL_H5_PATH = "model/plant_model.h5"
CLASS_NAMES_PATH = "model/class_names.json"

if not os.path.exists(DATASET_DIR):
    print(f"[ERROR] Dataset folder '{DATASET_DIR}' not found!")
    sys.exit(1)

classes = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])
num_classes = len(classes)
print(f"[INFO] Found {num_classes} classes in '{DATASET_DIR}'.")

print("[INFO] Loading training dataset...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=True
)

print("[INFO] Loading validation dataset...")
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=False
)

# Save class names dictionary
class_names_list = train_ds.class_names
class_names_dict = {i: name for i, name in enumerate(class_names_list)}
os.makedirs("model", exist_ok=True)
with open(CLASS_NAMES_PATH, "w") as f:
    json.dump(class_names_dict, f, indent=2)
print(f"[OK] Class names saved -> {CLASS_NAMES_PATH}")

# Optimize pipeline
AUTOTUNE = tf.data.AUTOTUNE

def preprocess(x, y):
    return preprocess_input(x), y

train_data = train_ds.map(preprocess, num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)
val_data = val_ds.map(preprocess, num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)

print("[INFO] Building MobileNetV2 architecture with transfer learning...")
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.4)(x)
predictions = Dense(num_classes, activation='softmax', dtype='float32')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True, verbose=1),
    ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
]

print("\n" + "=" * 60)
print(f">> STARTING FAST TRAINING ({EPOCHS} EPOCHS) ON CPU...")
print("=" * 60 + "\n")

start_time = time.time()

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

elapsed = time.time() - start_time
minutes, seconds = divmod(int(elapsed), 60)

# Save in both formats
model.save(MODEL_SAVE_PATH)
model.save(MODEL_H5_PATH)

print(f"\n[OK] Model successfully saved to {MODEL_SAVE_PATH}")
print(f"[OK] Model successfully saved to {MODEL_H5_PATH}")

loss, accuracy = model.evaluate(val_data, verbose=0)
print("\n" + "=" * 60)
print(f">> TRAINING COMPLETE!")
print(f">> Final Validation Accuracy : {accuracy * 100:.2f}%")
print(f">> Final Validation Loss     : {loss:.4f}")
print(f">> Total Training Time       : {minutes}m {seconds}s")
print("=" * 60)
