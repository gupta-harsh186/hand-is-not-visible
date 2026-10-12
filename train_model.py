import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import json
import os
import sys

# Fix encoding for Windows console
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
EPOCHS      = 30
DATASET_DIR = "dataset"
MODEL_SAVE_PATH  = "model/plant_model.keras"
CLASS_NAMES_PATH = "model/class_names.json"

# -------------------------------------------------------------------------
# Check dataset exists
# -------------------------------------------------------------------------
if not os.path.exists(DATASET_DIR):
    print(f"[ERROR] Dataset folder '{DATASET_DIR}' not found!")
    print("  Please create a 'dataset' folder with subfolders for each disease class.")
    print("  Example: dataset/Apple___healthy/, dataset/Tomato___Early_blight/, etc.")
    sys.exit(1)

classes = os.listdir(DATASET_DIR)
print(f"[OK] Found {len(classes)} classes in '{DATASET_DIR}'")
print("Sample classes:", classes[:5])

# -------------------------------------------------------------------------
# Data Augmentation
# -------------------------------------------------------------------------
train_gen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.6, 1.4],
    fill_mode='nearest'
)

val_gen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2
)

train_data = train_gen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_data = val_gen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

num_classes = train_data.num_classes
print(f"\n[OK] Total Classes  : {num_classes}")
print(f"[OK] Training images: {train_data.samples}")
print(f"[OK] Validation imgs: {val_data.samples}\n")

# -------------------------------------------------------------------------
# Save class names  (keeps labels in sync with the model)
# -------------------------------------------------------------------------
class_indices = train_data.class_indices           # {'Apple___scab': 0, ...}
class_names   = {v: k for k, v in class_indices.items()}  # {0: 'Apple___scab', ...}

os.makedirs("model", exist_ok=True)
with open(CLASS_NAMES_PATH, "w") as f:
    json.dump(class_names, f, indent=2)
print(f"[OK] Class names saved → {CLASS_NAMES_PATH}\n")

# -------------------------------------------------------------------------
# Build model  —  Transfer Learning with MobileNetV2
# -------------------------------------------------------------------------
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False   # freeze base first

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# -------------------------------------------------------------------------
# Compile
# -------------------------------------------------------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -------------------------------------------------------------------------
# Callbacks
# -------------------------------------------------------------------------
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),
    ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# -------------------------------------------------------------------------
# Phase 1 — Train only the top layers  (base frozen)
# -------------------------------------------------------------------------
print("\n" + "="*60)
print(">> PHASE 1: Training top layers (base frozen)...")
print("="*60 + "\n")

history1 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=15,
    callbacks=callbacks,
    verbose=1
)

# -------------------------------------------------------------------------
# Phase 2 — Fine-tune  (unfreeze last 50 layers of base)
# -------------------------------------------------------------------------
print("\n" + "="*60)
print(">> PHASE 2: Fine-tuning (unfreezing last 50 base layers)...")
print("="*60 + "\n")

base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history2 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    initial_epoch=15,
    callbacks=callbacks,
    verbose=1
)

# -------------------------------------------------------------------------
# Save final model (both formats for compatibility)
# -------------------------------------------------------------------------
model.save(MODEL_SAVE_PATH)
model.save("model/plant_model.h5")
print(f"\n[OK] Model saved → {MODEL_SAVE_PATH}")
print(f"[OK] Model saved → model/plant_model.h5")
print(f"[OK] Class names  → {CLASS_NAMES_PATH}")

# -------------------------------------------------------------------------
# Final evaluation
# -------------------------------------------------------------------------
print("\n" + "="*60)
print(">> FINAL EVALUATION ON VALIDATION SET")
print("="*60)
loss, accuracy = model.evaluate(val_data, verbose=1)
print(f"\n[RESULT] Validation Loss     : {loss:.4f}")
print(f"[RESULT] Validation Accuracy : {accuracy*100:.2f}%")
print("\n[DONE] Training complete! You can now run: python app.py")
