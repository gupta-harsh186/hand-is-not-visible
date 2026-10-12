"""
Fast Training Script for Plant Disease Detection
Uses mixed precision, tf.data pipelines, and aggressive optimizations
for significantly faster training while maintaining accuracy.
"""
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import json
import os
import sys
import time

# Fix encoding for Windows console
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# =========================================================================
# SPEED OPTIMIZATION 1: Enable Mixed Precision (FP16)
# Roughly 2x faster on GPUs with Tensor Cores (RTX series, T4, A100, etc.)
# Falls back gracefully on CPU — no harm if no GPU.
# =========================================================================
try:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print("[SPEED] Mixed precision (FP16) ENABLED")
except Exception:
    print("[INFO] Mixed precision not available, using FP32")

# =========================================================================
# Configuration
# =========================================================================
IMG_SIZE        = (224, 224)
BATCH_SIZE      = 64          # Larger batch = faster throughput
EPOCHS          = 15          # Fewer epochs, rely on early stopping
DATASET_DIR     = "dataset"
MODEL_SAVE_PATH = "model/plant_model.keras"
CLASS_NAMES_PATH = "model/class_names.json"

# =========================================================================
# GPU Memory Growth (prevent OOM)
# =========================================================================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"[GPU] Found {len(gpus)} GPU(s): {[g.name for g in gpus]}")
else:
    print("[CPU] No GPU detected — training on CPU (will be slower)")
    # On CPU, mixed precision can be slower — revert to float32
    tf.keras.mixed_precision.set_global_policy('float32')
    BATCH_SIZE = 32  # Smaller batch on CPU

# =========================================================================
# Check dataset
# =========================================================================
if not os.path.exists(DATASET_DIR):
    print(f"[ERROR] Dataset folder '{DATASET_DIR}' not found!")
    sys.exit(1)

classes = sorted([d for d in os.listdir(DATASET_DIR)
                  if os.path.isdir(os.path.join(DATASET_DIR, d))])
num_classes = len(classes)
print(f"[OK] Found {num_classes} classes in '{DATASET_DIR}'")

# =========================================================================
# SPEED OPTIMIZATION 2: Use tf.data instead of ImageDataGenerator
# tf.data is MUCH faster — supports prefetching, caching, parallelism
# =========================================================================
AUTOTUNE = tf.data.AUTOTUNE

# Load dataset using keras utility (returns tf.data.Dataset)
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

# Save class names
class_names_list = train_ds.class_names
class_names_dict = {i: name for i, name in enumerate(class_names_list)}
os.makedirs("model", exist_ok=True)
with open(CLASS_NAMES_PATH, "w") as f:
    json.dump(class_names_dict, f, indent=2)
print(f"[OK] Class names saved → {CLASS_NAMES_PATH}")

# =========================================================================
# SPEED OPTIMIZATION 3: Data augmentation as layers (runs on GPU!)
# This is faster than CPU-based ImageDataGenerator augmentation
# =========================================================================
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.15),
    tf.keras.layers.RandomZoom(0.15),
    tf.keras.layers.RandomContrast(0.2),
], name="data_augmentation")

# =========================================================================
# SPEED OPTIMIZATION 4: Preprocessing + Caching + Prefetching
# =========================================================================
# Use proper MobileNetV2 preprocessing (scales to [-1, 1]) instead of [0, 1]
def prepare_train(ds):
    ds = ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                num_parallel_calls=AUTOTUNE)
    ds = ds.prefetch(AUTOTUNE)
    return ds

def prepare_val(ds):
    ds = ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE)
    ds = ds.prefetch(AUTOTUNE)
    return ds

train_data = prepare_train(train_ds)
val_data = prepare_val(val_ds)

print(f"\n[OK] Training batches : {tf.data.experimental.cardinality(train_data).numpy()}")
print(f"[OK] Validation batches: {tf.data.experimental.cardinality(val_data).numpy()}\n")

# =========================================================================
# Build Model — MobileNetV2 Transfer Learning
# =========================================================================
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# OVERFITTING FIX: Freeze the entire base model. 
# Training base layers with a high learning rate destroys pre-trained features!
base_model.trainable = False

trainable_count = sum(1 for l in base_model.layers if l.trainable)
print(f"[MODEL] Base layers: {len(base_model.layers)} | Trainable: {trainable_count}")

from tensorflow.keras.regularizers import l2

x = base_model.output
x = GlobalAveragePooling2D()(x)
# OVERFITTING FIX: Added L2 regularization to penalize large weights
x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
# OVERFITTING FIX: Increased Dropout to 0.5
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax', dtype='float32')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# =========================================================================
# Compile with cosine decay for faster convergence
# =========================================================================
total_steps = tf.data.experimental.cardinality(train_data).numpy() * EPOCHS
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-3,
    decay_steps=total_steps,
    alpha=1e-6
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# =========================================================================
# Callbacks — aggressive early stopping for speed
# =========================================================================
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=4,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# =========================================================================
# Train — Single phase, fast
# =========================================================================
print("\n" + "=" * 60)
print(">> FAST TRAINING — Single Phase with Fine-Tuning")
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

# =========================================================================
# Save final model
# =========================================================================
model.save(MODEL_SAVE_PATH)
model.save("model/plant_model.h5")
print(f"\n[OK] Model saved → {MODEL_SAVE_PATH}")
print(f"[OK] Model saved → model/plant_model.h5")
print(f"[OK] Class names  → {CLASS_NAMES_PATH}")

# =========================================================================
# Final Evaluation
# =========================================================================
print("\n" + "=" * 60)
print(">> FINAL EVALUATION")
print("=" * 60)
loss, accuracy = model.evaluate(val_data, verbose=1)
print(f"\n[RESULT] Validation Loss     : {loss:.4f}")
print(f"[RESULT] Validation Accuracy : {accuracy*100:.2f}%")
print(f"[TIME]   Total training time : {minutes}m {seconds}s")
print("\n[DONE] Fast training complete! Run: python app.py")
