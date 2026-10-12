import os
import shutil
import random

SOURCE_DATASET = r"d:\hg\OneDrive\Desktop\plant-disease-project\dataset"
TARGET_DATASET = r"d:\hg\OneDrive\Desktop\project-live\plant-disease-main\dataset_sampled"
SAMPLES_PER_CLASS = 50

def create_sample():
    if not os.path.exists(SOURCE_DATASET):
        print(f"[ERROR] Source dataset not found: {SOURCE_DATASET}")
        return

    os.makedirs(TARGET_DATASET, exist_ok=True)
    classes = [d for d in os.listdir(SOURCE_DATASET) if os.path.isdir(os.path.join(SOURCE_DATASET, d))]
    print(f"[INFO] Found {len(classes)} classes in source dataset.")

    total_copied = 0
    for cls in sorted(classes):
        src_cls_dir = os.path.join(SOURCE_DATASET, cls)
        dst_cls_dir = os.path.join(TARGET_DATASET, cls)
        os.makedirs(dst_cls_dir, exist_ok=True)

        images = [f for f in os.listdir(src_cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        # Pick up to SAMPLES_PER_CLASS images
        random.seed(42)
        chosen = random.sample(images, min(len(images), SAMPLES_PER_CLASS))

        for img in chosen:
            src_file = os.path.join(src_cls_dir, img)
            dst_file = os.path.join(dst_cls_dir, img)
            if not os.path.exists(dst_file):
                shutil.copy2(src_file, dst_file)
                total_copied += 1

    print(f"[SUCCESS] Sampled dataset created with {total_copied} images across {len(classes)} classes at:")
    print(f"         {TARGET_DATASET}")

if __name__ == "__main__":
    create_sample()
