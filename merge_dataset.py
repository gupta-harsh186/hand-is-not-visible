import os
import shutil

def merge_directories():
    base_dir = r"d:\hg\OneDrive\Desktop\plant-disease\dataset"
    
    merges = [
        ("Pepper__bell___Bacterial_spot", "Pepper,_bell___Bacterial_spot"),
        ("Pepper__bell___healthy", "Pepper,_bell___healthy"),
        ("Tomato_Bacterial_spot", "Tomato___Bacterial_spot"),
        ("Tomato_Early_blight", "Tomato___Early_blight"),
        ("Tomato_Late_blight", "Tomato___Late_blight"),
        ("Tomato_Leaf_Mold", "Tomato___Leaf_Mold"),
        ("Tomato_Septoria_leaf_spot", "Tomato___Septoria_leaf_spot"),
        ("Tomato_Spider_mites_Two_spotted_spider_mite", "Tomato___Spider_mites Two-spotted_spider_mite"),
        ("Tomato__Target_Spot", "Tomato___Target_Spot"),
        ("Tomato__Tomato_mosaic_virus", "Tomato___Tomato_mosaic_virus"),
        ("Tomato_healthy", "Tomato___healthy")
    ]
    
    for src_name, dst_name in merges:
        src = os.path.join(base_dir, src_name)
        dst = os.path.join(base_dir, dst_name)
        
        if os.path.exists(src) and os.path.isdir(src):
            if not os.path.exists(dst):
                os.makedirs(dst)
                print(f"Created target directory: {dst_name}")
                
            print(f"Merging '{src_name}' into '{dst_name}'...")
            
            for filename in os.listdir(src):
                src_file = os.path.join(src, filename)
                dst_file = os.path.join(dst, filename)
                
                if os.path.isfile(src_file):
                    if os.path.exists(dst_file):
                        # Handle duplicate filenames if necessary
                        base, ext = os.path.splitext(filename)
                        import uuid
                        dst_file = os.path.join(dst, f"{base}_{uuid.uuid4().hex[:6]}{ext}")
                    
                    shutil.move(src_file, dst_file)
            
            # Remove empty source directory
            try:
                os.rmdir(src)
                print(f"Removed empty directory: {src_name}")
            except OSError as e:
                print(f"Warning: Could not remove directory '{src_name}': {e}")
                
    print("Merge operation completed successfully!")

if __name__ == "__main__":
    merge_directories()
