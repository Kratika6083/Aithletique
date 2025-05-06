import os
from PIL import Image

# CONFIGURATION
DATASET_DIR = "Dataset"  # Folder containing your yoga pose folders
TARGET_FORMAT = "jpg"     # You can choose "jpg", "jpeg", or "png"
DELETE_OLD = True         # Delete original files after converting

SUPPORTED_FORMATS = ["jpg", "jpeg", "png", "webp", "avif"]

def clean_and_convert_images():
    for root, dirs, files in os.walk(DATASET_DIR):
        for filename in files:
            ext = filename.split(".")[-1].lower()
            if ext in SUPPORTED_FORMATS:
                try:
                    filepath = os.path.join(root, filename)
                    img = Image.open(filepath).convert("RGB")

                    # ✅ Keep original image size (NO RESIZE)
                    new_filename = os.path.splitext(filename)[0] + "." + TARGET_FORMAT
                    new_filepath = os.path.join(root, new_filename)
                    img.save(new_filepath, quality=95)

                    if DELETE_OLD and ext != TARGET_FORMAT:
                        os.remove(filepath)

                    print(f"✅ Converted: {filepath} -> {new_filepath}")

                except Exception as e:
                    print(f"⚠️ Skipping {filename}: {e}")

if __name__ == "__main__":
    clean_and_convert_images()
    print("\n🎉 Dataset cleaning completed!")
