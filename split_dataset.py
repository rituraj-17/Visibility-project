import os
import shutil
import random

random.seed(42)

#PATH 

BASE_DIR = "/Users/mac/Desktop/VISIBILITY_PROJECT"
DEST_DIR = os.path.join(BASE_DIR, "dataset")


VAL_TEST_FOLDER = "2022_2023 CAM1"

TRAIN_FOLDERS = [
    "2023_2024 CAM1",
    "2024_2025 CAM1",
    "2024_2025 CAM2"
]

CLASSES = ["fog", "no_fog"]



for split in ["train", "val", "test"]:
    for cls in CLASSES:
        os.makedirs(os.path.join(DEST_DIR, split, cls), exist_ok=True)

print("dCreated dataset/train, dataset/val, dataset/test with class folders")


#HELPER FUNCTION

def get_images(folder):
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]


#TRAIN DATA 

for folder in TRAIN_FOLDERS:
    for cls in CLASSES:
        src = os.path.join(BASE_DIR, folder, cls)

        if not os.path.exists(src):
            print(f"⚠️ Skipping missing folder: {src}")
            continue

        images = get_images(src)

        for img in images:
            shutil.copy(img, os.path.join(DEST_DIR, "train", cls))

        print(f"TRAIN ← {folder}/{cls}: {len(images)} images")


#VALIDATION + TEST 

for cls in CLASSES:
    src = os.path.join(BASE_DIR, VAL_TEST_FOLDER, cls)

    if not os.path.exists(src):
        print(f"⚠️ Skipping missing folder: {src}")
        continue

    images = get_images(src)
    random.shuffle(images)

    mid = len(images) // 2
    val_imgs = images[:mid]
    test_imgs = images[mid:]

    for img in val_imgs:
        shutil.copy(img, os.path.join(DEST_DIR, "val", cls))

    for img in test_imgs:
        shutil.copy(img, os.path.join(DEST_DIR, "test", cls))

    print(f" 2022 → VAL:{len(val_imgs)} TEST:{len(test_imgs)} ({cls})")

print("\n🎉 Dataset split completed EXACTLY as per professor's instruction.")
