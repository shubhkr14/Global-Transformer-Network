import os
import random
import shutil

# Adjust this path to the folder where your extracted dataset is
source_dir = r"C:\Users\KIIT\Downloads\Brain Tumor MRI Dataset"

# Destination base folder (your GT-NET project)
dest_base = r"C:\Users\KIIT\Desktop\GT-NET\data"

# Percentage split
train_split = 0.8   # 80% train, 20% validation

# Class folder names — adjust based on what your dataset has
classes = ["glioma_tumor", "meningioma_tumor", "pituitary_tumor", "no_tumor"]

for cls in classes:
    src = os.path.join(source_dir, cls)
    if not os.path.exists(src):
        print(f"⚠️ Skipping {cls} (not found in dataset)")
        continue

    # get all image filenames
    images = [f for f in os.listdir(src) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    random.shuffle(images)
    split_idx = int(len(images) * train_split)
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    # make train/val folders
    train_dir = os.path.join(dest_base, "train", cls.replace("_tumor", ""))
    val_dir = os.path.join(dest_base, "val", cls.replace("_tumor", ""))

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # copy images
    for img in train_imgs:
        shutil.copy(os.path.join(src, img), os.path.join(train_dir, img))
    for img in val_imgs:
        shutil.copy(os.path.join(src, img), os.path.join(val_dir, img))

    print(f"✅ {cls}: {len(train_imgs)} train, {len(val_imgs)} val")

print("\n🎉 Dataset successfully split and copied to:")
print(dest_base)
