import os, random, shutil

# percent to copy from train -> val
val_fraction = 0.20  # 20% validation

data_root = r"C:\Users\KIIT\Desktop\GT-NET\data"  # adjust only if your project is elsewhere
train_root = os.path.join(data_root, "train")
val_root = os.path.join(data_root, "val")

os.makedirs(val_root, exist_ok=True)

classes = [d for d in os.listdir(train_root) if os.path.isdir(os.path.join(train_root, d))]
print("Detected classes:", classes)

for cls in classes:
    train_dir = os.path.join(train_root, cls)
    val_dir = os.path.join(val_root, cls)
    os.makedirs(val_dir, exist_ok=True)

    imgs = [f for f in os.listdir(train_dir) if f.lower().endswith((".jpg",".jpeg",".png"))]
    random.shuffle(imgs)
    n_val = max(1, int(len(imgs) * val_fraction))
    val_imgs = imgs[:n_val]

    for fn in val_imgs:
        src = os.path.join(train_dir, fn)
        dst = os.path.join(val_dir, fn)
        if not os.path.exists(dst):   # avoid overwriting if already copied
            shutil.copy2(src, dst)

    print(f"{cls}: {len(imgs)} total, copied {len(val_imgs)} to val")

print("\nDone. Now verify counts with the verify_counts.py script or below PowerShell commands.")
