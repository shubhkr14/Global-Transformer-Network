import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

CSV_FILE = "top_misclassified.csv"
OUT_FILE = "misclassified_grid.png"

if not os.path.isfile(CSV_FILE):
    raise SystemExit(f"Missing {CSV_FILE} - run eval_preds.py first")

df = pd.read_csv(CSV_FILE)

# Only show top 16 misclassified
n = min(16, len(df))
df = df.head(n)

fig, axes = plt.subplots(4, 4, figsize=(12, 12))
axes = axes.flatten()

for ax, (_, row) in zip(axes, df.iterrows()):
    img_path = row["filepath"]
    true_cls = row["true_class"]
    pred_cls = row["predicted_class"]
    conf = row["confidence"]
    if os.path.exists(img_path):
        img = mpimg.imread(img_path)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"T:{true_cls}\nP:{pred_cls} ({conf:.2f})",
                     fontsize=9, color="red" if true_cls != pred_cls else "green")
    else:
        ax.axis("off")
        ax.set_title("Missing", color="gray")

plt.tight_layout()
plt.savefig(OUT_FILE, dpi=150)
print(f"Saved misclassified grid to {OUT_FILE}")
