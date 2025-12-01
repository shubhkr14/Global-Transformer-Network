import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# files
CM_CSV = "confusion_matrix.csv"
OUT_PNG = "confusion_matrix.png"

if not os.path.isfile(CM_CSV):
    raise SystemExit(f"Missing {CM_CSV} - run eval_preds.py first")

cm_df = pd.read_csv(CM_CSV, index_col=0)
labels = list(cm_df.index)
cm = cm_df.values.astype(int)

fig, ax = plt.subplots(figsize=(8,6))
im = ax.imshow(cm, interpolation='nearest', aspect='auto')
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel("count", rotation=-90, va="bottom")

# ticks
ax.set(xticks=np.arange(len(labels)), yticks=np.arange(len(labels)),
       xticklabels=labels, yticklabels=labels,
       ylabel="True label", xlabel="Predicted label",
       title="Confusion Matrix")

plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# annotate counts
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")

fig.tight_layout()
plt.savefig(OUT_PNG, dpi=150)
print("Saved confusion matrix heatmap to", OUT_PNG)
