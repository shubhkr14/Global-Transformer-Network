import os
import csv
import numpy as np
import pandas as pd

# files
PRED_CSV = "predictions.csv"
OUT_METRICS = "metrics_summary.txt"
OUT_CONF = "confusion_matrix.csv"
OUT_ERRORS = "top_misclassified.csv"

# load predictions
df = pd.read_csv(PRED_CSV)
# ensure columns exist
assert set(["filepath","true_class","predicted_class","confidence"]).issubset(df.columns), "predictions.csv missing columns"

# labels
labels = sorted(df["true_class"].unique().tolist())
label_to_idx = {l:i for i,l in enumerate(labels)}
n = len(labels)

# confusion matrix
cm = np.zeros((n,n), dtype=int)
for _, row in df.iterrows():
    t = label_to_idx[row["true_class"]]
    p = label_to_idx[row["predicted_class"]]
    cm[t,p] += 1

# per-class metrics
precision = []
recall = []
f1 = []
support = cm.sum(axis=1)
for i in range(n):
    tp = cm[i,i]
    fp = cm[:,i].sum() - tp
    fn = cm[i,:].sum() - tp
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1s = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0
    precision.append(prec)
    recall.append(rec)
    f1.append(f1s)

# overall accuracy
accuracy = np.trace(cm) / cm.sum() if cm.sum() > 0 else 0.0

# save confusion matrix to CSV with labels
cm_df = pd.DataFrame(cm, index=labels, columns=labels)
cm_df.to_csv(OUT_CONF)

# save metrics summary
with open(OUT_METRICS, "w", encoding="utf-8") as f:
    f.write(f"Overall accuracy: {accuracy:.6f}\n\n")
    f.write("label,precision,recall,f1,support\n")
    for lbl, p, r, ff, sup in zip(labels, precision, recall, f1, support):
        f.write(f"{lbl},{p:.6f},{r:.6f},{ff:.6f},{int(sup)}\n")

# top misclassified examples (highest confidence but wrong)
wrong = df[df["true_class"] != df["predicted_class"]].copy()
if not wrong.empty:
    # confidence numeric
    wrong["confidence"] = wrong["confidence"].astype(float)
    wrong_sorted = wrong.sort_values(by="confidence", ascending=False).head(100)
    wrong_sorted.to_csv(OUT_ERRORS, index=False)
else:
    # create empty file
    pd.DataFrame(columns=df.columns).to_csv(OUT_ERRORS, index=False)

# print summary to console
print("Done. Results:")
print(f"Overall accuracy: {accuracy:.6f}")
print()
print("Per-class metrics:")
print("label\tprecision\trecall\tf1\tsupport")
for lbl, p, r, ff, sup in zip(labels, precision, recall, f1, support):
    print(f"{lbl}\t{p:.4f}\t{r:.4f}\t{ff:.4f}\t{int(sup)}")

print()
print(f"Confusion matrix saved to: {OUT_CONF}")
print(f"Per-class metrics saved to: {OUT_METRICS}")
print(f"Top misclassified saved to: {OUT_ERRORS}")
