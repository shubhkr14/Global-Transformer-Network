# predict_all.py  (auto-detects test folder)
import os
import csv
import numpy as np
from PIL import Image
from tensorflow import keras
from model import build_gt_net

def load_images_from_dir(val_dir, target_size=(224,224)):
    paths, X = [], []
    for cls in sorted(os.listdir(val_dir)):
        cls_path = os.path.join(val_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        for fname in sorted(os.listdir(cls_path)):
            p = os.path.join(cls_path, fname)
            if not os.path.isfile(p):
                continue
            try:
                img = Image.open(p).convert("RGB").resize(target_size)
                arr = np.array(img).astype("float32") / 255.0
                paths.append(p)
                X.append(arr)
            except Exception as e:
                print("Skipped", p, ":", e)
    return paths, np.stack(X, axis=0) if X else ([], np.zeros((0,)+target_size+(3,), dtype="float32"))

def load_model_and_weights(weights_path, backbone='densenet121', num_classes=4):
    model = build_gt_net(input_shape=(224,224,3), backbone_name=backbone, heads=4, num_classes=num_classes)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=2e-4),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.load_weights(weights_path)
    return model

def main():
    # auto-detect paths
    val_dir = "data/test" if os.path.isdir("data/test") else "data/val"
    weights_path = "./checkpoints/gt_net.weights.h5"
    out_csv = "test_predictions.csv"

    if not os.path.isdir(val_dir):
        raise SystemExit(f"❌ No folder found: {val_dir}")

    print(f"✅ Using dataset: {val_dir}")

    train_dir = 'data/train'
    class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    if not class_names:
        raise SystemExit("❌ No class folders found in data/train")

    print("📂 Loading images ...")
    paths, X = load_images_from_dir(val_dir, target_size=(224,224))
    print(f"Found {len(paths)} images to predict.")

    print("⚙️ Loading model & weights ...")
    # attempt to rebuild+load weights first; if it fails, fallback to full .keras model
    try:
        model = load_model_and_weights(weights_path, backbone='densenet121', num_classes=len(class_names))
        print("✅ Model loaded successfully (loaded weights by rebuilding architecture).")
    except Exception as e:
        print("⚠️ load_weights() failed with error:", str(e))
        print("⚙️ Falling back to loading full .keras model: ./checkpoints/gt_net_final.keras")
        from tensorflow import keras
        # allow deserialization for custom layers (same as predict.py)
        keras.config.enable_unsafe_deserialization()
        import model as model_module  # ensures GSB, GeM classes are available
        full_path = "./checkpoints/gt_net_final.keras"
        if not os.path.isfile(full_path):
            raise SystemExit(f"❌ Full .keras model not found at {full_path}; cannot continue.")
        model = keras.models.load_model(full_path, compile=False,
                                        custom_objects={'GSB': model_module.GSB, 'GeM': model_module.GeM})
        print("✅ Model loaded successfully (loaded full .keras model).")


    print("🔮 Predicting ...")
    preds = model.predict(X, batch_size=8, verbose=1)

    print("📝 Writing results to CSV ...")
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['filepath','true_class','predicted_class','confidence'])
        for pth, p in zip(paths, preds):
            pred_idx = int(np.argmax(p))
            conf = float(p[pred_idx])
            true_cls = os.path.basename(os.path.dirname(pth))
            writer.writerow([pth, true_cls, class_names[pred_idx], f"{conf:.6f}"])

    print(f"✅ Done. Predictions saved to {out_csv}")

if __name__ == "__main__":
    main()
