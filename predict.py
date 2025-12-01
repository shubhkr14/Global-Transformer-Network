# predict.py  (use this instead of loading the .keras file)
import sys
import os
import numpy as np
from PIL import Image
import model as model_module

# rebuild model architecture and load weights
from tensorflow import keras
from model import build_gt_net

def load_model_from_weights(weights_path, backbone='densenet121', num_classes=4):
    # Build same architecture used for training
    model = build_gt_net(input_shape=(224,224,3), backbone_name=backbone, heads=4, num_classes=num_classes)
    # compile is optional for prediction, but harmless
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=2e-4),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    # load weights
    model.load_weights(weights_path)
    return model

def main():
    if len(sys.argv) != 2:
        print("Usage: python predict.py /path/to/image.jpg")
        sys.exit(1)

    img_path = sys.argv[1]
    if not os.path.isfile(img_path):
        print("Image not found:", img_path)
        sys.exit(1)

    # find class names from train folder
    train_dir = 'data/train'
    class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    if not class_names:
        print("No class folders found in data/train")
        sys.exit(1)

    # change weights path if you renamed it
    weights_path = './checkpoints/gt_net.weights.h5'
    if not os.path.isfile(weights_path):
        print("Weights file not found:", weights_path)
        print("Make sure ./checkpoints/gt_net.weights.h5 exists")
        sys.exit(1)

    # load model (rebuild + weights)
    print("Loading model architecture and weights...")

# try strict rebuild+load_weights first (keeps original behavior)
    try:
        model = load_model_from_weights(weights_path, backbone='densenet121', num_classes=len(class_names))
        print("Model ready (loaded weights by rebuilding architecture).")
    except Exception as e:
    # if loading weights failed (e.g. missing GSB variables), fallback to full .keras model
        print("Warning: load_weights() failed with error:", str(e))
        print("Falling back to loading full .keras model (gt_net_final.keras)")

        from tensorflow import keras
    # allow unsafe deserialization (needed if model used lambdas/custom)
        keras.config.enable_unsafe_deserialization()

        full_path = './checkpoints/gt_net_final.keras'
        if not os.path.isfile(full_path):
            print("Full .keras model not found:", full_path)
            print("Cannot continue. Either restore original model.py or retrain to produce matching weights.")
            raise SystemExit(1)

    # import model module so custom classes (GSB, GeM) are available for deserialization

    model = keras.models.load_model(full_path, compile=False, custom_objects={'GSB': model_module.GSB, 'GeM': model_module.GeM})
    print("Model ready (loaded full .keras model).")


    # preprocess image (same as training)
    img = Image.open(img_path).convert('RGB').resize((224, 224))
    arr = np.array(img).astype('float32') / 255.0
    arr = np.expand_dims(arr, axis=0)

    # predict
    pred = model.predict(arr)
    pred_idx = int(pred.argmax(axis=-1)[0])
    prob = float(pred[0, pred_idx])

    print("Predicted class:", class_names[pred_idx])
    print(f"Confidence: {prob:.4f}")

if __name__ == "__main__":
    main()
