# gradcam_plus.py
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image

# === CONFIGURATION ===
MODEL_PATH = r"checkpoints\gt_net_test.h5"   # <-- our trained model
IMAGE_PATH = r"C:\Users\KIIT\Desktop\GT-NET\data\val\glioma\Tr-gl_0021.jpg" 
LAYER_NAME = None   # leave None to auto-detect last conv layer
# =====================

def gradcam_plus_plus(model, img_array, class_index, layer_name=None):
    # find the last convolutional layer if not specified
    if layer_name is None:
        for layer in reversed(model.layers):
            if len(layer.output_shape) == 4:
                layer_name = layer.name
                break

    grad_model = Model([model.inputs],
                       [model.get_layer(layer_name).output, model.output])

    with tf.GradientTape() as tape1:
        with tf.GradientTape() as tape2:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, class_index]
        grads = tape2.gradient(loss, conv_outputs)
    second_grad = tape1.gradient(grads, conv_outputs)

    first_grad = grads.numpy()
    second_grad = second_grad.numpy()
    conv_outputs = conv_outputs.numpy()

    alpha_num = second_grad
    alpha_denom = 2 * second_grad + np.sum(first_grad * conv_outputs, axis=(1, 2), keepdims=True)
    alpha_denom = np.where(alpha_denom != 0, alpha_denom, 1e-10)
    alphas = alpha_num / alpha_denom

    weights = np.maximum(first_grad, 0)
    deep_linear = np.sum(alphas * weights, axis=(1, 2))
    cam = np.sum(deep_linear[..., np.newaxis] * conv_outputs, axis=-1)[0]
    cam = np.maximum(cam, 0)
    cam /= np.max(cam) + 1e-8
    return cam

def overlay_heatmap(img, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    return overlay

if __name__ == "__main__":
    print("Loading model from:", MODEL_PATH)
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    print("Loading image from:", IMAGE_PATH)
    img = image.load_img(IMAGE_PATH, target_size=(224, 224))
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)
    class_idx = np.argmax(preds[0])
    print("Predicted class index:", class_idx, "   Probabilities:", preds[0])

    cam = gradcam_plus_plus(model, x, class_idx, layer_name=LAYER_NAME)
    original = cv2.cvtColor(cv2.imread(IMAGE_PATH), cv2.COLOR_BGR2RGB)
    result = overlay_heatmap(original, cam)

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(original)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Grad-CAM++")
    plt.imshow(result)
    plt.axis("off")

    plt.tight_layout()
    plt.show()