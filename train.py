# train.py
import os
import argparse
import tensorflow as tf
from model import build_gt_net
from dataloader import get_dataset
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default='data/train')
    parser.add_argument('--val_dir', type=str, default='data/val')
    parser.add_argument('--backbone', type=str, default='densenet121')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=36)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--out', type=str, default='checkpoints/gt_net.weights.h5')

    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    train_ds = get_dataset(args.train_dir, batch_size=args.batch, augment=True)
    val_ds = get_dataset(args.val_dir, batch_size=args.batch, augment=False, shuffle=False)

    num_classes = len([d for d in os.listdir(args.train_dir) if os.path.isdir(os.path.join(args.train_dir, d))])
    print(f"Detected {num_classes} classes from {args.train_dir}")
    model = build_gt_net(input_shape=(224,224,3), backbone_name=args.backbone, heads=4, num_classes=num_classes)


    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

# ---- callbacks: save weights only (filepath must end with .weights.h5) ----
    weights_path = os.path.splitext(args.out)[0] + '.weights.h5'

    callbacks = [
        ModelCheckpoint(weights_path,
                        monitor='val_accuracy',
                        save_best_only=True,
                        save_weights_only=True,
                        verbose=1),

        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
    ]
# --------------------------------------------------------------------------


    print("Callbacks:", callbacks)
    for i,c in enumerate(callbacks):
        print(i, type(c), getattr(c, "get_config", None))


    model.fit(train_ds,
              validation_data=val_ds,
              epochs=args.epochs,
              callbacks=callbacks)

    # After training, save the full model in native Keras format (.keras)
    try:
        model.save('./checkpoints/gt_net_final.keras')   # native format -> safer and recommended
        print("Saved full model to ./checkpoints/gt_net_final.keras")
    except Exception as e:
        print("Warning: could not save full model (.keras):", e)


if __name__ == "__main__":
    main()