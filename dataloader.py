import tensorflow as tf
IMG_SIZE = (224,224)
def get_dataset(data_dir, batch_size=32, shuffle=True, augment=True):
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        labels='inferred',
        label_mode='int',
        shuffle=shuffle,

        image_size=IMG_SIZE,
        batch_size=batch_size
    )
    AUTOTUNE = tf.data.AUTOTUNE
    def preprocess(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        return image, label
    ds = ds.map(preprocess, num_parallel_calls=AUTOTUNE)
    if augment:
        data_augmentation = tf.keras.Sequential([             
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.08),
            tf.keras.layers.RandomTranslation(0.05,0.05),
        ])
        def augment_fn(img, lbl):
            return data_augmentation(img, training=True), lbl
        ds = ds.map(augment_fn, num_parallel_calls=AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(1024)
    ds = ds.prefetch(AUTOTUNE)                                   # fast feeding to GPU
    return ds

