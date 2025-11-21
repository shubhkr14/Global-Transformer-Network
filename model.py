import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.applications import DenseNet121, ResNet50, EfficientNetB0
# ------------------------
# Your existing GeM pooling (kept as-is)
# ------------------------
class GeM(layers.Layer):
    def __init__(self, p=3.0, trainable=True, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.init_p = p
        self.trainable = trainable
        self.eps = eps

    def build(self, input_shape):
        p_init = tf.constant_initializer(self.init_p)
        self.p = self.add_weight(             
            name='p', shape=(1,), initializer=p_init, trainable=self.trainable, dtype=self.dtype
        )
        super().build(input_shape)

    def call(self, x):
        p = tf.maximum(self.p, self.eps)
        x = tf.clip_by_value(x, 1e-6, 1e6)
        x = tf.pow(x, p)
        x = tf.reduce_mean(x, axis=[1,2], keepdims=False)
        return tf.pow(x, 1.0 / p)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"p": float(self.init_p), "trainable": self.trainable})
        return cfg
# ------------------------
# New: Global Spatial Block as a tf.keras Layer (multi-head)
# ------------------------
class GSB(layers.Layer):
    """
    Global Spatial Block (GSB) implementing:
      - per-head Q/K/V projections,
      - spatial attention (softmax over HW),
      - channel branch (GAP -> Dense(sigmoid)),
      - combine via trainable scalars w1,w2,
      - concat heads -> 1x1 conv fuse,
      - residual + LayerNorm.
    """
    def __init__(self, num_heads=4, reduction=8, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = int(num_heads)
        self.reduction = int(reduction)

    def build(self, input_shape):
        # input_shape: (B, H, W, C)
        _, H, W, C = input_shape
        self.H_static = int(H) if H is not None else None
        self.W_static = int(W) if W is not None else None
        self.C = int(C)
        self.Cp = max(1, self.C // self.reduction)

        # per-head q,k,v convs (1x1)
        self.q_convs = [layers.Conv2D(self.Cp, 1, use_bias=True, name=f"gsb_q_conv_h{i}") for i in range(self.num_heads)]
        self.k_convs = [layers.Conv2D(self.Cp, 1, use_bias=True, name=f"gsb_k_conv_h{i}") for i in range(self.num_heads)]
        self.v_convs = [layers.Conv2D(self.Cp, 1, use_bias=True, name=f"gsb_v_conv_h{i}") for i in range(self.num_heads)]

        # channel projection (shared)
        self.channel_dense = layers.Dense(self.C, activation='sigmoid', name="gsb_channel_dense")

        # fusion conv to reduce channels after concat
        self.fuse_conv = layers.Conv2D(self.C, 1, activation='relu', padding='same', name="gsb_fuse_conv")

        # trainable scalars
        self.w1 = self.add_weight(name="gsb_w1", shape=(), initializer="ones", trainable=True, dtype=self.dtype)
        self.w2 = self.add_weight(name="gsb_w2", shape=(), initializer="ones", trainable=True, dtype=self.dtype)

        # LayerNorm
        self.ln = layers.LayerNormalization(epsilon=1e-6, name="gsb_layernorm")

        super().build(input_shape)

    def call(self, F):
        # dynamic shapes
        B = tf.shape(F)[0]
        H = tf.shape(F)[1]
        W = tf.shape(F)[2]
        HW = H * W

        head_outputs = []
        for i in range(self.num_heads):
            # Q: conv -> GAP -> (B, Cp)
            q = self.q_convs[i](F)                      # (B, H, W, Cp)
            q_gap = tf.reduce_mean(q, axis=[1,2])      # (B, Cp)
            q_exp = tf.expand_dims(q_gap, 1)           # (B,1,Cp)

            # K: conv -> reshape (B, HW, Cp)
            k = self.k_convs[i](F)
            k_resh = tf.reshape(k, (B, HW, self.Cp))

            # scores = q @ k^T  -> (B,1,HW)
            scores = tf.matmul(q_exp, k_resh, transpose_b=True)
            # scale
            scores = scores / tf.math.sqrt(tf.cast(self.Cp, tf.float32))
            attn = tf.nn.softmax(scores, axis=-1)      # (B,1,HW)

            # spatial branch: reshape to (B,H,W,1) and multiply with F (broadcast)
            attn_resh = tf.reshape(attn, (B, H, W, 1))
            F_sp = attn_resh * F                        # (B,H,W,C)

            # channel branch: v conv -> GAP -> Dense(sigmoid) -> (B,1,1,C)
            v = self.v_convs[i](F)                      # (B,H,W,Cp)
            v_gap = tf.reduce_mean(v, axis=[1,2])       # (B,Cp)
            channel_map = self.channel_dense(v_gap)     # (B,C) sigmoid
            channel_map = tf.reshape(channel_map, (B, 1, 1, self.C))
            F_ch = channel_map * F                      # (B,H,W,C)

            # combine with trainable scalars
            F_gs = self.w1 * F_sp + self.w2 * F_ch      # (B,H,W,C)
            head_outputs.append(F_gs)

        # concat heads and fuse
        concat = tf.concat(head_outputs, axis=-1)       # (B,H,W, C * heads)
        fused = self.fuse_conv(concat)                  # (B,H,W,C)

        # residual + layernorm
        out = fused + F
        out = self.ln(out)
        return out

# ------------------------
# Build function: uses GSB as GTM
# ------------------------
def build_gt_net(input_shape=(224,224,3), backbone_name='densenet121', heads=4, num_classes=4):
    inp = layers.Input(shape=input_shape)
    # choose backbone
    bname = backbone_name.lower()
    if bname == 'densenet121':
        backbone = DenseNet121(include_top=False, weights='imagenet', input_tensor=inp)
        feat = backbone.output
    elif bname == 'resnet50':
        backbone = ResNet50(include_top=False, weights='imagenet', input_tensor=inp)
        feat = backbone.output
    elif bname == 'efficientnetb0':
        backbone = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=inp)
        feat = backbone.output
    else:
        raise ValueError('Unknown backbone')

    # apply Global Spatial Block (multi-head) as GTM
    gsb_layer = GSB(num_heads=heads, reduction=8, name='global_spatial_block')
    gtm_out = gsb_layer(feat)  # same shape as feat

    # classifier: GeM pooling + Dropout + Dense softmax
    gem = GeM()(gtm_out)
    drop = layers.Dropout(0.4)(gem)
    out = layers.Dense(num_classes, activation='softmax', name='classifier')(drop)

    model = models.Model(inputs=inp, outputs=out, name='GT-Net')
    return model

# quick smoke test when run directly
if __name__ == "__main__":
    m = build_gt_net()
    m.summary()