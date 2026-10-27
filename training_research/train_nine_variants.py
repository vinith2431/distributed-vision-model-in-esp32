"""
Nine-Model Face Mask Training: Width Variants + Architecture Variants, Interleaved
------------------------------------------------------------------------------
Merges the "small/wide" width-scaling comparison with several architecture
comparisons into a single run. All 9 models train together, alternating
epoch-by-epoch:

  Epoch 1 - small, Epoch 1 - wide, Epoch 1 - baseline, Epoch 1 - deeper,
  Epoch 1 - residual, Epoch 1 - se, Epoch 1 - mbconv, Epoch 1 - cbam,
  Epoch 1 - dense
  Epoch 2 - small, Epoch 2 - wide, ...
  ...

Variants:
  small     - 3 depthwise-separable blocks, narrow channels   (~30k params)
  wide      - same topology as small, wide channels           (~139k params)
  baseline  - 3 depthwise-separable blocks, mid-width channels
  deeper    - 4 narrower depthwise-separable blocks instead of 3
  residual  - baseline topology + projected skip connections around each block
  se        - baseline topology + Squeeze-and-Excitation channel attention
  mbconv    - MobileNetV2-style inverted-residual (expand -> depthwise ->
              project) blocks with residual adds where shapes match
  cbam      - baseline topology + CBAM (channel attention AND spatial
              attention) after each block
  dense     - DenseNet-style blocks: each conv's output is concatenated
              onto the running feature map, with 1x1 transition/downsample
              layers between dense blocks

Manual early stopping (patience-based, monitors val_loss, restores best
weights) since Keras' EarlyStopping callback doesn't span multiple fit()
calls.

YOLO-style saving: auto-incremented run dir, results.csv appended live
every epoch, weights/last.keras + weights/best.keras per model, graphs
generated at the end.

  runs/train_nine_variants/exp/    <- exp2, exp3, ... on next runs
    small/
      weights/last.keras
      weights/best.keras
      results.csv
      results.png
      confusion_matrix.png
      classification_report.txt
    wide/      ... same layout ...
    baseline/  ... same layout ...
    deeper/    ... same layout ...
    residual/  ... same layout ...
    se/        ... same layout ...
    mbconv/    ... same layout ...
    cbam/      ... same layout ...
    dense/     ... same layout ...
    combined_val_accuracy.png
    combined_val_loss.png
    comparison_summary.txt

Install first (inside WSL / Linux):
  pip install tensorflow matplotlib scikit-learn --break-system-packages
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # suppress TF/XLA/cuDNN log spam
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # kill the oneDNN notice line
import csv
import copy
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
import logging
logging.getLogger("absl").setLevel(logging.ERROR)
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# ---------- Config ----------
# Copied into WSL's native filesystem (not /mnt/c/...) to avoid the 9p
# bridge slowness/RAM pressure from reading thousands of files off Windows.
DATA_DIR = os.path.expanduser("~/data/final")
IMG_SIZE = 96
BATCH_SIZE = 128
MAX_EPOCHS = 150
PATIENCE = 5
CLASS_NAMES = ["withmask", "withoutmask"]
VARIANTS = ["small", "wide", "baseline", "deeper", "residual", "se", "mbconv", "cbam", "dense"]
RUNS_ROOT = "runs/train_nine_variants"
CSV_FIELDS = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "best"]
# -----------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def configure_gpu():
    """Detect GPU(s) and enable memory growth so TF doesn't pre-allocate all VRAM."""
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("No GPU detected — training will run on CPU.")
        return
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            # Must be set before GPUs are initialized; safe to ignore if already set.
            print(f"Could not set memory growth for {gpu}: {e}")
    print(f"Detected {len(gpus)} GPU(s): {[g.name for g in gpus]}")


def next_run_dir():
    """runs/train_six_variants/exp, exp2, exp3, ... (first free name wins)."""
    root = os.path.join(SCRIPT_DIR, RUNS_ROOT)
    os.makedirs(root, exist_ok=True)
    n = 1
    while True:
        name = "exp" if n == 1 else f"exp{n}"
        path = os.path.join(root, name)
        if not os.path.exists(path):
            os.makedirs(path)
            return path
        n += 1


def load_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, "train"),
        image_size=(IMG_SIZE, IMG_SIZE), color_mode="rgb", batch_size=BATCH_SIZE,
        label_mode="int", class_names=CLASS_NAMES, shuffle=True,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, "val"),
        image_size=(IMG_SIZE, IMG_SIZE), color_mode="rgb", batch_size=BATCH_SIZE,
        label_mode="int", class_names=CLASS_NAMES, shuffle=False,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, "test"),
        image_size=(IMG_SIZE, IMG_SIZE), color_mode="rgb", batch_size=BATCH_SIZE,
        label_mode="int", class_names=CLASS_NAMES, shuffle=False,
    )

    # tf.data's .cache(filename) does NOT create the directory for you — it
    # throws a NotFoundError on the first write if the parent dir is missing.
    # That was silently killing the whole run at epoch 1. Use absolute paths
    # under SCRIPT_DIR too, so this doesn't depend on the CWD you launch from.
    cache_dir = os.path.join(SCRIPT_DIR, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    normalize = layers.Rescaling(1.0 / 255)
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache(os.path.join(cache_dir, "train")).map(lambda x, y: (normalize(x), y)).prefetch(2)
    val_ds = val_ds.cache(os.path.join(cache_dir, "val")).map(lambda x, y: (normalize(x), y)).prefetch(2)
    test_ds = test_ds.cache(os.path.join(cache_dir, "test")).map(lambda x, y: (normalize(x), y)).prefetch(2)
    return train_ds, val_ds, test_ds


# ---------- Shared building blocks ----------
#
# NOTE: BatchNormalization is inserted between every conv and its activation.
# Without it, wide layers (large channel jumps, e.g. 32->256 in a single 1x1
# conv) can produce activations large enough to saturate/kill ReLUs right at
# initialization, causing the whole network to collapse to predicting a
# single class (loss stuck flat, accuracy stuck at the majority-class rate).
# BN keeps activation scale in a healthy range regardless of layer width.

def conv_bn_act(x, filters, kernel, stride, prefix, activation="relu"):
    x = layers.Conv2D(filters, kernel, strides=stride, padding="same",
                       use_bias=False, name=f"{prefix}_conv")(x)
    x = layers.BatchNormalization(name=f"{prefix}_bn")(x)
    if activation:
        x = layers.Activation(activation, name=f"{prefix}_act")(x)
    return x


def dw_bn_act(x, stride, prefix, activation="relu"):
    x = layers.DepthwiseConv2D(3, strides=stride, padding="same",
                                use_bias=False, name=f"{prefix}_dw")(x)
    x = layers.BatchNormalization(name=f"{prefix}_dwbn")(x)
    if activation:
        x = layers.Activation(activation, name=f"{prefix}_dwact")(x)
    return x


def build_stem(inputs):
    x = conv_bn_act(inputs, 16, 3, 2, "stem_conv1")
    x = dw_bn_act(x, 1, "stem_dw1")
    x = conv_bn_act(x, 32, 1, 1, "stem_pw1")
    return x  # -> 48x48x32


def ds_block(x, out_ch, stride, prefix):
    """Depthwise-separable conv block, with BN after both the depthwise and
    pointwise convs."""
    x = dw_bn_act(x, stride, prefix)
    x = conv_bn_act(x, out_ch, 1, 1, f"{prefix}_pw")
    return x


def se_block(x, prefix, reduction=4):
    """Squeeze-and-Excitation channel attention."""
    channels = x.shape[-1]
    se = layers.GlobalAveragePooling2D(name=f"{prefix}_se_pool")(x)
    se = layers.Dense(max(channels // reduction, 4), activation="relu", name=f"{prefix}_se_fc1")(se)
    se = layers.Dense(channels, activation="sigmoid", name=f"{prefix}_se_fc2")(se)
    se = layers.Reshape((1, 1, channels), name=f"{prefix}_se_reshape")(se)
    return layers.Multiply(name=f"{prefix}_se_mul")([x, se])


def mbconv_block(x, expand_ch, out_ch, stride, prefix):
    """MobileNetV2-style inverted residual: 1x1 expand -> depthwise -> 1x1
    project (no activation on the project step). Adds a residual connection
    when stride is 1 and channel counts match."""
    in_ch = x.shape[-1]
    residual = x
    x = conv_bn_act(x, expand_ch, 1, 1, f"{prefix}_expand")
    x = dw_bn_act(x, stride, prefix)
    x = conv_bn_act(x, out_ch, 1, 1, f"{prefix}_project", activation=None)
    if stride == 1 and in_ch == out_ch:
        x = layers.Add(name=f"{prefix}_resadd")([x, residual])
    return x


def cbam_block(x, prefix, reduction=8):
    """Convolutional Block Attention Module: channel attention (shared MLP
    over avg- and max-pooled features) followed by spatial attention
    (conv over channel-pooled avg/max maps)."""
    channels = x.shape[-1]

    avg_pool = layers.GlobalAveragePooling2D(name=f"{prefix}_cbam_avgpool")(x)
    max_pool = layers.GlobalMaxPooling2D(name=f"{prefix}_cbam_maxpool")(x)
    fc1 = layers.Dense(max(channels // reduction, 4), activation="relu", name=f"{prefix}_cbam_fc1")
    fc2 = layers.Dense(channels, name=f"{prefix}_cbam_fc2")
    avg_out = fc2(fc1(avg_pool))
    max_out = fc2(fc1(max_pool))
    channel_att = layers.Add(name=f"{prefix}_cbam_chadd")([avg_out, max_out])
    channel_att = layers.Activation("sigmoid", name=f"{prefix}_cbam_chsig")(channel_att)
    channel_att = layers.Reshape((1, 1, channels), name=f"{prefix}_cbam_chreshape")(channel_att)
    x = layers.Multiply(name=f"{prefix}_cbam_chmul")([x, channel_att])

    avg_sp = layers.Lambda(lambda t: tf.reduce_mean(t, axis=-1, keepdims=True),
                            name=f"{prefix}_cbam_spavg")(x)
    max_sp = layers.Lambda(lambda t: tf.reduce_max(t, axis=-1, keepdims=True),
                            name=f"{prefix}_cbam_spmax")(x)
    concat = layers.Concatenate(axis=-1, name=f"{prefix}_cbam_spconcat")([avg_sp, max_sp])
    spatial_att = layers.Conv2D(1, 7, padding="same", activation="sigmoid",
                                 name=f"{prefix}_cbam_spconv")(concat)
    return layers.Multiply(name=f"{prefix}_cbam_spmul")([x, spatial_att])


def dense_block(x, growth_ch, num_layers, prefix):
    """DenseNet-style block: each layer's output is concatenated onto the
    running feature map so later layers see all earlier feature maps."""
    for i in range(num_layers):
        y = conv_bn_act(x, growth_ch, 3, 1, f"{prefix}_l{i}")
        x = layers.Concatenate(name=f"{prefix}_l{i}_concat")([x, y])
    return x


def build_head(x, head_ch, num_classes, prefix="head"):
    x = layers.GlobalAveragePooling2D(name=f"{prefix}_pool")(x)
    x = layers.Dense(head_ch, activation="relu", name=f"{prefix}_dense1")(x)
    return layers.Dense(num_classes, activation="softmax", name=f"{prefix}_dense2")(x)


# ---------- Architectures ----------

def build_width_variant(inputs, num_classes, mid_ch, head_ch):
    """Shared 3-block topology used by 'small' and 'wide' (stride 2,1,2)."""
    x = build_stem(inputs)
    x = ds_block(x, mid_ch[0], 2, "mid1")
    x = ds_block(x, mid_ch[1], 1, "mid2")
    x = ds_block(x, mid_ch[2], 2, "mid3")
    outputs = build_head(x, head_ch, num_classes)
    return outputs


def build_small(inputs, num_classes):
    outputs = build_width_variant(inputs, num_classes, mid_ch=(16, 16, 240), head_ch=96)
    return Model(inputs, outputs, name="face_mask_small")


def build_wide(inputs, num_classes):
    outputs = build_width_variant(inputs, num_classes, mid_ch=(32, 256, 400), head_ch=48)
    return Model(inputs, outputs, name="face_mask_wide")


def build_baseline(inputs, num_classes):
    outputs = build_width_variant(inputs, num_classes, mid_ch=(64, 96, 128), head_ch=48)
    return Model(inputs, outputs, name="face_mask_baseline")


def build_deeper(inputs, num_classes):
    """4 narrower blocks instead of 3."""
    x = build_stem(inputs)
    x = ds_block(x, 48, 2, "mid1")
    x = ds_block(x, 64, 1, "mid2")
    x = ds_block(x, 80, 1, "mid3")
    x = ds_block(x, 112, 2, "mid4")
    outputs = build_head(x, 48, num_classes)
    return Model(inputs, outputs, name="face_mask_deeper")


def build_residual(inputs, num_classes):
    """Baseline topology with projected (BN'd) skip connections around each block."""
    x = build_stem(inputs)

    shortcut = conv_bn_act(x, 64, 1, 2, "mid1_proj", activation=None)
    x = ds_block(x, 64, 2, "mid1")
    x = layers.Add(name="mid1_add")([x, shortcut])
    x = layers.Activation("relu", name="mid1_resact")(x)

    shortcut = conv_bn_act(x, 96, 1, 1, "mid2_proj", activation=None)
    x = ds_block(x, 96, 1, "mid2")
    x = layers.Add(name="mid2_add")([x, shortcut])
    x = layers.Activation("relu", name="mid2_resact")(x)

    shortcut = conv_bn_act(x, 128, 1, 2, "mid3_proj", activation=None)
    x = ds_block(x, 128, 2, "mid3")
    x = layers.Add(name="mid3_add")([x, shortcut])
    x = layers.Activation("relu", name="mid3_resact")(x)

    outputs = build_head(x, 48, num_classes)
    return Model(inputs, outputs, name="face_mask_residual")


def build_se(inputs, num_classes):
    """Baseline topology with SE attention after each block."""
    x = build_stem(inputs)
    x = ds_block(x, 64, 2, "mid1")
    x = se_block(x, "mid1")
    x = ds_block(x, 96, 1, "mid2")
    x = se_block(x, "mid2")
    x = ds_block(x, 128, 2, "mid3")
    x = se_block(x, "mid3")
    outputs = build_head(x, 48, num_classes)
    return Model(inputs, outputs, name="face_mask_se")


def build_mbconv(inputs, num_classes):
    """MobileNetV2-style inverted-residual bottleneck blocks."""
    x = build_stem(inputs)
    x = mbconv_block(x, 96, 64, 2, "mid1")     # 32->96 expand, downsample, project to 64
    x = mbconv_block(x, 192, 64, 1, "mid2")    # 64->192 expand, same-res, residual add (64==64)
    x = mbconv_block(x, 192, 96, 1, "mid3")    # 64->192 expand, project to 96, no residual (64!=96)
    x = mbconv_block(x, 288, 128, 2, "mid4")   # 96->288 expand, downsample, project to 128
    outputs = build_head(x, 48, num_classes)
    return Model(inputs, outputs, name="face_mask_mbconv")


def build_cbam(inputs, num_classes):
    """Baseline topology with CBAM (channel + spatial attention) after each block."""
    x = build_stem(inputs)
    x = ds_block(x, 64, 2, "mid1")
    x = cbam_block(x, "mid1")
    x = ds_block(x, 96, 1, "mid2")
    x = cbam_block(x, "mid2")
    x = ds_block(x, 128, 2, "mid3")
    x = cbam_block(x, "mid3")
    outputs = build_head(x, 48, num_classes)
    return Model(inputs, outputs, name="face_mask_cbam")


def build_dense(inputs, num_classes):
    """DenseNet-style dense blocks with 1x1 transition/downsample layers in between."""
    x = build_stem(inputs)                          # -> 32 channels
    x = dense_block(x, 24, 3, "dense1")              # -> 32 + 24*3 = 104 channels
    x = conv_bn_act(x, 64, 1, 2, "trans1")           # transition: compress + downsample
    x = dense_block(x, 32, 3, "dense2")              # -> 64 + 32*3 = 160 channels
    x = conv_bn_act(x, 96, 1, 2, "trans2")           # transition: compress + downsample
    outputs = build_head(x, 48, num_classes)
    return Model(inputs, outputs, name="face_mask_dense")


BUILDERS = {
    "small": build_small,
    "wide": build_wide,
    "baseline": build_baseline,
    "deeper": build_deeper,
    "residual": build_residual,
    "se": build_se,
    "mbconv": build_mbconv,
    "cbam": build_cbam,
    "dense": build_dense,
}


def build_model(variant, num_classes=2):
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    return BUILDERS[variant](inputs, num_classes)


# ---------- Training state / YOLO-style saving ----------

class TrainerState:
    def __init__(self, variant, model, run_dir):
        self.variant = variant
        self.model = model
        self.dir = os.path.join(run_dir, variant)
        self.weights_dir = os.path.join(self.dir, "weights")
        os.makedirs(self.weights_dir, exist_ok=True)

        self.csv_path = os.path.join(self.dir, "results.csv")
        with open(self.csv_path, "w", newline="") as f:
            csv.writer(f).writerow(CSV_FIELDS)

        self.history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
        self.best_val_loss = np.inf
        self.best_weights = None
        self.patience_counter = 0
        self.stopped = False
        self.stopped_epoch = None
        self.evaluated = False

    def log_epoch(self, epoch, logs, is_best):
        for k in self.history:
            self.history[k].append(logs[k])
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, logs["loss"], logs["accuracy"], logs["val_loss"], logs["val_accuracy"],
                int(is_best),
            ])
        self.model.save(os.path.join(self.weights_dir, "last.keras"))
        if is_best:
            self.model.save(os.path.join(self.weights_dir, "best.keras"))


def finalize_variant(state, test_ds, test_accs):
    """Restore best weights, plot curves, evaluate on test set, save report.
    Called as soon as a model stops (early-stopped or MAX_EPOCHS reached),
    and again from the finally-block for anything still unfinished if the
    run gets interrupted — so a crash/OOM on one variant never costs the
    results already earned by the others."""
    if state.evaluated:
        return
    if not state.history["loss"]:
        # never completed a single epoch — nothing to evaluate
        state.evaluated = True
        return

    if state.best_weights is not None:
        state.model.set_weights(state.best_weights)

    plot_results(state.history, state.variant, os.path.join(state.dir, "results.png"))
    test_accs[state.variant] = evaluate_on_test(state.model, test_ds, CLASS_NAMES, state.variant, state.dir)
    state.evaluated = True


def run_interleaved_training(train_ds, val_ds, test_ds, run_dir, test_accs):
    models = {v: build_model(v, num_classes=len(CLASS_NAMES)) for v in VARIANTS}
    for m in models.values():
        # clipnorm caps gradient magnitude per-step; cheap insurance against
        # occasional instability in the wider/deeper variants.
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)
        m.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        # m.summary() skipped for quieter output

    states = {v: TrainerState(v, models[v], run_dir) for v in VARIANTS}

    for epoch in range(1, MAX_EPOCHS + 1):
        if all(s.stopped for s in states.values()):
            print(f"\nAll models stopped early before epoch {epoch}.")
            break

        for variant in VARIANTS:
            state = states[variant]
            if state.stopped:
                continue

            h = state.model.fit(train_ds, validation_data=val_ds, epochs=1, verbose=0)
            logs = {k: h.history[k][0] for k in state.history}
            print(f"[{variant.upper():9s}] epoch {epoch:3d}  "
                  f"acc: {logs['accuracy']:.4f}  val_acc: {logs['val_accuracy']:.4f}")

            is_best = logs["val_loss"] < state.best_val_loss - 1e-5
            if is_best:
                state.best_val_loss = logs["val_loss"]
                state.best_weights = copy.deepcopy(state.model.get_weights())
                state.patience_counter = 0
            else:
                state.patience_counter += 1

            state.log_epoch(epoch, logs, is_best)

            if state.patience_counter >= PATIENCE:
                print(f"[{variant}] Early stopping at epoch {epoch} "
                      f"(best val_loss={state.best_val_loss:.4f})")
                state.stopped = True
                state.stopped_epoch = epoch
                # Finalize immediately — don't wait for the other 8 models.
                finalize_variant(state, test_ds, test_accs)

    # Anything that reached MAX_EPOCHS without early-stopping still needs
    # its results saved.
    for state in states.values():
        finalize_variant(state, test_ds, test_accs)

    return states


def plot_results(history, variant, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history["loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_title(f"Loss over Epochs ({variant})")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history["accuracy"], label="Train Accuracy")
    axes[1].plot(history["val_accuracy"], label="Val Accuracy")
    axes[1].set_title(f"Accuracy over Epochs ({variant})")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_confusion_matrix(cm, class_names, variant, out_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix ({variant}, Test Set)")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def evaluate_on_test(model, test_ds, class_names, variant, out_dir):
    y_true, y_pred = [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names)
    test_acc = float(np.mean(y_true == y_pred))


    report_path = os.path.join(out_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    plot_confusion_matrix(cm, class_names, variant, os.path.join(out_dir, "confusion_matrix.png"))
    return test_acc


def combine_results(states, test_accs, run_dir):
    fig, ax = plt.subplots(figsize=(9, 5))
    for variant, state in states.items():
        ax.plot(state.history["val_accuracy"], label=f"{variant} (val_acc)")
    ax.set_title("Combined Validation Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "combined_val_accuracy.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(9, 5))
    for variant, state in states.items():
        ax.plot(state.history["val_loss"], label=f"{variant} (val_loss)")
    ax.set_title("Combined Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "combined_val_loss.png"), dpi=150)
    plt.close()

    lines = ["Nine-Model Comparison Summary", "=" * 32]
    for variant, state in states.items():
        n_params = state.model.count_params()
        n_epochs = len(state.history["loss"])
        lines.append(f"\n{variant.upper()}")
        lines.append(f"  Parameters: {n_params:,}")
        lines.append(f"  Epochs trained: {n_epochs}"
                      + (f" (early-stopped at {state.stopped_epoch})" if state.stopped_epoch else ""))
        lines.append(f"  Best val_loss: {state.best_val_loss:.4f}")
        lines.append(f"  Test accuracy: {test_accs[variant]:.4f}")

    summary_path = os.path.join(run_dir, "comparison_summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


def main():
    configure_gpu()

    run_dir = next_run_dir()

    train_ds, val_ds, test_ds = load_datasets()

    test_accs = {}
    states = {}
    try:
        print(f"\nStarting interleaved training across {len(VARIANTS)} model variants...")
        states = run_interleaved_training(train_ds, val_ds, test_ds, run_dir, test_accs)
    finally:
        # Whatever crashes or gets interrupted (OOM, Ctrl+C, a variant going
        # NaN), finalize every model that already has training history but
        # wasn't evaluated yet, then write the combined comparison from
        # whatever's actually available — instead of losing everything.
        for state in states.values():
            finalize_variant(state, test_ds, test_accs)
        evaluated_states = {v: s for v, s in states.items() if s.evaluated and s.history["loss"]}
        if evaluated_states:
            combine_results(evaluated_states, test_accs, run_dir)
        else:
            print("\nNo model completed even one epoch — nothing to combine.")

    print(f"\nAll outputs saved under: {run_dir}")


if __name__ == "__main__":
    main()