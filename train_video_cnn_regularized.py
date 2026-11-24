
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

# ==================================================
# 1. CONFIGURATION
# ==================================================

SEED = 42
IMAGE_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 15
VALIDATION_PARTICIPANTS_PER_CLASS = 8

DATA_ROOT = (
    r"C:\ReSenseResearchData\video\processed\sampled_frames"
)

MANIFEST_PATH = os.path.join(
    DATA_ROOT, "frame_manifest.csv"
)

OUTPUT_DIR = (
    r"C:\ReSenseResearchData\video\processed\cnn_regularized"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(
    OUTPUT_DIR, "video_frame_cnn_regularized.h5"
)

# ==================================================
# 2. REPRODUCIBILITY
# ==================================================

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TensorFlow:", tf.__version__)
print("GPU devices:", tf.config.list_physical_devices("GPU"))

# ==================================================
# 3. LOAD MANIFEST
# ==================================================

print("\nLoading frame manifest...")

df = pd.read_csv(MANIFEST_PATH)

df["Participant"] = df["Participant"].astype(str)
df["Segment"] = df["Segment"].astype(str)
df["FrameIndex"] = pd.to_numeric(df["FrameIndex"])

# Composite participant identity.
df["ParticipantKey"] = (
    df["Class"].astype(str) + "_" +
    df["Participant"]
)

# One middle sampled frame per video.
df = df.sort_values("FrameIndex")

selected = (
    df.groupby(
        ["Split", "Class", "Participant", "Segment"],
        group_keys=False
    )
    .nth(4)
    .reset_index()
)

selected["Label"] = (
    selected["Class"] == "ASD"
).astype(np.int32)

print("Selected frames:", len(selected))
print("Unique video segments:", selected.groupby(
    ["Split", "Class", "Participant", "Segment"]
).ngroups)

# ==================================================
# 4. TRAIN / VALIDATION / TEST SPLIT
# ==================================================

train_all = selected[
    selected["Split"] == "training_set"
].copy()

test_df = selected[
    selected["Split"] == "testing_set"
].copy()

rng = np.random.RandomState(SEED)

train_parts = []
validation_parts = []

# Split participants separately within each class.
for label in ["ASD", "TD"]:

    class_data = train_all[
        train_all["Class"] == label
    ].copy()

    participants = sorted(
        class_data["ParticipantKey"].unique()
    )

    if len(participants) != 40:
        raise ValueError(
            f"Expected 40 {label} training participants, "
            f"found {len(participants)}"
        )

    shuffled = rng.permutation(participants)

    validation_ids = set(
        shuffled[:VALIDATION_PARTICIPANTS_PER_CLASS]
    )

    train_ids = set(shuffled) - validation_ids

    train_parts.append(
        class_data[
            class_data["ParticipantKey"].isin(train_ids)
        ]
    )

    validation_parts.append(
        class_data[
            class_data["ParticipantKey"].isin(validation_ids)
        ]
    )

train_df = pd.concat(train_parts, ignore_index=True)
validation_df = pd.concat(
    validation_parts, ignore_index=True
)

# Verify participant independence.
train_keys = set(train_df["ParticipantKey"])
validation_keys = set(validation_df["ParticipantKey"])
test_keys = set(test_df["ParticipantKey"])

assert train_keys.isdisjoint(validation_keys)
assert train_keys.isdisjoint(test_keys)
assert validation_keys.isdisjoint(test_keys)

assert len(train_df) == 7744
assert len(validation_df) == 1936
assert len(test_df) == 9680

print("\nDATASET SPLIT")
print("Training frames:", len(train_df))
print("Validation frames:", len(validation_df))
print("Testing frames:", len(test_df))

print("Training participants:", len(train_keys))
print("Validation participants:", len(validation_keys))
print("Testing participants:", len(test_keys))

# Save the exact split for reproducibility.
split_manifest = pd.concat([
    train_df.assign(Usage="training"),
    validation_df.assign(Usage="validation"),
    test_df.assign(Usage="testing")
], ignore_index=True)

split_manifest.to_csv(
    os.path.join(OUTPUT_DIR, "selected_frame_manifest.csv"),
    index=False
)

# ==================================================
# 5. TENSORFLOW DATA PIPELINE
# ==================================================

def create_dataset(data, training=False):

    paths = [
        os.path.join(
            DATA_ROOT,
            path.replace("/", os.sep)
        )
        for path in data["ImagePath"].tolist()
    ]

    labels = data["Label"].to_numpy(dtype=np.float32)

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    def load_image(path, label):

        image_data = tf.io.read_file(path)

        image = tf.io.decode_jpeg(
            image_data,
            channels=3
        )

        image = tf.image.resize(
            image,
            IMAGE_SIZE
        )

        image = tf.cast(image, tf.float32) / 255.0

        return image, label

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(data),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


print("\nPreparing TensorFlow datasets...")

train_ds = create_dataset(train_df, training=True)
validation_ds = create_dataset(validation_df)
test_ds = create_dataset(test_df)

# ==================================================
# 6. BUILD LIGHTWEIGHT CNN
# ==================================================

print("\nBuilding CNN...")

regularizer = tf.keras.regularizers.l2(1e-4)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(64, 64, 3)),

    tf.keras.layers.Conv2D(
        16, (3, 3), activation="relu",
        kernel_regularizer=regularizer
    ),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(
        32, (3, 3), activation="relu",
        kernel_regularizer=regularizer
    ),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(
        64, (3, 3), activation="relu",
        kernel_regularizer=regularizer
    ),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(
        32, activation="relu",
        kernel_regularizer=regularizer
    ),

    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==================================================
# 7. TRAIN CNN
# ==================================================

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )
]

print("\nStarting CNN training...")

history = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# ==================================================
# 8. SAVE MODEL AND TRAINING HISTORY
# ==================================================

model.save(MODEL_PATH)

pd.DataFrame(
    history.history
).to_csv(
    os.path.join(OUTPUT_DIR, "training_history.csv"),
    index=False
)

print("\nModel saved:", MODEL_PATH)

# ==================================================
# 9. PREDICT TEST VIDEO SEGMENTS
# ==================================================

print("\nPredicting test segments...")

segment_probabilities = model.predict(
    test_ds,
    verbose=1
).flatten()

test_results = test_df.copy()

test_results["Probability_ASD"] = segment_probabilities

test_results.to_csv(
    os.path.join(OUTPUT_DIR, "segment_predictions.csv"),
    index=False
)

# ==================================================
# 10. PARTICIPANT-LEVEL AGGREGATION
# ==================================================

print("\nAggregating predictions by participant...")

participant_results = (
    test_results.groupby(
        ["Class", "ParticipantKey"]
    )["Probability_ASD"]
    .mean()
    .reset_index()
)

participant_results["Actual"] = (
    participant_results["Class"] == "ASD"
).astype(int)

participant_results["Predicted"] = (
    participant_results["Probability_ASD"] >= 0.5
).astype(int)

y_true = participant_results["Actual"]
y_pred = participant_results["Predicted"]
y_prob = participant_results["Probability_ASD"]

# ==================================================
# 11. EVALUATION
# ==================================================

print("\n" + "=" * 50)
print("PARTICIPANT-LEVEL TEST RESULTS")
print("=" * 50)

print("Participants:", len(participant_results))

print(
    "Accuracy:",
    round(accuracy_score(y_true, y_pred), 4)
)

print(
    "Balanced accuracy:",
    round(balanced_accuracy_score(y_true, y_pred), 4)
)

print(
    "ROC-AUC:",
    round(roc_auc_score(y_true, y_prob), 4)
)

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_true, y_pred))

print("\nCLASSIFICATION REPORT")

print(classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=["TD", "ASD"],
    zero_division=0
))

participant_results.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "participant_predictions.csv"
    ),
    index=False
)

print("\n" + "=" * 50)
print("TRAINING EXPERIMENT COMPLETED")
print("=" * 50)

print("Model:", MODEL_PATH)
print("Results:", OUTPUT_DIR)
