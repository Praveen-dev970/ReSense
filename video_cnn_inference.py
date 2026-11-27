
import math
from functools import lru_cache
from pathlib import Path

import tensorflow as tf


MODEL_PATH = Path(
    r"C:\ReSenseResearchData\video\processed\cnn_regularized"
    r"\video_frame_cnn_regularized.h5"
)

IMAGE_SIZE = (64, 64)
SELECTED_FRAME_INDEX = 4
CLASS_THRESHOLD = 0.5


@lru_cache(maxsize=1)
def load_video_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained video CNN model not found: {MODEL_PATH}"
        )

    return tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False,
    )


def predict_video_folder(frame_dir):
    frame_dir = Path(frame_dir)

    if not frame_dir.is_dir():
        raise FileNotFoundError(
            f"Extracted frame directory not found: {frame_dir}"
        )

    frames = sorted(frame_dir.glob("frame_*.jpg"))

    if len(frames) <= SELECTED_FRAME_INDEX:
        raise ValueError(
            "At least five readable frames are required "
            "for the trained video CNN."
        )

    selected_frame = frames[SELECTED_FRAME_INDEX]

    # Match the training pipeline:
    # JPEG decoding -> RGB -> resize 64x64 -> divide by 255.
    image_data = tf.io.read_file(str(selected_frame))
    image = tf.io.decode_jpeg(image_data, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0

    batch = tf.expand_dims(image, axis=0)

    model = load_video_model()
    scores = model.predict(batch, verbose=0).reshape(-1)

    if len(scores) != 1:
        raise ValueError("The video CNN returned an unexpected output.")

    score = float(scores[0])

    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError("The video CNN returned an invalid model score.")

    predicted_class = "ASD" if score >= CLASS_THRESHOLD else "TD"

    return {
        "predicted_class": predicted_class,
        "model_score": score,
        "selected_frame": selected_frame.name,
        "frames_processed": len(frames),
    }
