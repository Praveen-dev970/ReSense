
import cv2
import csv
import numpy as np
import pandas as pd

from pathlib import Path

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

ROOT = Path(
    r"C:\ReSenseResearchData\video\raw\OpenPose_ADOS"
    r"\extracted\autism_data_anonymized"
)

SOURCE_CSV = Path(
    r"C:\ReSenseResearchData\metadata"
    r"\Phase2C7_5_color_channel_analysis"
    r"\color_component_coordinates.csv"
)

OUTPUT_DIR = Path(
    r"C:\ReSenseResearchData\metadata"
    r"\Phase2C7_6_noise_reduction_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLDS = [2, 5, 10, 20]

VIDEOS = [
    ("training_set", "ASD", "Subj_10_part_114.mp4"),
    ("training_set", "TD", "Subj_11_part_101.mp4"),
    ("testing_set", "ASD", "Subj_101_part_38.mp4"),
    ("testing_set", "TD", "Subj_121_part_116.mp4"),
]

COLOR_RANGES = {
    "red_low": (0, 10),
    "orange": (11, 24),
    "yellow": (25, 35),
    "green": (36, 85),
    "cyan": (86, 100),
    "blue": (101, 130),
    "magenta": (131, 169),
    "red_high": (170, 179),
}


# --------------------------------------------------
# LOAD EXISTING COMPONENT DATA
# --------------------------------------------------

if not SOURCE_CSV.exists():
    raise FileNotFoundError(
        f"Source CSV not found: {SOURCE_CSV}"
    )

df = pd.read_csv(SOURCE_CSV)

required_columns = {
    "split", "group", "video", "color", "area"
}

if not required_columns.issubset(df.columns):
    raise ValueError("Source CSV is missing required columns.")

print("Loaded component records:", len(df))


# --------------------------------------------------
# THRESHOLD COMPARISON
# --------------------------------------------------

summary = []
color_summary = []

for split, group, filename in VIDEOS:

    video_df = df[
        (df["split"] == split)
        & (df["group"] == group)
        & (df["video"] == filename)
    ]

    if video_df.empty:
        print("WARNING: No CSV records for", filename)
        continue

    total = len(video_df)

    print(f"\n{filename}")
    print("Original detected components:", total)

    for threshold in THRESHOLDS:

        retained = video_df[
            video_df["area"] >= threshold
        ]

        removed = total - len(retained)

        removed_percent = (
            removed / total * 100 if total else 0
        )

        summary.append({
            "split": split,
            "group": group,
            "video": filename,
            "minimum_area": threshold,
            "original_components": total,
            "retained_components": len(retained),
            "removed_components": removed,
            "removed_percent": round(removed_percent, 2),
        })

        print(
            f"Threshold {threshold}: "
            f"{len(retained)} retained, "
            f"{removed_percent:.2f}% removed"
        )

        for color_name in COLOR_RANGES:

            color_df = video_df[
                video_df["color"] == color_name
            ]

            color_retained = color_df[
                color_df["area"] >= threshold
            ]

            color_summary.append({
                "split": split,
                "group": group,
                "video": filename,
                "color": color_name,
                "minimum_area": threshold,
                "original_components": len(color_df),
                "retained_components": len(color_retained),
            })


# --------------------------------------------------
# SAVE COMPARISON REPORTS
# --------------------------------------------------

summary_path = OUTPUT_DIR / "threshold_comparison.csv"

pd.DataFrame(summary).to_csv(
    summary_path, index=False
)

color_path = OUTPUT_DIR / "color_threshold_comparison.csv"

pd.DataFrame(color_summary).to_csv(
    color_path, index=False
)


# --------------------------------------------------
# GENERATE VISUAL COMPARISONS
# --------------------------------------------------

for split, group, filename in VIDEOS:

    video_path = ROOT / split / group / filename

    if not video_path.exists():
        print("Video not found:", video_path)
        continue

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("Cannot open:", filename)
        cap.release()
        continue

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    middle_index = total_frames // 2

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        middle_index
    )

    success, frame = cap.read()
    cap.release()

    if not success:
        print("Could not read middle frame:", filename)
        continue

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    video_output = OUTPUT_DIR / (
        f"{split}_{group}_{Path(filename).stem}"
    )

    video_output.mkdir(parents=True, exist_ok=True)

    for threshold in THRESHOLDS:

        annotated = frame.copy()
        total_regions = 0

        for color_name, hue_range in COLOR_RANGES.items():

            lower_hue, upper_hue = hue_range

            lower = np.array([lower_hue, 60, 40])
            upper = np.array([upper_hue, 255, 255])

            mask = cv2.inRange(hsv, lower, upper)

            count, labels, stats, centroids = (
                cv2.connectedComponentsWithStats(
                    mask,
                    connectivity=8
                )
            )

            for component_id in range(1, count):

                area = int(
                    stats[component_id, cv2.CC_STAT_AREA]
                )

                if area < threshold:
                    continue

                x = int(
                    stats[component_id, cv2.CC_STAT_LEFT]
                )
                y = int(
                    stats[component_id, cv2.CC_STAT_TOP]
                )
                w = int(
                    stats[component_id, cv2.CC_STAT_WIDTH]
                )
                h = int(
                    stats[component_id, cv2.CC_STAT_HEIGHT]
                )

                cx, cy = centroids[component_id]

                cv2.rectangle(
                    annotated,
                    (x, y),
                    (x + w, y + h),
                    (255, 255, 255),
                    1
                )

                cv2.circle(
                    annotated,
                    (int(cx), int(cy)),
                    2,
                    (0, 0, 255),
                    -1
                )

                total_regions += 1

        cv2.putText(
            annotated,
            f"Area threshold: {threshold} | Regions: {total_regions}",
            (5, 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 255),
            1
        )

        image_path = video_output / (
            f"middle_frame_threshold_{threshold}.jpg"
        )

        cv2.imwrite(str(image_path), annotated)


# --------------------------------------------------
# COMPLETION
# --------------------------------------------------

print("\n" + "=" * 55)
print("NOISE REDUCTION INVESTIGATION COMPLETED")
print("=" * 55)

print("Threshold report:", summary_path)
print("Colour report:", color_path)
print("Visual comparisons:", OUTPUT_DIR)

print("\nOriginal videos and source CSV were not modified.")
