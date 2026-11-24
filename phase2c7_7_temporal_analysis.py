
import cv2
import csv
import numpy as np

from pathlib import Path

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

ROOT = Path(
    r"C:\ReSenseResearchData\video\raw\OpenPose_ADOS"
    r"\extracted\autism_data_anonymized"
)

OUTPUT_DIR = Path(
    r"C:\ReSenseResearchData\metadata"
    r"\Phase2C7_7_temporal_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

LOWER_SV = np.array([60, 40])
UPPER_SV = np.array([255, 255])


# --------------------------------------------------
# OUTPUT RECORDS
# --------------------------------------------------

frame_results = []
video_summaries = []


# --------------------------------------------------
# PROCESS EACH VIDEO
# --------------------------------------------------

for split, group, filename in VIDEOS:

    video_path = ROOT / split / group / filename

    print(f"\nAnalyzing: {filename} | {split} | {group}")

    if not video_path.exists():
        print("ERROR: Video not found")
        continue

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("ERROR: Cannot open video")
        cap.release()
        continue

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    diagonal = float(np.hypot(width, height))

    previous_masks = {}
    previous_centroids = {}

    displacement_values = {
        color: [] for color in COLOR_RANGES
    }

    overlap_values = {
        color: [] for color in COLOR_RANGES
    }

    frame_index = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        record = {
            "split": split,
            "group": group,
            "video": filename,
            "frame_index": frame_index,
            "timestamp_seconds": round(
                frame_index / fps, 4
            ) if fps > 0 else None,
        }

        for color_name, hue_range in COLOR_RANGES.items():

            lower_hue, upper_hue = hue_range

            lower = np.array([
                lower_hue,
                LOWER_SV[0],
                LOWER_SV[1]
            ])

            upper = np.array([
                upper_hue,
                UPPER_SV[0],
                UPPER_SV[1]
            ])

            mask = cv2.inRange(hsv, lower, upper)

            pixel_count = cv2.countNonZero(mask)

            record[f"{color_name}_pixels"] = pixel_count

            current_centroid = None

            if pixel_count > 0:

                moments = cv2.moments(mask)

                if moments["m00"] != 0:
                    center_x = (
                        moments["m10"] / moments["m00"]
                    )

                    center_y = (
                        moments["m01"] / moments["m00"]
                    )

                    current_centroid = (
                        center_x,
                        center_y
                    )

            # --------------------------------------
            # CENTROID DISPLACEMENT
            # --------------------------------------

            displacement = None

            previous_centroid = previous_centroids.get(
                color_name
            )

            if (
                current_centroid is not None
                and previous_centroid is not None
            ):

                distance = np.hypot(
                    current_centroid[0] - previous_centroid[0],
                    current_centroid[1] - previous_centroid[1]
                )

                displacement = distance / diagonal

                displacement_values[color_name].append(
                    displacement
                )

            record[f"{color_name}_centroid_x"] = (
                round(current_centroid[0], 2)
                if current_centroid is not None else None
            )

            record[f"{color_name}_centroid_y"] = (
                round(current_centroid[1], 2)
                if current_centroid is not None else None
            )

            record[f"{color_name}_normalized_displacement"] = (
                round(displacement, 6)
                if displacement is not None else None
            )

            # --------------------------------------
            # CONSECUTIVE FRAME MASK OVERLAP
            # --------------------------------------

            overlap = None

            previous_mask = previous_masks.get(color_name)

            if previous_mask is not None:

                intersection = cv2.countNonZero(
                    cv2.bitwise_and(mask, previous_mask)
                )

                union = cv2.countNonZero(
                    cv2.bitwise_or(mask, previous_mask)
                )

                if union > 0:
                    overlap = intersection / union

                    overlap_values[color_name].append(
                        overlap
                    )

            record[f"{color_name}_mask_iou"] = (
                round(overlap, 6)
                if overlap is not None else None
            )

            previous_masks[color_name] = mask.copy()
            previous_centroids[color_name] = current_centroid

        frame_results.append(record)

        frame_index += 1

    cap.release()

    # ----------------------------------------------
    # VIDEO SUMMARY
    # ----------------------------------------------

    summary = {
        "split": split,
        "group": group,
        "video": filename,
        "metadata_frames": total_frames,
        "decoded_frames": frame_index,
        "fps": fps,
        "width": width,
        "height": height,
    }

    for color_name in COLOR_RANGES:

        movements = displacement_values[color_name]
        overlaps = overlap_values[color_name]

        summary[f"{color_name}_valid_movements"] = len(
            movements
        )

        summary[f"{color_name}_mean_displacement"] = (
            round(float(np.mean(movements)), 6)
            if movements else None
        )

        summary[f"{color_name}_mean_mask_iou"] = (
            round(float(np.mean(overlaps)), 6)
            if overlaps else None
        )

    video_summaries.append(summary)

    print(f"Metadata frames: {total_frames}")
    print(f"Decoded frames: {frame_index}")

    print(
        "Valid colour displacement measurements:",
        sum(
            len(values)
            for values in displacement_values.values()
        )
    )


# --------------------------------------------------
# SAVE FRAME-LEVEL RESULTS
# --------------------------------------------------

if not frame_results:
    raise RuntimeError("No video frames were analyzed.")

frame_path = OUTPUT_DIR / "temporal_frame_metrics.csv"

with frame_path.open(
    "w", newline="", encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=frame_results[0].keys()
    )

    writer.writeheader()
    writer.writerows(frame_results)


# --------------------------------------------------
# SAVE VIDEO-LEVEL SUMMARY
# --------------------------------------------------

summary_path = OUTPUT_DIR / "temporal_video_summary.csv"

with summary_path.open(
    "w", newline="", encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=video_summaries[0].keys()
    )

    writer.writeheader()
    writer.writerows(video_summaries)


# --------------------------------------------------
# FINAL REPORT
# --------------------------------------------------

print("\n" + "=" * 55)
print("TEMPORAL ANALYSIS COMPLETED")
print("=" * 55)

print(f"Videos analyzed: {len(video_summaries)}")
print(f"Frames analyzed: {len(frame_results)}")

print("\nOUTPUT FILES")
print(frame_path)
print(summary_path)

print("\nOriginal videos were not modified.")
