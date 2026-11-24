
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
    r"\Phase2C7_5_color_channel_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VIDEOS = [
    ("training_set", "ASD", "Subj_10_part_114.mp4"),
    ("training_set", "TD", "Subj_11_part_101.mp4"),
    ("testing_set", "ASD", "Subj_101_part_38.mp4"),
    ("testing_set", "TD", "Subj_121_part_116.mp4"),
]

# OpenCV HSV hue ranges (0-179).
# These are exploratory colour bands, not anatomical labels.
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

MIN_COMPONENT_AREA = 2

SAMPLE_POSITIONS = [0.25, 0.50, 0.75]

component_results = []
frame_results = []


# --------------------------------------------------
# PROCESS VIDEOS
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

    video_output = OUTPUT_DIR / (
        f"{split}_{group}_{Path(filename).stem}"
    )

    video_output.mkdir(parents=True, exist_ok=True)

    sample_indices = {
        min(total_frames - 1, int(total_frames * p))
        for p in SAMPLE_POSITIONS
    }

    frame_index = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        annotated = frame.copy()

        frame_record = {
            "split": split,
            "group": group,
            "video": filename,
            "frame_index": frame_index,
        }

        total_components = 0

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

            valid_count = 0

            for component_id in range(1, count):

                x = int(stats[component_id, cv2.CC_STAT_LEFT])
                y = int(stats[component_id, cv2.CC_STAT_TOP])

                w = int(stats[component_id, cv2.CC_STAT_WIDTH])
                h = int(stats[component_id, cv2.CC_STAT_HEIGHT])

                area = int(stats[component_id, cv2.CC_STAT_AREA])

                if area < MIN_COMPONENT_AREA:
                    continue

                center_x = float(centroids[component_id][0])
                center_y = float(centroids[component_id][1])

                valid_count += 1
                total_components += 1

                component_results.append({
                    "split": split,
                    "group": group,
                    "video": filename,
                    "frame_index": frame_index,
                    "color": color_name,
                    "component_id": component_id,
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h,
                    "area": area,
                    "center_x": round(center_x, 2),
                    "center_y": round(center_y, 2),
                })

                if frame_index in sample_indices:

                    cv2.rectangle(
                        annotated,
                        (x, y),
                        (x + w, y + h),
                        (255, 255, 255),
                        1
                    )

                    cv2.circle(
                        annotated,
                        (int(center_x), int(center_y)),
                        2,
                        (0, 0, 255),
                        -1
                    )

                    cv2.putText(
                        annotated,
                        color_name,
                        (x, max(10, y - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3,
                        (255, 255, 255),
                        1
                    )

            frame_record[f"{color_name}_pixels"] = (
                cv2.countNonZero(mask)
            )

            frame_record[f"{color_name}_components"] = valid_count

        frame_record["total_components"] = total_components
        frame_results.append(frame_record)

        if frame_index in sample_indices:

            output_image = video_output / (
                f"frame_{frame_index:04d}_color_analysis.jpg"
            )

            cv2.imwrite(str(output_image), annotated)

        frame_index += 1

    cap.release()

    print(f"Metadata frames: {total_frames}")
    print(f"Decoded frames: {frame_index}")
    print(f"Colour-specific regions recorded: {len(component_results)}")


# --------------------------------------------------
# SAVE FRAME RESULTS
# --------------------------------------------------

if not frame_results:
    raise RuntimeError("No frames were successfully analyzed.")

frame_path = OUTPUT_DIR / "color_frame_metrics.csv"

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
# SAVE COMPONENT RESULTS
# --------------------------------------------------

component_path = OUTPUT_DIR / "color_component_coordinates.csv"

component_fields = [
    "split", "group", "video", "frame_index",
    "color", "component_id", "x", "y",
    "width", "height", "area", "center_x", "center_y"
]

with component_path.open(
    "w", newline="", encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=component_fields
    )

    writer.writeheader()
    writer.writerows(component_results)


# --------------------------------------------------
# FINAL SUMMARY
# --------------------------------------------------

print("\n" + "=" * 55)
print("COLOUR CHANNEL ANALYSIS COMPLETED")
print("=" * 55)

print(f"Videos configured: {len(VIDEOS)}")
print(f"Frames analyzed: {len(frame_results)}")
print(f"Colour-specific regions: {len(component_results)}")

print("\nOUTPUT FILES")
print(frame_path)
print(component_path)
print(OUTPUT_DIR)

print("\nOriginal videos were not modified.")
