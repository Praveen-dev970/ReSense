
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
    r"\Phase2C7_4_spatial_feature_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VIDEOS = [
    {
        "split": "training_set",
        "group": "ASD",
        "name": "Subj_10_part_114.mp4",
    },
    {
        "split": "training_set",
        "group": "TD",
        "name": "Subj_11_part_101.mp4",
    },
    {
        "split": "testing_set",
        "group": "ASD",
        "name": "Subj_101_part_38.mp4",
    },
    {
        "split": "testing_set",
        "group": "TD",
        "name": "Subj_121_part_116.mp4",
    },
]

LOWER_HSV = np.array([0, 60, 40])
UPPER_HSV = np.array([179, 255, 255])

MIN_COMPONENT_AREA = 2

# Annotated frames will be saved near these positions.
SAMPLE_POSITIONS = [0.25, 0.50, 0.75]


# --------------------------------------------------
# OUTPUT FILES
# --------------------------------------------------

FRAME_CSV = OUTPUT_DIR / "frame_spatial_metrics.csv"
COMPONENT_CSV = OUTPUT_DIR / "component_coordinates.csv"


frame_results = []
component_results = []


# --------------------------------------------------
# PROCESS EACH VIDEO
# --------------------------------------------------

for item in VIDEOS:

    video_path = ROOT / item["split"] / item["group"] / item["name"]

    print(f"\nAnalyzing: {video_path.name} | {item['split']} | {item['group']}")

    if not video_path.exists():
        print(f"ERROR: Video not found: {video_path}")
        continue

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("ERROR: Cannot open video")
        cap.release()
        continue

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    sample_indices = {
        min(total_frames - 1, int(total_frames * position))
        for position in SAMPLE_POSITIONS
    }

    video_output = OUTPUT_DIR / (
        item["split"] + "_" + item["group"] + "_" +
        Path(item["name"]).stem
    )

    video_output.mkdir(parents=True, exist_ok=True)

    frame_index = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(
            hsv,
            LOWER_HSV,
            UPPER_HSV
        )

        component_count, labels, stats, centroids = (
            cv2.connectedComponentsWithStats(
                mask,
                connectivity=8
            )
        )

        valid_components = []

        for component_id in range(1, component_count):

            x = int(stats[component_id, cv2.CC_STAT_LEFT])
            y = int(stats[component_id, cv2.CC_STAT_TOP])

            w = int(stats[component_id, cv2.CC_STAT_WIDTH])
            h = int(stats[component_id, cv2.CC_STAT_HEIGHT])

            area = int(stats[component_id, cv2.CC_STAT_AREA])

            if area < MIN_COMPONENT_AREA:
                continue

            center_x = float(centroids[component_id][0])
            center_y = float(centroids[component_id][1])

            valid_components.append({
                "component_id": component_id,
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "area": area,
                "center_x": round(center_x, 2),
                "center_y": round(center_y, 2),
            })

            component_results.append({
                "split": item["split"],
                "group": item["group"],
                "video": item["name"],
                "frame_index": frame_index,
                **valid_components[-1],
            })

        colored_pixels = cv2.countNonZero(mask)

        frame_results.append({
            "split": item["split"],
            "group": item["group"],
            "video": item["name"],
            "frame_index": frame_index,
            "colored_pixels": colored_pixels,
            "component_count": len(valid_components),
            "total_component_area": sum(
                component["area"]
                for component in valid_components
            ),
        })

        # ------------------------------------------
        # SAVE ANNOTATED SAMPLE FRAMES
        # ------------------------------------------

        if frame_index in sample_indices:

            annotated = frame.copy()

            for component in valid_components:

                x = component["x"]
                y = component["y"]
                w = component["width"]
                h = component["height"]

                center_x = int(component["center_x"])
                center_y = int(component["center_y"])

                cv2.rectangle(
                    annotated,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    1
                )

                cv2.circle(
                    annotated,
                    (center_x, center_y),
                    3,
                    (0, 0, 255),
                    -1
                )

            cv2.putText(
                annotated,
                f"Frame: {frame_index} | Regions: {len(valid_components)}",
                (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1
            )

            output_image = video_output / (
                f"frame_{frame_index:04d}_annotated.jpg"
            )

            cv2.imwrite(str(output_image), annotated)

        frame_index += 1

    cap.release()

    print(f"Metadata frames: {total_frames}")
    print(f"Decoded frames: {frame_index}")
    print(f"Resolution: {width}x{height}")
    print(f"Annotated frames saved: {len(sample_indices)}")


# --------------------------------------------------
# SAVE FRAME-LEVEL RESULTS
# --------------------------------------------------

if not frame_results:
    raise RuntimeError("No video frames were successfully analyzed.")

with FRAME_CSV.open(
    "w", newline="", encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=frame_results[0].keys()
    )

    writer.writeheader()
    writer.writerows(frame_results)


# --------------------------------------------------
# SAVE COMPONENT COORDINATES
# --------------------------------------------------

component_fields = [
    "split",
    "group",
    "video",
    "frame_index",
    "component_id",
    "x",
    "y",
    "width",
    "height",
    "area",
    "center_x",
    "center_y",
]

with COMPONENT_CSV.open(
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
print("SPATIAL FEATURE ANALYSIS COMPLETED")
print("=" * 55)

print(f"Videos configured: {len(VIDEOS)}")
print(f"Frames analyzed: {len(frame_results)}")
print(f"Connected regions recorded: {len(component_results)}")

print("\nOUTPUT FILES")
print(f"Frame metrics: {FRAME_CSV}")
print(f"Component coordinates: {COMPONENT_CSV}")
print(f"Annotated images: {OUTPUT_DIR}")

print("\nOriginal videos were not modified.")
