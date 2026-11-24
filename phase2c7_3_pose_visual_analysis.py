
import cv2
import csv
import numpy as np
from pathlib import Path

ROOT = Path(
    r"C:\ReSenseResearchData\video\raw\OpenPose_ADOS\extracted"
    r"\autism_data_anonymized"
)

VIDEOS = [
    ROOT / "training_set" / "ASD" / "Subj_2_part_20.mp4",
    ROOT / "training_set" / "TD" / "Subj_14_part_111.mp4",
]

OUTPUT = Path(
    r"C:\ReSenseResearchData\metadata"
    r"\Phase2C7_3_pose_visual_analysis.csv"
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

results = []

for video_path in VIDEOS:
    print(f"\nAnalyzing: {video_path.name}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("ERROR: Cannot open video")
        continue

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    colored_counts = []
    nonblack_counts = []
    component_counts = []

    for frame_index in range(total_frames):
        success, frame = cap.read()

        if not success:
            print(f"Frame read failed at {frame_index}")
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect saturated and sufficiently bright pixels.
        color_mask = cv2.inRange(
            hsv,
            np.array([0, 60, 40]),
            np.array([179, 255, 255])
        )

        colored_pixels = cv2.countNonZero(color_mask)
        nonblack_pixels = cv2.countNonZero(gray)

        # Count connected coloured regions.
        count, labels, stats, centroids = (
            cv2.connectedComponentsWithStats(
                color_mask,
                connectivity=8
            )
        )

        # Ignore background component.
        valid_components = sum(
            1 for area in stats[1:, cv2.CC_STAT_AREA]
            if area >= 2
        )

        colored_counts.append(colored_pixels)
        nonblack_counts.append(nonblack_pixels)
        component_counts.append(valid_components)

    cap.release()

    if not colored_counts:
        print("No frames analyzed")
        continue

    result = {
        "video": video_path.name,
        "group": video_path.parent.name,
        "frames_analyzed": len(colored_counts),
        "frames_with_color": sum(
            count > 0 for count in colored_counts
        ),
        "mean_colored_pixels": round(
            float(np.mean(colored_counts)), 2
        ),
        "max_colored_pixels": max(colored_counts),
        "mean_nonblack_pixels": round(
            float(np.mean(nonblack_counts)), 2
        ),
        "mean_connected_components": round(
            float(np.mean(component_counts)), 2
        ),
        "max_connected_components": max(component_counts),
    }

    results.append(result)

    for key, value in result.items():
        print(f"{key}: {value}")

if results:
    with OUTPUT.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys()
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nReport saved to: {OUTPUT}")
