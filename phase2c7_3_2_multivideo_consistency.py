
import cv2
import csv
import random
import re
import numpy as np

from pathlib import Path
from collections import defaultdict

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

ROOT = Path(
    r"C:\ReSenseResearchData\video\raw\OpenPose_ADOS"
    r"\extracted\autism_data_anonymized"
)

OUTPUT_DIR = Path(
    r"C:\ReSenseResearchData\metadata"
    r"\Phase2C7_3_2_multivideo_consistency"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GROUPS = [
    ("training_set", "ASD"),
    ("training_set", "TD"),
    ("testing_set", "ASD"),
    ("testing_set", "TD"),
]

VIDEOS_PER_GROUP = 5
RANDOM_SEED = 42

# Same HSV segmentation threshold as previous experiment.
LOWER_HSV = np.array([0, 60, 40])
UPPER_HSV = np.array([179, 255, 255])

MIN_COMPONENT_AREA = 2


# --------------------------------------------------
# SUBJECT IDENTIFICATION AND VIDEO SELECTION
# --------------------------------------------------

def get_subject_id(video_path):
    match = re.match(r"^(.*?)_part_", video_path.stem)

    if match:
        return match.group(1)

    return video_path.stem


rng = random.Random(RANDOM_SEED)

selected_videos = []

for split, group in GROUPS:

    folder = ROOT / split / group

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    all_videos = sorted(folder.glob("*.mp4"))

    # Group recordings by subject identifier.
    subject_videos = defaultdict(list)

    for video in all_videos:
        subject_id = get_subject_id(video)
        subject_videos[subject_id].append(video)

    subject_ids = sorted(subject_videos)

    if len(subject_ids) < VIDEOS_PER_GROUP:
        raise ValueError(
            f"Not enough unique subjects in {split}/{group}. "
            f"Found {len(subject_ids)}."
        )

    selected_subjects = sorted(
        rng.sample(subject_ids, VIDEOS_PER_GROUP)
    )

    for subject_id in selected_subjects:

        recordings = sorted(subject_videos[subject_id])
        selected_video = rng.choice(recordings)

        selected_videos.append({
            "split": split,
            "group": group,
            "subject_id": subject_id,
            "video_path": selected_video,
        })


# --------------------------------------------------
# SAVE REPRODUCIBLE SAMPLE MANIFEST
# --------------------------------------------------

manifest_path = OUTPUT_DIR / "sample_manifest.csv"

with manifest_path.open(
    "w", newline="", encoding="utf-8"
) as file:

    fields = [
        "split",
        "group",
        "subject_id",
        "video",
    ]

    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()

    for item in selected_videos:
        writer.writerow({
            "split": item["split"],
            "group": item["group"],
            "subject_id": item["subject_id"],
            "video": item["video_path"].name,
        })


# --------------------------------------------------
# VIDEO ANALYSIS
# --------------------------------------------------

video_results = []

for item in selected_videos:

    video_path = item["video_path"]

    print(
        f"\nAnalyzing: {video_path.name} "
        f"| {item['split']} | {item['group']}"
    )

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("ERROR: Cannot open video")
        cap.release()
        continue

    metadata_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = cap.get(cv2.CAP_PROP_FPS)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    colored_counts = []
    nonblack_counts = []
    component_counts = []

    while True:

        success, frame = cap.read()

        if not success:
            break

        hsv = cv2.cvtColor(
            frame, cv2.COLOR_BGR2HSV
        )

        gray = cv2.cvtColor(
            frame, cv2.COLOR_BGR2GRAY
        )

        color_mask = cv2.inRange(
            hsv,
            LOWER_HSV,
            UPPER_HSV
        )

        colored_pixels = cv2.countNonZero(
            color_mask
        )

        nonblack_pixels = cv2.countNonZero(
            gray
        )

        count, labels, stats, centroids = (
            cv2.connectedComponentsWithStats(
                color_mask,
                connectivity=8
            )
        )

        valid_components = sum(
            1
            for area in stats[1:, cv2.CC_STAT_AREA]
            if area >= MIN_COMPONENT_AREA
        )

        colored_counts.append(colored_pixels)
        nonblack_counts.append(nonblack_pixels)
        component_counts.append(valid_components)

    cap.release()

    decoded_frames = len(colored_counts)

    if decoded_frames == 0:
        print("WARNING: No frames decoded")
        continue

    frames_with_color = sum(
        value > 0 for value in colored_counts
    )

    result = {
        "split": item["split"],
        "group": item["group"],
        "subject_id": item["subject_id"],
        "video": video_path.name,
        "metadata_frames": metadata_frames,
        "decoded_frames": decoded_frames,
        "fps": fps,
        "width": width,
        "height": height,
        "frames_with_color": frames_with_color,
        "frames_with_color_percent": round(
            frames_with_color / decoded_frames * 100, 2
        ),
        "mean_colored_pixels": round(
            float(np.mean(colored_counts)), 2
        ),
        "std_colored_pixels": round(
            float(np.std(colored_counts)), 2
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

    video_results.append(result)

    print(
        f"Frames: {decoded_frames}/{metadata_frames}"
    )
    print(
        f"Frames with colour: "
        f"{result['frames_with_color_percent']}%"
    )
    print(
        f"Mean coloured pixels: "
        f"{result['mean_colored_pixels']}"
    )
    print(
        f"Mean connected components: "
        f"{result['mean_connected_components']}"
    )


# --------------------------------------------------
# SAVE INDIVIDUAL VIDEO RESULTS
# --------------------------------------------------

if not video_results:
    raise RuntimeError("No videos were successfully analyzed.")

details_path = OUTPUT_DIR / "video_level_results.csv"

with details_path.open(
    "w", newline="", encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=video_results[0].keys()
    )

    writer.writeheader()
    writer.writerows(video_results)


# --------------------------------------------------
# GROUP SUMMARY
# --------------------------------------------------

summary = []

for split, group in GROUPS:

    rows = [
        row for row in video_results
        if row["split"] == split
        and row["group"] == group
    ]

    if not rows:
        continue

    summary.append({
        "split": split,
        "group": group,
        "videos_analyzed": len(rows),
        "total_decoded_frames": sum(
            row["decoded_frames"] for row in rows
        ),
        "mean_frames_with_color_percent": round(
            float(np.mean([
                row["frames_with_color_percent"]
                for row in rows
            ])), 2
        ),
        "mean_colored_pixels": round(
            float(np.mean([
                row["mean_colored_pixels"]
                for row in rows
            ])), 2
        ),
        "mean_nonblack_pixels": round(
            float(np.mean([
                row["mean_nonblack_pixels"]
                for row in rows
            ])), 2
        ),
        "mean_connected_components": round(
            float(np.mean([
                row["mean_connected_components"]
                for row in rows
            ])), 2
        ),
    })


summary_path = OUTPUT_DIR / "group_summary.csv"

with summary_path.open(
    "w", newline="", encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=summary[0].keys()
    )

    writer.writeheader()
    writer.writerows(summary)


# --------------------------------------------------
# FINAL REPORT
# --------------------------------------------------

print("\n" + "=" * 55)
print("MULTI-VIDEO CONSISTENCY TEST COMPLETED")
print("=" * 55)

print(
    f"Videos selected: {len(selected_videos)}"
)

print(
    f"Videos successfully analyzed: {len(video_results)}"
)

print("\nGROUP SUMMARY")

for row in summary:
    print(
        f"\n{row['split']} / {row['group']}"
        f"\nVideos: {row['videos_analyzed']}"
        f"\nMean colour-visible frames: "
        f"{row['mean_frames_with_color_percent']}%"
        f"\nMean coloured pixels: "
        f"{row['mean_colored_pixels']}"
        f"\nMean connected components: "
        f"{row['mean_connected_components']}"
    )

print("\nOUTPUT FILES")
print(f"Manifest: {manifest_path}")
print(f"Video results: {details_path}")
print(f"Group summary: {summary_path}")
