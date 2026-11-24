
import cv2
import csv
import numpy as np

from pathlib import Path
from statistics import mean, median

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

ROOT = Path(
    r"C:\ReSenseResearchData\video\raw\OpenPose_ADOS"
    r"\extracted\autism_data_anonymized"
)

OUTPUT_DIR = Path(
    r"C:\ReSenseResearchData\metadata"
    r"\Phase2C7_9_tracking_quality"
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

AREA_THRESHOLDS = [2, 5, 10]
MATCH_DISTANCES = [10.0, 25.0, 40.0]


# --------------------------------------------------
# COMPONENT EXTRACTION
# --------------------------------------------------

def extract_components(frame, min_area):

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    detections = {}

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

        color_detections = []

        for component_id in range(1, count):

            area = int(stats[component_id, cv2.CC_STAT_AREA])

            if area < min_area:
                continue

            color_detections.append({
                "x": int(stats[component_id, cv2.CC_STAT_LEFT]),
                "y": int(stats[component_id, cv2.CC_STAT_TOP]),
                "width": int(stats[component_id, cv2.CC_STAT_WIDTH]),
                "height": int(stats[component_id, cv2.CC_STAT_HEIGHT]),
                "area": area,
                "center_x": float(centroids[component_id][0]),
                "center_y": float(centroids[component_id][1]),
            })

        detections[color_name] = color_detections

    return detections


# --------------------------------------------------
# BOUNDING BOX IOU
# --------------------------------------------------

def calculate_iou(a, b):

    ax, ay, aw, ah = a["x"], a["y"], a["width"], a["height"]
    bx, by, bw, bh = b["x"], b["y"], b["width"], b["height"]

    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    union = aw * ah + bw * bh - intersection

    return intersection / union if union > 0 else 0.0


# --------------------------------------------------
# GREEDY ONE-TO-ONE MATCHING
# --------------------------------------------------

def match_components(previous, current, max_distance):

    candidates = []

    for prev_index, prev in enumerate(previous):

        for curr_index, curr in enumerate(current):

            distance = float(np.hypot(
                prev["center_x"] - curr["center_x"],
                prev["center_y"] - curr["center_y"]
            ))

            if distance <= max_distance:

                iou = calculate_iou(prev, curr)

                candidates.append((
                    distance,
                    -iou,
                    prev_index,
                    curr_index
                ))

    candidates.sort()

    used_previous = set()
    used_current = set()
    matches = {}

    for distance, negative_iou, pi, ci in candidates:

        if pi in used_previous or ci in used_current:
            continue

        used_previous.add(pi)
        used_current.add(ci)

        matches[ci] = distance

    return matches


# --------------------------------------------------
# RUN ONE TRACKING CONFIGURATION
# --------------------------------------------------

def evaluate_configuration(
    frame_detections,
    min_area,
    max_distance
):

    previous_tracks = {
        color: [] for color in COLOR_RANGES
    }

    track_lengths = []
    next_track_id = 1

    total_detections = 0
    matched_detections = 0
    matching_distances = []

    for detections in frame_detections:

        for color_name in COLOR_RANGES:

            current = detections[color_name]
            previous = previous_tracks[color_name]

            matches = match_components(
                previous,
                current,
                max_distance
            )

            current_tracks = []

            total_detections += len(current)
            matched_detections += len(matches)

            for current_index, detection in enumerate(current):

                if current_index in matches:

                    previous_index = next(
                        i for i, old in enumerate(previous)
                        if old["track_id"] == previous[
                            matches[current_index]["previous_index"]
                        ]["track_id"]
                    ) if False else None

                    # The matching function returns the previous
                    # index through a separate mapping below.
                    pass

            # Recompute matches with explicit previous indices.
            # This avoids relying on component ordering or IDs.

            candidates = []

            for pi, old in enumerate(previous):

                for ci, new in enumerate(current):

                    distance = float(np.hypot(
                        old["center_x"] - new["center_x"],
                        old["center_y"] - new["center_y"]
                    ))

                    if distance <= max_distance:

                        candidates.append((
                            distance,
                            -calculate_iou(old, new),
                            pi,
                            ci
                        ))

            candidates.sort()

            used_previous = set()
            used_current = set()
            current_tracks = []

            for distance, negative_iou, pi, ci in candidates:

                if pi in used_previous or ci in used_current:
                    continue

                used_previous.add(pi)
                used_current.add(ci)

                track_id = previous[pi]["track_id"]
                track_lengths[track_id - 1] += 1

                current_tracks.append({
                    **current[ci],
                    "track_id": track_id
                })

                matching_distances.append(distance)

            for ci, detection in enumerate(current):

                if ci in used_current:
                    continue

                track_id = next_track_id
                next_track_id += 1

                track_lengths.append(1)

                current_tracks.append({
                    **detection,
                    "track_id": track_id
                })

            previous_tracks[color_name] = current_tracks

    total_tracks = len(track_lengths)

    if total_tracks:
        singleton_tracks = sum(
            length == 1 for length in track_lengths
        )

        mean_duration = mean(track_lengths)
        median_duration = median(track_lengths)
        longest_duration = max(track_lengths)
    else:
        singleton_tracks = 0
        mean_duration = 0
        median_duration = 0
        longest_duration = 0

    return {
        "minimum_area": min_area,
        "maximum_match_distance": max_distance,
        "total_detections": total_detections,
        "total_tracks": total_tracks,
        "matched_detections": matched_detections,
        "matched_percent": round(
            matched_detections / total_detections * 100, 2
        ) if total_detections else 0,
        "singleton_tracks": singleton_tracks,
        "singleton_track_percent": round(
            singleton_tracks / total_tracks * 100, 2
        ) if total_tracks else 0,
        "mean_track_duration": round(mean_duration, 2),
        "median_track_duration": round(median_duration, 2),
        "longest_track_duration": longest_duration,
        "mean_match_distance": round(
            mean(matching_distances), 3
        ) if matching_distances else None,
    }


# --------------------------------------------------
# PROCESS EACH VIDEO
# --------------------------------------------------

results = []

for split, group, filename in VIDEOS:

    video_path = ROOT / split / group / filename

    print(f"\nReading video: {filename}")

    if not video_path.exists():
        print("ERROR: Video not found:", video_path)
        continue

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("ERROR: Cannot open video:", filename)
        cap.release()
        continue

    frame_detections = []

    while True:

        success, frame = cap.read()

        if not success:
            break

        # Extract once at the smallest threshold.
        detections = extract_components(frame, min(AREA_THRESHOLDS))

        frame_detections.append(detections)

    cap.release()

    print("Frames decoded:", len(frame_detections))

    if not frame_detections:
        continue

    # Filter already-extracted detections for each threshold.
    for area_threshold in AREA_THRESHOLDS:

        filtered_frames = []

        for frame in frame_detections:

            filtered = {}

            for color_name, components in frame.items():

                filtered[color_name] = [
                    component for component in components
                    if component["area"] >= area_threshold
                ]

            filtered_frames.append(filtered)

        for max_distance in MATCH_DISTANCES:

            result = evaluate_configuration(
                filtered_frames,
                area_threshold,
                max_distance
            )

            result.update({
                "split": split,
                "group": group,
                "video": filename,
                "decoded_frames": len(frame_detections),
            })

            results.append(result)

            print(
                f"Area {area_threshold:2d} | "
                f"Distance {max_distance:4.0f} | "
                f"Tracks {result['total_tracks']:5d} | "
                f"Matched {result['matched_percent']:6.2f}% | "
                f"Singletons {result['singleton_track_percent']:6.2f}%"
            )


# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

if not results:
    raise RuntimeError("No tracking comparisons were completed.")

output_path = OUTPUT_DIR / "tracking_configuration_comparison.csv"

with output_path.open(
    "w", newline="", encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=results[0].keys()
    )

    writer.writeheader()
    writer.writerows(results)


# --------------------------------------------------
# FINAL REPORT
# --------------------------------------------------

print("\n" + "=" * 60)
print("TRACKING QUALITY COMPARISON COMPLETED")
print("=" * 60)

print("Videos analyzed:", len({
    row["video"] for row in results
}))

print("Configurations per video:", len(AREA_THRESHOLDS) * len(MATCH_DISTANCES))
print("Total comparison records:", len(results))

print("\nOutput:", output_path)
print("\nOriginal videos were not modified.")
