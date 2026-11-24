
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
    r"\Phase2C7_8_temporal_component_tracking"
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

MIN_COMPONENT_AREA = 2

# Exploratory matching distance in pixels per frame.
MAX_MATCH_DISTANCE = 25.0

SAMPLE_POSITIONS = [0.25, 0.50, 0.75]


# --------------------------------------------------
# BOUNDING BOX IOU
# --------------------------------------------------

def calculate_iou(box_a, box_b):

    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)

    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)

    intersection = (
        intersection_width * intersection_height
    )

    area_a = aw * ah
    area_b = bw * bh

    union = area_a + area_b - intersection

    if union == 0:
        return 0.0

    return intersection / union


# --------------------------------------------------
# COMPONENT EXTRACTION
# --------------------------------------------------

def extract_components(frame):

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

            if area < MIN_COMPONENT_AREA:
                continue

            x = int(stats[component_id, cv2.CC_STAT_LEFT])
            y = int(stats[component_id, cv2.CC_STAT_TOP])

            w = int(stats[component_id, cv2.CC_STAT_WIDTH])
            h = int(stats[component_id, cv2.CC_STAT_HEIGHT])

            center_x = float(centroids[component_id][0])
            center_y = float(centroids[component_id][1])

            color_detections.append({
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "area": area,
                "center_x": center_x,
                "center_y": center_y,
            })

        detections[color_name] = color_detections

    return detections


# --------------------------------------------------
# GREEDY COMPONENT MATCHING
# --------------------------------------------------

def match_components(previous, current):

    candidates = []

    for prev_index, prev in enumerate(previous):

        for curr_index, curr in enumerate(current):

            distance = float(np.hypot(
                prev["center_x"] - curr["center_x"],
                prev["center_y"] - curr["center_y"]
            ))

            if distance > MAX_MATCH_DISTANCE:
                continue

            prev_box = (
                prev["x"], prev["y"],
                prev["width"], prev["height"]
            )

            curr_box = (
                curr["x"], curr["y"],
                curr["width"], curr["height"]
            )

            iou = calculate_iou(prev_box, curr_box)

            candidates.append((
                distance,
                -iou,
                prev_index,
                curr_index,
                iou
            ))

    # Prioritize smaller centroid distance.
    # Use greater bounding-box IoU as a tie-breaker.
    candidates.sort()

    used_previous = set()
    used_current = set()
    matches = {}

    for distance, negative_iou, pi, ci, iou in candidates:

        if pi in used_previous or ci in used_current:
            continue

        used_previous.add(pi)
        used_current.add(ci)

        matches[ci] = {
            "previous_index": pi,
            "distance": distance,
            "iou": iou,
        }

    return matches


# --------------------------------------------------
# OUTPUT RECORDS
# --------------------------------------------------

detection_results = []
frame_results = []
video_summaries = []


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
    fps = cap.get(cv2.CAP_PROP_FPS)

    sample_indices = {
        min(total_frames - 1, int(total_frames * p))
        for p in SAMPLE_POSITIONS
    }

    video_output = OUTPUT_DIR / (
        f"{split}_{group}_{Path(filename).stem}"
    )

    video_output.mkdir(parents=True, exist_ok=True)

    # Separate tracks for each colour.
    previous_tracks = {
        color: [] for color in COLOR_RANGES
    }

    track_information = {}
    next_track_id = 1

    frame_index = 0
    total_detections = 0
    matched_detections = 0
    matching_distances = []

    while True:

        success, frame = cap.read()

        if not success:
            break

        detections = extract_components(frame)

        annotated = frame.copy()

        frame_record = {
            "split": split,
            "group": group,
            "video": filename,
            "frame_index": frame_index,
        }

        frame_detection_count = 0
        frame_match_count = 0

        for color_name, current in detections.items():

            previous = previous_tracks[color_name]

            matches = match_components(
                previous,
                current
            )

            current_tracks = []

            frame_record[f"{color_name}_detections"] = len(current)
            frame_record[f"{color_name}_matched"] = len(matches)

            frame_detection_count += len(current)
            frame_match_count += len(matches)

            for curr_index, detection in enumerate(current):

                if curr_index in matches:

                    match = matches[curr_index]

                    prev_index = match["previous_index"]

                    track_id = previous[prev_index]["track_id"]

                    track_information[track_id]["matched_frames"] += 1

                    matched_detections += 1
                    matching_distances.append(match["distance"])

                    match_distance = round(
                        match["distance"], 3
                    )

                    bbox_iou = round(match["iou"], 4)

                else:

                    track_id = next_track_id
                    next_track_id += 1

                    track_information[track_id] = {
                        "color": color_name,
                        "first_frame": frame_index,
                        "last_frame": frame_index,
                        "matched_frames": 0,
                    }

                    match_distance = None
                    bbox_iou = None

                track_information[track_id]["last_frame"] = frame_index

                track_information[track_id]["matched_frames"] += (
                    1 if curr_index not in matches else 0
                )

                track_record = {
                    **detection,
                    "track_id": track_id,
                    "color": color_name,
                }

                current_tracks.append(track_record)

                detection_results.append({
                    "split": split,
                    "group": group,
                    "video": filename,
                    "frame_index": frame_index,
                    "color": color_name,
                    "track_id": track_id,
                    "center_x": round(detection["center_x"], 2),
                    "center_y": round(detection["center_y"], 2),
                    "area": detection["area"],
                    "bbox_x": detection["x"],
                    "bbox_y": detection["y"],
                    "bbox_width": detection["width"],
                    "bbox_height": detection["height"],
                    "match_distance": match_distance,
                    "bbox_iou": bbox_iou,
                    "matched_previous_frame": (
                        curr_index in matches
                    ),
                })

                if frame_index in sample_indices:

                    cv2.rectangle(
                        annotated,
                        (
                            detection["x"],
                            detection["y"]
                        ),
                        (
                            detection["x"] + detection["width"],
                            detection["y"] + detection["height"]
                        ),
                        (255, 255, 255),
                        1
                    )

                    cv2.circle(
                        annotated,
                        (
                            int(detection["center_x"]),
                            int(detection["center_y"])
                        ),
                        2,
                        (0, 0, 255),
                        -1
                    )

                    cv2.putText(
                        annotated,
                        f"{color_name}:{track_id}",
                        (
                            detection["x"],
                            max(10, detection["y"] - 3)
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.28,
                        (255, 255, 255),
                        1
                    )

            previous_tracks[color_name] = current_tracks

        total_detections += frame_detection_count

        frame_results.append({
            **frame_record,
            "total_detections": frame_detection_count,
            "matched_detections": frame_match_count,
        })

        if frame_index in sample_indices:

            output_image = video_output / (
                f"frame_{frame_index:04d}_tracked.jpg"
            )

            cv2.imwrite(str(output_image), annotated)

        frame_index += 1

    cap.release()

    # ----------------------------------------------
    # VIDEO SUMMARY
    # ----------------------------------------------

    track_durations = [
        info["last_frame"] - info["first_frame"] + 1
        for info in track_information.values()
    ]

    summary = {
        "split": split,
        "group": group,
        "video": filename,
        "metadata_frames": total_frames,
        "decoded_frames": frame_index,
        "fps": fps,
        "total_detections": total_detections,
        "total_tracks": len(track_information),
        "matched_detections": matched_detections,
        "matched_detection_percent": round(
            matched_detections / total_detections * 100, 2
        ) if total_detections else 0,
        "mean_match_distance": round(
            float(np.mean(matching_distances)), 3
        ) if matching_distances else None,
        "mean_track_duration_frames": round(
            float(np.mean(track_durations)), 2
        ) if track_durations else None,
        "longest_track_frames": max(
            track_durations
        ) if track_durations else 0,
    }

    video_summaries.append(summary)

    print(f"Metadata frames: {total_frames}")
    print(f"Decoded frames: {frame_index}")
    print(f"Detections: {total_detections}")
    print(f"Provisional tracks: {len(track_information)}")
    print(
        f"Matched detections: "
        f"{summary['matched_detection_percent']}%"
    )


# --------------------------------------------------
# SAVE CSV FILES
# --------------------------------------------------

if not video_summaries:
    raise RuntimeError("No videos were successfully analyzed.")

def save_csv(path, records):

    if not records:
        return

    with path.open(
        "w", newline="", encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=records[0].keys()
        )

        writer.writeheader()
        writer.writerows(records)


save_csv(
    OUTPUT_DIR / "component_tracking_detections.csv",
    detection_results
)

save_csv(
    OUTPUT_DIR / "component_tracking_frames.csv",
    frame_results
)

save_csv(
    OUTPUT_DIR / "component_tracking_summary.csv",
    video_summaries
)


# --------------------------------------------------
# FINAL REPORT
# --------------------------------------------------

print("\n" + "=" * 55)
print("TEMPORAL COMPONENT TRACKING COMPLETED")
print("=" * 55)

print(f"Videos analyzed: {len(video_summaries)}")
print(f"Frames analyzed: {len(frame_results)}")
print(f"Component detections: {len(detection_results)}")

print("\nOUTPUT DIRECTORY")
print(OUTPUT_DIR)

print("\nOriginal videos were not modified.")
