
import cv2
import csv
from pathlib import Path

ROOT = Path(
    r"C:\ReSenseResearchData\video\raw\OpenPose_ADOS\extracted"
    r"\autism_data_anonymized"
)

VIDEOS = [
    ROOT / "testing_set" / "TD" / "Subj_122_part_58.mp4",
    ROOT / "training_set" / "ASD" / "Subj_2_part_20.mp4",
    ROOT / "training_set" / "TD" / "Subj_14_part_111.mp4",
]

OUTPUT = Path(
    r"C:\ReSenseResearchData\metadata"
    r"\Phase2C6_source_video_investigation.csv"
)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

results = []

for video_path in VIDEOS:
    print(f"\nInspecting: {video_path.name} | {video_path.parent.name}")

    if not video_path.exists():
        print(f"ERROR: File not found: {video_path}")
        continue

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("ERROR: Cannot open video")
        cap.release()
        continue

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    exact_black = []
    zero_foreground = []
    low_foreground = []
    visible = []
    read_errors = []

    index = 0

    # Read frames up to the reported metadata frame count.
    # A failed read before that count is recorded as an early read failure.
    while index < frame_count:
        success, frame = cap.read()

        if not success:
            read_errors.append(index)
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if cv2.countNonZero(gray) == 0:
            exact_black.append(index)
        else:
            foreground = cv2.countNonZero(
                cv2.threshold(
                    gray, 10, 255, cv2.THRESH_BINARY
                )[1]
            )

            percentage = foreground / (width * height) * 100

            if foreground == 0:
                zero_foreground.append(index)
            elif percentage < 0.5:
                low_foreground.append(index)
            else:
                visible.append(index)

        index += 1

    cap.release()

    results.append({
        "video": str(video_path),
        "frames_metadata": frame_count,
        "frames_decoded": index,
        "fps": fps,
        "width": width,
        "height": height,
        "exact_black_frames": len(exact_black),
        "zero_foreground_frames": len(zero_foreground),
        "low_foreground_frames": len(low_foreground),
        "visible_frames": len(visible),
        "exact_black_indices": ";".join(map(str, exact_black)),
        "read_errors": len(read_errors),
    })

    print(f"Metadata frames: {frame_count}")
    print(f"Decoded frames: {index}")
    print(f"FPS: {fps}")
    print(f"Resolution: {width}x{height}")
    print(f"Exact black: {len(exact_black)}")
    print(f"Zero foreground: {len(zero_foreground)}")
    print(f"Low foreground: {len(low_foreground)}")
    print(f"Visible: {len(visible)}")
    print(f"Early read failures: {len(read_errors)}")

with OUTPUT.open("w", newline="", encoding="utf-8") as f:
    if results:
        writer = csv.DictWriter(
            f,
            fieldnames=results[0].keys()
        )
        writer.writeheader()
        writer.writerows(results)

print(f"\nReport saved to: {OUTPUT}")
