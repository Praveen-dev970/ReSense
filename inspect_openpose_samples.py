
import cv2
from pathlib import Path

BASE = Path(
    r"C:\ReSenseResearchData\video\raw"
    r"\OpenPose_ADOS\extracted\autism_data_anonymized"
)

SAMPLES = [
    ("training_set", "ASD", 1, 1),
    ("training_set", "TD", 1, 1),
    ("testing_set", "ASD", 100, 1),
    ("testing_set", "TD", 121, 1),
]

for split, label, subject, part in SAMPLES:
    video_path = (
        BASE / split / label /
        f"Subj_{subject}_part_{part}.mp4"
    )

    print("\n" + "=" * 60)
    print(f"Group: {split} / {label}")
    print(f"File: {video_path}")

    if not video_path.exists():
        print("ERROR: Video file not found.")
        continue

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("ERROR: Cannot open video.")
        continue

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = cap.get(cv2.CAP_PROP_FPS)

    # Read a frame near the middle of the clip.
    target_frame = total_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

    success, frame = cap.read()
    cap.release()

    if not success or frame is None:
        print("ERROR: Cannot read representative frame.")
        continue

    print(f"Resolution: {frame.shape[1]}x{frame.shape[0]}")
    print(f"FPS: {fps:.3f}")
    print(f"Total frames: {total_frames}")
    print(f"Inspected frame: {target_frame}")

    cv2.imshow(
        f"{split} - {label}",
        frame
    )

    print("Press any key to close this frame.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print("\nVisual inspection completed.")
