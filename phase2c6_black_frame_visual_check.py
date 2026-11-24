
import cv2
from pathlib import Path

ROOT = Path(
    r"C:\ReSenseResearchData\video\raw\OpenPose_ADOS"
    r"\extracted\autism_data_anonymized"
)

OUTPUT = Path(
    r"C:\ReSenseResearchData\metadata"
    r"\Phase2C6_black_frame_visual_evidence"
)
OUTPUT.mkdir(parents=True, exist_ok=True)

VIDEOS = [
    ROOT / "testing_set" / "TD" / "Subj_122_part_58.mp4",
    ROOT / "training_set" / "ASD" / "Subj_2_part_20.mp4",
]

for path in VIDEOS:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print("Cannot open:", path)
        continue

    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for index in range(0, total, 5):
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()

        if not ok:
            continue

        frame = cv2.resize(frame, (320, 240))
        cv2.putText(
            frame, f"Frame {index}",
            (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 255, 0), 2
        )
        frames.append(frame)

    cap.release()

    columns = 4
    rows = (len(frames) + columns - 1) // columns
    sheet = cv2.hconcat([
        cv2.vconcat([
            frames[r * columns + c]
            if r * columns + c < len(frames)
            else frames[0] * 0
            for c in range(columns)
        ])
        for r in range(rows)
    ])

    output_file = OUTPUT / f"{path.stem}_{path.parent.name}_contact_sheet.jpg"
    cv2.imwrite(str(output_file), sheet)
    print("Saved:", output_file)

print("\nVisual evidence generation completed.")
