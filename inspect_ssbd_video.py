
import cv2
from pathlib import Path

video = Path(
    r"C:\ReSenseResearchData\raw\SSBD\authorized_videos"
    r"\v_ArmFlapping_01.mp4"
)

output = Path(
    r"C:\ReSenseResearchData\experiments\CNN"
    r"\synthetic_extraction_test\inspection"
)
output.mkdir(parents=True, exist_ok=True)

timestamps = [
    ("arm_flapping", 18),
    ("arm_flapping", 20),
    ("arm_flapping", 22),
    ("arm_flapping", 24),
    ("head_banging", 27),
    ("head_banging", 29),
    ("head_banging", 31),
    ("head_banging", 32),
]

cap = cv2.VideoCapture(str(video))

if not cap.isOpened():
    raise RuntimeError("Cannot open video")

fps = cap.get(cv2.CAP_PROP_FPS)
tiles = []

for category, second in timestamps:
    cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
    success, frame = cap.read()

    if not success:
        print("Frame read failed:", second)
        continue

    frame = cv2.resize(frame, (320, 240))
    cv2.putText(
        frame,
        f"{category} - {second}s",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )
    tiles.append(frame)

cap.release()

if not tiles:
    raise RuntimeError("No frames extracted")

while len(tiles) < 8:
    tiles.append(tiles[-1].copy())

rows = [
    cv2.hconcat(tiles[0:4]),
    cv2.hconcat(tiles[4:8])
]
sheet = cv2.vconcat(rows)

path = output / "v_ArmFlapping_01_inspection.jpg"

if not cv2.imwrite(str(path), sheet):
    raise RuntimeError("Could not save contact sheet")

print("FPS:", fps)
print("Inspection frames:", len(timestamps))
print("Contact sheet:", path)
