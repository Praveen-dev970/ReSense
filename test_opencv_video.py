import cv2
import subprocess
import sys
import tempfile
from pathlib import Path

project = Path.cwd()
video_path = Path(tempfile.gettempdir()) / "resense_opencv_test.avi"

writer = cv2.VideoWriter(
    str(video_path),
    cv2.VideoWriter_fourcc(*"MJPG"),
    10,
    (320, 240)
)

if not writer.isOpened():
    print("FAILED: Could not create synthetic test video.")
    sys.exit(1)

for i in range(20):
    frame = (i * 10) % 255
    image = __import__("numpy").full(
        (240, 320, 3), frame, dtype="uint8"
    )
    writer.write(image)

writer.release()

print("Test video created:", video_path)

result = subprocess.run(
    [sys.executable, str(project / "main.py"), str(video_path)],
    capture_output=True,
    text=True,
    cwd=str(project)
)

print("\n========== PROCESSING OUTPUT ==========")
print(result.stdout)
print(result.stderr)
print("Exit code:", result.returncode)

if result.returncode == 0:
    print("\nPASS: OpenCV video processing completed.")
else:
    print("\nFAIL: OpenCV video processing failed.")
