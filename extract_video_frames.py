
import os
import cv2
import csv
import time

ROOT = r"C:\ReSenseResearchData\video\raw\OpenPose_ADOS\extracted\autism_data_anonymized"
OUTPUT = r"C:\ReSenseResearchData\video\processed\sampled_frames"

FRAMES_PER_VIDEO = 8
IMAGE_SIZE = (64, 64)

GROUPS = [
    ("training_set", "ASD"),
    ("training_set", "TD"),
    ("testing_set", "ASD"),
    ("testing_set", "TD"),
]

os.makedirs(OUTPUT, exist_ok=True)

manifest_path = os.path.join(OUTPUT, "frame_manifest.csv")

total_videos = 0
total_frames = 0
failed_videos = []

start_time = time.time()

with open(manifest_path, "w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow([
        "Split", "Class", "Participant",
        "Segment", "FrameIndex", "ImagePath"
    ])

    for split, label in GROUPS:

        source_folder = os.path.join(ROOT, split, label)

        output_folder = os.path.join(OUTPUT, split, label)
        os.makedirs(output_folder, exist_ok=True)

        videos = sorted(
            [
                name for name in os.listdir(source_folder)
                if name.lower().endswith(".mp4")
            ],
            key=lambda name: (
                int(name.split("_")[1]),
                int(name.split("_")[3].split(".")[0])
            )
        )

        print(f"\nProcessing {split}/{label}: {len(videos)} videos")

        for filename in videos:

            name = os.path.splitext(filename)[0]

            parts = name.split("_")
            participant = parts[1]
            segment = parts[3]

            video_path = os.path.join(source_folder, filename)

            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                failed_videos.append(filename)
                continue

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if frame_count < FRAMES_PER_VIDEO:
                failed_videos.append(filename)
                cap.release()
                continue

            indices = [
                round(i * (frame_count - 1) / (FRAMES_PER_VIDEO - 1))
                for i in range(FRAMES_PER_VIDEO)
            ]

            video_ok = True

            for frame_index in indices:

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                success, frame = cap.read()

                if not success:
                    video_ok = False
                    break

                frame = cv2.resize(frame, IMAGE_SIZE)

                image_name = f"{name}_frame_{frame_index:04d}.jpg"
                image_path = os.path.join(output_folder, image_name)

                if not cv2.imwrite(
                    image_path,
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 90]
                ):
                    video_ok = False
                    break

                relative_path = os.path.relpath(image_path, OUTPUT)

                writer.writerow([
                    split, label, participant,
                    segment, frame_index, relative_path
                ])

                total_frames += 1

            cap.release()

            if video_ok:
                total_videos += 1
            else:
                failed_videos.append(filename)

            if total_videos % 500 == 0:
                print(
                    "Successfully processed:",
                    total_videos,
                    "| Frames:", total_frames
                )

print("\nEXTRACTION SUMMARY")
print("Successfully processed videos:", total_videos)
print("Extracted frames:", total_frames)
print("Failed videos:", len(failed_videos))
print("Manifest:", manifest_path)
print("Elapsed minutes:", round((time.time() - start_time) / 60, 2))

if failed_videos:
    failure_path = os.path.join(OUTPUT, "failed_videos.txt")

    with open(failure_path, "w") as file:
        file.write("\n".join(failed_videos))

    print("Failure list:", failure_path)
