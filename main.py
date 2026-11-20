"""Extract evenly sampled frames from a video.

This script only processes video frames. It does not detect autism or predict
behavior.
"""

import sys
import uuid
from pathlib import Path

import cv2


MAX_FRAMES = 30
PROJECT_DIR = Path(__file__).resolve().parent


def extract_frames(video_path):
    """Extract up to MAX_FRAMES evenly sampled frames into a unique folder."""
    video_path = Path(video_path).expanduser().resolve()
    if not video_path.is_file():
        raise ValueError("The input video path does not exist or is not a file.")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError("OpenCV could not open the input video.")

    try:
        output_dir = PROJECT_DIR / "output_frames" / (
            f"{video_path.stem}_{uuid.uuid4().hex}"
        )
        output_dir.mkdir(parents=True, exist_ok=False)

        # Count decodable frames first so sampling stays evenly spaced even
        # when the container's reported frame count is missing or inaccurate.
        total_frames = 0
        while True:
            ok, _ = capture.read()
            if not ok:
                break
            total_frames += 1

        if total_frames == 0:
            raise ValueError("The video opened, but contains no readable frames.")

        sample_count = min(MAX_FRAMES, total_frames)
        if sample_count == 1:
            frame_indices = [0]
        else:
            frame_indices = [
                round(index * (total_frames - 1) / (sample_count - 1))
                for index in range(sample_count)
            ]

        capture.release()
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError("The video could not be reopened to extract frames.")

        target_frames = set(frame_indices)
        frames_saved = 0
        frame_index = 0
        output_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index in target_frames:
                frame_path = output_dir / f"frame_{output_index:03d}.jpg"
                if not cv2.imwrite(str(frame_path), frame):
                    raise RuntimeError(f"Could not write frame to {frame_path}.")
                frames_saved += 1
                output_index += 1
            frame_index += 1

        if frames_saved != sample_count:
            raise RuntimeError(
                f"Expected {sample_count} sampled frames but extracted {frames_saved}."
            )

        return output_dir, total_frames, frames_saved
    finally:
        capture.release()


def main(argv):
    if len(argv) != 2:
        print("Usage: python main.py <video_path>", file=sys.stderr)
        return 2

    try:
        output_dir, total_frames, frames_saved = extract_frames(argv[1])
    except (OSError, ValueError, RuntimeError, cv2.error) as exc:
        print(f"Video frame processing failed: {exc}", file=sys.stderr)
        return 1

    print(f"Video: {Path(argv[1]).expanduser().resolve()}")
    print(f"Readable frames: {total_frames}")
    print(f"Frames extracted: {frames_saved} (maximum {MAX_FRAMES})")
    print(f"Output folder: {output_dir}")
    print("Processing only; no autism detection or behavioral prediction was performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
