
import json
import sys

from main import extract_frames
from video_cnn_inference import predict_video_folder


def main():
    if len(sys.argv) != 2:
        print("Usage: process_video.py <video_path>", file=sys.stderr)
        return 1

    video_path = sys.argv[1]

    try:
        output_dir, total_frames, frames_saved = extract_frames(
            video_path
        )

        prediction = predict_video_folder(output_dir)

        result = {
            "status": "success",
            "total_frames": total_frames,
            "frames_saved": frames_saved,
            "prediction": prediction,
        }

        print(json.dumps(result))
        return 0

    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
