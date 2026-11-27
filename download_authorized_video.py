
import argparse
import hashlib
from pathlib import Path
from urllib.parse import urlparse

import requests


def download_video(url, output_dir, filename):
    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        raise ValueError("Only HTTP or HTTPS URLs are accepted.")

    if parsed.netloc.lower() in ("youtube.com", "www.youtube.com", "youtu.be"):
        raise ValueError(
            "YouTube watch URLs are not direct video-file URLs. "
            "Use an officially authorized direct video URL."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / filename

    if output_path.suffix.lower() not in {
        ".mp4", ".mov", ".avi", ".mkv", ".webm"
    }:
        raise ValueError("Unsupported video extension.")

    digest = hashlib.sha256()

    with requests.get(url, stream=True, timeout=30) as response:
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "")
        if not content_type.startswith("video/"):
            raise ValueError(
                f"URL did not return a video. Content-Type: {content_type}"
            )

        with open(output_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)
                    digest.update(chunk)

    print("Download completed:", output_path)
    print("File size:", output_path.stat().st_size, "bytes")
    print("SHA-256:", digest.hexdigest())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--filename", default="authorized_video.mp4")

    args = parser.parse_args()

    download_video(args.url, args.output_dir, args.filename)
