"""
SSBD video-reference accessibility audit (metadata only).

Compatible with Python 3.9.
Reads SSBD XML files and checks public YouTube metadata using yt-dlp.
Does NOT download videos, frames, thumbnails, or media.

A metadata check is not proof of permission to download, extract frames,
reuse, or redistribute a video.
"""

import argparse
import csv
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urlparse, parse_qs

try:
    import yt_dlp
except ImportError:
    yt_dlp = None


class QuietLogger:
    """Prevent yt-dlp's repeated console messages; errors are captured in CSV."""
    def debug(self, message):
        pass

    def warning(self, message):
        pass

    def error(self, message):
        pass


def clean_text(element, path, default=""):
    node = element.find(path)
    return (node.text or "").strip() if node is not None else default


def parse_duration(value):
    value = (value or "").strip().lower()
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*s?", value)
    return float(match.group(1)) if match else None


def extract_video_id(url):
    parsed = urlparse(url)
    if parsed.hostname and "youtu.be" in parsed.hostname:
        return parsed.path.strip("/").split("/")[0]
    return parse_qs(parsed.query).get("v", [""])[0]


def inspect_xml(xml_path):
    root = ET.parse(str(xml_path)).getroot()
    video_id = root.attrib.get("id", xml_path.stem)
    url = clean_text(root, "url")
    events = root.findall("./behaviours/behaviour")

    categories = sorted(set(
        clean_text(event, "category").strip().lower()
        for event in events if clean_text(event, "category")
    ))

    duration_text = clean_text(root, "duration")

    return {
        "xml_file": xml_path.name,
        "video_id": video_id,
        "url": url,
        "youtube_id": extract_video_id(url),
        "video_duration": duration_text,
        "video_duration_seconds": parse_duration(duration_text),
        "event_count": len(events),
        "categories": ";".join(categories),
    }


def classify_error(message):
    msg = (message or "").lower()

    if "private video" in msg or "video is private" in msg:
        return "PRIVATE_VIDEO"
    if any(x in msg for x in (
        "video unavailable", "has been removed", "does not exist",
        "this video is unavailable"
    )):
        return "UNAVAILABLE_OR_REMOVED"
    if any(x in msg for x in (
        "please sign in", "sign in to confirm", "confirm your age",
        "age-restricted", "login required"
    )):
        return "LOGIN_OR_AGE_RESTRICTED"
    if any(x in msg for x in (
        "page needs to be reloaded", "po token", "bot",
        "automated requests", "confirm you're not a bot",
        "captcha", "challenge"
    )):
        return "YOUTUBE_CHALLENGE_OR_BLOCK"
    if "403" in msg or "forbidden" in msg:
        return "ACCESS_BLOCKED_OR_FORBIDDEN"
    if "timed out" in msg or "timeout" in msg:
        return "TIMEOUT"
    if "429" in msg or "too many requests" in msg:
        return "RATE_LIMITED"
    return "CHECK_FAILED_OR_UNKNOWN"


def check_metadata(url, timeout, retries):
    if not url:
        return "MISSING_URL", "", "", "No URL in XML"

    if yt_dlp is None:
        return (
            "NOT_CHECKED_YTDLP_MISSING", "", "",
            "Install with: py -m pip install -U yt-dlp"
        )

    options = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
        "socket_timeout": timeout,
        "logger": QuietLogger(),
    }

    last_error = ""
    for attempt in range(retries + 1):
        try:
            with yt_dlp.YoutubeDL(options) as ydl:
                info = ydl.extract_info(url, download=False)

            if not info:
                return "UNKNOWN_NO_METADATA", "", "", "No metadata returned"

            title = info.get("title") or ""
            availability = info.get("availability") or ""
            status = "METADATA_AVAILABLE"

            if availability and availability not in ("public", "unlisted"):
                status = "RESTRICTED_OR_NONPUBLIC_METADATA"

            return status, title, availability, ""

        except Exception as exc:
            last_error = str(exc).replace("\n", " ").strip()[:500]
            status = classify_error(last_error)

            # Retry only likely temporary/challenge/rate-limit failures once.
            retryable = status in (
                "YOUTUBE_CHALLENGE_OR_BLOCK", "TIMEOUT", "RATE_LIMITED"
            )
            if attempt < retries and retryable:
                time.sleep(3 * (attempt + 1))
                continue

            return status, "", "", last_error

    return classify_error(last_error), "", "", last_error


def main():
    parser = argparse.ArgumentParser(
        description="Check SSBD video metadata without downloading media."
    )
    parser.add_argument(
        "--annotations",
        default=r"C:\ReSenseResearchData\raw\SSBD\Annotations",
    )
    parser.add_argument(
        "--output",
        default=r"C:\ReSenseResearchData\documentation\SSBD\ssbd_video_access_audit.csv",
    )
    parser.add_argument("--timeout", type=int, default=25)
    parser.add_argument("--delay", type=float, default=2.0)
    parser.add_argument(
        "--retries", type=int, default=1,
        help="Retries for temporary YouTube challenge, timeout, or rate-limit errors."
    )
    args = parser.parse_args()

    annotations_dir = Path(args.annotations)
    output_path = Path(args.output)

    if not annotations_dir.is_dir():
        print("ERROR: Annotation folder not found: {}".format(annotations_dir))
        sys.exit(1)

    xml_files = sorted(annotations_dir.rglob("*.xml"))
    if not xml_files:
        print("ERROR: No XML files found under: {}".format(annotations_dir))
        sys.exit(1)

    if yt_dlp is None:
        print("ERROR: yt-dlp is not installed.")
        print("Run: py -m pip install -U yt-dlp")
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    seen_urls = {}

    print("Found {} XML files. Starting metadata-only checks...".format(len(xml_files)))
    print("No video or frame downloads will be performed.\n")

    for index, xml_path in enumerate(xml_files, start=1):
        try:
            row = inspect_xml(xml_path)
        except Exception as exc:
            rows.append({
                "xml_file": xml_path.name,
                "video_id": xml_path.stem,
                "url": "",
                "youtube_id": "",
                "video_duration": "",
                "video_duration_seconds": "",
                "event_count": "",
                "categories": "",
                "access_status": "XML_PARSE_ERROR",
                "video_title": "",
                "platform_availability": "",
                "check_details": str(exc)[:500],
                "duplicate_url_in_xml": "",
            })
            print("[{}/{}] {}: XML_PARSE_ERROR".format(
                index, len(xml_files), xml_path.name
            ))
            continue

        url = row["url"]
        duplicate_of = seen_urls.get(url, "") if url else ""
        if url and url not in seen_urls:
            seen_urls[url] = row["video_id"]

        status, title, availability, details = check_metadata(
            url, args.timeout, args.retries
        )

        row.update({
            "access_status": status,
            "video_title": title,
            "platform_availability": availability,
            "check_details": details,
            "duplicate_url_in_xml": duplicate_of,
        })
        rows.append(row)

        print("[{}/{}] {}: {} | events={}".format(
            index, len(xml_files), row["video_id"],
            status, row["event_count"]
        ))

        if index < len(xml_files) and args.delay > 0:
            time.sleep(args.delay)

    fields = [
        "xml_file", "video_id", "url", "youtube_id",
        "video_duration", "video_duration_seconds",
        "event_count", "categories", "access_status",
        "video_title", "platform_availability",
        "check_details", "duplicate_url_in_xml",
    ]

    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    counts = {}
    for row in rows:
        status = row.get("access_status", "UNKNOWN")
        counts[status] = counts.get(status, 0) + 1

    print("\nAudit completed.")
    print("XML records checked: {}".format(len(rows)))
    print("CSV report: {}".format(output_path))
    print("\nStatus summary:")
    for status in sorted(counts):
        print("  {}: {}".format(status, counts[status]))

    print("\nInterpretation:")
    print("- METADATA_AVAILABLE means metadata could be retrieved at check time.")
    print("- YOUTUBE_CHALLENGE_OR_BLOCK is inconclusive, not proof of removal.")
    print("- LOGIN_OR_AGE_RESTRICTED is inconclusive about the exact restriction.")
    print("- No media was downloaded.")
    print("- Metadata access does not establish reuse or frame-extraction permission.")


if __name__ == "__main__":
    main()
