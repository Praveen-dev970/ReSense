"""
SSBD video-reference accessibility audit (metadata only).

This script:
- Reads the original SSBD XML annotation files.
- Extracts video IDs, URLs, annotated categories, and event counts.
- Checks whether public video metadata can currently be retrieved.
- Does NOT download videos, frames, thumbnails, or media.

Important:
A successful metadata lookup does NOT grant permission to download or process a video.
Always check the original hosting platform's terms and the applicable rights/permissions.
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


def clean_text(element, path, default=""):
    node = element.find(path)
    return (node.text or "").strip() if node is not None else default


def parse_duration(value):
    """Convert values such as '46s' to seconds where possible."""
    value = (value or "").strip().lower()
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*s?", value)
    return float(match.group(1)) if match else None


def extract_video_id(url):
    parsed = urlparse(url)
    if parsed.hostname and "youtu.be" in parsed.hostname:
        return parsed.path.strip("/").split("/")[0]
    return parse_qs(parsed.query).get("v", [""])[0]


def inspect_xml(xml_path):
    root = ET.parse(xml_path).getroot()

    video_id = root.attrib.get("id", xml_path.stem)
    url = clean_text(root, "url")
    duration_text = clean_text(root, "duration")
    duration_seconds = parse_duration(duration_text)

    events = root.findall("./behaviours/behaviour")
    categories = sorted({
        clean_text(event, "category").strip().lower()
        for event in events
        if clean_text(event, "category")
    })

    return {
        "xml_file": xml_path.name,
        "video_id": video_id,
        "url": url,
        "youtube_id": extract_video_id(url),
        "video_duration": duration_text,
        "video_duration_seconds": duration_seconds,
        "event_count": len(events),
        "categories": ";".join(categories),
    }


def check_metadata(url, timeout):
    """Use yt-dlp metadata extraction only; never download media."""
    if not url:
        return "MISSING_URL", "", ""

    if yt_dlp is None:
        return "NOT_CHECKED_YTDLP_MISSING", "", (
            "Install yt-dlp: py -m pip install -U yt-dlp"
        )

    options = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "extract_flat": False,
        "socket_timeout": timeout,
        "noplaylist": True,
    }

    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=False)

        if not info:
            return "UNKNOWN_NO_METADATA", "", "No metadata returned"

        title = info.get("title") or ""
        availability = info.get("availability") or ""
        status = "METADATA_AVAILABLE"
        if availability and availability not in ("public", "unlisted"):
            status = "RESTRICTED_OR_NONPUBLIC_METADATA"

        return status, title, availability

    except Exception as exc:
        message = str(exc).replace("\n", " ").strip()
        # Keep the report readable and avoid storing huge exception strings.
        message = message[:300]
        lowered = message.lower()

        if any(term in lowered for term in ("private video", "video is private")):
            status = "PRIVATE_OR_UNAVAILABLE"
        elif any(term in lowered for term in ("video unavailable", "has been removed", "does not exist")):
            status = "UNAVAILABLE_OR_REMOVED"
        elif any(term in lowered for term in ("sign in", "age-restricted", "confirm your age")):
            status = "LOGIN_OR_AGE_RESTRICTED"
        elif any(term in lowered for term in ("http error 403", "forbidden")):
            status = "ACCESS_BLOCKED_OR_FORBIDDEN"
        elif "timed out" in lowered or "timeout" in lowered:
            status = "TIMEOUT"
        else:
            status = "CHECK_FAILED_OR_UNKNOWN"

        return status, "", message


def main():
    parser = argparse.ArgumentParser(
        description="Audit SSBD XML references and YouTube metadata without downloading videos."
    )
    parser.add_argument(
        "--annotations",
        default=r"C:\ReSenseResearchData\raw\SSBD\Annotations",
        help="Folder containing SSBD XML annotation files.",
    )
    parser.add_argument(
        "--output",
        default=r"C:\ReSenseResearchData\documentation\SSBD\ssbd_video_access_audit.csv",
        help="Output CSV path.",
    )
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Pause between metadata checks, in seconds.")
    args = parser.parse_args()

    annotations_dir = Path(args.annotations)
    output_path = Path(args.output)

    if not annotations_dir.is_dir():
        print(f"ERROR: Annotation folder not found: {annotations_dir}")
        sys.exit(1)

    xml_files = sorted(annotations_dir.rglob("*.xml"))
    if not xml_files:
        print(f"ERROR: No XML files found under: {annotations_dir}")
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    seen_urls = {}

    for index, xml_path in enumerate(xml_files, start=1):
        try:
            row = inspect_xml(xml_path)
        except Exception as exc:
            row = {
                "xml_file": xml_path.name,
                "video_id": xml_path.stem,
                "url": "",
                "youtube_id": "",
                "video_duration": "",
                "video_duration_seconds": "",
                "event_count": 0,
                "categories": "",
                "access_status": "XML_PARSE_ERROR",
                "video_title": "",
                "platform_availability": "",
                "check_details": str(exc)[:300],
                "duplicate_url_in_xml": "",
            }
            rows.append(row)
            print(f"[{index}/{len(xml_files)}] XML ERROR: {xml_path.name}")
            continue

        url = row["url"]
        duplicate_of = seen_urls.get(url, "") if url else ""
        if url and url not in seen_urls:
            seen_urls[url] = row["video_id"]

        status, title, availability = check_metadata(url, args.timeout)
        row.update({
            "access_status": status,
            "video_title": title,
            "platform_availability": availability,
            "check_details": "",
            "duplicate_url_in_xml": duplicate_of,
        })
        rows.append(row)

        print(
            f"[{index}/{len(xml_files)}] {row['video_id']}: "
            f"{status} | events={row['event_count']}"
        )

        if index < len(xml_files) and args.delay > 0:
            time.sleep(args.delay)

    fieldnames = [
        "xml_file", "video_id", "url", "youtube_id",
        "video_duration", "video_duration_seconds",
        "event_count", "categories", "access_status",
        "video_title", "platform_availability",
        "check_details", "duplicate_url_in_xml",
    ]

    with output_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\nAudit completed.")
    print(f"XML records: {len(rows)}")
    print(f"CSV report: {output_path}")
    print("\nReminder: metadata availability is not proof of permission to download,")
    print("extract frames, reuse, or redistribute the underlying video.")


if __name__ == "__main__":
    main()
