"""Nguồn RSS podcast: mỗi episode là một <enclosure> trỏ tới file audio.

Cách khai báo config: xem README.md cùng thư mục.
"""

import xml.etree.ElementTree as ET
from urllib.parse import urlparse

import requests

from ... import targets
from ...downloader import USER_AGENT, safe_filename


NS = {"itunes": "http://www.itunes.com/dtds/podcast-1.0.dtd"}


def parse_duration(text: str | None) -> float | None:
    """'HH:MM:SS' | 'MM:SS' | '1234' (giây) -> giây; không hợp lệ -> None."""
    if not text:
        return None
    try:
        parts = [float(x) for x in text.strip().split(":")]
        return sum(v * 60 ** i for i, v in enumerate(reversed(parts)))
    except ValueError:
        return None


def parse_feed(xml_text: str) -> list[dict]:
    """Trả [{'title', 'audio_url', 'pub_date', 'description', 'duration'}] từ RSS xml."""
    root = ET.fromstring(xml_text)
    items = []
    for item in root.iter("item"):
        enc = item.find("enclosure")
        if enc is None:
            continue
        url = enc.get("url")
        if not url:
            continue
        items.append({
            "title": (item.findtext("title") or "").strip(),
            "audio_url": url,
            "pub_date": (item.findtext("pubDate") or "").strip(),
            "description": (item.findtext("description") or "").strip()[:500],
            "duration": parse_duration(item.findtext("itunes:duration", namespaces=NS)),
        })
    return items


def feed_slug(feed_url: str) -> str:
    """URL feed -> tên thư mục con ổn định: 'vnexpress.net_rss_podcast_ban-on-khong.rss'."""
    u = urlparse(feed_url)
    return safe_filename(u.netloc + u.path.rstrip("/"), max_len=80) or "feed"


def list_items(cfg: dict) -> list[dict]:
    """Trả danh sách item chuẩn: {'url', 'title', 'subdir', 'meta'}. max_items áp cho từng feed.
    subdir = feed_slug(feed): cli xếp audio vào raw/<source>/<subdir>/ (mỗi feed một thư mục)."""
    items = []
    for feed_url in targets.resolve(cfg, "url"):  # url | urls | urls_file
        try:
            r = requests.get(feed_url, timeout=30, headers={"User-Agent": USER_AGENT})
            r.raise_for_status()
            episodes = parse_feed(r.text)
        except Exception as e:
            print(f"  lỗi feed {feed_url}: {e}")
            continue
        if cfg.get("max_items"):
            episodes = episodes[:cfg["max_items"]]
        items += [{
            "url": e["audio_url"],
            "title": e["title"],
            "subdir": feed_slug(feed_url),
            "duration": e["duration"],  # giây theo feed khai báo; cli dùng để bỏ qua ffprobe
            "meta": {"pub_date": e["pub_date"], "description": e["description"],
                     "duration_feed": e["duration"], "feed": feed_url},
        } for e in episodes]
    return items
