"""Nguồn RSS podcast: mỗi episode là một <enclosure> trỏ tới file audio.

Config:
  - name: ten-podcast
    type: rss
    url: https://feeds.example.com/podcast.xml
    max_items: 100        # tùy chọn, bỏ = lấy hết
"""

import xml.etree.ElementTree as ET

import requests

from ..downloader import USER_AGENT


def parse_feed(xml_text: str) -> list[dict]:
    """Trả [{'title', 'audio_url', 'pub_date', 'description'}] từ RSS xml."""
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
        })
    return items


def list_items(cfg: dict) -> list[dict]:
    """Trả danh sách item chuẩn: {'url', 'title', 'meta'}."""
    r = requests.get(cfg["url"], timeout=30, headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    episodes = parse_feed(r.text)
    if cfg.get("max_items"):
        episodes = episodes[:cfg["max_items"]]
    return [{
        "url": e["audio_url"],
        "title": e["title"],
        "meta": {"pub_date": e["pub_date"], "description": e["description"],
                 "feed": cfg["url"]},
    } for e in episodes]
