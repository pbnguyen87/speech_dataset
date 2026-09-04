"""Nguồn HTML tổng quát: quét các trang danh sách, nhặt link file audio.

Cách khai báo config: xem README.md cùng thư mục.
"""

import re
from urllib.parse import urljoin

import requests

from ... import targets
from ...downloader import USER_AGENT

DEFAULT_PATTERN = r"\.(mp3|m4a|wav|flac|ogg|opus|aac)(\?|$)"
HREF_RE = re.compile(r"""href=["']([^"']+)["']""", re.IGNORECASE)


def extract_links(html: str, base_url: str) -> list[str]:
    return [urljoin(base_url, m) for m in HREF_RE.findall(html)]


def audio_links(links: list[str], pattern: str) -> list[str]:
    rx = re.compile(pattern, re.IGNORECASE)
    return [u for u in links if rx.search(u)]


def _get(url: str) -> str:
    r = requests.get(url, timeout=30, headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    return r.text


def list_items(cfg: dict) -> list[dict]:
    pattern = cfg.get("link_pattern", DEFAULT_PATTERN)
    pages = []
    for u in targets.resolve(cfg, "start_url"):  # start_url | start_urls | start_urls_file
        if "{page}" in u:
            for p in range(1, cfg.get("page_count", 1) + 1):
                pages.append(u.replace("{page}", str(p)))
        else:
            pages.append(u)

    found: dict[str, dict] = {}
    for page_url in pages:
        try:
            links = extract_links(_get(page_url), page_url)
        except Exception as e:
            print(f"  lỗi trang {page_url}: {e}")
            continue
        for u in audio_links(links, pattern):
            found.setdefault(u, {"url": u, "title": u.rsplit("/", 1)[-1],
                                 "meta": {"listing_page": page_url}})
        if cfg.get("follow_detail"):
            detail = [u for u in links
                      if u not in found and u.startswith("http")
                      and not audio_links([u], pattern)]
            for du in detail[:cfg.get("detail_limit", 30)]:
                try:
                    dlinks = extract_links(_get(du), du)
                except Exception:
                    continue
                for u in audio_links(dlinks, pattern):
                    found.setdefault(u, {"url": u, "title": u.rsplit("/", 1)[-1],
                                         "meta": {"listing_page": du}})
    return list(found.values())
