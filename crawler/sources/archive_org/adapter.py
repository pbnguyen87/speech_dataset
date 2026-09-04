"""Nguồn Internet Archive: tải mọi file audio trong một item qua metadata API.

Cách khai báo config: xem README.md cùng thư mục.
"""

import os
import re

import requests

from ... import targets
from ...downloader import AUDIO_EXTS, USER_AGENT

_URL_RE = re.compile(r"archive\.org/(?:details|download|metadata)/([^/?#]+)")


def to_identifier(s: str) -> str:
    """Nhận identifier hoặc url archive.org (details/download/metadata) -> identifier."""
    m = _URL_RE.search(s)
    return m.group(1) if m else s.strip().strip("/")


def list_items(cfg: dict) -> list[dict]:
    items = []
    # identifier | identifiers | identifiers_file — mỗi mục là identifier hoặc url;
    # bỏ trùng sau khi chuẩn hóa (cùng item ghi dạng url và dạng identifier trần)
    idents = list(dict.fromkeys(to_identifier(r) for r in targets.resolve(cfg, "identifier")))
    for ident in idents:
        try:
            items += _list_item(ident)
        except Exception as e:
            print(f"  lỗi item {ident}: {e}")
    return items


def _list_item(ident: str) -> list[dict]:
    r = requests.get(f"https://archive.org/metadata/{ident}", timeout=30,
                     headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    data = r.json()
    if not data.get("files"):  # item không tồn tại: API trả {} thay vì 404
        raise ValueError("không có file (identifier sai?)")
    server, dirpath = data["server"], data["dir"]

    items = []
    for f in data.get("files", []):
        name = f.get("name", "")
        ext = os.path.splitext(name)[1].lower()
        # 'original' để tránh tải bản derivative (đuôi _64kb.mp3...) trùng nội dung
        if ext in AUDIO_EXTS and f.get("source") == "original":
            try:
                duration = float(f["length"]) if f.get("length") else None
            except ValueError:
                duration = None
            items.append({
                "url": f"https://{server}{dirpath}/{name}",
                "duration": duration,
                # server trong url đổi theo load-balancing mỗi lần gọi API ->
                # key resume phải là định danh ổn định, không phải url
                "key": f"archive.org/{ident}/{name}",
                "title": os.path.splitext(name)[0],
                "meta": {"archive_item": ident, "size": f.get("size"),
                         "length": f.get("length")},
            })
    return items
