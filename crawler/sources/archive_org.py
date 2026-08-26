"""Nguồn Internet Archive: tải mọi file audio trong một item qua metadata API.

Config:
  - name: uyen-uong-dao
    type: archive_org
    identifier: Uynngaowww.truyenaudio.net   # phần sau /details/ trong url
"""

import os

import requests

from ..downloader import AUDIO_EXTS, USER_AGENT


def list_items(cfg: dict) -> list[dict]:
    ident = cfg["identifier"]
    r = requests.get(f"https://archive.org/metadata/{ident}", timeout=30,
                     headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    data = r.json()
    server, dirpath = data["server"], data["dir"]

    items = []
    for f in data.get("files", []):
        name = f.get("name", "")
        ext = os.path.splitext(name)[1].lower()
        # 'original' để tránh tải bản derivative (đuôi _64kb.mp3...) trùng nội dung
        if ext in AUDIO_EXTS and f.get("source") == "original":
            items.append({
                "url": f"https://{server}{dirpath}/{name}",
                # server trong url đổi theo load-balancing mỗi lần gọi API ->
                # key resume phải là định danh ổn định, không phải url
                "key": f"archive.org/{ident}/{name}",
                "title": os.path.splitext(name)[0],
                "meta": {"archive_item": ident, "size": f.get("size"),
                         "length": f.get("length")},
            })
    return items
