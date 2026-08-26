"""Tải file audio lịch sự: retry + backoff, rate-limit, sidecar metadata.

Mỗi file audio tải xong được ghi kèm `<tên file>.json` (sidecar) chứa title,
url nguồn, tên source... — stage s0 của audio-pipeline tự đọc sidecar này và
giữ metadata đi suốt pipeline.
"""

import json
import os
import re
import time
from urllib.parse import urlparse

import requests

USER_AGENT = "audio-crawler/0.1 (research dataset collection)"
AUDIO_EXTS = {".mp3", ".m4a", ".wav", ".flac", ".ogg", ".opus", ".aac", ".wma"}
_UNSAFE = re.compile(r"[^\w\-.]+", re.UNICODE)


def safe_filename(name: str, max_len: int = 120) -> str:
    """Chuỗi bất kỳ -> tên file an toàn, giữ chữ có dấu tiếng Việt."""
    name = _UNSAFE.sub("_", name).strip("_.")
    return name[:max_len] or "untitled"


def ext_from_url(url: str, default: str = ".mp3") -> str:
    ext = os.path.splitext(urlparse(url).path)[1].lower()
    return ext if ext in AUDIO_EXTS else default


def download(url: str, dest: str, delay_s: float = 2.0, retries: int = 3,
             timeout: int = 60) -> bool:
    """Tải url -> dest (stream). Trả True nếu thành công. Luôn sleep delay_s
    sau mỗi lượt tải (kể cả lỗi) để lịch sự với server."""
    tmp = dest + ".part"
    ok = False
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=timeout,
                              headers={"User-Agent": USER_AGENT}) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 16):
                        f.write(chunk)
            os.replace(tmp, dest)
            ok = True
            break
        except Exception as e:
            print(f"    lỗi lần {attempt}/{retries}: {e}")
            if os.path.exists(tmp):
                os.remove(tmp)
            time.sleep(2 ** attempt)
    time.sleep(delay_s)
    return ok


def write_sidecar(audio_path: str, meta: dict) -> None:
    path = os.path.splitext(audio_path)[0] + ".json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
