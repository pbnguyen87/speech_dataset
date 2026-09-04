"""Tải file audio lịch sự: retry + backoff, rate-limit, sidecar metadata.

Mỗi file audio tải xong được ghi kèm `<tên file>.json` (sidecar) chứa title,
url nguồn, tên source... — stage s0 của audio-pipeline tự đọc sidecar này và
giữ metadata đi suốt pipeline.
"""

import json
import os
import re
import shutil
import subprocess
import time
from urllib.parse import urlparse

import requests

USER_AGENT = "audio-crawler/0.1 (research dataset collection)"
AUDIO_EXTS = {".mp3", ".m4a", ".wav", ".flac", ".ogg", ".opus", ".aac", ".wma"}
_UNSAFE = re.compile(r"[^\w\-.]+", re.UNICODE)

# format_name của ffprobe -> đuôi file; muxer ffmpeg cho từng đuôi (dùng khi -c copy)
_FORMAT_EXT = {"mp3": ".mp3", "mov": ".m4a", "mp4": ".m4a", "m4a": ".m4a", "ogg": ".ogg",
               "opus": ".opus", "flac": ".flac", "wav": ".wav", "aac": ".aac", "asf": ".wma"}
_MUXER = {".mp3": "mp3", ".m4a": "ipod", ".ogg": "ogg", ".opus": "opus", ".flac": "flac",
          ".wav": "wav", ".aac": "adts", ".wma": "asf"}


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


# ---------------------------------------------------------------- cắt file dài
def ext_from_format(format_name: str, default: str = ".mp3") -> str:
    """'mov,mp4,m4a,3gp,3g2,mj2' -> '.m4a'; 'mp3' -> '.mp3'."""
    for name in (format_name or "").split(","):
        if name.strip() in _FORMAT_EXT:
            return _FORMAT_EXT[name.strip()]
    return default


def probe(url: str, timeout: int = 120) -> dict | None:
    """ffprobe qua HTTP: {'duration': giây, 'ext': '.m4a'} — chỉ đọc header, không tải cả file.
    Trả None nếu không có ffprobe hoặc probe lỗi."""
    if not shutil.which("ffprobe"):
        return None
    cmd = ["ffprobe", "-v", "error"]
    if url.startswith(("http://", "https://")):
        cmd += ["-user_agent", USER_AGENT]
    cmd += ["-show_entries", "format=duration,format_name", "-of", "json", url]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
        fmt = json.loads(out)["format"]
        return {"duration": float(fmt["duration"]), "ext": ext_from_format(fmt.get("format_name"))}
    except Exception:
        return None


def trim_cmd(url: str, dest: str, max_seconds: float) -> list[str]:
    """ffmpeg đọc thẳng URL, dừng sau max_seconds, stream copy (không re-encode).
    Header ở đầu file (mp3, m4a faststart) -> chỉ tải phần cần."""
    final = dest[:-5] if dest.endswith(".part") else dest
    ext = os.path.splitext(final)[1].lower()
    cmd = ["ffmpeg", "-y", "-v", "error", "-nostdin"]
    if url.startswith(("http://", "https://")):  # tùy chọn giao thức HTTP, file cục bộ không nhận
        cmd += ["-user_agent", USER_AGENT, "-reconnect", "1", "-reconnect_streamed", "1",
                "-reconnect_delay_max", "10"]
    return cmd + ["-i", url, "-t", str(int(max_seconds)), "-map", "0:a:0", "-c", "copy",
                  "-f", _MUXER.get(ext, "mp3"), dest]


def download_trimmed(url: str, dest: str, max_seconds: float, delay_s: float = 2.0,
                     retries: int = 3) -> bool:
    """Tải tối đa max_seconds đầu của url -> dest bằng ffmpeg. Cùng hợp đồng với download()."""
    tmp = dest + ".part"
    ok = False
    for attempt in range(1, retries + 1):
        try:
            subprocess.run(trim_cmd(url, tmp, max_seconds), check=True,
                           capture_output=True, text=True)
            os.replace(tmp, dest)
            ok = True
            break
        except subprocess.CalledProcessError as e:
            print(f"    ffmpeg lỗi lần {attempt}/{retries}: {e.stderr.strip()[-200:]}")
        except Exception as e:
            print(f"    lỗi lần {attempt}/{retries}: {e}")
        if os.path.exists(tmp):
            os.remove(tmp)
        time.sleep(2 ** attempt)
    time.sleep(delay_s)
    return ok
