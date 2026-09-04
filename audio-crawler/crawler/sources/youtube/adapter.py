"""Nguồn YouTube (channel / playlist / video) qua yt-dlp.

Khác các nguồn còn lại, adapter này không trả `list_items` mà tự tải bằng
`crawl()`: yt-dlp lo tải, resume (`ytdlp_archive.txt`) và metadata
(`--write-info-json`). Sau khi yt-dlp xong, `.info.json` được chuyển thành
sidecar `.json` cùng format với các nguồn khác và ghi vào ledger.

Cần: pip install yt-dlp, ffmpeg trong PATH.
Cách khai báo config: xem README.md cùng thư mục.
"""

import importlib.util
import json
import os
import subprocess
import sys

from ... import targets
from ...downloader import AUDIO_EXTS, write_sidecar
from ...state import Ledger

ARCHIVE_FILE = "ytdlp_archive.txt"


def info_to_sidecar(info: dict, source: str) -> dict:
    """Rút gọn info.json của yt-dlp thành sidecar {title, url, source, ...}."""
    return {
        "title": info.get("title") or info.get("id", ""),
        "url": info.get("webpage_url") or f"https://www.youtube.com/watch?v={info.get('id')}",
        "source": source,
        "youtube_id": info.get("id"),
        "channel": info.get("channel") or info.get("uploader"),
        "channel_id": info.get("channel_id"),
        "upload_date": info.get("upload_date"),
        "duration": info.get("duration"),
        "language": info.get("language"),
        "description": (info.get("description") or "")[:500],
    }


def build_cmd(cfg: dict, out_dir: str, delay_s: float, limit: int | None) -> list[str]:
    fmt = cfg.get("audio_format", "m4a")
    cmd = [
        sys.executable, "-m", "yt_dlp",
        # ưu tiên stream sẵn đúng định dạng để ffmpeg chỉ remux, không re-encode
        "-f", f"bestaudio[ext={fmt}]/bestaudio",
        "--extract-audio", "--audio-format", fmt,
        "--write-info-json", "--no-write-playlist-metafiles",
        "--download-archive", os.path.join(out_dir, ARCHIVE_FILE),
        "--sleep-interval", str(delay_s),
        "-o", os.path.join(out_dir, "%(id)s.%(ext)s"),
        "--no-warnings", "--ignore-errors", "--no-overwrites",
    ]
    n = limit or cfg.get("max_items")
    if n:
        cmd += ["--playlist-end", str(n)]
    if cfg.get("max_hours"):  # cắt video quá dài (cần ffmpeg)
        cmd += ["--download-sections", f"*0-{int(float(cfg['max_hours']) * 3600)}", "--force-keyframes-at-cuts"]
    if cfg.get("match_filter"):
        cmd += ["--match-filter", cfg["match_filter"]]
    for extra in cfg.get("extra_args", []):
        cmd.append(str(extra))
    cmd += targets.resolve(cfg, "url")  # url | urls | urls_file
    return cmd


def finalize(out_dir: str, source: str) -> int:
    """Chuyển mọi <id>.info.json có audio đi kèm thành sidecar + ledger.
    Trả số file mới được ghi nhận."""
    got = 0
    with Ledger(out_dir) as ledger:
        for fn in sorted(os.listdir(out_dir)):
            if not fn.endswith(".info.json"):
                continue
            vid = fn[: -len(".info.json")]
            audio = next((os.path.join(out_dir, vid + ext) for ext in AUDIO_EXTS
                          if os.path.exists(os.path.join(out_dir, vid + ext))), None)
            info_path = os.path.join(out_dir, fn)
            if audio is None:  # yt-dlp lỗi giữa chừng, để lần sau tải tiếp
                continue
            with open(info_path, encoding="utf-8") as f:
                info = json.load(f)
            key = f"youtube/{info.get('id', vid)}"
            if not ledger.has(key):
                write_sidecar(audio, info_to_sidecar(info, source))
                ledger.add(key, audio, source)
                got += 1
            os.remove(info_path)
    return got


def crawl(cfg: dict, out_dir: str, delay_s: float, limit: int | None) -> int:
    if importlib.util.find_spec("yt_dlp") is None:
        raise SystemExit("Nguồn youtube cần yt-dlp: pip install yt-dlp (và ffmpeg trong PATH)")
    os.makedirs(out_dir, exist_ok=True)
    cmd = build_cmd(cfg, out_dir, delay_s, limit)
    urls = targets.resolve(cfg, "url")
    print(f"  yt-dlp: {len(urls)} url" + (f" ({urls[0]})" if len(urls) == 1 else ""))
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        print(f"  yt-dlp kết thúc với mã {proc.returncode} (một số video có thể lỗi)")
    return finalize(out_dir, cfg["name"])
