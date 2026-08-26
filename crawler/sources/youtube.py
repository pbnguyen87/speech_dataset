"""Nguồn YouTube (channel/playlist/video) qua yt-dlp — cần `pip install yt-dlp`.

Không đi qua downloader chung: yt-dlp tự quản lý tải, resume
(--download-archive) và ghi metadata (--write-info-json).

Config:
  - name: kenh-sach-noi-x
    type: youtube
    url: https://www.youtube.com/@channel/videos   # hoặc playlist/video
    max_items: 50          # tùy chọn
    audio_format: m4a      # giữ m4a gốc (không re-encode); pipeline đọc được
"""

import os
import subprocess
import sys


def crawl(cfg: dict, out_dir: str, delay_s: float, limit: int | None) -> int:
    os.makedirs(out_dir, exist_ok=True)
    n = limit or cfg.get("max_items")
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-f", "bestaudio",
        "--extract-audio", "--audio-format", cfg.get("audio_format", "m4a"),
        "--write-info-json",
        "--download-archive", os.path.join(out_dir, "ytdlp_archive.txt"),
        "--sleep-interval", str(delay_s),
        "-o", os.path.join(out_dir, "%(id)s.%(ext)s"),
        "--no-warnings", "--ignore-errors",
    ]
    if n:
        cmd += ["--playlist-end", str(n)]
    cmd.append(cfg["url"])

    print(f"  yt-dlp: {cfg['url']}")
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        print(f"  yt-dlp kết thúc với mã {proc.returncode} (một số video có thể lỗi)")
    return sum(1 for f in os.listdir(out_dir)
               if os.path.splitext(f)[1] in (".m4a", ".mp3", ".opus"))
