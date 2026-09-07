"""CLI: python -m crawler run --config config/sources.yaml --out ./raw [--source X] [--limit N]

Output: raw/<source_name>/[<feed>/]*.mp3 + sidecar .json — đưa thẳng vào audio-pipeline:
  python -m pipeline run --raw-dir raw/<source_name> --workdir work
"""

import argparse
import itertools
import os
import shutil

import yaml

from . import downloader, sources
from .state import Ledger


def needs_probe(known_duration: float | None, max_seconds: float) -> bool:
    """Bỏ qua ffprobe khi nguồn đã khai thời lượng và nó dưới ngưỡng cắt."""
    return known_duration is None or known_duration > max_seconds


def crawl_source(cfg: dict, out_root: str, limit: int | None) -> None:
    name, stype = cfg["name"], cfg["type"]
    # dir: thư mục con dưới --out (mặc định = name); dir: "." -> ghi thẳng vào --out
    out_dir = os.path.normpath(os.path.join(out_root, cfg.get("dir") or name))
    delay = float(cfg.get("delay_s", 2.0))
    print(f"[{name}] type={stype}")

    try:
        adapter = sources.get(stype)
    except KeyError as e:
        raise SystemExit(e.args[0])

    if hasattr(adapter, "crawl"):  # adapter tự tải (youtube/yt-dlp)
        got = adapter.crawl(cfg, out_dir, delay, limit)
        print(f"[{name}] xong: {got} file mới trong {out_dir}")
        return

    # iter_items (generator, rss) -> tải ngay từng feed; list_items -> liệt kê hết rồi tải
    if hasattr(adapter, "iter_items"):
        items = adapter.iter_items(cfg)
    else:
        items = adapter.list_items(cfg)
        print(f"[{name}] {len(items)} item")
    if limit:
        items = itertools.islice(items, limit)

    max_seconds = float(cfg["max_hours"]) * 3600 if cfg.get("max_hours") else None
    if max_seconds and not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        # Không có ffmpeg thì không cắt được -> sẽ tải nguyên file (có tập dài hàng chục giờ).
        raise SystemExit(f"[{name}] max_hours={cfg['max_hours']} cần ffmpeg + ffprobe trong PATH "
                         "(apt install ffmpeg / conda install ffmpeg)")
    os.makedirs(out_dir, exist_ok=True)
    got = 0
    with Ledger(out_dir) as ledger:
        for i, item in enumerate(items):
            key = item.get("key", item["url"])  # key ổn định nếu adapter cung cấp
            if ledger.has(key):
                continue
            ext, extra = downloader.ext_from_url(item["url"]), {}
            trim = False
            if max_seconds and needs_probe(item.get("duration"), max_seconds):
                # ffprobe chỉ khi feed không khai thời lượng hoặc khai vượt ngưỡng
                info = downloader.probe(item["url"])
                if info is None:
                    # probe lỗi (mạng/timeout/ffprobe không hỗ trợ https): KHÔNG tải nguyên file,
                    # vẫn cắt bằng ffmpeg -t với đuôi đoán từ url; ffmpeg lỗi thì báo rõ ở dưới.
                    print(f"    cảnh báo: ffprobe lỗi, cắt {max_seconds:g}s bằng ffmpeg với đuôi {ext}")
                    info = {"duration": item.get("duration") or float("inf"), "ext": ext}
                if info["duration"] > max_seconds:
                    trim, ext = True, info["ext"]
                    extra = {"truncated_to_seconds": int(max_seconds)}
                    if info["duration"] != float("inf"):
                        extra["duration_original"] = round(info["duration"], 1)
            base = downloader.safe_filename(item["title"])
            subdir = item.get("subdir")  # rss: mỗi feed một thư mục con
            dest_dir = os.path.join(out_dir, subdir) if subdir else out_dir
            os.makedirs(dest_dir, exist_ok=True)
            dest = os.path.join(dest_dir, base + ext)
            if os.path.exists(dest):  # tên trùng nhưng url khác -> thêm hậu tố
                dest = os.path.join(dest_dir, f"{base}_{i:04d}" + ext)
            note = ""
            if trim:
                orig = f"{extra['duration_original'] / 3600:.1f}h" if "duration_original" in extra else "?h"
                note = f" (cắt {max_seconds / 3600:g}h / {orig})"
            print(f"  [{i + 1}] {item['title'][:60]}{note}")
            ok = (downloader.download_trimmed(item["url"], dest, max_seconds, delay_s=delay) if trim
                  else downloader.download(item["url"], dest, delay_s=delay))
            if ok:
                downloader.write_sidecar(dest, {
                    "title": item["title"], "url": item["url"],
                    "source": name, **item.get("meta", {}), **extra,
                })
                ledger.add(key, dest, name, subdir)
                got += 1
    print(f"[{name}] xong: {got} file mới trong {out_dir}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="crawler",
                                     description="Thu thập audio cho audio-pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--config", default="config/sources.yaml")
    p.add_argument("--out", required=True, help="thư mục gốc chứa audio tải về")
    p.add_argument("--source", help="chỉ chạy source có name này (mặc định: tất cả)")
    p.add_argument("--limit", type=int, help="tối đa N item mỗi source (chạy thử)")
    args = parser.parse_args(argv)

    with open(args.config, encoding="utf-8") as f:
        sources = yaml.safe_load(f)["sources"]
    config_dir = os.path.dirname(os.path.abspath(args.config))
    for s in sources:
        s["_config_dir"] = config_dir  # để *_file tính tương đối theo file config
    if args.source:
        sources = [s for s in sources if s["name"] == args.source]
        if not sources:
            raise SystemExit(f"Không có source tên '{args.source}' trong config")

    for cfg in sources:
        crawl_source(cfg, args.out, args.limit)


if __name__ == "__main__":
    main()
