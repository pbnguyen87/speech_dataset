"""CLI: python -m crawler run --config config/sources.yaml --out ./raw [--source X] [--limit N]

Output: raw/<source_name>/*.mp3 + sidecar .json — đưa thẳng vào audio-pipeline:
  python -m pipeline run --raw-dir raw/<source_name> --workdir work
"""

import argparse
import os

import yaml

from . import downloader
from .state import Ledger
from .sources import archive_org, html_listing, rss

LIST_SOURCES = {"rss": rss, "archive_org": archive_org, "html": html_listing}


def crawl_source(cfg: dict, out_root: str, limit: int | None) -> None:
    name, stype = cfg["name"], cfg["type"]
    out_dir = os.path.join(out_root, name)
    delay = float(cfg.get("delay_s", 2.0))
    print(f"[{name}] type={stype}")

    if stype not in LIST_SOURCES:
        raise SystemExit(f"Không hỗ trợ type '{stype}' (rss|archive_org|html)")

    items = LIST_SOURCES[stype].list_items(cfg)
    if limit:
        items = items[:limit]
    print(f"[{name}] {len(items)} item")

    os.makedirs(out_dir, exist_ok=True)
    got = 0
    with Ledger(out_dir) as ledger:
        for i, item in enumerate(items):
            key = item.get("key", item["url"])  # key ổn định nếu adapter cung cấp
            if ledger.has(key):
                continue
            base = downloader.safe_filename(item["title"])
            dest = os.path.join(out_dir, base + downloader.ext_from_url(item["url"]))
            if os.path.exists(dest):  # tên trùng nhưng url khác -> thêm hậu tố
                dest = os.path.join(
                    out_dir, f"{base}_{i:04d}" + downloader.ext_from_url(item["url"]))
            print(f"  [{i + 1}/{len(items)}] {item['title'][:60]}")
            if downloader.download(item["url"], dest, delay_s=delay):
                downloader.write_sidecar(dest, {
                    "title": item["title"], "url": item["url"],
                    "source": name, **item.get("meta", {}),
                })
                ledger.add(key, dest, name)
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
    if args.source:
        sources = [s for s in sources if s["name"] == args.source]
        if not sources:
            raise SystemExit(f"Không có source tên '{args.source}' trong config")

    for cfg in sources:
        crawl_source(cfg, args.out, args.limit)


if __name__ == "__main__":
    main()
