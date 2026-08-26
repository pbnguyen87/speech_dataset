"""Runner chạy MỘT stage — được cli.py gọi qua subprocess.

Mỗi stage chạy trong process riêng để cách ly runtime: torch (s1/s4) và
ctranslate2 (s5) cùng nhúng OpenMP, load chung một process gây segfault/deadlock
trên macOS Intel.

Usage: python -m pipeline.stage_runner <stage_name> <cfg.json> <workdir> <limit|->
"""

import importlib
import json
import sys


def main() -> None:
    stage_name, cfg_path, workdir, limit_s = sys.argv[1:5]
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    limit = None if limit_s == "-" else int(limit_s)
    mod = importlib.import_module(f"pipeline.stages.{stage_name}")
    mod.run(cfg, workdir, limit)


if __name__ == "__main__":
    main()
