"""CLI: python -m pipeline run   --raw-dir ./raw --workdir ./work [--stages s0-s8]   # tuần tự theo lô
     python -m pipeline serve --raw-dir ./raw --workdir ./work [--input-done]      # 8 stage song song
     python -m pipeline status --workdir ./work

Mỗi stage được chạy trong một SUBPROCESS riêng (qua pipeline.stage_runner):
torch (s1/s4/s5-verify) và ctranslate2 (s5-primary) cùng nhúng OpenMP runtime,
load chung một process gây segfault/deadlock trên macOS Intel.
"""

import argparse
import json
import subprocess
import sys
import tempfile

import yaml

STAGES = ["s0_ingest", "s1_separate", "s2_segment", "s3_quality",
          "s4_speaker", "s5_transcribe", "s6_textnorm", "s7_loudnorm",
          "s8_package"]


def parse_stages(spec: str) -> list[str]:
    """'all' | 's0,s2,s5' | 's2-s6'"""
    if spec == "all":
        return STAGES
    short = {s.split("_")[0]: s for s in STAGES}
    if "-" in spec and "," not in spec:
        a, b = spec.split("-")
        ia, ib = STAGES.index(short[a]), STAGES.index(short[b])
        return STAGES[ia:ib + 1]
    return [short[s.strip()] for s in spec.split(",")]


def deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(paths: list[str]) -> dict:
    cfg: dict = {}
    for p in paths:
        with open(p, encoding="utf-8") as f:
            cfg = deep_merge(cfg, yaml.safe_load(f) or {})
    return cfg


def main(argv=None):
    parser = argparse.ArgumentParser(prog="pipeline",
                                     description="Pipeline làm sạch audio crawl -> dữ liệu training")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="chạy các stage")
    p.add_argument("--config", action="append", default=None,
                   help="file yaml config (lặp lại được, file sau override file trước; mặc định config/default.yaml)")
    p.add_argument("--raw-dir", help="thư mục audio thô (bắt buộc khi chạy s0)")
    p.add_argument("--workdir", required=True, help="thư mục làm việc chứa output các stage")
    p.add_argument("--stages", default="all", help='"all" | "s0,s2" | "s2-s6" (mặc định all)')
    p.add_argument("--limit", type=int, default=None, help="chỉ xử lý N item đầu (chạy thử)")
    p.add_argument("--device", default=None, help="override device: cpu|cuda|mps")

    p = sub.add_parser("serve", help="8 stage chạy song song, mỗi stage một tiến trình (xem pipeline.stream)")
    p.add_argument("--config", action="append", default=None)
    p.add_argument("--raw-dir", required=True, help="thư mục audio thô; quét lặp lại để nhận file mới")
    p.add_argument("--workdir", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--input-done", action="store_true",
                   help="raw không có thêm file mới (không chờ INPUT_DONE); mặc định chờ crawler tạo <workdir>/INPUT_DONE")
    p.add_argument("--cleanup", action="store_true",
                   help="xóa audio raw + wav trung gian ngay khi stage cuối dùng xong; sau s8 xóa wav tier C")

    p = sub.add_parser("status", help="số dòng manifest + cờ DONE của từng stage")
    p.add_argument("--workdir", required=True)

    args = parser.parse_args(argv)

    import os
    if args.cmd == "status":
        from .stream import status_line
        print(status_line(args.workdir))
        return

    default_cfg = os.path.join(os.path.dirname(__file__), "..", "config", "default.yaml")
    cfg = load_config(args.config or [default_cfg])
    if args.device:
        cfg["device"] = args.device
    if args.raw_dir:
        cfg["raw_dir"] = args.raw_dir

    if args.cmd == "serve":
        from .stream import serve
        st = cfg.get("stream", {})
        cfg["stream"] = {**st, "cleanup": bool(st.get("cleanup")) or args.cleanup}
        sys.exit(serve(cfg, args.workdir, input_done=args.input_done))

    stages = parse_stages(args.stages)
    if "s0_ingest" in stages and "raw_dir" not in cfg:
        sys.exit("Cần --raw-dir khi chạy s0_ingest")

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(cfg, f, ensure_ascii=False)
        cfg_path = f.name

    limit_s = "-" if args.limit is None else str(args.limit)
    for name in stages:
        proc = subprocess.run(
            [sys.executable, "-m", "pipeline.stage_runner",
             name, cfg_path, args.workdir, limit_s],
        )
        if proc.returncode != 0:
            sys.exit(f"Stage {name} lỗi (exit {proc.returncode}) — dừng pipeline. "
                     f"Chạy lại sẽ resume từ chỗ dở.")

    print("\nHoàn tất các stage:", ", ".join(stages))


if __name__ == "__main__":
    main()
