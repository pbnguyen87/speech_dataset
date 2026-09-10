"""Dọn đĩa hồi tố: python -m pipeline cleanup --workdir work [--raw-dir raw] [--dry-run]

Áp cùng quy tắc với `serve --cleanup` nhưng cho dữ liệu đã có sẵn (chạy khi đĩa đầy
giữa chừng, hoặc lần trước chạy không --cleanup):
  * raw audio đã qua s0 (giữ sidecar .json + ledger nên crawler không tải lại)
  * wav s0/s1 của file đã qua s2 đủ segment
  * wav s2 của segment đã qua s7 (s7 mode != off)
  * wav s7 của segment tier C theo manifest s8
Chạy được trong lúc serve đang chạy: chỉ xóa thứ đã có dấu hoàn tất trong manifest.
"""

import os

from . import manifest
from .stages.s2_segment import done_files


def _rm(path: str, dry: bool) -> int:
    if not os.path.isfile(path):
        return 0
    size = os.path.getsize(path)
    if not dry:
        os.remove(path)
    return size


def run(workdir: str, raw_dir: str | None, dry_run: bool = False) -> dict:
    gb = 1 << 30
    freed = {"raw": 0, "s0_s1": 0, "s2": 0, "tier_c": 0}

    # 1) raw đã qua s0
    if raw_dir:
        for r in manifest.iter_records(manifest.manifest_path(workdir, "s0_ingest")):
            freed["raw"] += _rm(os.path.join(raw_dir, r["source_path"]), dry_run)

    # 2) wav s0/s1 của file đã qua s2
    for fid in done_files(manifest.manifest_path(workdir, "s2_segment")):
        for st in ("s0_ingest", "s1_separate"):
            freed["s0_s1"] += _rm(os.path.join(workdir, st, "audio", f"{fid}.wav"), dry_run)

    # 3) wav s2 của segment đã qua s7 (s7 ghi audio riêng)
    s2_audio = os.path.join(workdir, "s2_segment", "audio")
    for r in manifest.iter_records(manifest.manifest_path(workdir, "s7_loudnorm")):
        if r.get("loudnorm") != "off":
            freed["s2"] += _rm(os.path.join(s2_audio, os.path.basename(r["audio_path"])), dry_run)

    # 4) wav s7 của tier C
    s7_audio = os.path.abspath(os.path.join(workdir, "s7_loudnorm", "audio"))
    for r in manifest.iter_records(manifest.manifest_path(workdir, "s8_package")):
        p = os.path.abspath(r.get("audio_path", ""))
        if r.get("tier") == "C" and p.startswith(s7_audio + os.sep):
            freed["tier_c"] += _rm(p, dry_run)

    tag = "sẽ giải phóng" if dry_run else "đã giải phóng"
    for k, v in freed.items():
        print(f"  {k:7s}: {v / gb:8.2f} GB")
    print(f"[cleanup] {tag} {sum(freed.values()) / gb:.2f} GB")
    return freed
