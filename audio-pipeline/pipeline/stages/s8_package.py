"""s8: gán quality tier, chia train/val/test theo speaker, xuất dataset.

Output trong workdir/s8_package/:
  dataset/wav/{split}/... + metadata.csv   (kiểu HF audiofolder)
  dataset/parquet/{split}-XXXXX.parquet    (audio bytes nhúng)
  report.html                              (thống kê)

Stage này rẻ — đổi ngưỡng tier trong config chỉ cần chạy lại mình s8.
"""

import csv
import hashlib
import os
import shutil

from .. import manifest

STAGE = "s8_package"
PREV = "s7_loudnorm"

META_COLS = ["file_name", "text", "text_normalized", "speaker_id", "duration",
             "tier", "snr_db", "dnsmos", "cer", "clipping", "bandwidth_hz",
             "multi_speaker", "source_path"]


def normalize_source_path(path: str | None) -> str | None:
    """Đưa source_path về một kiểu tên: '<feed>/<tập>.mp3'.

    Chế độ lô cũ (tools/crawl_and_process.py không --stream) đặt tên staging
    'nguon__feed__tap.mp3'; chế độ stream ghi đường dẫn thật trong raw 'feed/tap.mp3'.
    Tên staging (không có '/', có '__') -> bỏ phần tên nguồn, nối các phần còn lại bằng '/'.
    """
    if not path or "/" in path or "__" not in path:
        return path
    parts = path.split("__")
    return "/".join(parts[1:]) if len(parts) > 1 else path


def assign_tier(rec: dict, tiers_cfg: dict) -> str:
    """Xét lần lượt các tier theo thứ tự khai báo; không đạt tier nào -> 'C'."""
    for name, t in tiers_cfg.items():
        cer = rec.get("cer")
        if cer is None or cer > t["max_cer"]:
            continue
        if not (t["min_seconds"] <= rec["duration"] <= t["max_seconds"]):
            continue
        if rec.get("clipping", 0) > t["max_clipping"]:
            continue
        if rec.get("multi_speaker") and not t["allow_multi_speaker"]:
            continue
        # có điểm dnsmos thì xét dnsmos, không thì xét snr
        if rec.get("dnsmos") is not None:
            if rec["dnsmos"] < t["min_dnsmos"]:
                continue
        elif rec.get("snr_db", 0) < t["min_snr_db"]:
            continue
        if not (rec.get("text") or "").strip():
            continue
        return name
    return "C"


def keep_record(rec: dict, keep_tiers: list | None) -> bool:
    """Segment được xuất vào dataset: tier != C, và nằm trong keep_tiers nếu có khai."""
    if rec["tier"] == "C":
        return False
    return keep_tiers is None or rec["tier"] in keep_tiers


def assign_split(speaker_id: str, val_ratio: float, test_ratio: float,
                 seed: int) -> str:
    """Chia split THEO SPEAKER (hash ổn định) để tránh leak giọng giữa các split."""
    h = hashlib.sha1(f"{seed}:{speaker_id}".encode()).digest()
    u = int.from_bytes(h[:8], "big") / 2**64
    if u < test_ratio:
        return "test"
    if u < test_ratio + val_ratio:
        return "val"
    return "train"


def run(cfg: dict, workdir: str, limit: int | None = None) -> str:
    pcfg = cfg["package"]
    records = manifest.read_records(manifest.manifest_path(workdir, PREV))
    out_dir = manifest.stage_dir(workdir, STAGE)
    mpath = manifest.manifest_path(workdir, STAGE)

    # gán tier + split (chạy lại từ đầu mỗi lần — stage này rẻ); chuẩn hóa source_path
    for rec in records:
        rec["source_path"] = normalize_source_path(rec.get("source_path"))
        rec["tier"] = assign_tier(rec, pcfg["tiers"])
        rec["split"] = assign_split(rec.get("speaker_id", rec["file_id"]),
                                    pcfg["split"]["val_ratio"],
                                    pcfg["split"]["test_ratio"],
                                    pcfg["split"]["seed"])

    keep_tiers = pcfg.get("keep_tiers")
    kept = [r for r in records if keep_record(r, keep_tiers)]
    tier_note = "tier " + "/".join(keep_tiers) if keep_tiers else "mọi tier trừ C"
    print(f"[{STAGE}] {len(records)} segment -> giữ {len(kept)} ({tier_note}), loại {len(records) - len(kept)}")

    # 1) wav + metadata.csv
    ds_dir = os.path.join(out_dir, "dataset")
    if os.path.exists(ds_dir):
        shutil.rmtree(ds_dir)
    meta_rows = []
    for rec in kept:
        rel = os.path.join("wav", rec["split"], os.path.basename(rec["audio_path"]))
        dst = os.path.join(ds_dir, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(rec["audio_path"], dst)
        meta_rows.append({c: rec.get(c) for c in META_COLS} | {"file_name": rel, "split": rec["split"]})

    with open(os.path.join(ds_dir, "metadata.csv"), "w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=META_COLS + ["split"])
        wtr.writeheader()
        wtr.writerows(meta_rows)

    # 2) parquet shards
    if pcfg["parquet"]["enabled"]:
        _write_parquet(kept, os.path.join(ds_dir, "parquet"),
                       pcfg["parquet"]["shard_max_mb"])

    # 3) manifest đầy đủ (kể cả tier C, để đối chiếu / đổi ngưỡng)
    if os.path.exists(mpath):
        os.remove(mpath)
    with manifest.ManifestWriter(mpath) as w:
        for rec in records:
            w.write(rec)

    # 4) report
    _write_report(records, keep_tiers, os.path.join(out_dir, "report.html"))

    # 5) stream.cleanup: xóa wav s7 của tier C (không vào dataset; tier A/B giữ vì s8 xây lại từ đó)
    if cfg.get("stream", {}).get("cleanup"):
        s7_audio = os.path.abspath(os.path.join(workdir, "s7_loudnorm", "audio"))
        n = 0
        for rec in records:
            p = os.path.abspath(rec.get("audio_path", ""))
            if rec.get("tier") == "C" and p.startswith(s7_audio + os.sep) and os.path.exists(p):
                os.remove(p)
                n += 1
        print(f"[{STAGE}] cleanup: xóa {n} wav tier C")
    print(f"[{STAGE}] xong — dataset: {ds_dir}")
    return mpath


def _write_parquet(records: list[dict], out_dir: str, shard_max_mb: int) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    os.makedirs(out_dir, exist_ok=True)
    max_bytes = shard_max_mb * 1024 * 1024

    by_split: dict[str, list[dict]] = {}
    for r in records:
        by_split.setdefault(r["split"], []).append(r)

    for split, recs in by_split.items():
        rows, size, shard = [], 0, 0

        def flush():
            nonlocal rows, size, shard
            if not rows:
                return
            table = pa.table({
                "audio": [{"bytes": r["_bytes"], "path": r["_name"]} for r in rows],
                "text": [r["text"] for r in rows],
                "text_normalized": [r["text_normalized"] for r in rows],
                "speaker_id": [r["speaker_id"] for r in rows],
                "duration": [r["duration"] for r in rows],
                "tier": [r["tier"] for r in rows],
            })
            pq.write_table(table, os.path.join(out_dir, f"{split}-{shard:05d}.parquet"))
            rows, size, shard = [], 0, shard + 1

        for r in recs:
            with open(r["audio_path"], "rb") as f:
                b = f.read()
            rows.append({
                "_bytes": b, "_name": os.path.basename(r["audio_path"]),
                "text": r.get("text"), "text_normalized": r.get("text_normalized"),
                "speaker_id": r.get("speaker_id"), "duration": r["duration"],
                "tier": r["tier"],
            })
            size += len(b)
            if size >= max_bytes:
                flush()
        flush()
        print(f"  parquet[{split}]: {shard} shard")


def _write_report(records: list[dict], keep_tiers: list | None, path: str) -> None:
    def hours(recs):
        return sum(r["duration"] for r in recs) / 3600

    tiers = sorted({r["tier"] for r in records})
    speakers = {r.get("speaker_id") for r in records}
    rows = []
    for t in tiers:
        recs = [r for r in records if r["tier"] == t]
        cers = [r["cer"] for r in recs if r.get("cer") is not None]
        snrs = [r["snr_db"] for r in recs if r.get("snr_db") is not None]
        rows.append(
            f"<tr><td>{t}</td><td>{len(recs)}</td><td>{hours(recs):.2f}</td>"
            f"<td>{(sum(cers)/len(cers)):.3f}</td>" if cers else
            f"<tr><td>{t}</td><td>{len(recs)}</td><td>{hours(recs):.2f}</td><td>-</td>"
        )
        rows[-1] += (f"<td>{(sum(snrs)/len(snrs)):.1f}</td></tr>" if snrs else "<td>-</td></tr>")

    split_rows = ""
    for sp in ("train", "val", "test"):
        recs = [r for r in records if r.get("split") == sp and keep_record(r, keep_tiers)]
        split_rows += f"<tr><td>{sp}</td><td>{len(recs)}</td><td>{hours(recs):.2f}</td></tr>"

    html = f"""<!doctype html><meta charset="utf-8"><title>Pipeline report</title>
<style>body{{font-family:sans-serif;max-width:720px;margin:2rem auto}}
table{{border-collapse:collapse;margin:1rem 0}}td,th{{border:1px solid #999;padding:4px 12px}}</style>
<h1>Báo cáo pipeline</h1>
<p>Tổng: {len(records)} segment, {hours(records):.2f} giờ, {len(speakers)} speaker.</p>
<h2>Theo tier</h2>
<table><tr><th>Tier</th><th>Segment</th><th>Giờ</th><th>CER TB</th><th>SNR TB (dB)</th></tr>{''.join(rows)}</table>
<h2>Theo split (chỉ tier được xuất: {'/'.join(keep_tiers) if keep_tiers else 'mọi tier trừ C'})</h2>
<table><tr><th>Split</th><th>Segment</th><th>Giờ</th></tr>{split_rows}</table>
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
