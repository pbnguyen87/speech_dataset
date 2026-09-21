"""s8: gán quality tier, chia train/val/test theo speaker, xuất dataset.

Output trong workdir/s8_package/:
  dataset/wav/{split}/... + metadata.csv   (kiểu HF audiofolder)
  dataset/parquet/{split}-XXXXX.parquet    (audio bytes nhúng)
  report.html                              (thống kê)

Stage này rẻ — đổi ngưỡng tier trong config chỉ cần chạy lại mình s8.

Hai cách chạy:
  run()                 : xây lại toàn bộ từ manifest s7 (pipeline run --stages s8 / cuối serve).
  package_incremental() : `python -m pipeline package` — chỉ gói segment s7 chưa có trong
                          manifest s8, nối thêm vào dataset đang có; chạy tay mỗi khi s7 có
                          thêm file. Wav s7 đã gói có thể xóa, lệnh không đọc lại chúng.
"""

import csv
import glob
import hashlib
import io
import json
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


CONFIG_HASH_FILE = "config_hash"


def config_hash(pcfg: dict) -> str:
    """Hash phần config quyết định tier/split — đổi thì dataset đang có không còn nhất quán."""
    key = {"tiers": pcfg["tiers"], "keep_tiers": pcfg.get("keep_tiers"), "split": pcfg["split"]}
    return hashlib.sha1(json.dumps(key, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]


def _write_config_hash(out_dir: str, pcfg: dict) -> None:
    with open(os.path.join(out_dir, CONFIG_HASH_FILE), "w") as f:
        f.write(config_hash(pcfg) + "\n")


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
    os.makedirs(ds_dir, exist_ok=True)  # kể cả khi không segment nào được giữ -> metadata.csv rỗng
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

    # 4) report + hash config (để `pipeline package` biết dataset được gán tier theo config nào)
    _write_report(records, keep_tiers, os.path.join(out_dir, "report.html"))
    _write_config_hash(out_dir, pcfg)

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


def _write_parquet(records: list[dict], out_dir: str, shard_max_mb: int, prefix: str = "",
                   quiet: bool = False) -> None:
    """Shard `{split}-{prefix}{NNNNN}.parquet`; ghi .tmp rồi rename để không có shard cụt."""
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
            path = os.path.join(out_dir, f"{split}-{prefix}{shard:05d}.parquet")
            pq.write_table(table, path + ".tmp")
            os.replace(path + ".tmp", path)
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
        if not quiet:
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


# ---------------------------------------------------------------- gói phần mới (chạy tay)
def _link_or_copy(src: str, dst: str, copy: bool) -> None:
    """Hardlink (không tốn đĩa, cùng filesystem) hoặc copy; đã có đích thì bỏ qua."""
    if os.path.exists(dst):
        return
    if not copy:
        try:
            os.link(src, dst)
            return
        except OSError:  # khác filesystem / fs không hỗ trợ -> copy
            pass
    shutil.copy2(src, dst)


def reconcile_csv(csv_path: str, done_ids: set) -> int:
    """Bỏ dòng metadata.csv mà segment không có trong manifest s8 (crash giữa lúc ghi csv và
    ghi manifest của một lô). Trả số dòng đã bỏ."""
    if not os.path.exists(csv_path):
        return 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        fields = rdr.fieldnames
        rows = list(rdr)
    keep = [r for r in rows if os.path.splitext(os.path.basename(r["file_name"]))[0] in done_ids]
    removed = len(rows) - len(keep)
    if removed and fields:
        tmp = csv_path + ".tmp"
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(keep)
        os.replace(tmp, csv_path)
    return removed


def _append_csv(csv_path: str, rows: list[dict]) -> None:
    """Nối các dòng bằng một lần write; tạo header nếu file chưa có / rỗng."""
    fields = META_COLS + ["split"]
    need_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields)
    if need_header:
        w.writeheader()
    w.writerows(rows)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        f.write(buf.getvalue())
        f.flush()


def package_incremental(cfg: dict, workdir: str, cleanup: bool = False, copy: bool = False,
                        dry_run: bool = False, chunk: int = 2000, drop_wav: bool = False) -> dict:
    """Gói segment s7 chưa có trong manifest s8, nối vào dataset đang có (cùng layout với run()).

    Mỗi lô `chunk` segment: gán tier/split -> link wav tier giữ vào dataset/wav/{split}/ ->
    parquet `{split}-inc-{hash8}-NNNNN.parquet` (hash8 từ id đầu lô: chạy lại sau crash ra
    đúng tên, ghi đè) -> csv một lần write -> manifest một lần write (dấu đã gói) -> xóa wav
    s7 tier C nếu cleanup. Wav của segment đã gói không được đọc lại nên xóa ở s7 được.
    drop_wav: dataset chỉ là parquet — không link wav vào dataset/wav/, và sau khi lô đã nằm
    trong parquet + manifest thì xóa wav s7 của segment đó (đĩa chỉ cần dư ~1 lô). Cần
    parquet bật; wav khôi phục lại được từ parquet (extract_audio.py --raw).
    Trả thống kê {"new", "kept", "skipped_missing", "csv_removed", "dropped"}.
    """
    pcfg = cfg["package"]
    if drop_wav and not pcfg["parquet"]["enabled"]:
        raise SystemExit(f"[{STAGE}] --drop-wav cần package.parquet.enabled: true (parquet là bản duy nhất còn lại)")
    out_dir = manifest.stage_dir(workdir, STAGE)
    mpath = manifest.manifest_path(workdir, STAGE)
    ds_dir = os.path.join(out_dir, "dataset")
    csv_path = os.path.join(ds_dir, "metadata.csv")
    keep_tiers = pcfg.get("keep_tiers")
    s7_audio = os.path.abspath(os.path.join(workdir, "s7_loudnorm", "audio"))
    stats = {"new": 0, "kept": 0, "skipped_missing": 0, "csv_removed": 0, "dropped": 0}

    hpath = os.path.join(out_dir, CONFIG_HASH_FILE)
    cur = config_hash(pcfg)
    if os.path.exists(hpath):
        old = open(hpath).read().strip()
        if old != cur:
            raise SystemExit(f"[{STAGE}] config tier/split đã đổi ({old} -> {cur}) so với dataset đang có; "
                             f"nối thêm sẽ không nhất quán. Chạy full: python -m pipeline run --stages s8")

    done = manifest.done_ids(mpath)
    w = None
    if not dry_run:
        stats["csv_removed"] = reconcile_csv(csv_path, done)
        if stats["csv_removed"]:
            print(f"[{STAGE}] bỏ {stats['csv_removed']} dòng metadata.csv không có trong manifest (lô ghi dở)")
        w = manifest.ManifestWriter(mpath)

    new = [r for r in manifest.iter_records(manifest.manifest_path(workdir, PREV)) if r["id"] not in done]
    stats["new"] = len(new)
    print(f"[{STAGE}] {len(new)} segment mới ở s7 chưa gói (đã gói: {len(done)})")
    if not new:
        if w:
            w.close()
        return stats

    for c0 in range(0, len(new), chunk):
        batch = []
        for rec in new[c0:c0 + chunk]:
            rec = dict(rec)
            rec["source_path"] = normalize_source_path(rec.get("source_path"))
            rec["tier"] = assign_tier(rec, pcfg["tiers"])
            rec["split"] = assign_split(rec.get("speaker_id", rec["file_id"]), pcfg["split"]["val_ratio"],
                                        pcfg["split"]["test_ratio"], pcfg["split"]["seed"])
            if keep_record(rec, keep_tiers) and not os.path.exists(rec["audio_path"]):
                # thiếu wav của segment sẽ vào dataset: không ghi manifest để lần sau thử lại
                print(f"  thiếu wav {rec['audio_path']} ({rec['id']}, tier {rec['tier']}) — bỏ qua lần này")
                stats["skipped_missing"] += 1
                continue
            batch.append(rec)
        kept = [r for r in batch if keep_record(r, keep_tiers)]
        stats["kept"] += len(kept)
        n_c = sum(1 for r in batch if r["tier"] == "C")
        tag = "(dry-run) " if dry_run else ""
        print(f"  {tag}lô {c0 // chunk + 1}: {len(batch)} segment, giữ {len(kept)}, tier C {n_c}")
        if dry_run or not batch:
            continue

        # 1) wav vào dataset/wav/{split}/ (cùng layout run()); drop_wav: chỉ ghi tên, không link
        rows = []
        for r in kept:
            rel = os.path.join("wav", r["split"], os.path.basename(r["audio_path"]))
            if not drop_wav:
                dst = os.path.join(ds_dir, rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                _link_or_copy(r["audio_path"], dst, copy)
            rows.append({c: r.get(c) for c in META_COLS} | {"file_name": rel, "split": r["split"]})
        # 2) parquet: tên suy từ id đầu lô -> chạy lại lô này ghi đè đúng file cũ
        if pcfg["parquet"]["enabled"] and kept:
            h8 = hashlib.sha1(batch[0]["id"].encode()).hexdigest()[:8]
            _write_parquet(kept, os.path.join(ds_dir, "parquet"), pcfg["parquet"]["shard_max_mb"],
                           prefix=f"inc-{h8}-", quiet=True)
        # 3) csv rồi 4) manifest — mỗi thứ một lần write; manifest là dấu đã gói
        os.makedirs(ds_dir, exist_ok=True)
        if rows:
            _append_csv(csv_path, rows)
        w.write_many(batch)  # dấu đã gói
        # 5) cleanup: wav s7 của tier C trong lô; drop_wav: cả wav A/B đã nằm trong parquet
        n_c = n_d = 0
        for r in batch:
            p = os.path.abspath(r.get("audio_path", ""))
            if not p.startswith(s7_audio + os.sep) or not os.path.isfile(p):
                continue
            if r["tier"] == "C" and cleanup:
                os.remove(p)
                n_c += 1
            elif r["tier"] != "C" and drop_wav and keep_record(r, keep_tiers):
                os.remove(p)
                n_d += 1
        stats["dropped"] += n_d
        if n_c or n_d:
            print(f"  xóa wav s7: {n_c} tier C" + (f", {n_d} đã vào parquet" if n_d else ""))
    if w:
        w.close()

    if not dry_run:
        _write_config_hash(out_dir, pcfg)
        _write_report(manifest.read_records(mpath), keep_tiers, os.path.join(out_dir, "report.html"))
        print(f"[{STAGE}] xong — gói thêm {stats['new'] - stats['skipped_missing']} segment "
              f"(giữ {stats['kept']}), dataset: {ds_dir}")
    return stats
