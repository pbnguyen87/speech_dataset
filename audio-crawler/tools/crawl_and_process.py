"""Crawl và xử lý nối tiếp: mỗi file vừa tải xong được đưa ngay vào audio-pipeline.

    python tools/crawl_and_process.py --config config/sources.yaml --out ./raw \\
        --workdir ../work [--source X] [--limit N] [--batch 1] [--pipeline-dir ../audio-pipeline]

Cơ chế:
  * Crawler chạy như tiến trình con. Mỗi file tải xong được ghi một dòng vào
    `<out>/<source>/ledger.jsonl` (append + flush) — script tail các ledger này
    làm nguồn sự kiện, không cần hook trong crawler.
  * File mới gom thành lô (`--batch`, mặc định 1 = xử lý ngay từng file), symlink
    vào `<workdir>/_staging/<lô>/` rồi gọi
    `python -m pipeline run --raw-dir <staging> --workdir <workdir> --stages s0-s7`.
    Pipeline resume theo hash nên gọi lặp lại là an toàn.
  * Crawler xong và hàng đợi cạn -> chạy `--final-stages` (mặc định s8: gán tier,
    chia split, xuất dataset) một lần trên toàn bộ workdir.
  * `<workdir>/processed_files.jsonl` ghi file đã qua pipeline; chạy lại script
    bỏ qua phần đã làm. `--no-crawl` chỉ xử lý ledger có sẵn, không crawl.
  * `--cleanup`: tiết kiệm đĩa — sau mỗi lô xóa audio gốc trong raw/ (giữ sidecar
    + ledger) và wav trung gian s0/s1/s2 của lô; sau s8 xóa wav s7 của segment
    tier C. Wav s7 của tier A/B phải giữ vì s8 xây lại dataset từ đó mỗi lần chạy.
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

CRAWLER_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ------------------------------------------------------------------ ledger tail
class LedgerTail:
    """Theo dõi mọi <out>/*/ledger.jsonl, trả các record mới kể từ lần đọc trước."""

    def __init__(self, out_root: str):
        self.out_root = out_root
        self.offsets: dict[str, int] = {}

    def poll(self) -> list[dict]:
        new = []
        for path in sorted(glob.glob(os.path.join(self.out_root, "*", "ledger.jsonl"))):
            with open(path, encoding="utf-8") as f:
                f.seek(self.offsets.get(path, 0))
                for line in f:
                    if not line.endswith("\n"):  # dòng đang ghi dở, đọc ở vòng sau
                        break
                    self.offsets[path] = self.offsets.get(path, 0) + len(line.encode("utf-8"))
                    line = line.strip()
                    if line:
                        rec = json.loads(line)
                        rec["path"] = os.path.abspath(os.path.join(CRAWLER_ROOT, rec["path"]))
                        new.append(rec)
        return new


# ------------------------------------------------------------------ staging + pipeline
def stage_batch(records: list[dict], workdir: str) -> str:
    """Symlink audio + sidecar của lô vào <workdir>/_staging/<ts>/; trả đường dẫn staging."""
    stage_dir = os.path.join(workdir, "_staging", time.strftime("%Y%m%d-%H%M%S") + f"-{len(records)}")
    os.makedirs(stage_dir, exist_ok=True)
    for rec in records:
        base, ext = os.path.splitext(os.path.basename(rec["path"]))
        # tránh trùng tên giữa các nguồn / giữa các feed trong cùng nguồn (rss: subdir = feed)
        name = "__".join(x for x in (rec["source"], rec.get("subdir"), base) if x)
        os.symlink(rec["path"], os.path.join(stage_dir, name + ext))
        sidecar = os.path.join(os.path.dirname(rec["path"]), base + ".json")
        if os.path.exists(sidecar):
            os.symlink(sidecar, os.path.join(stage_dir, name + ".json"))
    return stage_dir


def pipeline_python(pipeline_dir: str) -> str:
    venv = os.path.join(pipeline_dir, ".venv", "bin", "python")
    return venv if os.path.exists(venv) else sys.executable


def run_pipeline(pipeline_dir: str, workdir: str, stages: str, raw_dir: str | None,
                 configs: list[str], device: str | None) -> bool:
    cmd = [pipeline_python(pipeline_dir), "-m", "pipeline", "run",
           "--workdir", os.path.abspath(workdir), "--stages", stages]
    if raw_dir:
        cmd += ["--raw-dir", os.path.abspath(raw_dir)]
    if configs:  # pipeline thay hẳn default khi có --config -> luôn chèn default.yaml trước
        default_cfg = os.path.join(pipeline_dir, "config", "default.yaml")
        for c in [default_cfg] + list(configs):
            cmd += ["--config", os.path.abspath(c)]
    if device:
        cmd += ["--device", device]
    print(f"[pipeline] {stages} <- {raw_dir or workdir}", flush=True)
    return subprocess.run(cmd, cwd=pipeline_dir).returncode == 0


class Processed:
    def __init__(self, workdir: str):
        os.makedirs(workdir, exist_ok=True)
        self.path = os.path.join(workdir, "processed_files.jsonl")
        self.done: set[str] = set()
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as f:
                self.done = {json.loads(l)["path"] for l in f if l.strip()}

    def add(self, records: list[dict]) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps({"path": r["path"], "source": r["source"], "key": r.get("key")},
                                   ensure_ascii=False) + "\n")
                self.done.add(r["path"])


# ------------------------------------------------------------------ dọn đĩa
def _read_jsonl(path: str):
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _remove(path: str) -> int:
    try:
        os.remove(path)
        return 1
    except FileNotFoundError:
        return 0


def cleanup_batch(workdir: str, staged_names: set[str], raw_paths: list[str]) -> dict:
    """Sau khi lô qua s0-s7: xóa audio gốc (giữ sidecar/ledger) và wav s0/s1/s2 của lô.
    file_id lấy từ s0 manifest qua source_path (= tên trong staging)."""
    n = {"raw": 0, "work": 0}
    for p in raw_paths:
        n["raw"] += _remove(p)
    ids = [r["id"] for r in _read_jsonl(os.path.join(workdir, "s0_ingest", "manifest.jsonl"))
           if os.path.basename(r.get("source_path", "")) in staged_names]
    for fid in ids:
        for stage in ("s0_ingest", "s1_separate"):
            n["work"] += _remove(os.path.join(workdir, stage, "audio", f"{fid}.wav"))
        for seg in glob.glob(os.path.join(workdir, "s2_segment", "audio", f"{fid}_*.wav")):
            n["work"] += _remove(seg)
    return n


def cleanup_tier_c(workdir: str) -> int:
    """Sau s8: xóa wav s7 của segment tier C (không vào dataset). Chỉ xóa trong s7_loudnorm/audio."""
    s7_audio = os.path.abspath(os.path.join(workdir, "s7_loudnorm", "audio"))
    n = 0
    for r in _read_jsonl(os.path.join(workdir, "s8_package", "manifest.jsonl")):
        p = os.path.abspath(r.get("audio_path", ""))
        if r.get("tier") == "C" and p.startswith(s7_audio + os.sep):
            n += _remove(p)
    return n


# ------------------------------------------------------------------ main loop
def main(argv=None):
    ap = argparse.ArgumentParser(description="Crawl + đưa ngay từng file vào audio-pipeline")
    ap.add_argument("--config", default="config/sources.yaml")
    ap.add_argument("--out", required=True, help="thư mục gốc audio tải về (như crawler --out)")
    ap.add_argument("--source", help="chỉ crawl source này")
    ap.add_argument("--limit", type=int, help="tối đa N item mỗi source (chạy thử)")
    ap.add_argument("--no-crawl", action="store_true", help="không crawl, chỉ xử lý ledger có sẵn")
    ap.add_argument("--pipeline-dir", default=os.path.join(CRAWLER_ROOT, "..", "audio-pipeline"))
    ap.add_argument("--workdir", required=True, help="workdir của audio-pipeline")
    ap.add_argument("--pipeline-config", action="append", default=[],
                    help="yaml override cho pipeline (lặp lại được)")
    ap.add_argument("--stages", default="s0-s7", help="stage chạy cho từng lô (mặc định s0-s7)")
    ap.add_argument("--final-stages", default="s8", help="stage chạy một lần cuối ('' = bỏ)")
    ap.add_argument("--batch", type=int, default=1, help="gom N file rồi mới chạy pipeline")
    ap.add_argument("--device", help="override device cho pipeline: cpu|cuda|mps")
    ap.add_argument("--poll", type=float, default=3.0, help="giây giữa 2 lần đọc ledger")
    ap.add_argument("--cleanup", action="store_true",
                    help="xóa audio gốc + wav trung gian sau khi xử lý; sau s8 xóa wav tier C (giữ tier A/B)")
    args = ap.parse_args(argv)

    pipeline_dir = os.path.abspath(args.pipeline_dir)
    if not os.path.isdir(os.path.join(pipeline_dir, "pipeline")):
        sys.exit(f"Không thấy audio-pipeline tại {pipeline_dir} (--pipeline-dir)")
    workdir = os.path.abspath(args.workdir)
    processed = Processed(workdir)
    tail = LedgerTail(os.path.abspath(args.out))

    crawler = None
    if not args.no_crawl:
        cmd = [sys.executable, "-m", "crawler", "run", "--config", args.config, "--out", args.out]
        if args.source:
            cmd += ["--source", args.source]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        crawler = subprocess.Popen(cmd, cwd=CRAWLER_ROOT)
        print(f"[crawler] pid {crawler.pid}", flush=True)

    queue: list[dict] = []
    n_ok = n_fail = 0

    def flush(batch: list[dict]) -> None:
        nonlocal n_ok, n_fail
        stage_dir = stage_batch(batch, workdir)
        ok = run_pipeline(pipeline_dir, workdir, args.stages, stage_dir, args.pipeline_config, args.device)
        if ok:
            processed.add(batch)
            if args.cleanup:
                staged = {f for f in os.listdir(stage_dir) if not f.endswith(".json")}  # = source_path trong s0
                n = cleanup_batch(workdir, staged, [r["path"] for r in batch])
                print(f"[cleanup] xóa {n['raw']} audio gốc, {n['work']} wav trung gian", flush=True)
            shutil.rmtree(stage_dir, ignore_errors=True)  # chỉ symlink
            n_ok += len(batch)
        else:
            n_fail += len(batch)
            print(f"[pipeline] LỖI lô {os.path.basename(stage_dir)} — giữ staging để chạy lại; "
                  f"chạy lại script sẽ xử lý lại các file này", flush=True)

    try:
        while True:
            for rec in tail.poll():
                if rec["path"] not in processed.done and os.path.exists(rec["path"]):
                    queue.append(rec)
            crawler_done = crawler is None or crawler.poll() is not None
            while len(queue) >= args.batch or (crawler_done and queue):
                batch, queue = queue[:args.batch], queue[args.batch:]
                flush(batch)
            if crawler_done:
                break
            time.sleep(args.poll)
    except KeyboardInterrupt:
        print("\n[dừng] Ctrl-C — tắt crawler, xử lý xong lô hiện tại", flush=True)
        if crawler and crawler.poll() is None:
            crawler.terminate()
            crawler.wait()
        if queue:
            flush(queue)

    if args.final_stages and (n_ok or processed.done):  # chạy cả khi không có file mới: hoàn tất s8 dở
        ok = run_pipeline(pipeline_dir, workdir, args.final_stages, None, args.pipeline_config, args.device)
        if ok and args.cleanup:
            print(f"[cleanup] xóa {cleanup_tier_c(workdir)} wav tier C", flush=True)
    print(f"\n[xong] pipeline: {n_ok} file ok, {n_fail} file lỗi | workdir: {workdir}", flush=True)
    if crawler and crawler.returncode not in (None, 0):
        print(f"[crawler] kết thúc với mã {crawler.returncode}")


if __name__ == "__main__":
    main()
