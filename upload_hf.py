"""Đẩy dataset đã đóng gói (s8_package của audio-pipeline) lên Hugging Face, chạy lặp được.

  python upload_hf.py --workdir <work> --repo nguyenvanba/vi-podcast [--public] [--with-manifest]

Mỗi lần `pipeline package` sinh thêm shard, chạy lại lệnh này: huggingface_hub băm từng file
và chỉ đẩy file mới / đã đổi (shard cũ giữ nguyên tên nên bị bỏ qua). Ngắt giữa chừng chạy
lại là tiếp (upload_large_folder lưu tiến độ trong <staging>/.cache).

Đẩy theo đợt `--batch-shards` (mặc định 10): upload_large_folder với < 150 file chỉ commit
MỘT lần ở cuối, còn shard đã lên mà chưa commit chỉ được client tin trong 20 giờ (sau đó
coi như HF đã dọn, đẩy lại). Chia đợt thì mỗi đợt kết thúc bằng một commit thật, ngắt lúc
nào cũng chỉ mất tối đa một đợt.

Bố cục trên repo:
  data/train-*.parquet      # audio bytes nhúng, load_dataset tự nhận split train
  metadata.csv              # mọi cột chỉ số (tier, cer, snr...) — file_name = wav/train/<id>.wav
  manifest.jsonl            # tùy chọn (--with-manifest): đủ mọi segment kể cả tier C
  README.md                 # dataset card khai features -> cột audio tự thành kiểu Audio

Cần: pip install huggingface_hub; đã `huggingface-cli login`.
"""

import argparse
import csv
import glob
import os
import shutil
import sys
import time

from huggingface_hub import HfApi

CARD = """---
language:
- vi
pretty_name: {name}
license: other
task_categories:
- automatic-speech-recognition
- text-to-speech
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*.parquet
dataset_info:
  features:
  - name: audio
    dtype: audio
  - name: text
    dtype: string
  - name: text_normalized
    dtype: string
  - name: speaker_id
    dtype: string
  - name: duration
    dtype: float64
  - name: tier
    dtype: string
---

# {name}

Giọng nói tiếng Việt cắt từ podcast, đã qua audio-pipeline (tách nhạc nền, VAD, đo chất lượng,
gán speaker, transcribe 2 model + CER, chuẩn hóa text, loudnorm −23 LUFS). Chỉ gồm segment
tier A (TTS-grade) và B (ASR-grade); WAV mono 24 kHz nhúng trong parquet.

| | |
|---|---|
| Segment | {n_seg} |
| Giờ | {hours:.1f} |
| Speaker | {n_spk} |
| Tier A / B | {n_a} / {n_b} |

`metadata.csv` có thêm `snr_db`, `dnsmos`, `cer`, `clipping`, `bandwidth_hz`, `multi_speaker`,
`source_path`; khớp với parquet qua `file_name` = `wav/train/<path trong cột audio>`.

```python
from datasets import load_dataset
ds = load_dataset("{repo}", split="train")   # cột audio đã là kiểu Audio (24 kHz)
```

Không có split val/test: người dùng tự chia lúc training (nên chia theo `speaker_id`).
"""


def card_stats(csv_path: str) -> dict:
    n = hours = 0.0
    spk, tiers = set(), {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            n += 1
            hours += float(r["duration"] or 0) / 3600
            spk.add(r["speaker_id"])
            tiers[r["tier"]] = tiers.get(r["tier"], 0) + 1
    return {"n_seg": int(n), "hours": hours, "n_spk": len(spk),
            "n_a": tiers.get("A", 0), "n_b": tiers.get("B", 0)}


def build_staging(staging: str, s8: str, with_manifest: bool, repo: str) -> int:
    """Hardlink parquet + metadata vào staging (0 byte thêm, cùng đĩa); giữ .cache để resume."""
    ds = os.path.join(s8, "dataset")
    data = os.path.join(staging, "data")
    os.makedirs(data, exist_ok=True)
    for stale in glob.glob(os.path.join(data, "*.parquet")):  # link cũ không còn nguồn (đổi tên/xóa)
        if not os.path.exists(os.path.join(ds, "parquet", os.path.basename(stale))):
            os.remove(stale)
    n = 0
    for src in sorted(glob.glob(os.path.join(ds, "parquet", "*.parquet"))):
        dst = os.path.join(data, os.path.basename(src))
        if os.path.exists(dst) and os.path.samefile(src, dst):
            continue
        if os.path.exists(dst):
            os.remove(dst)
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
        n += 1
    for name, src in [("metadata.csv", os.path.join(ds, "metadata.csv")),
                      ("manifest.jsonl", os.path.join(s8, "manifest.jsonl"))]:
        dst = os.path.join(staging, name)
        if name == "manifest.jsonl" and not with_manifest:
            if os.path.exists(dst):
                os.remove(dst)
            continue
        shutil.copy2(src, dst)  # copy chứ không link: file này bị ghi lại mỗi lần package
    with open(os.path.join(staging, "README.md"), "w", encoding="utf-8") as f:
        f.write(CARD.format(name=repo.split("/")[-1], repo=repo, **card_stats(os.path.join(ds, "metadata.csv"))))
    return n


def upload_state(staging: str, rel: str) -> tuple[bool, bool]:
    """Đọc file trạng thái của upload_large_folder: (is_uploaded, is_committed) = dòng 7, 8."""
    meta = os.path.join(staging, ".cache", "huggingface", "upload", rel + ".metadata")
    try:
        with open(meta) as f:
            lines = f.read().splitlines()
        return (len(lines) >= 7 and lines[6].strip() == "1", len(lines) >= 8 and lines[7].strip() == "1")
    except OSError:
        return False, False


def is_committed(staging: str, rel: str) -> bool:
    return upload_state(staging, rel)[1]


def upload_with_retry(api: HfApi, retries: int = 30, **kw) -> None:
    """Mạng chậm/đứt thoáng qua (HF đóng kết nối) làm upload_large_folder ném lỗi ở bước
    kiểm tra repo hoặc commit; chạy lại là tiếp vì tiến độ đã lưu trong .cache."""
    for attempt in range(1, retries + 1):
        try:
            api.upload_large_folder(**kw)
            return
        except KeyboardInterrupt:
            raise
        except Exception as e:  # ConnectionError, HfHubHTTPError 5xx, timeout...
            if attempt == retries:
                raise
            wait = min(300, 15 * attempt)
            print(f"  [lỗi lần {attempt}/{retries}] {str(e)[:200]} — thử lại sau {wait}s", flush=True)
            time.sleep(wait)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True, help="workdir của audio-pipeline (chứa s8_package/)")
    ap.add_argument("--repo", required=True, help="user/ten-repo trên Hugging Face")
    ap.add_argument("--public", action="store_true", help="tạo repo public (mặc định private)")
    ap.add_argument("--with-manifest", action="store_true", help="đẩy cả s8_package/manifest.jsonl")
    ap.add_argument("--staging", help="thư mục tạm chứa hardlink + tiến độ upload "
                                      "(mặc định <workdir>/s8_package/_hf_staging)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--batch-shards", type=int, default=10,
                    help="số shard mỗi đợt upload_large_folder (mỗi đợt một commit); 0 = đẩy hết một lần")
    args = ap.parse_args()

    s8 = os.path.join(os.path.abspath(args.workdir), "s8_package")
    if not os.path.exists(os.path.join(s8, "dataset", "metadata.csv")):
        sys.exit(f"Không thấy {s8}/dataset/metadata.csv — chạy pipeline package / --stages s8 trước")
    staging = args.staging or os.path.join(s8, "_hf_staging")
    n_new = build_staging(staging, s8, args.with_manifest, args.repo)
    n_all = len(glob.glob(os.path.join(staging, "data", "*.parquet")))
    print(f"[staging] {staging}: {n_all} shard ({n_new} link mới)", flush=True)

    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", private=not args.public, exist_ok=True)
    print(f"[hf] repo {args.repo} ({'public' if args.public else 'private'}) — bắt đầu upload", flush=True)
    small = [f for f in ("README.md", "metadata.csv", "manifest.jsonl") if os.path.exists(os.path.join(staging, f))]
    shards = sorted(os.path.basename(p) for p in glob.glob(os.path.join(staging, "data", "*.parquet")))
    if args.batch_shards <= 0:
        upload_with_retry(api, repo_id=args.repo, repo_type="dataset", folder_path=staging,
                                num_workers=args.workers, print_report_every=60)
    else:
        k = 0
        while True:
            todo = [f"data/{s}" for s in shards if not is_committed(staging, f"data/{s}")]
            if not todo:
                break
            # shard đã lên xong (lần trước ngắt trước khi commit) đi trước để được commit ngay
            todo.sort(key=lambda r: (not upload_state(staging, r)[0], r))
            batch = todo[:args.batch_shards]
            k += 1
            print(f"[đợt {k}] {len(batch)} shard, còn {len(todo) - len(batch)} chưa commit", flush=True)
            upload_with_retry(api, repo_id=args.repo, repo_type="dataset", folder_path=staging,
                                    allow_patterns=small + batch,
                                    num_workers=args.workers, print_report_every=60)
        if k == 0:  # mọi shard đã commit từ trước: vẫn đẩy card/csv mới nhất
            upload_with_retry(api, repo_id=args.repo, repo_type="dataset", folder_path=staging,
                                    allow_patterns=small, num_workers=args.workers, print_report_every=60)
    print(f"[xong] https://huggingface.co/datasets/{args.repo}", flush=True)


if __name__ == "__main__":
    main()
