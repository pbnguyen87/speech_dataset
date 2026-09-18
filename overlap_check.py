"""Kiểm tra trùng nội dung giữa dữ liệu podcast đã transcribe (manifest s5/s8 của audio-pipeline)
và các dataset HF (PhoAudiobook, viVoice) — CHỈ đọc cột text/speaker/channel của parquet từ xa,
không tải audio.

  python overlap_check.py --manifest work/s5_transcribe/manifest.jsonl \\
      --datasets thivux/phoaudiobook capleaf/viVoice [--limit-files 5] [--out overlap.json]

Cần: pip install duckdb huggingface_hub (đọc parquet từ xa bằng DuckDB, chỉ lấy column chunk text).

Cách làm: chuẩn hóa text (thường, bỏ dấu câu), băm n-gram từ (mặc định 8) của mọi segment
của ta thành chỉ mục; quét từng utterance của dataset, tỷ lệ n-gram của utterance có trong
chỉ mục >= min-frac thì tính là trùng. 8 từ liên tiếp giống nhau gần như chỉ xảy ra khi
cùng một bản thu / cùng một văn bản được đọc.
"""

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def words(text: str) -> list[str]:
    text = unicodedata.normalize("NFC", (text or "").lower())
    return _PUNCT.sub(" ", text).split()


def shingles(ws: list[str], n: int):
    return [hash(tuple(ws[i:i + n])) for i in range(len(ws) - n + 1)]


def load_ours(path: str, n: int):
    """Chỉ mục shingle -> [chỉ số segment]; kèm info segment."""
    index: dict[int, list[int]] = defaultdict(list)
    segs = []
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        ws = words(r.get("text") or "")
        if len(ws) < n:
            continue
        i = len(segs)
        segs.append({"id": r["id"], "file_id": r.get("file_id"), "n_sh": len(ws) - n + 1,
                     "source": (r.get("source_meta") or {}).get("feed") or r.get("source_path"),
                     "text": r.get("text")})
        for h in set(shingles(ws, n)):
            index[h].append(i)
    return index, segs


_local = __import__("threading").local()


def scan_file(repo: str, fname: str, token: str, cols: list[str]):
    """Đọc đúng các cột cần từ một parquet trên HF bằng DuckDB (range request theo column chunk,
    nhanh hơn nhiều so với pyarrow + fsspec). Mỗi thread một connection."""
    import duckdb

    con = getattr(_local, "con", None)
    if con is None:
        con = duckdb.connect()
        con.execute(f"CREATE SECRET hf (TYPE HUGGINGFACE, TOKEN '{token}')")
        con.execute("SET threads=2")
        _local.con = con
    q = f"SELECT {', '.join(cols)} FROM read_parquet('hf://datasets/{repo}/{fname}')"
    for row in con.execute(q).fetchall():
        yield dict(zip(cols, row))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--datasets", nargs="+", default=["thivux/phoaudiobook", "capleaf/viVoice"])
    ap.add_argument("--n", type=int, default=8, help="số từ mỗi n-gram")
    ap.add_argument("--min-frac", type=float, default=0.6, help="tỷ lệ n-gram trùng để tính một utterance là trùng")
    ap.add_argument("--min-words", type=int, default=12,
                    help="bỏ qua utterance ngắn hơn (câu 8 từ thông dụng như 'điều này làm sao có thể xảy ra' trùng ngẫu nhiên)")
    ap.add_argument("--limit-files", type=int, help="chỉ quét N parquet đầu mỗi dataset (chạy thử)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="overlap.json")
    args = ap.parse_args()

    from huggingface_hub import HfApi, get_token

    index, segs = load_ours(args.manifest, args.n)
    print(f"[ta] {len(segs)} segment, {len(index)} n-gram ({args.n} từ) từ {args.manifest}", flush=True)
    api, token = HfApi(), get_token()
    report = {"n": args.n, "min_frac": args.min_frac, "ours_segments": len(segs), "datasets": {}}

    for repo in args.datasets:
        files = sorted(f for f in api.list_repo_files(repo, repo_type="dataset") if f.endswith(".parquet"))
        if args.limit_files:
            files = files[:args.limit_files]
        key_col = "speaker" if "phoaudiobook" in repo else "channel"
        cols = ["text", key_col]
        n_utt = 0
        hits = []  # (key, frac, text, [seg idx])
        by_key = Counter()
        utt_by_key = Counter()

        def work(fname):
            local = []
            cnt = Counter()
            n = 0
            for r in scan_file(repo, fname, token, cols):
                n += 1
                cnt[r.get(key_col)] += 1
                ws = words(r.get("text"))
                if len(ws) < args.min_words:
                    continue
                sh = shingles(ws, args.n)
                found = [h for h in sh if h in index]
                frac = len(found) / len(sh)
                if frac >= args.min_frac:
                    seg_ids = Counter(i for h in found for i in index[h])
                    local.append((r.get(key_col), round(frac, 2), r.get("text"), [i for i, _ in seg_ids.most_common(3)]))
            return n, cnt, local

        with ThreadPoolExecutor(args.workers) as ex:
            for k, (n, cnt, local) in enumerate(ex.map(work, files), 1):
                n_utt += n
                utt_by_key.update(cnt)
                hits += local
                for h in local:
                    by_key[h[0]] += 1
                if k % 20 == 0 or k == len(files):
                    print(f"  [{repo}] {k}/{len(files)} file, {n_utt} utterance, trùng {len(hits)}", flush=True)

        seg_hit = Counter()
        file_hit = Counter()
        for _, _, _, idxs in hits:
            for i in idxs[:1]:
                seg_hit[i] += 1
                file_hit[segs[i]["file_id"]] += 1
        rep = {
            "files": len(files), "utterances": n_utt, "hits": len(hits),
            "hit_ratio": round(len(hits) / n_utt, 5) if n_utt else 0,
            "top_keys": [{"key": k, "hits": v, "utterances": utt_by_key[k]} for k, v in by_key.most_common(15)],
            "our_files_hit": [{"file_id": f, "hits": v} for f, v in file_hit.most_common(15)],
            "examples": [{"key": k, "frac": fr, "their": t[:120], "ours": segs[idxs[0]]["text"][:120]}
                         for k, fr, t, idxs in sorted(hits, key=lambda x: -x[1])[:10]],
        }
        report["datasets"][repo] = rep
        print(f"[{repo}] {n_utt} utterance, trùng {len(hits)} ({rep['hit_ratio'] * 100:.3f}%), "
              f"top: {by_key.most_common(3)}", flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f"[xong] báo cáo: {args.out}")


if __name__ == "__main__":
    main()
