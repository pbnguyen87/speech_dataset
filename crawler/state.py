"""Ledger JSONL: ghi lại mọi URL đã tải để resume và khử trùng lặp.

Mỗi thư mục output có một `ledger.jsonl`; mỗi dòng một record
{"id", "url", "path", "source"}. Ghi append + flush ngay (giống manifest
của audio-pipeline) — ngắt giữa chừng chạy lại là bỏ qua phần đã tải.
"""

import hashlib
import json
import os


def url_id(url: str) -> str:
    return hashlib.sha1(url.encode()).hexdigest()[:16]


class Ledger:
    def __init__(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(out_dir, "ledger.jsonl")
        self.done: set[str] = set()
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.done.add(json.loads(line)["id"])
        self._fh = open(self.path, "a", encoding="utf-8")

    def has(self, key: str) -> bool:
        return url_id(key) in self.done

    def add(self, key: str, path: str, source: str) -> None:
        rec = {"id": url_id(key), "key": key, "path": path, "source": source}
        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._fh.flush()
        self.done.add(rec["id"])

    def close(self) -> None:
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
