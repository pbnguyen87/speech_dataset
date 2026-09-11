"""Ledger JSONL: ghi lại mọi URL đã tải để resume và khử trùng lặp.

Mỗi thư mục output có một `ledger.jsonl`; mỗi dòng một record
{"id", "key", "path", "source", "subdir"?}. Ghi append + flush ngay (giống manifest
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
            with open(self.path, encoding="utf-8", errors="replace") as f:
                for i, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self.done.add(json.loads(line)["id"])
                    except (json.JSONDecodeError, KeyError) as e:
                        # dòng ghi dở (đĩa đầy / kill): coi như chưa tải, url đó sẽ tải lại
                        print(f"  [ledger] bỏ qua dòng hỏng {self.path}:{i} ({e})", flush=True)
        self._fh = open(self.path, "a", encoding="utf-8")
        if os.path.exists(self.path) and os.path.getsize(self.path) > 0:
            with open(self.path, "rb") as f:  # dòng cuối thiếu '\n' -> chèn, khỏi dính vào bản ghi mới
                f.seek(-1, os.SEEK_END)
                if f.read(1) != b"\n":
                    self._fh.write("\n")
                    self._fh.flush()

    def has(self, key: str) -> bool:
        return url_id(key) in self.done

    def add(self, key: str, path: str, source: str, subdir: str | None = None) -> None:
        rec = {"id": url_id(key), "key": key, "path": path, "source": source}
        if subdir:
            rec["subdir"] = subdir
        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._fh.flush()
        self.done.add(rec["id"])

    def close(self) -> None:
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
