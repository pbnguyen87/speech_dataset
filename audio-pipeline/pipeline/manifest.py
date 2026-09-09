"""Manifest JSONL: mỗi stage đọc manifest stage trước, ghi manifest mới.

Ghi append từng dòng + flush ngay để resume an toàn: chạy lại thì các id đã
có trong manifest được skip.
"""

import json
import os


def stage_dir(workdir: str, stage: str) -> str:
    d = os.path.join(workdir, stage)
    os.makedirs(d, exist_ok=True)
    return d


def audio_dir(workdir: str, stage: str) -> str:
    d = os.path.join(workdir, stage, "audio")
    os.makedirs(d, exist_ok=True)
    return d


def manifest_path(workdir: str, stage: str) -> str:
    return os.path.join(workdir, stage, "manifest.jsonl")


def iter_records(path: str):
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_records(path: str, limit: int | None = None) -> list[dict]:
    out = []
    for rec in iter_records(path):
        out.append(rec)
        if limit is not None and len(out) >= limit:
            break
    return out


def done_ids(path: str, key: str = "id") -> set:
    return {rec[key] for rec in iter_records(path) if key in rec}


class ManifestWriter:
    """Ghi manifest append-only, hỗ trợ resume qua tập id đã xong."""

    def __init__(self, path: str, key: str = "id"):
        self.path = path
        self.key = key
        self.done = done_ids(path, key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._fh = open(path, "a", encoding="utf-8")

    def is_done(self, rec_id: str) -> bool:
        return rec_id in self.done

    def write(self, rec: dict) -> None:
        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._fh.flush()
        self.done.add(rec[self.key])

    def close(self) -> None:
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class ManifestTail:
    """Đọc dần một manifest đang được stage khác ghi (chế độ stream).

    Nhớ offset byte; dòng chưa có '\\n' là dòng ghi dở -> để lần sau. File chưa tồn tại
    coi như rỗng. Đọc ở chế độ nhị phân để offset là byte thật."""

    def __init__(self, path: str):
        self.path = path
        self.offset = 0

    def poll(self) -> list[dict]:
        if not os.path.exists(self.path):
            return []
        out = []
        with open(self.path, "rb") as f:
            f.seek(self.offset)
            while True:
                line = f.readline()
                if not line or not line.endswith(b"\n"):
                    break
                self.offset += len(line)
                s = line.decode("utf-8").strip()
                if s:
                    out.append(json.loads(s))
        return out
