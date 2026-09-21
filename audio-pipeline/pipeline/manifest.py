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


def parse_line(line: str, path: str, lineno: int):
    """json.loads một dòng manifest; dòng hỏng (ghi dở khi đĩa đầy / bị kill) -> None + cảnh báo.
    Bản ghi hỏng coi như chưa làm: stage sẽ làm lại, an toàn vì mọi stage đều idempotent."""
    try:
        return json.loads(line)
    except json.JSONDecodeError as e:
        print(f"  [manifest] bỏ qua dòng hỏng {os.path.relpath(path)}:{lineno} ({e.msg} tại cột {e.pos}); "
              f"chạy `python -m pipeline repair` để dọn", flush=True)
        return None


def iter_records(path: str):
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if line:
                rec = parse_line(line, path, i)
                if rec is not None:
                    yield rec


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
        # dòng cuối ghi dở (đĩa đầy / kill) không có '\n' -> chèn để bản ghi mới không dính vào rác
        if os.path.getsize(path) > 0:
            with open(path, "rb") as f:
                f.seek(-1, os.SEEK_END)
                if f.read(1) != b"\n":
                    self._fh.write("\n")
                    self._fh.flush()

    def is_done(self, rec_id: str) -> bool:
        return rec_id in self.done

    def write(self, rec: dict) -> None:
        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._fh.flush()
        self.done.add(rec[self.key])

    def write_many(self, recs: list[dict]) -> None:
        """Ghi nhiều record bằng MỘT lần write + flush: tiến trình bị kill giữa chừng thì
        hoặc cả lô có đủ, hoặc không có gì (đủ cho kill/OOM; không bảo đảm khi mất điện)."""
        if not recs:
            return
        self._fh.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs))
        self._fh.flush()
        self.done.update(r[self.key] for r in recs)

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
                s = line.decode("utf-8", errors="replace").strip()
                if s:
                    rec = parse_line(s, self.path, -1)
                    if rec is not None:
                        out.append(rec)
        return out


def relocate_paths(rec: dict, old_root: str, new_root: str, fields=("audio_path", "path")) -> int:
    """Đổi tiền tố đường dẫn tuyệt đối trong record (audio_path của manifest, path của ledger).
    Trả số trường đã đổi."""
    n = 0
    old_root = old_root.rstrip("/") + "/"
    new_root = new_root.rstrip("/") + "/"
    for k in fields:
        v = rec.get(k)
        if isinstance(v, str) and v.startswith(old_root):
            rec[k] = new_root + v[len(old_root):]
            n += 1
    return n


def repair(path: str, dry_run: bool = False, relocate: tuple[str, str] | None = None) -> tuple[int, int, int]:
    """Ghi lại manifest không còn dòng hỏng; relocate=(gốc cũ, gốc mới) thì đổi luôn đường dẫn
    tuyệt đối (chuyển workdir sang máy/đĩa khác). Trả (số dòng giữ, số dòng bỏ, số dòng đổi đường dẫn)."""
    if not os.path.exists(path):
        return 0, 0, 0
    keep, bad, moved = [], 0, 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                rec = json.loads(s)
            except json.JSONDecodeError:
                bad += 1
                continue
            if relocate and relocate_paths(rec, *relocate):
                moved += 1
                s = json.dumps(rec, ensure_ascii=False)
            keep.append(s)
    if (bad or moved) and not dry_run:
        tmp = path + ".repair"
        with open(tmp, "w", encoding="utf-8") as f:
            for s in keep:
                f.write(s + "\n")
        os.replace(tmp, path)
    return len(keep), bad, moved
