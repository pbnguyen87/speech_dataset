"""s6: chuẩn hóa transcript (số -> chữ, viết tắt, khoảng trắng).

Backend `vinorm` nếu cài được (pip install vinorm), fallback `basic`:
chuẩn hóa unicode NFC, số nguyên -> chữ tiếng Việt, gọn khoảng trắng.
"""

import re
import unicodedata

from .. import manifest

STAGE = "s6_textnorm"
PREV = "s5_transcribe"

_DIGITS = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]


def _three_digits(n: int, full: bool) -> str:
    """Đọc nhóm 3 chữ số. full=True khi cần đọc cả 'không trăm'."""
    tram, rest = divmod(n, 100)
    chuc, dv = divmod(rest, 10)
    parts = []
    if tram or full:
        parts += [_DIGITS[tram], "trăm"]
    if chuc == 0 and dv and (tram or full):
        parts.append("lẻ")
    elif chuc == 1:
        parts.append("mười")
    elif chuc > 1:
        parts += [_DIGITS[chuc], "mươi"]
    if dv:
        if chuc >= 1 and dv == 5:
            parts.append("lăm")
        elif chuc > 1 and dv == 1:
            parts.append("mốt")
        else:
            parts.append(_DIGITS[dv])
    return " ".join(parts)


def number_to_words_vi(n: int) -> str:
    if n == 0:
        return "không"
    if n < 0:
        return "âm " + number_to_words_vi(-n)
    groups = []
    while n:
        groups.append(n % 1000)
        n //= 1000
    units = ["", "nghìn", "triệu", "tỷ", "nghìn tỷ", "triệu tỷ"]
    parts = []
    for i in range(len(groups) - 1, -1, -1):
        g = groups[i]
        if g == 0:
            continue
        word = _three_digits(g, full=(i < len(groups) - 1))
        parts.append((word + " " + units[i]).strip())
    return " ".join(parts)


def basic_normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    # số nguyên (tối đa 15 chữ số, tránh chuỗi số điện thoại dài vô nghĩa vẫn đọc được)
    text = re.sub(r"\d{1,15}", lambda m: number_to_words_vi(int(m.group())), text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def pick_backend(cfg: dict):
    """Trả (hàm chuẩn hóa, tên backend) theo textnorm.backend: vinorm | basic | auto."""
    backend = cfg["textnorm"]["backend"]
    if backend in ("vinorm", "auto"):
        try:
            from vinorm import TTSnorm

            return (lambda t: TTSnorm(t).strip()), "vinorm"
        except ImportError:
            if backend == "vinorm":
                raise SystemExit("textnorm.backend=vinorm nhưng chưa cài vinorm")
    return basic_normalize, "basic"


class Worker:
    batch_n = 1
    flush_s = 0

    def __init__(self, cfg: dict, workdir: str):
        self.norm_fn, self.backend = pick_backend(cfg)
        self.w = manifest.ManifestWriter(manifest.manifest_path(workdir, STAGE))
        print(f"[{STAGE}] backend={self.backend}")

    def is_done(self, rec: dict) -> bool:
        return self.w.is_done(rec["id"])

    def ready(self, pending: list, upstream_done: bool):
        return pending, []

    def process(self, records: list[dict]) -> None:
        for rec in records:
            if self.w.is_done(rec["id"]):
                continue
            self.w.write({**rec,
                          "text_normalized": self.norm_fn(rec.get("text") or ""),
                          "textnorm_backend": self.backend})

    def close(self) -> None:
        self.w.close()


def run(cfg: dict, workdir: str, limit: int | None = None) -> str:
    records = manifest.read_records(manifest.manifest_path(workdir, PREV))
    worker = Worker(cfg, workdir)
    print(f"[{STAGE}] {len(records)} segment")
    worker.process(records)
    worker.close()
    print(f"[{STAGE}] xong")
    return manifest.manifest_path(workdir, STAGE)
