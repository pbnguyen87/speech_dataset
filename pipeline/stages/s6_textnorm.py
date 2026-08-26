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


def run(cfg: dict, workdir: str, limit: int | None = None) -> str:
    backend = cfg["textnorm"]["backend"]
    norm_fn = None
    if backend in ("vinorm", "auto"):
        try:
            from vinorm import TTSnorm

            norm_fn = lambda t: TTSnorm(t).strip()  # noqa: E731
            backend = "vinorm"
        except ImportError:
            if backend == "vinorm":
                raise SystemExit("textnorm.backend=vinorm nhưng chưa cài vinorm")
            backend = "basic"
    if norm_fn is None:
        norm_fn = basic_normalize
        backend = "basic"

    records = manifest.read_records(manifest.manifest_path(workdir, PREV))
    mpath = manifest.manifest_path(workdir, STAGE)
    print(f"[{STAGE}] backend={backend}, {len(records)} segment")

    with manifest.ManifestWriter(mpath) as w:
        for rec in records:
            if w.is_done(rec["id"]):
                continue
            w.write({**rec,
                     "text_normalized": norm_fn(rec.get("text") or ""),
                     "textnorm_backend": backend})
    print(f"[{STAGE}] xong")
    return mpath
