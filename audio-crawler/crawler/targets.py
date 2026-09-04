"""Gom danh sách "mục tiêu" (url / identifier) của một source từ 3 dạng khai báo.

Với tên cơ sở `url`, source có thể dùng bất kỳ tổ hợp nào:

    url: https://a            # một mục
    urls: [https://a, ...]    # nhiều mục
    urls_file: links.txt      # file text, mỗi dòng một mục; bỏ dòng trống và dòng bắt đầu '#'

`*_file` tính tương đối theo thư mục chứa file config (cli đặt `cfg["_config_dir"]`),
không có thì theo thư mục hiện tại. Kết quả giữ thứ tự, bỏ trùng.
"""

import os


def read_list_file(path: str) -> list[str]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                out.append(line)
    return out


def resolve(cfg: dict, base: str) -> list[str]:
    """base="url" -> đọc cfg["url"], cfg["urls"], cfg["urls_file"]."""
    plural, file_key = base + "s", base + "s_file"
    targets: list[str] = []
    if cfg.get(base):
        targets.append(str(cfg[base]))
    if cfg.get(plural):
        v = cfg[plural]
        targets += [str(v)] if isinstance(v, str) else [str(x) for x in v]
    if cfg.get(file_key):
        path = cfg[file_key]
        if not os.path.isabs(path):
            path = os.path.join(cfg.get("_config_dir", "."), path)
        targets += read_list_file(path)

    seen, uniq = set(), []
    for t in targets:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    if not uniq:
        raise SystemExit(
            f"[{cfg.get('name')}] cần ít nhất một trong: {base} | {plural} | {file_key}")
    return uniq
