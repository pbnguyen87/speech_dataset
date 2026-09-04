"""Registry các nguồn crawl. Mỗi nguồn là một package con trong thư mục này:

    crawler/sources/<type>/
        __init__.py   # export TYPE + list_items (hoặc crawl)
        adapter.py    # xem 2 kiểu bên dưới
        README.md     # cách khai báo trong config/sources.yaml

Hai kiểu adapter:
  - list_items(cfg) -> [{"url", "title", "meta", ("key")}]: chỉ liệt kê,
    phần tải / ledger / sidecar dùng chung trong cli.py (archive_org, rss, html).
  - crawl(cfg, out_dir, delay_s, limit) -> int: tự tải bằng công cụ riêng
    (youtube qua yt-dlp), tự ghi sidecar + ledger, trả số file mới.

Thêm nguồn mới = tạo thư mục mới theo mẫu trên; registry tự phát hiện,
không cần sửa cli.py.
"""

import importlib
import pkgutil

REGISTRY: dict[str, object] = {}

for _mod in pkgutil.iter_modules(__path__):
    if _mod.ispkg:
        _pkg = importlib.import_module(f"{__name__}.{_mod.name}")
        REGISTRY[getattr(_pkg, "TYPE", _mod.name)] = _pkg


def get(stype: str):
    """Trả module nguồn theo type trong config, hoặc raise KeyError kèm gợi ý."""
    if stype not in REGISTRY:
        raise KeyError(f"Không hỗ trợ type '{stype}' ({'|'.join(sorted(REGISTRY))})")
    return REGISTRY[stype]
