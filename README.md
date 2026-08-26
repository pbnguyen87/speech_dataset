# audio-crawler

Bộ thu thập audio tiếng Việt (sách nói, truyện audio, podcast) làm
đầu vào cho [audio-pipeline](https://github.com/pbnguyen87/audio-pipeline).
Output là thư mục audio thô + **sidecar `.json` metadata** cạnh mỗi file —
đúng format mà stage s0 của pipeline tự nhận và giữ suốt các bước xử lý.

## Nguồn hỗ trợ (3 adapter)

| type | Dùng cho | Ghi chú |
|---|---|---|
| `archive_org` | Item trên Internet Archive | An toàn nhất, tải hàng loạt qua metadata API, chỉ lấy file `original` |
| `rss` | Podcast (RSS feed công khai) | Mỗi episode một enclosure mp3 — không cần scrape HTML |
| `html` | Trang danh sách có link mp3 trực tiếp | Tổng quát: quét trang (hỗ trợ `{page}`), nhặt href khớp regex, tùy chọn quét thêm 1 cấp trang con |

Trang phức tạp (link ẩn sau player/javascript) cần viết adapter riêng trong
`crawler/sources/` — mỗi adapter chỉ cần hàm `list_items(cfg)` trả về
`[{"url", "title", "meta"}]`, phần tải/ledger/sidecar dùng chung.

## Cài đặt

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Chạy

```bash
# khai báo nguồn trong config/sources.yaml rồi:
python -m crawler run --config config/sources.yaml --out ./raw

# chạy thử 3 item mỗi nguồn
python -m crawler run --config config/sources.yaml --out ./raw --limit 3

# chỉ chạy một nguồn
python -m crawler run --config config/sources.yaml --out ./raw --source uyen-uong-dao
```

Output:

```
raw/
└── <tên source>/
    ├── Tập_1_xxx.mp3
    ├── Tập_1_xxx.json     # sidecar: title, url, source, metadata riêng của nguồn
    ├── ...
    └── ledger.jsonl       # mọi url đã tải — chạy lại tự bỏ qua (resume/dedup)
```

Nối vào pipeline:

```bash
cd ../audio-pipeline
python -m pipeline run --raw-dir ../audio-crawler/raw/uyen-uong-dao --workdir work
```

## Cơ chế

- **Resume/dedup**: mỗi thư mục nguồn có `ledger.jsonl` (append + flush từng
  dòng) — ngắt giữa chừng chạy lại là tiếp tục; url đã tải không tải lại.
- **Lịch sự với server**: nghỉ `delay_s` giây giữa 2 lượt tải (mặc định 2s,
  chỉnh riêng từng nguồn), retry 3 lần với backoff, User-Agent tự nhận diện.
- **Tải an toàn**: stream về file `.part`, thành công mới đổi tên — không bao
  giờ có file cụt trong thư mục output.

## Test

```bash
pip install pytest && pytest tests/ -v   # hàm thuần, không cần mạng
```

## Lưu ý pháp lý

Công cụ này phục vụ thu thập dữ liệu cho **nghiên cứu/huấn luyện model**.
Người dùng tự chịu trách nhiệm: tôn trọng điều khoản sử dụng và robots.txt
của từng trang, giữ delay hợp lý để không tạo tải lên server, và **không phân
phối lại audio gốc** đã tải (tham khảo cách các dataset viVoice/PhoAudiobook
chỉ phát hành dữ liệu đã xử lý kèm điều khoản nghiên cứu).
