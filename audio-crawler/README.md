# audio-crawler

Bộ thu thập audio tiếng Việt (sách nói, truyện audio, podcast, YouTube) làm
đầu vào cho [audio-pipeline](https://github.com/pbnguyen87/audio-pipeline).
Output là thư mục audio thô + **sidecar `.json` metadata** cạnh mỗi file —
đúng format mà stage s0 của pipeline tự nhận và giữ suốt các bước xử lý.

## Nguồn hỗ trợ (4 adapter)

| type | Dùng cho | Ghi chú |
|---|---|---|
| `archive_org` | Item trên Internet Archive | An toàn nhất, tải hàng loạt qua metadata API, chỉ lấy file `original` |
| `rss` | Podcast (RSS feed công khai) | Mỗi episode một enclosure mp3 — không cần scrape HTML |
| `youtube` | Channel / playlist / video | Qua `yt-dlp` (tự resume bằng download-archive), giữ m4a gốc không re-encode, info.json → sidecar |
| `html` | Trang danh sách có link mp3 trực tiếp | Tổng quát: quét trang (hỗ trợ `{page}`), nhặt href khớp regex, tùy chọn quét thêm 1 cấp trang con |

Mỗi nguồn là một thư mục riêng trong `crawler/sources/`, có README
hướng dẫn khai báo config của nguồn đó:

```
crawler/sources/
├── __init__.py          # registry: tự phát hiện mọi package con
├── archive_org/         # adapter.py + README.md
├── rss/
├── html/
└── youtube/            # cần yt-dlp + ffmpeg
```

Thêm nguồn mới (vd. trang phức tạp, link ẩn sau player/javascript): tạo
thư mục `crawler/sources/<type>/` với `adapter.py` chứa hàm `list_items(cfg)`
trả về `[{"url", "title", "meta"}]` (tùy chọn `"key"` làm định danh resume) và
`__init__.py` export `TYPE` + `list_items`. Không cần sửa `cli.py`; phần
tải/ledger/sidecar dùng chung. Nguồn cần công cụ tải riêng (như `youtube`)
thay `list_items` bằng `crawl(cfg, out_dir, delay_s, limit)` và tự ghi
sidecar + ledger.

### Khai báo nhiều link cho một nguồn

Mọi nguồn đều nhận mục tiêu ở 3 dạng, dùng riêng hoặc kết hợp (gộp, bỏ trùng,
giữ thứ tự): giá trị đơn, danh sách, hoặc **file text mỗi dòng một link**
(dòng trống và dòng bắt đầu `#` bị bỏ; đường dẫn tương đối tính theo thư mục
chứa file config).

| type | đơn | danh sách | file |
|---|---|---|---|
| `youtube` | `url` | `urls` | `urls_file` |
| `rss` | `url` | `urls` | `urls_file` |
| `archive_org` | `identifier` | `identifiers` | `identifiers_file` |
| `html` | `start_url` | `start_urls` | `start_urls_file` |

Ví dụ 100 video YouTube: ghi link vào `config/youtube_links.txt` rồi khai
báo một source `type: youtube` với `urls_file: youtube_links.txt`. Toàn bộ
tải vào chung một thư mục, chung ledger, chạy lại chỉ tải phần còn thiếu.
File mẫu cho từng nguồn nằm trong `config/*.example.txt`.

Danh sách podcast tiếng Việt có sẵn (khám phá qua iTunes Search + bảng xếp
hạng Apple VN, đo tổng `itunes:duration` trong RSS, ngày 3/9/2026):
`config/podcast_feeds_2h.txt` (1693 kênh ≥ 2 giờ) và
`config/podcast_feeds_10h.txt` (882 kênh ≥ 10 giờ), kèm `.csv` cùng tên có
tên kênh, số giờ, số tập, link Apple.

## Cài đặt

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# nếu dùng nguồn youtube: pip install yt-dlp  (+ ffmpeg trong PATH)
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
└── <tên source>/          # hoặc bỏ tầng này khi source khai `dir: .`
    ├── <feed>/            # nguồn rss: mỗi feed một thư mục (host_path của url feed)
    │   ├── Tập_1_xxx.mp3
    │   └── Tập_1_xxx.json # sidecar: title, url, source, metadata riêng của nguồn
    ├── ...                # nguồn khác (html/archive/youtube): file nằm thẳng ở đây
    └── ledger.jsonl       # mọi url đã tải — chạy lại tự bỏ qua (resume/dedup)
```

Nối vào pipeline:

```bash
cd ../audio-pipeline
python -m pipeline run --raw-dir ../audio-crawler/raw/uyen-uong-dao --workdir work
```

### Crawl và xử lý nối tiếp

Hướng dẫn cài đặt + chạy thật trên máy GPU: [docs/RUN_GPU.md](docs/RUN_GPU.md).

`tools/crawl_and_process.py` chạy crawler rồi đưa **ngay từng file vừa tải
xong** vào audio-pipeline, không đợi crawl hết:

```bash
python tools/crawl_and_process.py --config config/sources.yaml --out ./raw \
    --workdir ../work --pipeline-dir ../audio-pipeline \
    [--source X] [--limit N] [--batch 1] [--pipeline-config ../audio-pipeline/config/test_cpu.yaml]
```

- Nguồn sự kiện là `ledger.jsonl` của từng source (crawler ghi ngay khi tải xong
  một file); script tail các ledger, gom `--batch` file (mặc định 1) vào
  `<workdir>/_staging/` bằng symlink rồi gọi `python -m pipeline run --stages s0-s7`.
- Crawler xong và hàng đợi cạn thì chạy `--final-stages` (mặc định `s8`: gán
  tier, chia split theo speaker, xuất dataset) một lần trên toàn bộ workdir.
- `<workdir>/processed_files.jsonl` ghi file đã xử lý; chạy lại script bỏ qua
  phần đã làm. Pipeline resume theo hash nên gọi lặp là an toàn. `--no-crawl`
  chỉ xử lý ledger có sẵn. Ctrl-C tắt crawler và xử lý nốt lô đang dở.
- `--cleanup`: tiết kiệm đĩa. Sau mỗi lô xóa audio gốc trong `raw/` (giữ
  sidecar + ledger nên không tải lại) và wav trung gian s0/s1/s2 của lô; sau s8
  xóa wav s7 của segment tier C. Wav s7 của tier A/B được giữ vì s8 xây lại
  dataset từ đó mỗi lần chạy — muốn đổi ngưỡng tier sau này vẫn chạy lại được
  s8, nhưng segment đã là C thì không quay lại được.
- Mỗi lô khởi động lại model của s1/s4/s5, nên với podcast tập ngắn nên đặt
  `--batch 20` trở lên; với file dài 10 giờ thì `--batch 1` là hợp lý.
- `--stream`: thay toàn bộ cơ chế lô bằng `python -m pipeline serve` — 8 stage
  của pipeline chạy song song, mỗi stage một tiến trình, quét thẳng thư mục
  `--out`, model nạp một lần, GPU không nghỉ giữa các lô. Crawler xong thì
  script tạo `<workdir>/INPUT_DONE`; bị kill thì chạy lại lệnh cũ là tiếp tục
  (dấu "đã làm" là dòng manifest của từng stage). `--batch/--stages/--poll`
  không dùng trong chế độ này. Xem README của audio-pipeline, mục "Chế độ stream".
- `--min-free-gb N` (mặc định 20, cả hai chế độ): đĩa chứa `--out` trống dưới N GB
  thì script tạo `<out>/PAUSE`, crawler ngừng tải trước tập kế tiếp (in `[tạm dừng]`),
  pipeline vẫn chạy và giải phóng đĩa; trống lại trên 1.5N GB thì gỡ. Chạy crawler
  tay cũng dừng được bằng `touch <out>/PAUSE`.

## Cơ chế

- **Resume/dedup**: mỗi thư mục nguồn có `ledger.jsonl` (append + flush từng
  dòng) — ngắt giữa chừng chạy lại là tiếp tục; url đã tải không tải lại.
  Nguồn youtube dùng thêm cơ chế tương đương của yt-dlp (`ytdlp_archive.txt`).
- **Lịch sự với server**: nghỉ `delay_s` giây giữa 2 lượt tải (mặc định 2s,
  chỉnh riêng từng nguồn), retry 3 lần với backoff, User-Agent tự nhận diện.
- **Tải an toàn**: stream về file `.part`, thành công mới đổi tên — không bao
  giờ có file cụt trong thư mục output.
- **Cắt file quá dài** (`max_hours` trong config nguồn, mặc định tắt): trước khi
  tải, dùng thời lượng nguồn khai báo (`itunes:duration` của rss, `length` của
  archive.org); thiếu hoặc vượt ngưỡng mới gọi `ffprobe` qua HTTP. Vượt ngưỡng thì `ffmpeg` đọc thẳng URL,
  dừng sau `max_hours` giờ và stream-copy (không re-encode) — với mp3 / m4a
  faststart chỉ tải phần cần, không tải cả file. Sidecar ghi thêm
  `duration_original` và `truncated_to_seconds`. Cần ffmpeg trong PATH. Nguồn
  youtube dùng `--download-sections` của yt-dlp.

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
