# Nguồn `youtube` — Channel / playlist / video qua yt-dlp

Tải audio từ YouTube bằng [yt-dlp](https://github.com/yt-dlp/yt-dlp). Đây là
cách viVoice thu 1000 giờ từ 186 kênh tiếng Việt.

Cài thêm:

```bash
pip install yt-dlp
brew install ffmpeg      # hoặc apt install ffmpeg — cần có trong PATH
```

```yaml
- name: kenh-sach-noi-x
  type: youtube
  url: https://www.youtube.com/@tenkenh/videos   # hoặc playlist / video đơn
  # nhiều link: dùng thêm/thay bằng
  urls:
    - https://www.youtube.com/watch?v=xxx
    - https://www.youtube.com/playlist?list=yyy
  urls_file: youtube_links.txt   # mỗi dòng 1 link, '#' là comment; đường dẫn tương đối theo file config
  max_items: 50            # tùy chọn, số video mới nhất
  audio_format: m4a        # tùy chọn, mặc định m4a (giữ AAC gốc, không re-encode)
  delay_s: 5.0             # tùy chọn, nghỉ giữa 2 video
  match_filter: "duration > 300"     # tùy chọn, cú pháp --match-filter của yt-dlp
  max_hours: 10            # tùy chọn, chỉ tải N giờ đầu (--download-sections, cần ffmpeg)
  extra_args: ["--cookies-from-browser", "chrome"]   # tùy chọn, truyền thẳng cho yt-dlp
```

Cơ chế:

- yt-dlp tự resume bằng `ytdlp_archive.txt` trong thư mục nguồn; chạy lại
  chỉ tải video mới.
- File audio đặt tên theo video id (`<id>.m4a`), title nằm trong sidecar.
- Sau khi yt-dlp xong, `<id>.info.json` được rút gọn thành sidecar `<id>.json`
  (title, url, channel, upload_date, duration, description...) và ghi vào
  `ledger.jsonl` như các nguồn khác, rồi xóa info.json.
- `--limit N` của CLI tương đương `max_items: N` (áp cho từng url/playlist).
- Mọi link trong `url` / `urls` / `urls_file` được gộp vào một lần chạy yt-dlp,
  tải chung một thư mục, chung ledger và archive.

Lưu ý: YouTube hay đổi cơ chế chống bot; nếu bị lỗi "Sign in to confirm",
nâng cấp yt-dlp hoặc dùng `extra_args` với `--cookies-from-browser`.
