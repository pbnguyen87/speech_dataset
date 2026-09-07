# Nguồn `rss` — Podcast RSS feed

Mỗi `<item>` có `<enclosure>` là một episode audio. Không cần scrape HTML.

```yaml
- name: ten-podcast
  type: rss
  url: https://feeds.example.com/podcast.xml
  # nhiều feed: dùng thêm/thay bằng
  urls: [https://feeds.example.com/a.xml, https://feeds.example.com/b.xml]
  urls_file: rss_feeds.txt   # mỗi dòng 1 feed, '#' là comment; tương đối theo file config
  dir: .                # tùy chọn, thư mục dưới --out (mặc định = name); "." = ghi thẳng vào --out
  max_items: 100        # tùy chọn, áp cho từng feed; bỏ = lấy hết
  max_hours: 10         # tùy chọn, cắt file dài hơn 10h (cần ffmpeg)
  delay_s: 2.0          # tùy chọn
```

Mỗi feed một thư mục con `raw/<name>/<host_path-của-feed>/` (vd `vnexpress.net_rss_podcast_ban-on-khong.rss`);
ledger vẫn một file chung `raw/<name>/ledger.jsonl`.
Đọc và tải theo từng feed (đọc feed 1 -> tải hết tập của nó -> đọc feed 2...), nên với `urls_file`
hàng nghìn feed, audio đầu tiên về ngay sau vài giây thay vì chờ liệt kê hết.
Feed lỗi chỉ bị báo và bỏ qua, các feed khác vẫn chạy. Sidecar lưu thêm `pub_date`, `description` (cắt 500 ký tự), `feed`.
Tìm feed của podcast qua trang podcast hoặc podcastindex.org.
