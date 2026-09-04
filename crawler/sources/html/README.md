# Nguồn `html` — Trang danh sách có link audio trực tiếp

Quét các trang danh sách, nhặt mọi `href` khớp regex đuôi audio. Dùng cho
trang kiểu "một trang chứa nhiều link .mp3 trực tiếp". Trang phức tạp hơn
(link ẩn sau javascript/player) cần viết adapter riêng.

```yaml
- name: trang-x
  type: html
  start_url: https://example.com/sach-noi        # một trang
  start_urls:                                    # hoặc nhiều trang
    - https://example.com/sach-noi/page/{page}   # {page} thay bằng 1..page_count
  start_urls_file: listing_pages.txt             # hoặc file text, mỗi dòng 1 url, '#' là comment
  page_count: 5           # tùy chọn, mặc định 1
  link_pattern: '\.mp3'   # tùy chọn, mặc định: các đuôi audio phổ biến
  follow_detail: true     # tùy chọn, quét thêm 1 cấp trang con
  detail_limit: 30        # tùy chọn, số trang con tối đa mỗi trang danh sách
  delay_s: 3.0            # tùy chọn
  max_hours: 10         # tùy chọn, cắt file dài hơn 10h (cần ffmpeg)
```

Title lấy từ tên file trong url; sidecar lưu thêm `listing_page`.
