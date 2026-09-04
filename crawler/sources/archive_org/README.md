# Nguồn `archive_org` — Internet Archive

Tải mọi file audio **bản gốc** (`source == original`) trong một item trên
archive.org qua metadata API. Bỏ các bản derivative (ogg, png, spectrogram)
archive.org tự sinh.

```yaml
- name: uyen-uong-dao
  type: archive_org
  identifier: Uynngaowww.truyenaudio.net   # phần sau /details/ trong url
  # nhiều item: dùng thêm/thay bằng
  identifiers: [item-a, https://archive.org/details/item-b]
  identifiers_file: archive_items.txt      # mỗi dòng 1 item, '#' là comment; tương đối theo file config
  delay_s: 2.0                             # tùy chọn
  max_hours: 10         # tùy chọn, cắt file dài hơn 10h (cần ffmpeg)
```

Mỗi mục có thể là identifier trần hoặc url dạng
`https://archive.org/details/<id>` (cũng nhận `/download/`, `/metadata/`),
adapter tự rút identifier. Item lỗi hoặc không tồn tại chỉ bị báo và bỏ qua.

Key resume là `archive.org/<identifier>/<tên file>` (không phải url) vì
server trong url đổi theo load-balancing mỗi lần gọi API.
