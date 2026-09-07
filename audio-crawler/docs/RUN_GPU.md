# Hướng dẫn cài đặt và chạy crawl + xử lý trên máy GPU

Chạy `tools/crawl_and_process.py`: crawl podcast tiếng Việt từ danh sách
`config/podcast_feeds_2h.txt`, đưa ngay từng lô file vào audio-pipeline, dọn đĩa
sau mỗi lô, xuất dataset tier A/B.

## 1. Yêu cầu

- Linux, GPU NVIDIA có driver CUDA (kiểm tra: `nvidia-smi`).
- Python 3.10 - 3.12, `git`, `ffmpeg` (`sudo apt install ffmpeg`).
- Hai repo nằm cạnh nhau:

```
<thư mục gốc>/
├── audio-crawler/      # repo này
└── audio-pipeline/     # https://github.com/pbnguyen87/audio-pipeline
```

- Đĩa: xem mục 6 trước khi chạy full.

## 2. Cài audio-pipeline

```bash
cd audio-pipeline
python3 -m venv .venv && source .venv/bin/activate
# torch bản CUDA trước (đổi cu121 theo CUDA của máy: cu118 / cu121 / cu124)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install vinorm                     # tùy chọn: chuẩn hóa text tiếng Việt tốt hơn (s6)
python -c "import torch; print(torch.cuda.is_available())"   # phải in True
deactivate
```

Lần chạy đầu sẽ tự tải model (Whisper large-v3 ~3 GB, PhoWhisper-large ~3 GB,
demucs, ECAPA, silero-vad) về `~/.cache`. Máy cần mạng ở lần đầu.

## 3. Cài audio-crawler

```bash
cd ../audio-crawler
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install pytest && pytest tests/ -q    # 30 test, không cần mạng
```

`tools/crawl_and_process.py` tự dùng `../audio-pipeline/.venv/bin/python` cho
pipeline, nên chỉ cần kích hoạt venv của audio-crawler.

## 4. Chạy thử trước (bắt buộc)

Ba kênh khác thể loại, mỗi kênh 1 tập cắt 10 phút, đủ 8 stage, có cleanup:

```bash
cd audio-crawler && source .venv/bin/activate
OUT=/data/vi_podcast/test30            # đổi theo máy
mkdir -p $OUT

python tools/crawl_and_process.py \
    --config config/test30.yaml \
    --out $OUT/raw --workdir $OUT/work \
    --pipeline-dir ../audio-pipeline \
    --device cuda --limit 1 --batch 3 --cleanup \
    2>&1 | tee $OUT/run.log
```

Kiểm tra:

```bash
ls $OUT/work/s8_package/dataset/            # metadata.csv, wav/, parquet/
head -5 $OUT/work/s8_package/dataset/metadata.csv
open $OUT/work/s8_package/report.html       # xdg-open trên Linux
```

Với GPU và model large, tier A phải chiếm phần lớn (trên CPU với model small
chỉ ~4% đạt A vì CER cao). Nếu tier A vẫn thấp, xem `cer` trong metadata.csv.

## 5. Chạy thật

```bash
cd audio-crawler && source .venv/bin/activate
OUT=/data/vi_podcast
mkdir -p $OUT

nohup python tools/crawl_and_process.py \
    --config config/vi_podcast.yaml \
    --out $OUT/raw \
    --workdir $OUT/work \
    --pipeline-dir ../audio-pipeline \
    --device cuda \
    --batch 20 \
    --cleanup \
    > $OUT/run.log 2>&1 &
echo $! > $OUT/run.pid
```

Không truyền `--pipeline-config` để dùng `config/default.yaml` của pipeline
(Whisper large-v3 + PhoWhisper-large).

Tham số:

| Tham số | Ý nghĩa |
|---|---|
| `--config config/vi_podcast.yaml` | 1 source `rss`, `urls_file: podcast_feeds_2h.txt` (1694 feed), `max_hours: 10` |
| `--batch 20` | gom 20 file rồi chạy pipeline một lần (mỗi lô nạp lại model; tập podcast ngắn nên gom nhiều) |
| `--cleanup` | xóa audio gốc + wav trung gian sau mỗi lô; sau s8 xóa wav tier C |
| `--source X` | chỉ chạy source tên X |
| `--limit N` | mỗi source chỉ N item đầu (chạy thử) |
| `--no-crawl` | không crawl, chỉ xử lý ledger có sẵn |
| `--stages s0-s7` / `--final-stages s8` | stage chạy theo lô / chạy một lần cuối |

## 6. Theo dõi

```bash
tail -f $OUT/run.log
wc -l $OUT/raw/ledger.jsonl      # file đã tải
wc -l $OUT/work/processed_files.jsonl        # file đã qua pipeline
du -sh $OUT/raw $OUT/work $OUT/work/s8_package/dataset
nvidia-smi                                   # GPU đang dùng khi pipeline chạy
```

Dừng: `kill $(cat $OUT/run.pid)` — script tắt crawler, xử lý nốt lô đang dở rồi
chạy s8. Chạy lại đúng lệnh ở mục 5 để tiếp tục; ba lớp resume (ledger của
crawler, `processed_files.jsonl` của script, manifest của pipeline) bỏ qua phần
đã làm.

## 7. Output

```
$OUT/
├── raw/                      # vi_podcast.yaml khai dir: "." nên không có tầng vi-podcast/
│   ├── ledger.jsonl          # mọi url đã tải (resume) — KHÔNG xóa
│   └── <feed>/               # mỗi feed một thư mục, tên = host_path của url feed
│       ├── *.json            # sidecar metadata (giữ lại sau cleanup)
│       └── *.mp3 / *.m4a     # audio gốc, chỉ tồn tại đến khi lô xử lý xong
├── work/
│   ├── processed_files.jsonl # file đã qua pipeline (resume)
│   ├── s0_ingest ... s7_loudnorm/manifest.jsonl   # metadata từng stage
│   ├── s7_loudnorm/audio/    # wav tier A/B (s8 xây dataset từ đây)
│   └── s8_package/
│       ├── dataset/          # <-- DATASET
│       │   ├── metadata.csv  # file_name, text, text_normalized, speaker_id, duration, tier, snr_db, cer, split...
│       │   ├── wav/{train,val,test}/*.wav   # 24 kHz mono
│       │   └── parquet/
│       ├── manifest.jsonl    # mọi segment kể cả tier C
│       └── report.html
└── run.log
```

## 8. Dung lượng và thời gian — đọc trước khi chạy full

- Danh sách 2h: 1693 kênh, ~118.000 giờ theo feed khai báo; sau `max_hours: 10`
  còn ~103.000 giờ ≈ 5 TB m4a. Dataset tier A/B dạng wav 24 kHz có thể vài TB.
- Pipeline trên một GPU tầm 1x realtime với large-v3 + PhoWhisper-large, tức
  ~100.000 giờ máy cho full danh sách. Nên chạy theo phần:
  - dùng `config/podcast_feeds_10h.txt` (882 kênh) trước, hoặc
  - tách `podcast_feeds_2h.txt` thành nhiều file nhỏ, mỗi file một source trong
    yaml, chạy `--source` lần lượt.
- Feed đầu danh sách là "Truyện Audio Sưu Tầm" (790 tập × 10 giờ = 7.900 giờ, 1
  giọng/tập). Muốn đa dạng giọng sớm thì đảo thứ tự dòng trong file txt
  (`shuf`).
- Crawler tải nhanh hơn pipeline xử lý, `raw/` sẽ phình rồi giảm khi cleanup.
  Đĩa hạn chế thì tăng `delay_s` trong `config/vi_podcast.yaml` hoặc crawl từng phần.
- Đã xóa raw thì không chạy lại được s0-s2 cho file đó (muốn thì xóa dòng
  tương ứng trong ledger để crawler tải lại). Segment tier C đã xóa không quay
  lại được khi nới ngưỡng tier — muốn thử ngưỡng, chạy mẫu nhỏ không `--cleanup`.

## 9. Lỗi thường gặp

| Hiện tượng | Xử lý |
|---|---|
| `torch.cuda.is_available()` = False | cài lại torch đúng CUDA index-url; kiểm tra `nvidia-smi` |
| `Stage s5_transcribe lỗi` OOM | giảm `transcribe.batch_size` trong config pipeline, hoặc dùng `medium` |
| `KeyError: 'ingest'` | truyền `--pipeline-config` mà thiếu default.yaml — script đã tự chèn, chỉ gặp khi gọi pipeline tay |
| Nhiều `lỗi feed ...` trong log | feed chết/đổi host, crawler bỏ qua, các feed khác vẫn chạy |
| yt-dlp / youtube | không dùng trong cấu hình này |
