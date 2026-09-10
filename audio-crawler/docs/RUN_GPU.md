# Hướng dẫn cài đặt và chạy crawl + xử lý trên máy GPU

Chạy `tools/crawl_and_process.py --stream`: crawl podcast tiếng Việt từ danh sách
`config/podcast_feeds_2h.txt`, audio-pipeline chạy 8 stage song song và xử lý ngay
từng file vừa tải, dọn đĩa ngay khi không cần nữa, xuất dataset tier A/B.

## 1. Yêu cầu

- Linux, GPU NVIDIA có driver CUDA (kiểm tra: `nvidia-smi`).
- Python 3.10 - 3.12, `git`, `ffmpeg` + `ffprobe` (`sudo apt install ffmpeg`; crawler
  cần cả hai để cắt tập dài, thiếu thì dừng ngay thay vì tải nguyên file 38 giờ).
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
# torch bản CUDA trước, GIỮ 2.8 (torchaudio 2.9+ đòi torchcodec, demucs 4.0.1 lỗi
# "TorchCodec is required"). cu128 chạy được với driver CUDA 12.8 trở lên, kể cả 13.x.
pip install "torch==2.8.0" "torchaudio==2.8.0" --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install vinorm                     # tùy chọn: chuẩn hóa text tiếng Việt tốt hơn (s6)
python -c "import torch; print(torch.cuda.is_available())"   # phải in True
deactivate
```

Lần chạy đầu sẽ tự tải model (Whisper large-v3 ~3 GB, PhoWhisper-large ~3 GB,
demucs, ECAPA, silero-vad) về `~/.cache`. Máy cần mạng ở lần đầu. Dòng cảnh báo
`unauthenticated requests to the HF Hub` là vô hại; muốn tải nhanh hơn thì
`huggingface-cli login` một lần (token Read tạo tại huggingface.co/settings/tokens).

## 3. Cài audio-crawler

```bash
cd ../audio-crawler
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install pytest && pytest tests/ -q    # 42 test, không cần mạng
```

`tools/crawl_and_process.py` tự dùng `../audio-pipeline/.venv/bin/python` cho
pipeline, nên chỉ cần kích hoạt venv của audio-crawler.

## 4. Chạy thử trước (bắt buộc)

Ba kênh khác thể loại, mỗi kênh 1 tập cắt 10 phút, đủ 9 stage, có cleanup, cùng
chế độ `--stream` như chạy thật:

```bash
cd audio-crawler && source .venv/bin/activate
OUT=/data/vi_podcast/test30            # đổi theo máy
mkdir -p $OUT

python tools/crawl_and_process.py \
    --config config/test30.yaml \
    --out $OUT/raw --workdir $OUT/work \
    --pipeline-dir ../audio-pipeline \
    --device cuda --limit 1 --stream --cleanup \
    2>&1 | tee $OUT/run.log
```

Trong log phải thấy `[s5] ... primary=large-v3@cuda/float16` và `verify: vinai/PhoWhisper-large@cuda`;
kết thúc là dòng `[serve] hoàn tất: IN✓ s0=3✓ ... s8=…✓`.

Kiểm tra:

```bash
ls $OUT/work/s8_package/dataset/            # metadata.csv, wav/, parquet/
head -5 $OUT/work/s8_package/dataset/metadata.csv
xdg-open $OUT/work/s8_package/report.html
nvidia-smi                                  # trong lúc chạy: VRAM ~7 GB nền + batch s5
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
    --stream \
    --cleanup \
    > $OUT/run.log 2>&1 &
echo $! > $OUT/run.pid
```

Không truyền `--pipeline-config` để dùng `config/default.yaml` của pipeline
(Whisper large-v3 + PhoWhisper-large, batch 64 cho cả hai; hết VRAM thì s5 tự giảm
một nửa và in `OOM, giảm batch_size`, khi đó đặt hẳn giá trị đó vào
`transcribe.primary.batch_size` / `transcribe.verify.batch_size` để khỏi thử lại mỗi lô).

Tham số:

| Tham số | Ý nghĩa |
|---|---|
| `--config config/vi_podcast.yaml` | 1 source `rss`, `urls_file: podcast_feeds_2h.txt` (1694 feed), `max_hours: 10` |
| `--stream` | pipeline chạy 8 stage song song (`python -m pipeline serve`), model nạp một lần, GPU không nghỉ; bị kill chạy lại là tiếp (xem bên dưới) |
| `--cleanup` | xóa audio gốc ngay khi s0 xong, wav trung gian ngay khi stage cuối dùng xong; sau s8 xóa wav tier C |
| `--min-free-gb 20` | đĩa chứa `--out` trống dưới 20 GB thì tạo `$OUT/raw/PAUSE`, crawler ngừng tải trước tập kế cho pipeline giải phóng; trống lại trên 30 GB thì gỡ. 0 = tắt |
| `--source X` | chỉ chạy source tên X |
| `--limit N` | mỗi source chỉ N item đầu (chạy thử) |
| `--no-crawl` | không crawl, chỉ xử lý ledger có sẵn |
| `--batch N`, `--stages`, `--final-stages`, `--poll` | chỉ dùng cho chế độ lô cũ (bỏ `--stream`): gom N file rồi gọi pipeline một lần, mỗi lô nạp lại model |

### Chế độ stream hoạt động thế nào

Mỗi stage s0–s7 của pipeline là một tiến trình, đọc dần manifest của stage trước và
ghi manifest của mình từng dòng — dòng manifest là dấu "đã làm". Bị kill (máy tắt,
OOM, Ctrl-C) thì chạy lại đúng lệnh ở trên là tiếp tục, không làm lại việc đã xong.
Stage nào chết thì supervisor tự khởi động lại (tối đa 5 lần). Crawler xong thì script
tạo `$OUT/work/INPUT_DONE`, các stage lần lượt cạn hàng đợi và tạo cờ `DONE`, rồi s8
chạy một lần. Trên một GPU, s1/s4/s5 thay phiên dùng GPU nhờ khóa `$OUT/work/.gpu.lock`.

## 6. Theo dõi

```bash
tail -f $OUT/run.log                         # log 8 stage gộp, mỗi dòng có tiền tố [s0]..[s7]
grep "^\[serve\]" $OUT/run.log | tail -3     # dòng trạng thái mỗi phút: IN✓ s0=…✓ s1=… (✓ = stage đã DONE)
wc -l $OUT/raw/ledger.jsonl                  # file đã tải
../audio-pipeline/.venv/bin/python -m pipeline status --workdir $OUT/work   # số dòng manifest từng stage
du -sh $OUT/raw $OUT/work $OUT/work/s8_package/dataset
nvidia-smi                                   # GPU đang dùng khi pipeline chạy
```

Dừng: `kill $(cat $OUT/run.pid)` — script tắt crawler và cả 8 stage ngay (không
chạy s8). Chạy lại đúng lệnh ở mục 5 để tiếp tục; hai lớp resume (ledger của
crawler, manifest từng stage của pipeline) bỏ qua phần đã làm. Muốn xuất dataset
từ những gì đã có mà không crawl thêm: thêm `--no-crawl` (pipeline xử lý nốt rồi chạy s8).

## 7. Output

```
$OUT/
├── raw/                      # vi_podcast.yaml khai dir: "." nên không có tầng vi-podcast/
│   ├── ledger.jsonl          # mọi url đã tải (resume) — KHÔNG xóa
│   └── <feed>/               # mỗi feed một thư mục, tên = host_path của url feed
│       ├── *.json            # sidecar metadata (giữ lại sau cleanup)
│       └── *.mp3 / *.m4a     # audio gốc, với --cleanup chỉ tồn tại đến khi s0 chuyển xong sang wav
├── work/
│   ├── INPUT_DONE            # cờ: crawler đã xong, không còn file mới (script tạo)
│   ├── s0_ingest ... s7_loudnorm/manifest.jsonl   # metadata từng stage = dấu đã làm (resume)
│   ├── s*/DONE               # cờ stage đã cạn hàng đợi (xóa khi chạy lại)
│   ├── s0_ingest/skipped.jsonl   # file raw bị bỏ qua (trùng hash / lỗi convert)
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
- Tốc độ pipeline trên một GPU chưa đo lại sau khi gộp batch cho s5 và chạy stream;
  ước tính cũ (từng segment một) là ~1x realtime với large-v3 + PhoWhisper-large.
  Đo ở lần chạy thử mục 4 (tổng giờ audio trong report.html chia thời gian chạy)
  rồi nhân lên. Nên chạy theo phần:
  - dùng `config/podcast_feeds_10h.txt` (882 kênh) trước, hoặc
  - tách `podcast_feeds_2h.txt` thành nhiều file nhỏ, mỗi file một source trong
    yaml, chạy `--source` lần lượt.
- Feed đầu danh sách là "Truyện Audio Sưu Tầm" (790 tập × 10 giờ = 7.900 giờ, 1
  giọng/tập). Muốn đa dạng giọng sớm thì đảo thứ tự dòng trong file txt
  (`shuf`).
- Crawler tải nhanh hơn pipeline xử lý nhiều lần. Dữ liệu đang chờ nằm ở stage nút
  thắt (s5) dưới dạng wav s2, khoảng 170 MB mỗi giờ audio, cộng raw chưa qua s0.
  `--min-free-gb` (mặc định 20) tự tạm dừng crawler khi đĩa sắp đầy; muốn giữ
  đĩa dư nhiều hơn thì tăng lên. Đĩa đã đầy giữa chừng: dừng, chạy
  `../audio-pipeline/.venv/bin/python -m pipeline cleanup --workdir $OUT/work --raw-dir $OUT/raw`
  (thêm `--dry-run` để xem trước) rồi chạy lại với `--cleanup`.
- Đã xóa raw thì không chạy lại được s0-s2 cho file đó (muốn thì xóa dòng
  tương ứng trong ledger để crawler tải lại và dòng trong `s0_ingest/manifest.jsonl`). Segment tier C đã xóa không quay
  lại được khi nới ngưỡng tier — muốn thử ngưỡng, chạy mẫu nhỏ không `--cleanup`.

## 9. Lỗi thường gặp

| Hiện tượng | Xử lý |
|---|---|
| `torch.cuda.is_available()` = False | cài lại torch đúng CUDA index-url; kiểm tra `nvidia-smi` |
| `[s5] ... OOM, giảm batch_size -> N` lặp lại mỗi lô | đặt `transcribe.primary.batch_size` / `transcribe.verify.batch_size` = N trong config pipeline (16 GB: 32/32; 24 GB: 64/48) |
| `TorchCodec is required for save_with_torchcodec` (demucs, s1) | torchaudio >= 2.9; cài lại đúng mục 2: `torch==2.8.0 torchaudio==2.8.0` |
| `demucs lỗi (mã -9)` | OOM killer hết RAM hệ thống (`dmesg -T \| grep -i killed`): demucs nạp cả khúc vào RAM, ~10 GB cho khúc 2 giờ; hạ `separate.chunk_seconds` (vd 1800) trong config pipeline |
| `CalledProcessError ... ffmpeg ... exit status 228` | ENOSPC, hết đĩa: `df -h /tmp $OUT`. File tạm của s1 nằm ở `$OUT/work/_tmp` (không dùng /tmp); đĩa $OUT đầy thì `python -m pipeline cleanup` (mục 8) rồi chạy lại với `--cleanup --min-free-gb` |
| `demucs lỗi (mã N)` kèm stderr | đọc stderr in ngay sau: thiếu mạng tải model htdemucs, thiếu ffmpeg, hoặc CUDA OOM |
| `[serve] sN thoát mã ... khởi động lại (k/5)` | stage chết, xem traceback ngay trước dòng đó; hết 5 lần thì serve dừng, sửa rồi chạy lại |
| Crawler tải nguyên tập dài hàng chục giờ | thiếu `ffprobe`/`ffmpeg` trong PATH của venv crawler; bản hiện tại dừng ngay và báo, cài ffmpeg rồi chạy lại |
| `KeyError: 'ingest'` | truyền `--pipeline-config` mà thiếu default.yaml — script đã tự chèn, chỉ gặp khi gọi pipeline tay |
| Nhiều `lỗi feed ...` trong log | feed chết/đổi host, crawler bỏ qua, các feed khác vẫn chạy |
| yt-dlp / youtube | không dùng trong cấu hình này |
