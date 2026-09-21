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
| `--no-s8` | không chạy s8 khi s7 xong; dataset được gói dần bằng `python -m pipeline package` (mục 11). **Bắt buộc** nếu gói với `--drop-wav` |
| `--max-raw-gb 20` / `--resume-raw-gb 5` | `raw/` (audio chưa qua s0) vượt 20 GB thì tạo `$OUT/raw/PAUSE`, crawler ngừng tải trước tập kế; s0 xử lý và xóa dần (cần `--cleanup`), giảm dưới 5 GB thì crawler chạy tiếp. `--max-raw-gb 0` = tắt |
| `--min-free-gb N` | tùy chọn thêm: đĩa chứa `--out` trống dưới N GB cũng dừng crawler; mặc định 0 = bỏ qua |
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
  Ba chốt chặn cùng ngưỡng 20 GB, chạy lại khi giảm dưới 5 GB: crawler dừng khi
  `raw/` vượt `--max-raw-gb`; s0 dừng khi `work/s0_ingest/audio/` và s2 dừng khi
  `work/s2_segment/audio/` vượt `stream.max_dir_gb` trong config pipeline. Chỉ có tác
  dụng khi có `--cleanup` (s0 xóa raw đã chuyển, s2 xóa wav s0 đã cắt, s7 xóa wav s2
  đã chuẩn âm), không thì thư mục không bao giờ giảm và các tiến trình dừng mãi.
  s5 chậm thì s2 đầy trước, kéo theo s0 rồi crawler dừng; dữ liệu tạm tối đa ~60 GB. Đĩa đã đầy giữa chừng: dừng, chạy
  `../audio-pipeline/.venv/bin/python -m pipeline cleanup --workdir $OUT/work --raw-dir $OUT/raw`
  (thêm `--dry-run` để xem trước) rồi chạy lại với `--cleanup`.
- Đã xóa raw thì không chạy lại được s0-s2 cho file đó (muốn thì xóa dòng
  tương ứng trong ledger để crawler tải lại và dòng trong `s0_ingest/manifest.jsonl`). Segment tier C đã xóa không quay
  lại được khi nới ngưỡng tier — muốn thử ngưỡng, chạy mẫu nhỏ không `--cleanup`.

## 9. Chuyển sang máy hoặc đĩa khác giữa chừng

Mọi tiến độ nằm trong manifest và ledger, nên chuyển `$OUT` đi rồi chạy tiếp được.
Ví dụ dưới đây: máy cũ `$OUT=/data/vi_podcast`, máy mới `$OUT_MỚI=/mnt/big/vi_podcast`.

1. **Dừng run trên máy cũ**: `kill $(cat $OUT/run.pid)` rồi chờ dòng `[serve] dừng`
   trong run.log. Máy chết đột ngột hay bị kill -9 cũng không sao: dòng manifest chỉ được
   ghi sau khi output hoàn chỉnh, nên phần dở (wav viết dở, `.part` trong raw, khúc trong
   `_tmp`, dòng JSON cụt) sẽ bị bỏ qua hoặc làm lại, nhiều nhất mất một chunk s5 (32 segment)
   và một file đang ở s1/s2. Bước 4 dọn luôn dòng JSON cụt. Ghi lại tiến độ để so sánh sau:
   ```bash
   ../audio-pipeline/.venv/bin/python -m pipeline status --workdir $OUT/work
   ```

2. **Cài hai repo trên máy mới** theo mục 1 đến 3 (venv, torch cu128, ffmpeg). Muốn
   khỏi tải lại ~10 GB model thì copy luôn cache: `rsync -a ~/.cache/huggingface ~/.cache/torch máy-mới:~/.cache/`.

3. **Copy dữ liệu**, bỏ hai thư mục tạm:
   ```bash
   rsync -a --info=progress2 --exclude _tmp --exclude _staging $OUT/ máy-mới:$OUT_MỚI/
   ```
   `work/_tmp` là file tạm của s1 khi đang tách nhạc, `work/_staging` chỉ chế độ lô cũ dùng.
   Các cờ `INPUT_DONE`, `s*/DONE`, `.gpu.lock`, `raw/PAUSE` copy theo cũng không sao, serve
   tự xóa lúc khởi động. Phần bắt buộc phải có đủ: `raw/ledger.jsonl` + sidecar `.json`
   (crawler không tải lại), `raw/<feed>/*.mp3` còn lại (chưa qua s0), `work/s*/manifest.jsonl`
   (dấu đã làm), `work/s*/audio/` (wav đang chờ stage sau; wav s7 là sản phẩm),
   `work/s0_ingest/skipped.jsonl`.

4. **Đổi đường dẫn tuyệt đối** trong manifest và ledger (chỉ khi `$OUT_MỚI` khác `$OUT`;
   cùng đường dẫn thì bỏ qua bước này):
   ```bash
   cd audio-pipeline && source .venv/bin/activate
   python -m pipeline repair --workdir $OUT_MỚI/work --raw-dir $OUT_MỚI/raw \
       --relocate $OUT $OUT_MỚI --dry-run          # in số dòng sẽ đổi từng file, chưa ghi
   python -m pipeline repair --workdir $OUT_MỚI/work --raw-dir $OUT_MỚI/raw \
       --relocate $OUT $OUT_MỚI
   ```
   Lệnh này đổi tiền tố của `audio_path` trong 9 manifest và `path` trong ledger,
   đường dẫn tương đối như `source_path` giữ nguyên, đồng thời bỏ luôn dòng JSON hỏng nếu có.

5. **Kiểm tra rồi chạy tiếp** với lệnh mục 5 nhưng `OUT=$OUT_MỚI`:
   ```bash
   python -m pipeline status --workdir $OUT_MỚI/work     # số dòng phải bằng bước 1
   ls $OUT_MỚI/work/s2_segment/audio | head            # wav còn nguyên
   ```
   Crawler đọc ledger nên không tải lại tập đã có; các stage đọc manifest nên chỉ làm phần
   còn thiếu. Nếu quên bước 4, stage đầu tiên cần wav sẽ báo `No such file or directory`
   với đường dẫn máy cũ, chạy lại bước 4 là xong.

## 10. Lỗi thường gặp

| Hiện tượng | Xử lý |
|---|---|
| `torch.cuda.is_available()` = False | cài lại torch đúng CUDA index-url; kiểm tra `nvidia-smi` |
| `[s5] ... OOM, giảm batch_size -> N` lặp lại mỗi lô | đặt `transcribe.primary.batch_size` / `transcribe.verify.batch_size` = N trong config pipeline (16 GB: 32/32; 24 GB: 64/48) |
| `TorchCodec is required for save_with_torchcodec` (demucs, s1) | torchaudio >= 2.9; cài lại đúng mục 2: `torch==2.8.0 torchaudio==2.8.0` |
| `demucs lỗi (mã -9)` | OOM killer hết RAM hệ thống (`dmesg -T \| grep -i killed`): demucs nạp cả khúc vào RAM, ~10 GB cho khúc 2 giờ; hạ `separate.chunk_seconds` (vd 1800) trong config pipeline |
| `CalledProcessError ... ffmpeg ... exit status 228` | ENOSPC, hết đĩa: `df -h /tmp $OUT`. File tạm của s1 nằm ở `$OUT/work/_tmp` (không dùng /tmp); đĩa $OUT đầy thì `python -m pipeline cleanup` (mục 8) rồi chạy lại với `--cleanup` |
| `[đĩa] ... tạm dừng crawler` hoặc `[stream] s0_ingest/s2_segment tạm dừng` kéo dài | raw/, s0 audio hoặc s2 audio vượt 20 GB và stage sau chưa tiêu kịp: bình thường nếu s5 là nút thắt; bất thường nếu thiếu `--cleanup` (thư mục không bao giờ giảm) |
| `json.decoder.JSONDecodeError: Unterminated string` | manifest/ledger có dòng ghi dở (đĩa đầy hoặc kill giữa lúc ghi). Bản hiện tại tự bỏ qua dòng hỏng và làm lại bản ghi đó; dọn hẳn bằng `python -m pipeline repair --workdir $OUT/work` |
| `demucs lỗi (mã N)` kèm stderr | đọc stderr in ngay sau: thiếu mạng tải model htdemucs, thiếu ffmpeg, hoặc CUDA OOM |
| `[serve] sN thoát mã ... khởi động lại (k/5)` | stage chết, xem traceback ngay trước dòng đó; hết 5 lần thì serve dừng, sửa rồi chạy lại |
| Crawler tải nguyên tập dài hàng chục giờ | thiếu `ffprobe`/`ffmpeg` trong PATH của venv crawler; bản hiện tại dừng ngay và báo, cài ffmpeg rồi chạy lại |
| `KeyError: 'ingest'` | truyền `--pipeline-config` mà thiếu default.yaml — script đã tự chèn, chỉ gặp khi gọi pipeline tay |
| Nhiều `lỗi feed ...` trong log | feed chết/đổi host, crawler bỏ qua, các feed khác vẫn chạy |
| yt-dlp / youtube | không dùng trong cấu hình này |

## 11. Gói dần khi s7 có thêm file mới và đẩy lên Hugging Face

Full s8 (`--stages s8`, hoặc bước cuối của `serve`) xây lại toàn bộ dataset từ wav s7 mỗi
lần chạy: với hàng chục nghìn giờ thì mất nhiều giờ copy, phải giữ wav A/B ở s7 và mỗi lần
chạy lại đổi tên shard parquet nên upload lại từ đầu. Thay vào đó dùng
`python -m pipeline package`: chỉ gói những segment s7 **chưa có trong manifest s8**, nối
thêm vào dataset đang có, chạy tay bất cứ lúc nào s7 có thêm file (kể cả khi `serve` đang chạy).

### 11.1. Chạy pipeline với `--no-s8`

```bash
cd audio-crawler && source .venv/bin/activate
OUT=/data/vi_podcast

nohup python tools/crawl_and_process.py \
    --config config/vi_podcast.yaml \
    --out $OUT/raw --workdir $OUT/work \
    --pipeline-dir ../audio-pipeline \
    --device cuda --stream --cleanup --no-s8 \
    > $OUT/run.log 2>&1 &
echo $! > $OUT/run.pid
```

`--no-s8` để `serve` dừng sau s7 (in `bỏ qua s8_package (--no-s8)`), không xây lại dataset
đè lên phần đã gói. Thiếu cờ này mà đã gói với `--drop-wav` thì s8 cuối sẽ lỗi vì wav s7
không còn; nếu wav còn thì nó xóa `dataset/` và đổi tên toàn bộ shard, lần upload sau phải
đẩy lại 100%.

### 11.2. Gói phần mới của s7

Chạy mỗi khi muốn (ví dụ mỗi ngày, hoặc mỗi khi `python -m pipeline status` thấy s7 tăng):

```bash
cd audio-pipeline && source .venv/bin/activate
python -m pipeline package --workdir $OUT/work --dry-run          # xem số segment mới, tier, chưa ghi gì
python -m pipeline package --workdir $OUT/work --cleanup --drop-wav
```

| Tham số | Ý nghĩa |
|---|---|
| (mặc định) | gói segment s7 chưa có trong `s8_package/manifest.jsonl`; wav A/B vào `dataset/wav/{split}/` bằng hardlink (không tốn đĩa), parquet mỗi lô một shard `train-inc-<hash>-NNNNN.parquet`, nối `metadata.csv`, ghi manifest, sinh lại `report.html` |
| `--cleanup` | xóa wav s7 của segment tier C ngay sau khi gán tier |
| `--drop-wav` | dataset chỉ giữ parquet: không link wav vào `dataset/wav/`, xóa wav s7 của segment A/B ngay sau khi lô đã nằm trong parquet + manifest. Đĩa chỉ cần dư ~1 lô (2000 segment ≈ 1 GB). Wav khôi phục được từ parquet: `python extract_audio.py --input $OUT/work/s8_package/dataset/parquet --out wav --raw` |
| `--chunk N` | số segment mỗi lô (mặc định 2000, ≈ 1 GB wav) |
| `--copy` | copy thay vì hardlink khi không dùng `--drop-wav` |

Tính chất:

- Dấu "đã gói" là dòng trong `s8_package/manifest.jsonl`, ghi **sau** parquet và csv. Kill
  giữa chừng rồi chạy lại: lô dở được làm lại, tên shard suy từ id đầu lô nên ghi đè đúng
  file cũ, dòng csv mồ côi bị dọn lúc khởi động. Không sinh dòng trùng.
- Wav của segment đã gói không bao giờ được đọc lại, nên xóa ở s7 được (đó là việc
  `--drop-wav` làm). Sau khi đã xóa thì **không chạy full `--stages s8` nữa**.
- `s8_package/config_hash` ghi `package.tiers / keep_tiers / split` đã dùng; đổi ngưỡng tier
  thì lệnh từ chối nối thêm. Muốn đổi ngưỡng phải xây lại từ đầu, tức là cần wav s7 còn đủ
  (hoặc trích lại từ parquet).
- **Không chia train/val/test.** `config/default.yaml` đặt `package.split.val_ratio: 0` và
  `test_ratio: 0` nên mọi segment vào `train` (một thư mục `dataset/wav/train/`, shard
  `train-*.parquet`, cột `split` trong `metadata.csv` luôn là `train`). Việc chia để lúc
  training tự làm, nên chia theo `speaker_id` để cùng một giọng không nằm ở cả train lẫn test.
  Code chia theo speaker (`assign_split`, hash SHA1 của `seed:speaker_id`) vẫn giữ trong
  `s8_package.py`: muốn chia lại chỉ cần đặt tỷ lệ trong config, nhưng đó là đổi config
  tier/split nên `pipeline package` sẽ từ chối nối thêm vào dataset đang có (xem `config_hash`
  ở trên), phải xây lại từ đầu.

### 11.3. Đẩy lên Hugging Face

Một lần: `pip install huggingface_hub` (đã có trong venv audio-pipeline) và
`huggingface-cli login` bằng token Write. Script `upload_hf.py` nằm ở thư mục gốc
`speech_dataset/` (cạnh hai script download).

```bash
cd speech_dataset
audio-pipeline/.venv/bin/python upload_hf.py \
    --workdir $OUT/work --repo <user>/<ten-repo> --batch-shards 1
```

- Hardlink parquet vào `$OUT/work/s8_package/_hf_staging/data/` (0 byte thêm), sinh `README.md`
  (dataset card khai cột `audio` kiểu Audio, số giờ / segment / speaker từ `metadata.csv`),
  copy `metadata.csv`, rồi `upload_large_folder`: file đã có trên repo (cùng hash) bị bỏ qua,
  chỉ shard mới được đẩy. Bố cục repo: `data/train-*.parquet`, `metadata.csv`, `README.md`;
  `load_dataset("<user>/<ten-repo>", split="train")` dùng được ngay.
- `--batch-shards 1`: mỗi shard một commit. `upload_large_folder` với < 150 file chỉ commit
  một lần ở cuối, và shard đã lên mà chưa commit chỉ được client tin trong 20 giờ; chia đợt
  thì ngắt lúc nào cũng chỉ mất phần dở của một shard (~6 phút ở 1,4 MB/s). Mạng nhanh có
  thể tăng lên 5–10.
- Ngắt (Ctrl-C, mất mạng) rồi chạy lại đúng lệnh là tiếp: trạng thái băm / đã lên / đã commit
  từng file nằm trong `_hf_staging/.cache`. Lỗi mạng thoáng qua (HF đóng kết nối) được thử
  lại tự động tới 30 lần, chờ 15 s × số lần. Tiến trình treo ở `committing: 1` quá 10 phút
  không có lưu lượng thì kill và chạy lại.
- `--with-manifest` đẩy thêm `manifest.jsonl` (đủ mọi segment kể cả tier C); `--public` tạo
  repo public (mặc định private).

Vòng lặp mỗi khi s7 có thêm file:

```bash
python -m pipeline status --workdir $OUT/work                       # s7=N tăng?
python -m pipeline package --workdir $OUT/work --cleanup --drop-wav  # gói phần mới
audio-pipeline/.venv/bin/python upload_hf.py --workdir $OUT/work --repo <user>/<ten-repo> --batch-shards 1
```

Hai lệnh sau gọi lặp bao nhiêu lần cũng được: không có gì mới thì in `0 segment mới` /
`còn 0 chưa commit` rồi thoát.

