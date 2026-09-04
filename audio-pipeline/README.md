# audio-pipeline

> Mô tả chi tiết từng stage (tại sao cần, thuật toán, cấu hình, output):
> xem **[PIPELINE.md](PIPELINE.md)**.

Pipeline làm sạch audio tiếng Việt crawl từ internet (sách nói, truyện audio...)
thành dữ liệu sẵn sàng training TTS/ASR. Thiết kế mô phỏng pipeline của
PhoAudiobook (paper ACL 2025, VinAI) nhưng tổng quát hóa cho audio crawl từ
nguồn bất kỳ.

## Bài toán

Audio crawl từ các trang sách nói/truyện audio ở dạng thô: mp3/m4a/wav lẫn lộn,
có nhạc nền và hiệu ứng, nhiều người nói, không có transcript. Để dùng được cho
training cần: tách nhạc, cắt thành đoạn ngắn, đo chất lượng, gán người nói,
sinh transcript đáng tin cậy, chuẩn hóa văn bản và âm lượng, rồi đóng gói.

## Nguyên tắc thiết kế

1. **Không stage nào vứt dữ liệu trừ bước cuối** — các stage giữa chỉ *đo và
   ghi* chỉ số vào manifest; quyết định lọc dồn hết về s8 qua rule trong config.
   Nhờ đó đổi ngưỡng lọc chỉ cần chạy lại s8 (rẻ), không phải transcribe lại.
2. **Manifest-driven, resume mọi cấp** — mỗi stage đọc manifest JSONL của stage
   trước, xử lý, ghi manifest mới (append từng dòng + flush). Bị ngắt giữa chừng
   thì chạy lại, phần đã xong tự skip. An toàn với dữ liệu nghìn giờ.
3. **Quality tier thay vì lọc cứng** — phục vụ cả TTS lẫn ASR: mỗi segment được
   gắn tier, lúc training tự chọn ngưỡng theo mục đích.
4. **Fallback không GPU** — demucs tắt được, DNSMOS thay bằng SNR, pyannote thay
   bằng ECAPA, whisper large thay bằng model nhỏ — tất cả qua config; pipeline
   vẫn chạy trọn trên CPU với chất lượng lọc thấp hơn.

## Sơ đồ pipeline

```
raw audio ──► s0 ingest ──► s1 separate ──► s2 segment ──► s3 quality ──► s4 speaker ──► s5 transcribe ──► s6 textnorm ──► s7 loudnorm ──► s8 package
              (chuẩn hóa     (demucs tách    (silero-VAD    (SNR/clip/     (lọc đa       (whisper +        (chuẩn hóa     (chuẩn âm      (wav+csv, parquet,
               format,        nhạc nền)       cắt 1–30s)     DNSMOS)        speaker,      cross-check       số/viết tắt)   lượng LUFS)     report thống kê)
               dedup)                                                       gán spk id)   → CER tier)
```

## Chi tiết từng stage

| Stage | Việc làm | Công cụ | Ghi chú |
|---|---|---|---|
| **s0 ingest** | Quét thư mục raw đệ quy, hash SHA1 dedup, convert về WAV mono 24kHz, sinh `manifest.jsonl` (id, source_path, duration, hash) | ffmpeg, soundfile | Giữ metadata nguồn (tên sách, url) nếu có file sidecar `.json` cùng tên từ crawler |
| **s1 separate** | Tách vocal khỏi nhạc nền/hiệu ứng | demucs (htdemucs) | Bật/tắt qua config `separate.enabled`; chế độ `auto` — tách thử mẫu 30s giữa file, đo tỷ lệ RMS nhạc nền/tổng, vượt ngưỡng mới chạy full để tiết kiệm GPU |
| **s2 segment** | VAD tìm vùng có tiếng nói, gộp thành segment 1–30s (config được), cắt tại khoảng lặng, gộp đoạn ngắn liền kề | silero-vad | Segment id = `{file_id}_{start_ms}_{end_ms}`; chunk đơn lẻ dài quá max bị cắt cứng |
| **s3 quality** | Đo và ghi chỉ số mỗi segment: tỷ lệ clipping, SNR ước lượng (chênh lệch năng lượng frame p90−p10), bandwidth thực (phát hiện mp3 upsample), DNSMOS P.835 nếu có model onnx | numpy, onnxruntime (tùy chọn) | **Không lọc** — chỉ ghi chỉ số vào manifest, tier quyết định ở s8 |
| **s4 speaker** | Phát hiện segment nhiều người nói (flag), tính speaker embedding, cluster trong phạm vi từng file nguồn → gán `speaker_id` | ECAPA (speechbrain) mặc định; pyannote diarization tùy chọn | pyannote cần HF token (model gated) — ECAPA là fallback không cần token; flag đa người nói bằng khoảng cách embedding nửa đầu vs nửa sau segment |
| **s5 transcribe** | Transcribe bằng model chính (faster-whisper large-v3), transcribe lại bằng model kiểm chứng (PhoWhisper-large), tính CER giữa 2 bản → độ tin cậy transcript | faster-whisper, transformers | Bước tốn nhất; resume theo từng segment; model/beam/compute_type chọn trong config; mps rơi về cpu cho faster-whisper (ctranslate2 không hỗ trợ mps) |
| **s6 textnorm** | Chuẩn hóa transcript: số → chữ tiếng Việt, viết tắt, khoảng trắng; giữ cả bản raw (`text`) và bản chuẩn hóa (`text_normalized`) | vinorm, fallback bộ chuyển số→chữ tự viết | Chỉ chạy trên transcript của model chính |
| **s7 loudnorm** | Chuẩn hóa âm lượng về mức thống nhất | ffmpeg loudnorm (−23 LUFS) hoặc peak norm, config được | Tắt được qua `loudnorm.mode: off` |
| **s8 package** | Gán **quality tier** theo rule config; chia train/val/test **theo speaker** (hash ổn định, tránh leak giọng); xuất dataset + report | pyarrow, csv | Stage rẻ nhất — đổi ngưỡng tier chỉ chạy lại mình s8 |

## Rule gán tier (mặc định, chỉnh trong `config/default.yaml`)

Xét lần lượt A rồi B; không đạt tier nào → C:

- **Tier A (TTS-grade)**: CER giữa 2 model < 5%, một người nói, DNSMOS ≥ 3.0
  (không có điểm DNSMOS thì thay bằng SNR ≥ 20dB), clipping ≤ 0.1%, duration 1–20s
- **Tier B (ASR-grade)**: CER < 15%, SNR ≥ 10dB (hoặc DNSMOS ≥ 2.5),
  clipping ≤ 1%, duration 1–30s, cho phép đa người nói
- **Tier C (loại)**: còn lại — vẫn nằm trong manifest cuối để đối chiếu,
  không copy vào dataset

Segment không có CER (verify tắt hoặc lỗi) rơi vào tier C — bật
`transcribe.verify` nếu muốn dữ liệu được xếp tier.

## Cài đặt

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# có GPU NVIDIA: cài thêm torch bản CUDA + requirements-gpu.txt
```

Cần `ffmpeg` trong PATH (`brew install ffmpeg` / `apt install ffmpeg`).

Tùy chọn (pipeline tự fallback nếu thiếu):
- `pip install vinorm` — chuẩn hóa văn bản tiếng Việt tốt hơn (s6)
- `pip install onnxruntime` + tải model DNSMOS `sig_bak_ovr.onnx` từ repo
  [microsoft/DNS-Challenge](https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS)
  rồi khai báo `quality.dnsmos.model_path` (s3)
- `pip install pyannote.audio` + HF token — diarization đầy đủ (s4)

## Chạy

```bash
# chạy thử trên 3 file đầu
./scripts/run_sample.sh /path/to/raw_audio ./work_sample

# chạy full
python -m pipeline run --raw-dir /path/to/raw_audio --workdir ./work

# chạy lại một stage (ví dụ đổi ngưỡng tier -> chỉ cần chạy lại s8)
python -m pipeline run --workdir ./work --stages s8

# chạy một dải stage, ép device
python -m pipeline run --workdir ./work --stages s2-s5 --device cuda

# override config
python -m pipeline run --raw-dir raw --workdir work \
  --config config/default.yaml --config my_overrides.yaml
```

CLI:
- `--stages`: `all` | danh sách `s0,s2,s5` | dải `s2-s6` (mặc định `all`)
- `--limit N`: chỉ xử lý N item đầu mỗi stage — dùng chạy thử trước khi chạy full
- `--device`: override `cpu|cuda|mps` (mặc định `auto` trong config)
- `--config`: lặp lại được, file sau override file trước theo từng key

## Input

Thư mục `--raw-dir` chứa audio bất kỳ định dạng (mp3/m4a/wav/flac/ogg/opus/aac/wma),
lồng thư mục con thoải mái. Nếu crawler ghi kèm file metadata cùng tên đuôi
`.json` (vd `book01_ep1.mp3` + `book01_ep1.json`) thì nội dung đó đi theo suốt
pipeline trong trường `source_meta`.

## Cấu trúc thư mục làm việc

```
workdir/
├── s0_ingest/{audio/, manifest.jsonl}
├── s1_separate/...                     # mỗi stage một thư mục + manifest riêng
├── ...
└── s8_package/
    ├── dataset/
    │   ├── wav/{train,val,test}/*.wav  # audio sạch (kiểu HF audiofolder)
    │   ├── metadata.csv                # file_name, text, text_normalized, speaker_id,
    │   │                               # duration, tier, snr_db, dnsmos, cer, clipping,
    │   │                               # bandwidth_hz, multi_speaker, source_path, split
    │   └── parquet/{split}-XXXXX.parquet   # audio bytes nhúng, ~500MB/shard
    ├── report.html                     # thống kê giờ/tier/speaker/split
    └── manifest.jsonl                  # đầy đủ mọi segment kể cả tier C
```

Load bằng HF datasets:

```python
from datasets import load_dataset
ds = load_dataset("parquet", data_files="work/s8_package/dataset/parquet/train-*.parquet")
# hoặc audiofolder từ dataset/ (wav + metadata.csv)
```

## Cấu trúc repo

```
audio-pipeline/
├── README.md
├── requirements.txt / requirements-gpu.txt
├── config/default.yaml        # toàn bộ tham số
├── pipeline/
│   ├── cli.py                 # entry: python -m pipeline run ...
│   ├── manifest.py            # JSONL append-only + resume
│   ├── device.py              # resolve auto|cpu|cuda|mps
│   ├── audio_utils.py         # ffmpeg wrapper, hash, resample
│   └── stages/s0_ingest.py ... s8_package.py
│       # interface chung: run(cfg, workdir, limit) -> đường dẫn manifest
├── scripts/run_sample.sh      # demo end-to-end trên 3 file
└── tests/test_stages.py       # test hàm thuần, không cần GPU/mạng/model
```

## Test

```bash
pip install pytest
pytest tests/ -v
```

Test phủ: resume manifest, logic gộp segment, chỉ số chất lượng (clipping/SNR/
bandwidth), CER, chuyển số→chữ tiếng Việt, peak normalize, rule gán tier, chia
split theo speaker.

## Verify end-to-end sau khi cài

1. Lấy vài file mẫu (ví dụ trích từ phoaudiobook: `python extract_audio.py
   --input data/phoaudiobook --out raw_test --raw --limit 3`)
2. `./scripts/run_sample.sh ./raw_test` → mở `work_sample/s8_package/report.html`
3. Nghe thử vài wav tier A trong `dataset/wav/`, đối chiếu transcript trong
   `metadata.csv`
4. Kiểm tra resume: ngắt giữa s5 (Ctrl-C), chạy lại, xác nhận phần đã
   transcribe không chạy lại
