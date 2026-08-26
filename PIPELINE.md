# Mô tả chi tiết các stage trong pipeline

Tài liệu này mô tả từng bước của pipeline ở mức "tại sao cần — hoạt động thế nào
— cấu hình gì — ghi ra gì — bước sau dùng thế nào". Tổng quan kiến trúc và cách
chạy xem `README.md`.

## Kiến trúc chung trước khi đi vào từng stage

**Manifest JSONL xuyên suốt.** Mỗi stage đọc `manifest.jsonl` của stage trước,
xử lý, rồi ghi manifest mới trong thư mục riêng (`workdir/s3_quality/manifest.jsonl`...).
Mỗi dòng là một record JSON; stage sau = record stage trước + các trường mới.
Manifest được ghi **append từng dòng + flush ngay**, nên khi bị ngắt giữa chừng,
chạy lại là các id đã có trong manifest tự động được bỏ qua (resume).

**Mỗi stage một subprocess.** CLI không import stage vào process chính mà spawn
`python -m pipeline.stage_runner <stage>` cho từng stage. Lý do: torch (dùng ở
s1/s4 và bước verify của s5) và ctranslate2 (bước primary của s5) mỗi bên nhúng
một bản OpenMP runtime riêng — load cả hai vào cùng một process gây
segfault/deadlock trên macOS Intel. Cách ly process cũng khiến một stage crash
không kéo sập cả pipeline: CLI dừng có kiểm soát, chạy lại sẽ resume.

**`--limit N`** chỉ áp ở cấp *file nguồn* (s0–s2). Các stage làm việc trên
segment (s3 trở đi) luôn xử lý toàn bộ manifest của stage trước — nhờ đó
`--limit 3` nghĩa là "chạy thử trên 3 file đầu" chứ không cắt cụt dữ liệu giữa chừng.

**`device`** (`auto|cpu|cuda|mps`): resolve một lần ở `pipeline/device.py`.
`auto` ưu tiên cuda → mps → cpu. Riêng faster-whisper (ctranslate2) chỉ hỗ trợ
cpu/cuda nên mps tự rơi về cpu; speechbrain và demucs cũng bị ép cpu khi không
có cuda vì MPS chưa ổn định với chúng.

---

## s0 — ingest: chuẩn hóa đầu vào và khử trùng lặp

**Tại sao cần:** audio crawl về ở đủ định dạng (mp3/m4a/wav/flac/ogg...), sample
rate và số kênh lung tung, và crawl nhiều nguồn thường dính file trùng nhau
(cùng một tập truyện đăng lại ở nhiều trang). Mọi stage sau cần một đầu vào
đồng nhất và không trùng lặp.

**Cách hoạt động:**
1. Quét đệ quy thư mục `--raw-dir`, nhận file theo đuôi trong
   `ingest.extensions` (mặc định 8 định dạng phổ biến).
2. Tính **SHA1 của toàn bộ nội dung file**. Hash này dùng cho 2 việc: khử trùng
   lặp (2 file byte-giống-hệt chỉ giữ 1) và sinh `id` = 12 ký tự hex đầu của
   hash — id ổn định, chạy lại không đổi, là nền của cơ chế resume.
3. Convert bằng ffmpeg về **WAV mono, sample rate `ingest.target_sr`**
   (mặc định 24kHz — chuẩn phổ biến của TTS hiện đại; ASR dùng 16k có thể
   downsample sau, còn upsample từ 16k lên thì không cứu được).
4. Nếu cạnh file audio có file `.json` cùng tên (vd `ep01.mp3` + `ep01.json` do
   crawler ghi), nội dung được giữ nguyên vào trường `source_meta` và đi theo
   record suốt pipeline — giữ được tên sách, url nguồn, tên giọng đọc...

**Ghi ra manifest:** `id`, `hash`, `source_path` (tương đối so với raw-dir),
`audio_path`, `sr`, `duration`, `source_meta`.

**Lưu ý:** file convert lỗi (hỏng, DRM) chỉ bị báo và bỏ qua, không làm dừng stage.

---

## s1 — separate: tách giọng khỏi nhạc nền

**Tại sao cần:** sách nói/truyện audio hay có nhạc hiệu mở đầu, nhạc nền và hiệu
ứng (đặc biệt truyện ma, tiên hiệp). Nhạc lẫn trong giọng làm hỏng training TTS
(model học cả nhạc) và giảm độ chính xác transcribe. Đây là bước đầu tiên trong
pipeline PhoAudiobook.

**Cách hoạt động:** dùng **demucs** (model `htdemucs`) chế độ two-stems: tách
audio thành `vocals` (giọng) và `no_vocals` (mọi thứ còn lại), chỉ giữ vocals.
Demucs xuất 44.1kHz stereo nên kết quả được đưa lại về mono 24kHz chuẩn pipeline.
Demucs chạy trong subprocess con (một lớp cách ly nữa, và tận dụng CLI chính chủ
của demucs).

Ba chế độ qua `separate.enabled`:
- `false` — bỏ qua, manifest trỏ thẳng audio của s0 (nguồn biết chắc là sạch).
- `true` — tách mọi file.
- `auto` (mặc định) — vì demucs rất đắt (file 2 phút mất ~1.5 phút CPU), chỉ
  tách file thực sự có nhạc: cắt **mẫu 30 giây ở giữa file** (`auto_sample_seconds`),
  chạy demucs trên mẫu, tính `music_ratio = RMS(no_vocals) / RMS(toàn mẫu)`.
  Vượt `auto_music_ratio` (mặc định 0.10) → tách cả file; dưới → copy nguyên.
  Trong thực nghiệm: file có nhạc hiệu cho ratio ~0.2+, file thuần giọng đọc cho
  ratio 0.006–0.03 — ngưỡng 0.10 phân tách rất rõ. Nếu bước dò mẫu lỗi thì an
  toàn là trên hết: tách full.

**Ghi ra manifest:** `separated` (bool), `music_ratio` (null nếu mode true/false),
`audio_path` mới nếu đã tách.

**Lưu ý:** mẫu lấy ở *giữa* file chứ không phải đầu file — tránh bị nhạc hiệu
mở đầu đánh lừa theo cả hai hướng (file chỉ có nhạc ở 15s đầu mà phần còn lại
sạch thì không đáng tách full; ngược lại lấy mẫu đầu có thể gặp đúng đoạn giới
thiệu không nhạc).

---

## s2 — segment: cắt audio dài thành đoạn ngắn theo khoảng lặng

**Tại sao cần:** model TTS/ASR train trên utterance vài giây đến vài chục giây,
không train trên file 40 phút. Cắt phải rơi vào **khoảng lặng** — cắt giữa từ
tạo dữ liệu rác.

**Cách hoạt động:**
1. Chạy **silero-VAD** trên bản 16kHz của audio (VAD yêu cầu 8k/16k) → danh
   sách các vùng có tiếng nói, với `min_silence_ms` (300ms) quyết định hai vùng
   cách nhau bao lâu thì tính là tách biệt.
2. Gộp các vùng speech thành segment bằng thuật toán greedy
   (`merge_speech_chunks`): nối vùng kế tiếp vào segment hiện tại chừng nào
   tổng độ dài chưa vượt `max_seconds`; vượt thì chốt segment tại *cuối vùng
   trước đó* — tức là luôn cắt ở khoảng lặng. Một vùng speech đơn lẻ dài quá
   `max_seconds` (người đọc nói liền không nghỉ) bị cắt cứng thành nhiều khúc —
   trường hợp bất khả kháng duy nhất.
3. Mỗi segment được nới `speech_pad_ms` (100ms) hai đầu cho khỏi cụt tiếng,
   segment ngắn hơn `min_seconds` (1s) bị bỏ. Audio segment được cắt từ bản
   24kHz (không phải bản 16k của VAD).

**Cấu hình đáng chú ý:** `max_seconds` quyết định phân bố độ dài — mặc định
hiện tại là **13s** (đổi từ 30s): thực nghiệm cho thấy 30s làm mọi segment
vượt ngưỡng duration của tier A (≤20s), còn 13s cho độ dài trung bình ~11s,
đúng vùng lý tưởng của TTS (XTTS và các model zero-shot dùng 3–15s).

**Ghi ra manifest:** mỗi segment một record mới — `id` dạng
`{file_id}_{start_ms:08d}_{end_ms:08d}`, `file_id`, `start`, `end`, `duration`,
`audio_path` riêng; kế thừa `source_path`, `source_meta`, `separated` từ file cha.

**Lưu ý resume:** vì một file sinh nhiều segment, resume ở đây theo *file*
(file đã có segment nào trong manifest thì bỏ qua cả file) — ngắt giữa chừng một
file thì file đó được làm lại từ đầu, chấp nhận được vì VAD rẻ.

---

## s3 — quality: đo chất lượng, tuyệt đối không lọc

**Tại sao cần:** để cuối pipeline phân tier được thì mỗi segment phải có bộ chỉ
số khách quan. Triết lý quan trọng nhất: **s3 chỉ đo và ghi, không vứt gì cả** —
mọi quyết định lọc dồn về s8, nhờ đó đổi ngưỡng lọc chỉ cần chạy lại s8 (giây)
thay vì đo lại từ đầu (giờ).

**Bốn chỉ số, cách tính:**
- `clipping` — tỷ lệ mẫu có |biên độ| ≥ 0.999. Clipping nghĩa là audio bị vỡ
  do thu/khuếch đại quá mức; TTS rất nhạy với lỗi này.
- `snr_db` — SNR *ước lượng* không cần bản sạch tham chiếu: chia audio thành
  frame 30ms (hop 10ms), tính năng lượng từng frame theo dB, lấy
  **percentile 90 trừ percentile 10**. Trực giác: frame to nhất ~ tiếng nói,
  frame nhỏ nhất ~ nhiễu nền trong khoảng lặng giữa các từ; chênh lệch càng lớn
  nền càng sạch. Sách nói studio đo được 30–50dB; thu ngoài trời ồn sẽ tụt dưới 15dB.
- `bandwidth_hz` — tần số roll-off chứa 99% năng lượng phổ. Dùng để bắt "fake
  sample rate": file mp3 64kbps (cắt phổ ở ~8kHz) bị ai đó upsample lên 24kHz —
  duration và sr trông đẹp nhưng chất lượng thật kém; bandwidth thấp bất thường
  sẽ tố cáo.
- `dnsmos` — điểm chất lượng cảm quan (thang MOS 1–5) từ model DNSMOS P.835 của
  Microsoft, chạy qua ONNX. **Tùy chọn**: cần `pip install onnxruntime` và tải
  file `sig_bak_ovr.onnx` rồi khai `quality.dnsmos.model_path`; chế độ `auto`
  tự tắt nếu thiếu (ghi `null`), s8 khi đó tự dùng SNR thay thế.

**Ghi ra manifest:** 4 trường trên, cộng vào record của s2.

---

## s4 — speaker: gán danh tính người nói, phát hiện đa giọng

**Tại sao cần:** hai lý do. (1) Dữ liệu TTS cần biết segment nào cùng một giọng,
và s8 cần `speaker_id` để chia train/val/test **theo speaker** — cùng một giọng
không được nằm ở cả train lẫn test (leak). (2) Segment có hai người nói chồng
nhau làm hỏng training TTS — cần cờ đánh dấu để tier A loại ra.

**Cách hoạt động (backend `ecapa` — mặc định), ba việc:**
1. **Embedding từng segment:** audio resample 16kHz, đưa qua ECAPA-TDNN
   (`speechbrain/spkrec-ecapa-voxceleb`) → vector 192 chiều đặc trưng cho
   *giọng* (âm sắc, cao độ), không phụ thuộc nội dung câu. Hai segment cùng
   người đọc → hai vector gần nhau theo cosine.
2. **Phát hiện đa người nói** (heuristic nhẹ): cắt đôi segment, embedding riêng
   từng nửa; khoảng cách cosine hai nửa vượt `multi_speaker_threshold` (0.45)
   → nhiều khả năng đổi giọng giữa chừng → `multi_speaker: true`. Segment
   dưới 2s bỏ qua kiểm tra.
3. **Gom cụm trong phạm vi từng file nguồn:** embedding của mọi segment cùng
   file gốc đưa vào Agglomerative Clustering (cosine, ngưỡng `cluster_threshold`
   0.35). Mỗi cụm một người nói → `speaker_id = {file_id}_spk{n}`. Sách nói một
   giọng ra đúng 1 cụm; phỏng vấn 2 người ra 2 cụm.

Phạm vi cluster là *từng file*, không so giọng giữa các file — cùng một MC đọc
2 bộ truyện sẽ mang 2 speaker_id. Lựa chọn an toàn (giống PhoAudiobook): thà
tách nhầm 1 người thành 2 id còn hơn gộp nhầm 2 người làm 1.

**Backend thay thế** (`speaker.backend`):
- `pyannote` — diarization đầy đủ (`pyannote/speaker-diarization-3.1`): đếm
  được số speaker thật trong segment, chính xác hơn hẳn, nhưng model gated
  (cần HF token) và nặng. Speaker chính = người chiếm nhiều thời lượng nhất.
- `off` — `speaker_id = file_id`, `multi_speaker = null`; dùng khi biết chắc
  nguồn một giọng và muốn nhanh.

**Ghi ra manifest:** `speaker_id`, `multi_speaker` (không tạo audio mới).
s8 dùng `multi_speaker` trong rule tier và hash `speaker_id` để chia split.

---

## s5 — transcribe: sinh transcript kèm thước đo độ tin cậy

**Tại sao cần:** audio crawl không có transcript — phải tự sinh (pseudo-label).
Nhưng transcript máy sinh có sai, và *không có ground truth để biết sai bao
nhiêu*. Giải pháp (theo PhoAudiobook): transcribe bằng **hai model độc lập** và
đo độ lệch giữa hai bản — hai model khác kiến trúc, khác dữ liệu train mà cho
ra cùng một văn bản thì văn bản đó gần như chắc đúng.

**Cách hoạt động — xử lý theo chunk 32 segment, mỗi chunk 3 pha:**
1. **Primary** (in-process): faster-whisper (mặc định `large-v3`, config
   `transcribe.primary.model`) — bản ctranslate2 tối ưu, `compute_type: auto`
   chọn float16 trên cuda / int8 trên cpu, beam search 5.
2. **Verify** (subprocess riêng — `s5_verify_worker.py`): PhoWhisper
   (`vinai/PhoWhisper-large`, model whisper fine-tune riêng cho tiếng Việt của
   VinAI) transcribe lại từng segment qua transformers. *Vì sao subprocess:*
   transformers kéo torch vào, mà torch + ctranslate2 chung process là xung đột
   OpenMP (xem phần kiến trúc). Worker nhận/trả JSON qua file tạm; worker lỗi
   thì cả chunk chỉ mất phần verify (cer = null), không mất transcript chính.
3. **CER + ghi manifest:** hai transcript được chuẩn hóa (NFC, lowercase, bỏ
   dấu câu, gọn khoảng trắng) rồi tính **CER** (khoảng cách Levenshtein mức ký
   tự / độ dài bản chính). Ghi từng chunk xong mới sang chunk sau → resume mất
   tối đa 1 chunk.

**Diễn giải CER cho đúng:** đây là độ *vênh giữa hai model*, không phải tỷ lệ
lỗi so với sự thật. CER thấp → hai model đồng thuận → transcript đáng tin. CER
cao có thể do transcript sai, audio khó, hoặc một trong hai model yếu — đằng nào
cũng là segment không nên cho vào tier cao. Thực nghiệm với cặp model *small*:
CER 4–12% trên audio sạch; cặp large sẽ thấp hơn đáng kể.

**Ghi ra manifest:** `text` (bản primary — bản chính thức), `text_verify`,
`cer` (null nếu verify tắt/lỗi — s8 sẽ xếp các segment này vào tier C).

**Lưu ý:** đây là stage đắt nhất toàn pipeline (2 lần inference mọi segment).
Muốn nhanh gấp đôi và chấp nhận không có thước đo tin cậy: `verify.enabled: false`.

---

## s6 — textnorm: chuẩn hóa văn bản về dạng đọc

**Tại sao cần:** TTS học ánh xạ văn bản → âm thanh, nên văn bản phải ở *dạng
được đọc ra*: "25/12" phải thành "hai mươi lăm tháng mười hai", "1.500đ" thành
"một nghìn năm trăm đồng". Whisper thường xuất số ở dạng ký số → phải chuyển.

**Cách hoạt động** (`textnorm.backend`):
- `vinorm` — thư viện chuẩn hóa văn bản tiếng Việt chuyên dụng (xử lý số, ngày
  tháng, viết tắt, đơn vị, ký hiệu). Được ưu tiên nếu cài được.
- `basic` — fallback tự viết khi thiếu vinorm: chuẩn hóa Unicode NFC, chuyển
  **số nguyên → chữ tiếng Việt** đúng ngữ pháp (mười *lăm*, hai mươi *mốt*,
  một trăm *lẻ* năm, nghìn/triệu/tỷ), gọn khoảng trắng. Không xử lý ngày tháng,
  đơn vị, viết tắt.
- `auto` (mặc định) — thử vinorm, thiếu thì basic.

Giữ **cả hai bản**: `text` (nguyên gốc từ ASR — dùng cho ASR training, đối
chiếu) và `text_normalized` (dạng đọc — dùng cho TTS training).

**Ghi ra manifest:** `text_normalized`, `textnorm_backend` (để biết dataset được
chuẩn hóa bằng gì).

---

## s7 — loudnorm: chuẩn hóa âm lượng

**Tại sao cần:** dữ liệu gom từ nhiều nguồn có âm lượng chênh nhau rất lớn
(file thu -30dB cạnh file thu sát 0dB). Training trên dữ liệu lệch âm lượng làm
model học sai; nghe kiểm tra cũng khổ.

**Cách hoạt động** (`loudnorm.mode`):
- `lufs` (mặc định) — ffmpeg filter `loudnorm` đưa mọi segment về
  `target_lufs` (-23 LUFS, chuẩn phát thanh EBU R128), true-peak trần -2dB.
  LUFS đo *cảm nhận độ to* của tai người, tốt hơn hẳn peak đơn thuần.
- `peak` — scale tuyến tính cho đỉnh biên độ chạm `peak_dbfs` (-3dB); rẻ hơn
  (thuần numpy, không qua ffmpeg) nhưng không phản ánh cảm nhận độ to.
- `off` — giữ nguyên.

**Ghi ra manifest:** `audio_path` trỏ sang bản đã chuẩn hóa, `loudnorm` (mode).
Đây là bản audio cuối cùng đi vào dataset.

---

## s8 — package: phân tier, chia split, đóng gói

**Tại sao cần:** đây là nơi *duy nhất* quyết định giữ/bỏ, và là nơi biến manifest
thành dataset dùng được ngay cho training.

**Ba việc:**

1. **Gán quality tier** — xét từng segment qua các tier theo thứ tự khai báo
   trong config (A trước, B sau), đạt tier nào nhận tier đó, trượt hết → C:
   - Điều kiện mỗi tier: `cer ≤ max_cer`, duration trong `[min_seconds,
     max_seconds]`, `clipping ≤ max_clipping`, đa người nói chỉ khi
     `allow_multi_speaker`, text không rỗng, và chất lượng âm: **có điểm dnsmos
     thì xét dnsmos, không có thì xét snr** (vì dnsmos là tùy chọn).
   - Mặc định: **A** (TTS) = CER<5%, một giọng, DNSMOS≥3.0 / SNR≥20dB,
     clipping≤0.1%, 1–20s. **B** (ASR) = CER<15%, SNR≥10dB, cho phép đa giọng,
     tới 30s. Segment `cer = null` (verify tắt/lỗi) rơi thẳng xuống C.
   - Tier C **không bị xóa** — vẫn nằm đủ trong `manifest.jsonl` cuối để đối
     chiếu; chỉ không được copy vào dataset.
   - **Chọn tier để xuất** qua `package.keep_tiers`: `null` (mặc định) = xuất
     mọi tier trừ C; hoặc danh sách, vd `keep_tiers: [A]` chỉ xuất tier A —
     tiện chuyển qua lại "A+B cho ASR" / "chỉ A cho TTS" bằng một dòng trong
     file override, không phải xóa block tier. Tier vẫn được gán đầy đủ trong
     manifest; keep_tiers chỉ lọc lúc copy vào dataset. Tier C không bao giờ
     được xuất kể cả khi liệt kê.
2. **Chia train/val/test theo speaker:** hash SHA1 của `"{seed}:{speaker_id}"`
   → số thực u ∈ [0,1) → so với `test_ratio`/`val_ratio` (mặc định 2%/2%).
   Chia theo *speaker* chứ không theo segment để giọng trong test là giọng model
   chưa từng nghe; hash ổn định nên chạy lại không xáo split.
3. **Xuất ba dạng:**
   - `dataset/wav/{split}/*.wav` + `metadata.csv` — chuẩn HF audiofolder, đủ
     mọi cột chỉ số (text, text_normalized, speaker_id, duration, tier, snr_db,
     dnsmos, cer, clipping, bandwidth_hz, multi_speaker, source_path, split) —
     duyệt tay, nghe kiểm tra được ngay.
   - `dataset/parquet/{split}-XXXXX.parquet` — audio bytes nhúng trong cột
     struct `{bytes, path}` (cùng format viVoice/phoaudiobook), shard tự cắt ở
     `shard_max_mb` (500MB) — tiện lưu trữ, upload HF.
   - `report.html` — tổng giờ/segment/speaker, bảng theo tier (số lượng, giờ,
     CER trung bình, SNR trung bình) và theo split.

**Tính chất quan trọng nhất:** s8 rẻ (không inference). Muốn thử ngưỡng lọc
khác — đổi `package.tiers` trong config rồi chạy `--stages s8`, vài giây có
dataset mới, không đụng gì các stage đắt phía trước.
