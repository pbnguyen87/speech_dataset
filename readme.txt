HƯỚNG DẪN CHUẨN BỊ TRƯỚC KHI CHẠY 2 SCRIPT DOWNLOAD DATASET
=============================================================

2 script trong thư mục này:
  - download_vivoice.py       : tải dataset capleaf/viVoice       -> ./data/viVoice
  - download_phoaudiobook.py  : tải dataset thivux/phoaudiobook   -> ./data/phoaudiobook


BƯỚC 1: CÀI PYTHON VÀ THƯ VIỆN
------------------------------
Yêu cầu Python 3.8 trở lên. Kiểm tra:

    python3 --version

Cài thư viện cần thiết:

    pip install -U huggingface_hub

(Khuyến nghị cài thêm để tải nhanh hơn:  pip install -U "huggingface_hub[hf_transfer]" )


BƯỚC 2: TẠO TÀI KHOẢN HUGGING FACE VÀ LẤY TOKEN
-----------------------------------------------
1. Tạo tài khoản (miễn phí) tại: https://huggingface.co/join
2. Tạo Access Token tại: https://huggingface.co/settings/tokens
   - Bấm "New token", chọn quyền "Read", copy token lại (dạng hf_xxxx...).


BƯỚC 3: CHẤP NHẬN ĐIỀU KHOẢN DATASET viVoice (BẮT BUỘC)
-------------------------------------------------------
viVoice là dataset "gated" (có kiểm soát truy cập):
1. Đăng nhập Hugging Face trên trình duyệt.
2. Vào trang: https://huggingface.co/datasets/capleaf/viVoice
3. Bấm nút đồng ý / "Agree and access repository".

Nếu bỏ qua bước này, download_vivoice.py sẽ báo lỗi 401/403.
(phoaudiobook là dataset public, không cần bước này.)


BƯỚC 4: ĐĂNG NHẬP TOKEN TRÊN MÁY
--------------------------------
Chọn 1 trong 2 cách:

Cách A - đăng nhập 1 lần (khuyến nghị):

    huggingface-cli login
    # dán token hf_xxxx... khi được hỏi

Cách B - dùng biến môi trường mỗi lần chạy:

    export HF_TOKEN=hf_xxxx...


BƯỚC 5: KIỂM TRA DUNG LƯỢNG Ổ ĐĨA
---------------------------------
Cả 2 dataset đều rất lớn (tổng cộng có thể tới hàng trăm GB).
Kiểm tra dung lượng trống:

    df -h .

Nếu chỉ muốn tải thử một phần, dùng --allow-patterns, ví dụ:

    python download_vivoice.py --allow-patterns "data/train-000*"


BƯỚC 6: CHẠY SCRIPT
-------------------
    python download_vivoice.py --workers 8
    python download_phoaudiobook.py --workers 8

Tùy chọn:
  --out <thư mục>      đổi thư mục lưu (mặc định ./data/viVoice và ./data/phoaudiobook)
  --workers <số>       số luồng tải song song (mặc định 4)
  --token <hf_xxx>     truyền token trực tiếp thay cho bước 4
  --allow-patterns     chỉ tải các file khớp mẫu glob

BƯỚC 7 (TÙY CHỌN): TRÍCH XUẤT FILE .WAV TỪ PARQUET
--------------------------------------------------
Dữ liệu tải về ở dạng parquet (audio nằm bên trong). Nếu cần file .wav rời:

    pip install pyarrow soundfile numpy

    python extract_audio.py --input ./data/viVoice --out ./wav/viVoice
    python extract_audio.py --input ./data/phoaudiobook --out ./wav/phoaudiobook

Tùy chọn:
  --limit 20    chỉ trích 20 dòng đầu để chạy thử
  --raw         giữ nguyên định dạng gốc (không re-encode sang wav, nhanh hơn)

Kết quả: các file audio + metadata.csv (tên file wav -> transcript).
CHÚ Ý: wav không nén nên có thể tốn dung lượng gấp vài lần parquet gốc.

LƯU Ý:
- Nếu mạng bị ngắt giữa chừng, chỉ cần chạy lại script — nó sẽ tự tải tiếp
  phần còn dở, không tải lại từ đầu.
- Nên chạy trong tmux/screen nếu tải qua SSH để tránh mất phiên.
