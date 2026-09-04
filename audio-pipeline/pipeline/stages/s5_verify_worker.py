"""Worker chạy trong subprocess riêng: transcribe kiểm chứng bằng PhoWhisper.

Tách process vì torch (transformers) và ctranslate2 (faster-whisper) cùng nhúng
OpenMP runtime — load chung một process trên macOS Intel gây abort/deadlock.

Usage: python -m pipeline.stages.s5_verify_worker <in.json> <out.json> <model> <device>
  in.json : {"seg_id": "audio_path", ...}
  out.json: {"seg_id": "transcript hoặc null", ...}
"""

import json
import sys


def main() -> None:
    in_path, out_path, model_name, device = sys.argv[1:5]
    with open(in_path, encoding="utf-8") as f:
        items: dict = json.load(f)

    from transformers import pipeline as hf_pipeline

    hf_dev = 0 if device == "cuda" else (device if device == "mps" else -1)
    asr = hf_pipeline("automatic-speech-recognition", model=model_name, device=hf_dev)

    out = {}
    for seg_id, audio_path in items.items():
        try:
            out[seg_id] = asr(audio_path)["text"].strip()
        except Exception as e:
            print(f"verify lỗi ({seg_id}): {e}", file=sys.stderr)
            out[seg_id] = None

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
