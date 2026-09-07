"""Worker chạy trong subprocess riêng: transcribe kiểm chứng bằng PhoWhisper.

Tách process vì torch (transformers) và ctranslate2 (faster-whisper) cùng nhúng
OpenMP runtime — load chung một process trên macOS Intel gây abort/deadlock.

Usage: python -m pipeline.stages.s5_verify_worker <in.json> <out.json> <model> <device> [batch_size]
  in.json : {"seg_id": "audio_path", ...}
  out.json: {"seg_id": "transcript hoặc null", ...}
  batch_size: số segment mỗi lần forward (mặc định 1); hết VRAM thì tự giảm một nửa.
"""

import json
import sys


def _is_oom(e: Exception) -> bool:
    return "out of memory" in str(e).lower()


def transcribe_batched(asr, items: dict, batch_size: int) -> dict:
    """asr(list, batch_size=N) -> {seg_id: text|None}. Lỗi OOM -> chia đôi batch; lỗi khác -> None."""
    ids, paths = list(items), list(items.values())
    out = {}
    i = 0
    while i < len(ids):
        n = min(batch_size, len(ids) - i)
        try:
            res = asr(paths[i:i + n], batch_size=n)
            for sid, r in zip(ids[i:i + n], res):
                out[sid] = (r.get("text") or "").strip()
            i += n
        except Exception as e:
            if _is_oom(e) and n > 1:
                batch_size = max(1, n // 2)  # chia đôi từ batch vừa OOM
                print(f"verify OOM, giảm batch_size -> {batch_size}", file=sys.stderr)
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                continue
            if n > 1:  # 1 file hỏng làm cả batch lỗi -> chạy lẻ từng file để chỉ mất file đó
                batch_size = 1
                continue
            print(f"verify lỗi ({ids[i]}): {e}", file=sys.stderr)
            out[ids[i]] = None
            i += 1
    return out


def main() -> None:
    in_path, out_path, model_name, device = sys.argv[1:5]
    batch_size = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    with open(in_path, encoding="utf-8") as f:
        items: dict = json.load(f)

    from transformers import pipeline as hf_pipeline

    hf_dev = 0 if device == "cuda" else (device if device == "mps" else -1)
    kw = {}
    if device == "cuda":  # fp16: nửa VRAM, nhanh ~2x, chất lượng ASR gần như không đổi
        import torch
        kw["torch_dtype"] = torch.float16
    asr = hf_pipeline("automatic-speech-recognition", model=model_name, device=hf_dev, **kw)

    out = transcribe_batched(asr, items, batch_size)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
