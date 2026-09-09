"""Worker chạy trong subprocess riêng: transcribe kiểm chứng bằng PhoWhisper.

Tách process vì torch (transformers) và ctranslate2 (faster-whisper) cùng nhúng
OpenMP runtime — load chung một process trên macOS Intel gây abort/deadlock.

Hai cách gọi:
  1) một lượt:  python -m pipeline.stages.s5_verify_worker <in.json> <out.json> <model> <device> [batch_size]
  2) thường trú: python -m pipeline.stages.s5_verify_worker --serve <model> <device>
     nạp model một lần, đọc từng dòng JSON {"in":..., "out":..., "batch_size":N} từ stdin,
     xử lý xong in "ok\\n" (hoặc "err <msg>\\n") ra stdout. Chỉ giao thức mới được in ra stdout.
  in.json : {"seg_id": "audio_path", ...}
  out.json: {"seg_id": "transcript hoặc null", ...}
  batch_size: số segment mỗi lần forward (mặc định 1); hết VRAM thì tự giảm một nửa.
"""

import json
import os
import sys

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")


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


def load_asr(model_name: str, device: str):
    from transformers import pipeline as hf_pipeline

    hf_dev = 0 if device == "cuda" else (device if device == "mps" else -1)
    kw = {}
    if device == "cuda":  # fp16: nửa VRAM, nhanh ~2x, chất lượng ASR gần như không đổi
        import torch
        kw["torch_dtype"] = torch.float16
    return hf_pipeline("automatic-speech-recognition", model=model_name, device=hf_dev, **kw)


def _handle(asr, in_path: str, out_path: str, batch_size: int) -> None:
    with open(in_path, encoding="utf-8") as f:
        items: dict = json.load(f)
    out = transcribe_batched(asr, items, batch_size)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)


def serve(model_name: str, device: str) -> None:
    asr = load_asr(model_name, device)
    print("ready", flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            _handle(asr, req["in"], req["out"], int(req.get("batch_size", 1)))
            print("ok", flush=True)
        except Exception as e:
            print(f"err {str(e)[-300:]}".replace("\n", " "), flush=True)


def main() -> None:
    if sys.argv[1:2] == ["--serve"]:
        serve(sys.argv[2], sys.argv[3])
        return
    in_path, out_path, model_name, device = sys.argv[1:5]
    batch_size = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    _handle(load_asr(model_name, device), in_path, out_path, batch_size)


if __name__ == "__main__":
    main()
