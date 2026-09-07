"""s5: transcribe 2 lượt — model chính (faster-whisper) + model kiểm chứng
(PhoWhisper), tính CER giữa 2 bản để đánh giá độ tin cậy transcript.

Bước verify chạy trong SUBPROCESS riêng (s5_verify_worker): torch và
ctranslate2 cùng nhúng OpenMP runtime, load chung một process trên macOS Intel
gây abort/deadlock. Xử lý theo chunk để giữ khả năng resume.
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import unicodedata

from .. import device as device_mod, manifest

# safety net nếu vẫn có 2 OpenMP runtime trong một process
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

STAGE = "s5_transcribe"
PREV = "s4_speaker"
CHUNK = 32  # segment mỗi chunk tối thiểu: transcribe -> verify -> ghi manifest (thực tế >= 2*batch_size)

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")


def norm_for_cer(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "").lower()
    text = _PUNCT.sub(" ", text)
    return _WS.sub(" ", text).strip()


def cer(ref: str, hyp: str) -> float:
    """Character error rate (Levenshtein / len(ref)); 2 chuỗi rỗng -> 0."""
    ref, hyp = norm_for_cer(ref), norm_for_cer(hyp)
    if not ref and not hyp:
        return 0.0
    if not ref or not hyp:
        return 1.0
    prev = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        cur = [i]
        for j, hc in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1] / len(ref)


def _verify_chunk(chunk: list[dict], model: str, dev: str, batch_size: int = 1) -> dict:
    """Chạy PhoWhisper trên chunk segment trong subprocess riêng, batch_size segment mỗi forward."""
    with tempfile.TemporaryDirectory() as td:
        in_path = os.path.join(td, "in.json")
        out_path = os.path.join(td, "out.json")
        with open(in_path, "w", encoding="utf-8") as f:
            json.dump({r["id"]: r["audio_path"] for r in chunk}, f)
        proc = subprocess.run(
            [sys.executable, "-m", "pipeline.stages.s5_verify_worker",
             in_path, out_path, model, dev, str(batch_size)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            print(f"  verify worker lỗi (chunk bỏ verify): {proc.stderr.strip()[-300:]}")
            return {}
        with open(out_path, encoding="utf-8") as f:
            return json.load(f)


def _is_oom(e: Exception) -> bool:
    return "out of memory" in str(e).lower()


def primary_batched(model, paths: list[str], language: str, beam_size: int, batch_size: int) -> list[str]:
    """Transcribe nhiều segment ngắn (<=30s) độc lập trong MỘT lượt encode + generate.

    Đi thẳng vào encoder/generate của ctranslate2 giống BatchedInferencePipeline của
    faster-whisper, nhưng gộp nhiều file thay vì chia 1 file dài — vì segment ở đây
    chỉ 1-13s nên transcribe() từng file bỏ phí GPU. OOM -> chia đôi batch."""
    import numpy as np
    from faster_whisper.audio import decode_audio
    from faster_whisper.tokenizer import Tokenizer
    from faster_whisper.transcribe import get_suppressed_tokens

    fe = model.feature_extractor
    tok = Tokenizer(model.hf_tokenizer, model.model.is_multilingual, task="transcribe", language=language)
    prompt = model.get_prompt(tok, [], without_timestamps=True)
    suppress = get_suppressed_tokens(tok, [-1])
    n_samples, n_frames = fe.n_samples, fe.nb_max_frames

    def features(path):
        audio = decode_audio(path, sampling_rate=fe.sampling_rate)[:n_samples]
        audio = np.pad(audio, (0, n_samples - len(audio)))  # pad 30s như whisper gốc
        return fe(audio, padding=0)[..., :n_frames]

    texts: list[str] = []
    i = 0
    while i < len(paths):
        n = min(batch_size, len(paths) - i)
        try:
            enc = model.encode(np.stack([features(p) for p in paths[i:i + n]]))
            results = model.model.generate(
                enc, [prompt] * n, beam_size=beam_size, max_length=model.max_length,
                suppress_blank=True, suppress_tokens=suppress,
            )
            texts += [tok.decode(r.sequences_ids[0]).strip() for r in results]
            i += n
        except Exception as e:
            if _is_oom(e) and n > 1:
                batch_size = max(1, n // 2)
                print(f"  primary OOM, giảm batch_size -> {batch_size}")
                continue
            raise
    return texts


def run(cfg: dict, workdir: str, limit: int | None = None) -> str:
    from faster_whisper import WhisperModel

    tcfg = cfg["transcribe"]
    dev = device_mod.resolve(cfg["device"])
    ct_dev = device_mod.for_ctranslate2(dev)
    compute = tcfg["primary"]["compute_type"]
    if compute == "auto":
        compute = "float16" if ct_dev == "cuda" else "int8"

    # limit áp ở cấp file nguồn (s0-s2); stage segment-level xử lý toàn bộ
    records = manifest.read_records(manifest.manifest_path(workdir, PREV))
    mpath = manifest.manifest_path(workdir, STAGE)
    todo = [r for r in records if r["id"] not in manifest.done_ids(mpath)]
    p_batch = int(tcfg["primary"].get("batch_size", 1))
    print(f"[{STAGE}] {len(todo)}/{len(records)} segment cần transcribe "
          f"(primary={tcfg['primary']['model']}@{ct_dev}/{compute}, batch_size={p_batch})")
    if not todo:
        return mpath

    primary = WhisperModel(tcfg["primary"]["model"], device=ct_dev, compute_type=compute)
    verify_on = tcfg["verify"]["enabled"]
    batch_size = int(tcfg["verify"].get("batch_size", 1))
    chunk_n = max(CHUNK, 2 * batch_size, 2 * p_batch)  # mỗi lần spawn worker (nạp lại model) xử lý >= 2 batch
    if verify_on:
        print(f"  verify: {tcfg['verify']['model']}@{dev} (subprocess, batch_size={batch_size}, chunk={chunk_n})")

    with manifest.ManifestWriter(mpath) as w:
        for c0 in range(0, len(todo), chunk_n):
            chunk = todo[c0:c0 + chunk_n]

            # pha 1: transcribe chính (in-process, chỉ ctranslate2), gộp batch nhiều segment
            texts = {}
            if p_batch > 1:
                outs = primary_batched(primary, [r["audio_path"] for r in chunk],
                                       tcfg["primary"]["language"], tcfg["primary"]["beam_size"], p_batch)
                texts = {r["id"]: t for r, t in zip(chunk, outs)}
            else:
                for rec in chunk:
                    segs, _ = primary.transcribe(
                        rec["audio_path"],
                        language=tcfg["primary"]["language"],
                        beam_size=tcfg["primary"]["beam_size"],
                    )
                    texts[rec["id"]] = " ".join(s.text for s in segs).strip()
            print(f"  primary {min(c0 + chunk_n, len(todo))}/{len(todo)}")

            # pha 2: verify trong subprocess riêng (chỉ torch)
            verifies = _verify_chunk(chunk, tcfg["verify"]["model"], dev, batch_size) if verify_on else {}

            # pha 3: ghi manifest cho chunk (resume theo chunk)
            for rec in chunk:
                text = texts[rec["id"]]
                tv = verifies.get(rec["id"])
                w.write({**rec, "text": text, "text_verify": tv,
                         "cer": round(cer(text, tv), 4) if tv is not None else None})
            print(f"  ghi {min(c0 + chunk_n, len(todo))}/{len(todo)} segment")

    print(f"[{STAGE}] xong")
    return mpath
