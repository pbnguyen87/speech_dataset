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
from ..gpulock import gpu_lock, wants_lock

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


class VerifyClient:
    """Giữ subprocess PhoWhisper thường trú (nạp model một lần); chết thì tự khởi động lại."""

    def __init__(self, model: str, dev: str, batch_size: int):
        self.model, self.dev, self.batch_size = model, dev, batch_size
        self.proc = None

    def _start(self) -> None:
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "pipeline.stages.s5_verify_worker", "--serve", self.model, self.dev],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
        )
        line = self.proc.stdout.readline().strip()
        if line != "ready":
            raise RuntimeError(f"verify worker không khởi động được: {line!r}")

    def run(self, chunk: list[dict]) -> dict:
        with tempfile.TemporaryDirectory() as td:
            in_path, out_path = os.path.join(td, "in.json"), os.path.join(td, "out.json")
            with open(in_path, "w", encoding="utf-8") as f:
                json.dump({r["id"]: r["audio_path"] for r in chunk}, f)
            for attempt in (1, 2):
                try:
                    if self.proc is None or self.proc.poll() is not None:
                        self._start()
                    self.proc.stdin.write(json.dumps({"in": in_path, "out": out_path,
                                                      "batch_size": self.batch_size}) + "\n")
                    self.proc.stdin.flush()
                    line = self.proc.stdout.readline().strip()
                    if line == "ok":
                        with open(out_path, encoding="utf-8") as f:
                            return json.load(f)
                    print(f"  verify worker lỗi: {line or 'chết giữa chừng'}")
                    if not line:  # process chết -> khởi động lại rồi thử lại một lần
                        self.proc = None
                        continue
                    return {}
                except (BrokenPipeError, OSError) as e:
                    print(f"  verify worker lỗi I/O: {e}")
                    self.proc = None
            print("  verify worker lỗi 2 lần — chunk bỏ verify")
            return {}

    def close(self) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.stdin.close()
                self.proc.wait(timeout=30)
            except Exception:
                self.proc.kill()


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


class Worker:
    flush_s = 30

    def __init__(self, cfg: dict, workdir: str):
        from faster_whisper import WhisperModel

        self.tcfg = cfg["transcribe"]
        self.workdir = workdir
        self.lock = wants_lock(cfg)
        dev = device_mod.resolve(cfg["device"])
        ct_dev = device_mod.for_ctranslate2(dev)
        compute = self.tcfg["primary"]["compute_type"]
        if compute == "auto":
            compute = "float16" if ct_dev == "cuda" else "int8"
        self.p_batch = int(self.tcfg["primary"].get("batch_size", 1))
        self.v_batch = int(self.tcfg["verify"].get("batch_size", 1))
        self.chunk_n = max(CHUNK, 2 * self.v_batch, 2 * self.p_batch)  # mỗi chunk >= 2 batch
        self.batch_n = self.chunk_n
        self.flush_s = float(cfg.get("stream", {}).get("flush_seconds", 30))
        self.w = manifest.ManifestWriter(manifest.manifest_path(workdir, STAGE))
        print(f"[{STAGE}] primary={self.tcfg['primary']['model']}@{ct_dev}/{compute} batch_size={self.p_batch}")
        self.primary = WhisperModel(self.tcfg["primary"]["model"], device=ct_dev, compute_type=compute)
        self.verify = None
        if self.tcfg["verify"]["enabled"]:
            self.verify = VerifyClient(self.tcfg["verify"]["model"], dev, self.v_batch)
            print(f"  verify: {self.tcfg['verify']['model']}@{dev} (subprocess thường trú, "
                  f"batch_size={self.v_batch}, chunk={self.chunk_n})")

    def is_done(self, rec: dict) -> bool:
        return self.w.is_done(rec["id"])

    def ready(self, pending: list, upstream_done: bool):
        return pending, []

    def process(self, records: list[dict]) -> None:
        todo = [r for r in records if not self.w.is_done(r["id"])]
        pcfg = self.tcfg["primary"]
        for c0 in range(0, len(todo), self.chunk_n):
            chunk = todo[c0:c0 + self.chunk_n]

            # pha 1: transcribe chính (in-process, chỉ ctranslate2), gộp batch nhiều segment
            texts = {}
            with gpu_lock(self.workdir, self.lock):
                if self.p_batch > 1:
                    outs = primary_batched(self.primary, [r["audio_path"] for r in chunk],
                                           pcfg["language"], pcfg["beam_size"], self.p_batch)
                    texts = {r["id"]: t for r, t in zip(chunk, outs)}
                else:
                    for rec in chunk:
                        segs, _ = self.primary.transcribe(rec["audio_path"], language=pcfg["language"],
                                                          beam_size=pcfg["beam_size"])
                        texts[rec["id"]] = " ".join(s.text for s in segs).strip()
            print(f"  primary {min(c0 + self.chunk_n, len(todo))}/{len(todo)}")

            # pha 2: verify trong subprocess riêng (chỉ torch)
            verifies = {}
            if self.verify:
                with gpu_lock(self.workdir, self.lock):
                    verifies = self.verify.run(chunk)

            # pha 3: ghi manifest cho chunk (resume theo chunk)
            for rec in chunk:
                text = texts[rec["id"]]
                tv = verifies.get(rec["id"])
                self.w.write({**rec, "text": text, "text_verify": tv,
                              "cer": round(cer(text, tv), 4) if tv is not None else None})
            print(f"  ghi {min(c0 + self.chunk_n, len(todo))}/{len(todo)} segment")

    def close(self) -> None:
        if self.verify:
            self.verify.close()
        self.w.close()


def run(cfg: dict, workdir: str, limit: int | None = None) -> str:
    # limit áp ở cấp file nguồn (s0-s2); stage segment-level xử lý toàn bộ
    records = manifest.read_records(manifest.manifest_path(workdir, PREV))
    mpath = manifest.manifest_path(workdir, STAGE)
    todo = [r for r in records if r["id"] not in manifest.done_ids(mpath)]
    print(f"[{STAGE}] {len(todo)}/{len(records)} segment cần transcribe")
    if not todo:
        return mpath
    worker = Worker(cfg, workdir)
    worker.process(todo)
    worker.close()
    print(f"[{STAGE}] xong")
    return mpath
