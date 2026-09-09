"""s2: silero-VAD tìm vùng có tiếng nói, gộp thành segment min–max giây.

Mỗi segment ghi kèm n_segments của file nguồn -> stage sau (s4) biết khi nào đã nhận đủ
segment của một file; resume theo file: file xong khi số segment trong manifest == n_segments.
"""

import os
from collections import Counter

from .. import audio_utils, manifest

STAGE = "s2_segment"
PREV = "s1_separate"


def merge_speech_chunks(chunks: list[dict], min_s: float, max_s: float,
                        pad_s: float = 0.1) -> list[tuple[float, float]]:
    """Gộp các vùng speech (giây) thành segment trong khoảng [min_s, max_s].

    chunks: [{"start": s, "end": s}, ...] đã sort theo start.
    Greedy: nối chunk liên tiếp vào segment hiện tại chừng nào chưa vượt max_s;
    vượt thì chốt segment tại end của chunk trước (cắt ở khoảng lặng).
    """
    segments = []
    cur_start = cur_end = None
    for c in chunks:
        if cur_start is None:
            cur_start, cur_end = c["start"], c["end"]
        elif c["end"] - cur_start <= max_s:
            cur_end = c["end"]
        else:
            segments.append((cur_start, cur_end))
            cur_start, cur_end = c["start"], c["end"]
        # một chunk đơn lẻ dài quá max_s: cắt cứng thành nhiều khúc
        while cur_end - cur_start > max_s:
            segments.append((cur_start, cur_start + max_s))
            cur_start = cur_start + max_s
    if cur_start is not None:
        segments.append((cur_start, cur_end))

    out = []
    for s, e in segments:
        s = max(0.0, s - pad_s)
        e = e + pad_s
        if e - s >= min_s:
            out.append((s, e))
    return out


def done_files(mpath: str) -> set[str]:
    """file_id đã có đủ segment (count == n_segments; manifest cũ không có n_segments -> coi là xong)."""
    count: Counter = Counter()
    expected: dict = {}
    for r in manifest.iter_records(mpath):
        count[r["file_id"]] += 1
        expected[r["file_id"]] = r.get("n_segments")
    return {f for f, n in count.items() if expected[f] is None or n >= expected[f]}


class Worker:
    batch_n = 1
    flush_s = 0

    def __init__(self, cfg: dict, workdir: str):
        from silero_vad import load_silero_vad

        self.scfg = cfg["segment"]
        self.vad_sr = self.scfg["vad_sr"]
        self.sr = cfg["ingest"]["target_sr"]
        self.cleanup = bool(cfg.get("stream", {}).get("cleanup"))
        self.workdir = workdir
        self.out_audio = manifest.audio_dir(workdir, STAGE)
        mpath = manifest.manifest_path(workdir, STAGE)
        self.done_files = done_files(mpath)
        self.w = manifest.ManifestWriter(mpath)
        self.model = load_silero_vad()

    def is_done(self, rec: dict) -> bool:
        return rec["id"] in self.done_files

    def ready(self, pending: list, upstream_done: bool):
        return pending, []

    def process(self, records: list[dict]) -> None:
        from silero_vad import get_speech_timestamps, read_audio

        for i, rec in enumerate(records):
            if rec["id"] in self.done_files:
                continue
            wav16 = read_audio(rec["audio_path"], sampling_rate=self.vad_sr)
            ts = get_speech_timestamps(
                wav16, self.model, sampling_rate=self.vad_sr,
                min_silence_duration_ms=self.scfg["min_silence_ms"],
                speech_pad_ms=0, return_seconds=True,
            )
            segs = merge_speech_chunks(
                ts, self.scfg["min_seconds"], self.scfg["max_seconds"],
                pad_s=self.scfg["speech_pad_ms"] / 1000.0,
            )

            x, _ = audio_utils.load_wav(rec["audio_path"], self.sr)
            for k, (s, e) in enumerate(segs):
                seg_id = f"{rec['id']}_{int(s*1000):08d}_{int(e*1000):08d}"
                if self.w.is_done(seg_id):  # resume giữa file: chỉ cắt phần còn thiếu
                    continue
                a, b = int(s * self.sr), min(int(e * self.sr), len(x))
                seg_path = os.path.join(self.out_audio, f"{seg_id}.wav")
                audio_utils.save_wav(seg_path, x[a:b], self.sr)
                self.w.write({
                    "id": seg_id,
                    "file_id": rec["id"],
                    "seg_index": k,
                    "n_segments": len(segs),
                    "audio_path": seg_path,
                    "sr": self.sr,
                    "start": round(s, 3),
                    "end": round(e, 3),
                    "duration": round(e - s, 3),
                    "source_path": rec["source_path"],
                    "source_meta": rec.get("source_meta"),
                    "separated": rec.get("separated"),
                })
            self.done_files.add(rec["id"])
            if self.cleanup:  # s2 là stage cuối đọc wav s0/s1 của file này
                for stage in ("s0_ingest", "s1_separate"):
                    p = os.path.join(self.workdir, stage, "audio", f"{rec['id']}.wav")
                    if os.path.exists(p):
                        os.remove(p)
            print(f"  [{i+1}/{len(records)}] {rec['id']}: {len(segs)} segment")

    def close(self) -> None:
        self.w.close()


def run(cfg: dict, workdir: str, limit: int | None = None) -> str:
    records = manifest.read_records(manifest.manifest_path(workdir, PREV), limit)
    worker = Worker(cfg, workdir)
    print(f"[{STAGE}] {len(records)} file")
    worker.process(records)
    worker.close()
    mpath = manifest.manifest_path(workdir, STAGE)
    print(f"[{STAGE}] xong: {len(manifest.read_records(mpath))} segment")
    return mpath
