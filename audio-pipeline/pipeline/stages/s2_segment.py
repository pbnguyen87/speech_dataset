"""s2: silero-VAD tìm vùng có tiếng nói, gộp thành segment min–max giây."""

import os

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


def run(cfg: dict, workdir: str, limit: int | None = None) -> str:
    from silero_vad import get_speech_timestamps, load_silero_vad, read_audio

    scfg = cfg["segment"]
    vad_sr = scfg["vad_sr"]
    sr = cfg["ingest"]["target_sr"]
    records = manifest.read_records(manifest.manifest_path(workdir, PREV), limit)
    out_audio = manifest.audio_dir(workdir, STAGE)
    mpath = manifest.manifest_path(workdir, STAGE)

    model = load_silero_vad()
    print(f"[{STAGE}] {len(records)} file")

    # resume theo file_id (mỗi file sinh nhiều segment, ghi marker khi xong cả file)
    done_files = {r["file_id"] for r in manifest.iter_records(mpath)}

    with manifest.ManifestWriter(mpath) as w:
        for i, rec in enumerate(records):
            if rec["id"] in done_files:
                continue
            wav16 = read_audio(rec["audio_path"], sampling_rate=vad_sr)
            ts = get_speech_timestamps(
                wav16, model, sampling_rate=vad_sr,
                min_silence_duration_ms=scfg["min_silence_ms"],
                speech_pad_ms=0, return_seconds=True,
            )
            segs = merge_speech_chunks(
                ts, scfg["min_seconds"], scfg["max_seconds"],
                pad_s=scfg["speech_pad_ms"] / 1000.0,
            )

            x, _ = audio_utils.load_wav(rec["audio_path"], sr)
            for s, e in segs:
                a, b = int(s * sr), min(int(e * sr), len(x))
                seg_id = f"{rec['id']}_{int(s*1000):08d}_{int(e*1000):08d}"
                seg_path = os.path.join(out_audio, f"{seg_id}.wav")
                audio_utils.save_wav(seg_path, x[a:b], sr)
                w.write({
                    "id": seg_id,
                    "file_id": rec["id"],
                    "audio_path": seg_path,
                    "sr": sr,
                    "start": round(s, 3),
                    "end": round(e, 3),
                    "duration": round(e - s, 3),
                    "source_path": rec["source_path"],
                    "source_meta": rec.get("source_meta"),
                    "separated": rec.get("separated"),
                })
            print(f"  [{i+1}/{len(records)}] {rec['id']}: {len(segs)} segment")

    n = len(manifest.read_records(mpath))
    print(f"[{STAGE}] xong: {n} segment")
    return mpath
