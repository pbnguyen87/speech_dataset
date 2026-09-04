"""s4: gán speaker_id + phát hiện segment đa người nói.

Backend:
  ecapa   : embedding ECAPA (speechbrain), cluster trong phạm vi từng file nguồn;
            flag đa người nói bằng khoảng cách embedding nửa đầu vs nửa sau.
  pyannote: diarization đầy đủ (cần HF token, model gated).
  off     : pass-through, speaker_id = file_id.
"""

from collections import defaultdict

from .. import audio_utils, device as device_mod, manifest

STAGE = "s4_speaker"
PREV = "s3_quality"


def _cosine_dist(a, b) -> float:
    import numpy as np

    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(1 - np.dot(a, b) / denom)


class EcapaBackend:
    def __init__(self, dev: str):
        import torch
        from speechbrain.inference.speaker import EncoderClassifier

        # speechbrain chưa ổn định trên mps -> dùng cpu
        run_dev = dev if dev == "cuda" else "cpu"
        self.torch = torch
        self.clf = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": run_dev},
        )

    def embed(self, x16):
        t = self.torch.tensor(x16)[None]
        with self.torch.no_grad():
            emb = self.clf.encode_batch(t)
        return emb.squeeze().cpu().numpy()


def run(cfg: dict, workdir: str, limit: int | None = None) -> str:
    import numpy as np

    scfg = cfg["speaker"]
    backend = scfg["backend"]
    # limit áp ở cấp file nguồn (s0-s2); stage segment-level xử lý toàn bộ
    records = manifest.read_records(manifest.manifest_path(workdir, PREV))
    mpath = manifest.manifest_path(workdir, STAGE)

    if backend == "off":
        with manifest.ManifestWriter(mpath) as w:
            for rec in records:
                if not w.is_done(rec["id"]):
                    w.write({**rec, "speaker_id": rec["file_id"], "multi_speaker": None})
        print(f"[{STAGE}] off — speaker_id = file_id")
        return mpath

    dev = device_mod.resolve(cfg["device"])

    if backend == "pyannote":
        return _run_pyannote(cfg, records, mpath, dev)

    # --- backend ecapa ---
    # Resume theo file nguồn: cluster cần đủ mọi segment của một file, nên chỉ xử lý
    # các file còn segment chưa có trong manifest; file đã xong bỏ qua hoàn toàn
    # (không đọc lại audio -> chạy theo lô + xóa wav trung gian của lô trước vẫn ổn).
    done = manifest.done_ids(mpath)
    pending_files = {r["file_id"] for r in records if r["id"] not in done}
    records = [r for r in records if r["file_id"] in pending_files]
    print(f"[{STAGE}] ecapa device={dev}, {len(records)} segment / {len(pending_files)} file cần xử lý")
    if not records:
        print(f"[{STAGE}] xong (không có gì mới)")
        return mpath
    enc = EcapaBackend(dev)
    ms_thr = scfg["multi_speaker_threshold"]

    # 1) embedding + flag đa người nói cho từng segment
    embs: dict[str, "np.ndarray"] = {}
    multi: dict[str, bool] = {}
    for i, rec in enumerate(records):
        x, sr = audio_utils.load_wav(rec["audio_path"])
        x16 = audio_utils.resample(x, sr, 16000)
        embs[rec["id"]] = enc.embed(x16)
        if len(x16) >= 2 * 16000:
            half = len(x16) // 2
            d = _cosine_dist(enc.embed(x16[:half]), enc.embed(x16[half:]))
            multi[rec["id"]] = d > ms_thr
        else:
            multi[rec["id"]] = False
        if (i + 1) % 100 == 0:
            print(f"  embed {i+1}/{len(records)}")

    # 2) cluster trong phạm vi từng file nguồn
    from sklearn.cluster import AgglomerativeClustering

    by_file = defaultdict(list)
    for rec in records:
        by_file[rec["file_id"]].append(rec)

    speaker_of = {}
    for file_id, recs in by_file.items():
        X = np.stack([embs[r["id"]] for r in recs])
        if len(recs) == 1:
            labels = [0]
        else:
            labels = AgglomerativeClustering(
                n_clusters=None, metric="cosine", linkage="average",
                distance_threshold=scfg["cluster_threshold"],
            ).fit_predict(X)
        for r, lb in zip(recs, labels):
            speaker_of[r["id"]] = f"{file_id}_spk{lb}"

    with manifest.ManifestWriter(mpath) as w:
        for rec in records:
            if not w.is_done(rec["id"]):
                w.write({**rec, "speaker_id": speaker_of[rec["id"]],
                         "multi_speaker": multi[rec["id"]]})
    print(f"[{STAGE}] xong")
    return mpath


def _run_pyannote(cfg, records, mpath, dev):
    import torch
    from pyannote.audio import Pipeline

    pipe = Pipeline.from_pretrained(cfg["speaker"]["pyannote_model"])
    if dev != "cpu":
        pipe.to(torch.device(dev))

    print(f"[{STAGE}] pyannote device={dev}, {len(records)} segment")
    with manifest.ManifestWriter(mpath) as w:
        for i, rec in enumerate(records):
            if w.is_done(rec["id"]):
                continue
            dia = pipe(rec["audio_path"])
            speakers = dia.labels()
            durations = {s: sum(seg.duration for seg, _, sp in
                                dia.itertracks(yield_label=True) if sp == s)
                         for s in speakers}
            main = max(durations, key=durations.get) if durations else "spk0"
            w.write({
                **rec,
                "speaker_id": f"{rec['file_id']}_{main}",
                "multi_speaker": len(speakers) > 1,
            })
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(records)}")
    print(f"[{STAGE}] xong")
    return mpath
