"""s4: gán speaker_id + phát hiện segment đa người nói.

Backend:
  ecapa   : embedding ECAPA (speechbrain), cluster trong phạm vi từng file nguồn;
            flag đa người nói bằng khoảng cách embedding nửa đầu vs nửa sau.
  pyannote: diarization đầy đủ (cần HF token, model gated).
  off     : pass-through, speaker_id = file_id.

Cluster cần đủ mọi segment của một file, nên ở chế độ stream Worker.ready() chỉ nhả
một file khi số segment nhận được (cộng số đã ghi) == n_segments do s2 khai.
"""

import re
from collections import Counter, defaultdict

from .. import audio_utils, device as device_mod, manifest
from ..gpulock import gpu_lock, wants_lock

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


def split_ready(pending: list[dict], done_count: Counter, upstream_done: bool):
    """Chia pending thành (đủ file để xử lý, còn chờ). File đủ khi nhận đủ n_segments;
    thiếu n_segments (manifest cũ) hoặc upstream đã xong -> nhả hết."""
    if upstream_done:
        return pending, []
    by_file = defaultdict(list)
    for r in pending:
        by_file[r["file_id"]].append(r)
    ready, rest = [], []
    for fid, recs in by_file.items():
        n = recs[0].get("n_segments")
        if n is None or len(recs) + done_count.get(fid, 0) >= n:
            ready += recs
        else:
            rest += recs
    return ready, rest


class Worker:
    batch_n = 1
    flush_s = 0

    def __init__(self, cfg: dict, workdir: str):
        self.scfg = cfg["speaker"]
        self.backend = self.scfg["backend"]
        self.workdir = workdir
        self.lock = wants_lock(cfg)
        mpath = manifest.manifest_path(workdir, STAGE)
        # số segment đã ghi + nhãn speaker lớn nhất mỗi file (resume giữa file -> nhãn mới không đè)
        self.done_count: Counter = Counter()
        self.max_label: dict[str, int] = {}
        for r in manifest.iter_records(mpath):
            self.done_count[r["file_id"]] += 1
            m = re.search(r"_spk(\d+)$", r.get("speaker_id") or "")
            if m:
                self.max_label[r["file_id"]] = max(self.max_label.get(r["file_id"], -1), int(m.group(1)))
        self.w = manifest.ManifestWriter(mpath)
        self.enc = None
        self.pipe = None
        if self.backend == "off":
            print(f"[{STAGE}] off — speaker_id = file_id")
            return
        self.dev = device_mod.resolve(cfg["device"])
        if self.backend == "pyannote":
            import torch
            from pyannote.audio import Pipeline

            self.pipe = Pipeline.from_pretrained(self.scfg["pyannote_model"])
            if self.dev != "cpu":
                self.pipe.to(torch.device(self.dev))
            print(f"[{STAGE}] pyannote device={self.dev}")
        else:
            self.enc = EcapaBackend(self.dev)
            print(f"[{STAGE}] ecapa device={self.dev}")

    def is_done(self, rec: dict) -> bool:
        return self.w.is_done(rec["id"])

    def ready(self, pending: list, upstream_done: bool):
        if self.backend != "ecapa":
            return pending, []
        return split_ready(pending, self.done_count, upstream_done)

    def _write(self, rec: dict) -> None:
        self.w.write(rec)
        self.done_count[rec["file_id"]] += 1

    def process(self, records: list[dict]) -> None:
        records = [r for r in records if not self.w.is_done(r["id"])]
        if not records:
            return
        if self.backend == "off":
            for rec in records:
                self._write({**rec, "speaker_id": rec["file_id"], "multi_speaker": None})
            return
        if self.backend == "pyannote":
            self._process_pyannote(records)
            return

        import numpy as np
        from sklearn.cluster import AgglomerativeClustering

        ms_thr = self.scfg["multi_speaker_threshold"]
        # 1) embedding + flag đa người nói cho từng segment
        embs, multi = {}, {}
        for i, rec in enumerate(records):
            x, sr = audio_utils.load_wav(rec["audio_path"])
            x16 = audio_utils.resample(x, sr, 16000)
            with gpu_lock(self.workdir, self.lock):
                embs[rec["id"]] = self.enc.embed(x16)
                if len(x16) >= 2 * 16000:
                    half = len(x16) // 2
                    d = _cosine_dist(self.enc.embed(x16[:half]), self.enc.embed(x16[half:]))
                    multi[rec["id"]] = d > ms_thr
                else:
                    multi[rec["id"]] = False
            if (i + 1) % 100 == 0:
                print(f"  embed {i+1}/{len(records)}")

        # 2) cluster trong phạm vi từng file nguồn
        by_file = defaultdict(list)
        for rec in records:
            by_file[rec["file_id"]].append(rec)
        for file_id, recs in by_file.items():
            X = np.stack([embs[r["id"]] for r in recs])
            if len(recs) == 1:
                labels = [0]
            else:
                labels = AgglomerativeClustering(
                    n_clusters=None, metric="cosine", linkage="average",
                    distance_threshold=self.scfg["cluster_threshold"],
                ).fit_predict(X)
            base = self.max_label.get(file_id, -1) + 1  # file đã có segment từ lần chạy trước
            for r, lb in zip(recs, labels):
                self._write({**r, "speaker_id": f"{file_id}_spk{base + int(lb)}",
                             "multi_speaker": multi[r["id"]]})
            self.max_label[file_id] = base + int(max(labels))

    def _process_pyannote(self, records: list[dict]) -> None:
        for i, rec in enumerate(records):
            with gpu_lock(self.workdir, self.lock):
                dia = self.pipe(rec["audio_path"])
            speakers = dia.labels()
            durations = {s: sum(seg.duration for seg, _, sp in
                                dia.itertracks(yield_label=True) if sp == s)
                         for s in speakers}
            main = max(durations, key=durations.get) if durations else "spk0"
            self._write({**rec, "speaker_id": f"{rec['file_id']}_{main}",
                         "multi_speaker": len(speakers) > 1})
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(records)}")

    def close(self) -> None:
        self.w.close()


def run(cfg: dict, workdir: str, limit: int | None = None) -> str:
    # limit áp ở cấp file nguồn (s0-s2); stage segment-level xử lý toàn bộ
    records = manifest.read_records(manifest.manifest_path(workdir, PREV))
    mpath = manifest.manifest_path(workdir, STAGE)
    if cfg["speaker"]["backend"] == "ecapa":
        # Resume theo file nguồn: cluster cần đủ mọi segment của một file -> lấy cả file
        # còn segment chưa xong; file đã xong bỏ qua hoàn toàn (không đọc lại audio).
        done = manifest.done_ids(mpath)
        pending_files = {r["file_id"] for r in records if r["id"] not in done}
        records = [r for r in records if r["file_id"] in pending_files]
        print(f"[{STAGE}] {len(records)} segment / {len(pending_files)} file cần xử lý")
        if not records:
            print(f"[{STAGE}] xong (không có gì mới)")
            return mpath
    worker = Worker(cfg, workdir)
    worker.process(records)
    worker.close()
    print(f"[{STAGE}] xong")
    return mpath
