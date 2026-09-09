"""s3: đo chỉ số chất lượng từng segment — KHÔNG lọc, chỉ ghi vào manifest.

Chỉ số: clipping_ratio, snr_db (ước lượng percentile năng lượng frame),
bandwidth_hz (phát hiện mp3 upsample), dnsmos_ovr (tùy chọn, cần onnx model).
"""

import os

from .. import audio_utils, manifest

STAGE = "s3_quality"
PREV = "s2_segment"


def clipping_ratio(x, threshold: float = 0.999) -> float:
    import numpy as np

    if len(x) == 0:
        return 0.0
    return float(np.mean(np.abs(x) >= threshold))


def estimate_snr_db(x, sr: int, frame_ms: int = 30, hop_ms: int = 10) -> float:
    """SNR ước lượng: năng lượng frame p90 (speech) so với p10 (nhiễu nền)."""
    import numpy as np

    frame, hop = int(sr * frame_ms / 1000), int(sr * hop_ms / 1000)
    if len(x) < frame:
        return 0.0
    n = 1 + (len(x) - frame) // hop
    energies = np.array([
        np.mean(x[i * hop:i * hop + frame] ** 2) for i in range(n)
    ]) + 1e-12
    db = 10 * np.log10(energies)
    return float(np.percentile(db, 90) - np.percentile(db, 10))


def bandwidth_hz(x, sr: int, energy_ratio: float = 0.99) -> float:
    """Tần số roll-off chứa 99% năng lượng phổ — thấp bất thường => audio bị nén/upsample."""
    import numpy as np

    if len(x) == 0:
        return 0.0
    mag = np.abs(np.fft.rfft(x)) ** 2
    cum = np.cumsum(mag)
    if cum[-1] <= 0:
        return 0.0
    idx = int(np.searchsorted(cum, energy_ratio * cum[-1]))
    freqs = np.fft.rfftfreq(len(x), 1 / sr)
    return float(freqs[min(idx, len(freqs) - 1)])


class DNSMOSScorer:
    """DNSMOS P.835 (sig_bak_ovr.onnx). Trả None nếu không khả dụng."""

    def __init__(self, model_path: str | None, device: str):
        self.session = None
        if not model_path or not os.path.exists(model_path):
            return
        try:
            import onnxruntime as ort

            providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                         if device == "cuda" else ["CPUExecutionProvider"])
            self.session = ort.InferenceSession(model_path, providers=providers)
        except Exception as e:
            print(f"  DNSMOS không khả dụng: {e}")

    def score(self, x, sr: int) -> float | None:
        if self.session is None:
            return None
        import numpy as np

        x16 = audio_utils.resample(x, sr, 16000)
        seg_len = 9 * 16000  # model nhận cửa sổ ~9s
        if len(x16) < seg_len:
            x16 = np.pad(x16, (0, seg_len - len(x16)))
        scores = []
        for i in range(0, len(x16) - seg_len + 1, seg_len):
            inp = x16[i:i + seg_len][None].astype("float32")
            out = self.session.run(None, {self.session.get_inputs()[0].name: inp})
            scores.append(float(out[0][0][-1]))  # OVR là giá trị cuối
        return round(sum(scores) / len(scores), 3) if scores else None


class Worker:
    batch_n = 1
    flush_s = 0

    def __init__(self, cfg: dict, workdir: str):
        from .. import device as device_mod

        self.qcfg = cfg["quality"]
        self.w = manifest.ManifestWriter(manifest.manifest_path(workdir, STAGE))
        dns_enabled = self.qcfg["dnsmos"]["enabled"]
        self.scorer = None
        if dns_enabled in (True, "auto"):
            scorer = DNSMOSScorer(self.qcfg["dnsmos"]["model_path"], device_mod.resolve(cfg["device"]))
            if scorer.session is None and dns_enabled is True:
                raise SystemExit("dnsmos.enabled=true nhưng model/onnxruntime không khả dụng")
            self.scorer = scorer if scorer.session is not None else None
        print(f"[{STAGE}] dnsmos={'on' if self.scorer else 'off'}")

    def is_done(self, rec: dict) -> bool:
        return self.w.is_done(rec["id"])

    def ready(self, pending: list, upstream_done: bool):
        return pending, []

    def process(self, records: list[dict]) -> None:
        for i, rec in enumerate(records):
            if self.w.is_done(rec["id"]):
                continue
            x, sr = audio_utils.load_wav(rec["audio_path"])
            self.w.write({
                **rec,
                "clipping": round(clipping_ratio(x, self.qcfg["clipping_threshold"]), 5),
                "snr_db": round(estimate_snr_db(x, sr), 2),
                "bandwidth_hz": round(bandwidth_hz(x, sr), 1),
                "dnsmos": self.scorer.score(x, sr) if self.scorer else None,
            })
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(records)}")

    def close(self) -> None:
        self.w.close()


def run(cfg: dict, workdir: str, limit: int | None = None) -> str:
    # limit áp ở cấp file nguồn (s0-s2); stage segment-level xử lý toàn bộ
    records = manifest.read_records(manifest.manifest_path(workdir, PREV))
    worker = Worker(cfg, workdir)
    print(f"[{STAGE}] {len(records)} segment")
    worker.process(records)
    worker.close()
    print(f"[{STAGE}] xong")
    return manifest.manifest_path(workdir, STAGE)
