"""Test các hàm thuần của pipeline — không cần mạng, GPU hay model."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline import manifest
from pipeline.stages.s2_segment import merge_speech_chunks
from pipeline.stages.s3_quality import bandwidth_hz, clipping_ratio, estimate_snr_db
from pipeline.stages.s5_transcribe import cer
from pipeline.stages.s6_textnorm import basic_normalize, number_to_words_vi
from pipeline.stages.s7_loudnorm import peak_normalize
from pipeline.stages.s8_package import assign_split, assign_tier, keep_record

TIERS = {
    "A": {"max_cer": 0.05, "min_snr_db": 20.0, "min_dnsmos": 3.0,
          "max_clipping": 0.001, "min_seconds": 1.0, "max_seconds": 20.0,
          "allow_multi_speaker": False},
    "B": {"max_cer": 0.15, "min_snr_db": 10.0, "min_dnsmos": 2.5,
          "max_clipping": 0.01, "min_seconds": 1.0, "max_seconds": 30.0,
          "allow_multi_speaker": True},
}


def rec(**kw):
    base = {"cer": 0.01, "duration": 5.0, "clipping": 0.0, "multi_speaker": False,
            "snr_db": 25.0, "dnsmos": None, "text": "xin chào"}
    base.update(kw)
    return base


class TestManifest:
    def test_resume(self, tmp_path):
        p = str(tmp_path / "m.jsonl")
        with manifest.ManifestWriter(p) as w:
            w.write({"id": "a", "x": 1})
            w.write({"id": "b", "x": 2})
        with manifest.ManifestWriter(p) as w:
            assert w.is_done("a") and w.is_done("b")
            assert not w.is_done("c")
            w.write({"id": "c", "x": 3})
        assert [r["id"] for r in manifest.read_records(p)] == ["a", "b", "c"]

    def test_limit(self, tmp_path):
        p = str(tmp_path / "m.jsonl")
        with manifest.ManifestWriter(p) as w:
            for i in range(10):
                w.write({"id": str(i)})
        assert len(manifest.read_records(p, limit=3)) == 3


class TestSegmentMerge:
    def test_merge_within_max(self):
        chunks = [{"start": 0.0, "end": 2.0}, {"start": 2.5, "end": 4.0}]
        segs = merge_speech_chunks(chunks, min_s=1.0, max_s=10.0, pad_s=0.0)
        assert segs == [(0.0, 4.0)]

    def test_split_at_silence_when_exceeding_max(self):
        chunks = [{"start": 0.0, "end": 8.0}, {"start": 9.0, "end": 15.0}]
        segs = merge_speech_chunks(chunks, min_s=1.0, max_s=10.0, pad_s=0.0)
        assert segs == [(0.0, 8.0), (9.0, 15.0)]

    def test_hard_cut_long_chunk(self):
        chunks = [{"start": 0.0, "end": 25.0}]
        segs = merge_speech_chunks(chunks, min_s=1.0, max_s=10.0, pad_s=0.0)
        assert segs[0] == (0.0, 10.0) and segs[1] == (10.0, 20.0)
        assert all(e - s <= 10.0 for s, e in segs)

    def test_drop_short(self):
        chunks = [{"start": 0.0, "end": 0.3}]
        assert merge_speech_chunks(chunks, min_s=1.0, max_s=10.0, pad_s=0.0) == []


class TestQualityMetrics:
    def test_clipping(self):
        x = np.zeros(1000, dtype="float32")
        assert clipping_ratio(x) == 0.0
        x[:100] = 1.0
        assert clipping_ratio(x) == pytest.approx(0.1)

    def test_snr_clean_vs_noisy(self):
        sr = 16000
        t = np.arange(sr * 2) / sr
        speech = np.sin(2 * np.pi * 220 * t).astype("float32")
        # mô phỏng speech: burst 200ms xen kẽ khoảng lặng 200ms
        gate = (t % 0.4 < 0.2).astype("float32")
        clean = speech * gate
        noisy = clean + np.random.RandomState(0).randn(len(t)).astype("float32") * 0.1
        assert estimate_snr_db(clean, sr) > estimate_snr_db(noisy, sr)

    def test_bandwidth(self):
        sr = 24000
        t = np.arange(sr) / sr
        low = np.sin(2 * np.pi * 500 * t).astype("float32")
        assert bandwidth_hz(low, sr) < 1000


class TestCER:
    def test_identical(self):
        assert cer("xin chào Việt Nam", "Xin chào, Việt Nam!") == 0.0

    def test_different(self):
        assert 0 < cer("xin chào", "xin chề") <= 0.5

    def test_empty(self):
        assert cer("", "") == 0.0
        assert cer("abc", "") == 1.0


class TestTextNorm:
    def test_numbers(self):
        assert number_to_words_vi(0) == "không"
        assert number_to_words_vi(15) == "mười lăm"
        assert number_to_words_vi(21) == "hai mươi mốt"
        assert number_to_words_vi(105) == "một trăm lẻ năm"
        assert number_to_words_vi(1234) == "một nghìn hai trăm ba mươi bốn"
        assert number_to_words_vi(2000000) == "hai triệu"

    def test_basic_normalize(self):
        assert basic_normalize("có  5 người") == "có năm người"


class TestLoudnorm:
    def test_peak(self):
        x = np.array([0.1, -0.2, 0.05], dtype="float32")
        y = peak_normalize(x, peak_dbfs=-3.0)
        assert np.max(np.abs(y)) == pytest.approx(10 ** (-3 / 20), abs=1e-4)


class TestPackage:
    def test_tier_a(self):
        assert assign_tier(rec(), TIERS) == "A"

    def test_tier_b_on_cer(self):
        assert assign_tier(rec(cer=0.10), TIERS) == "B"

    def test_tier_c_on_bad_cer(self):
        assert assign_tier(rec(cer=0.5), TIERS) == "C"

    def test_no_cer_is_c(self):
        assert assign_tier(rec(cer=None), TIERS) == "C"

    def test_multi_speaker_demoted(self):
        assert assign_tier(rec(multi_speaker=True), TIERS) == "B"

    def test_dnsmos_used_when_present(self):
        assert assign_tier(rec(dnsmos=2.0, snr_db=99.0), TIERS) == "C"
        assert assign_tier(rec(dnsmos=3.5), TIERS) == "A"

    def test_long_segment_demoted(self):
        assert assign_tier(rec(duration=25.0), TIERS) == "B"

    def test_keep_tiers_none_keeps_all_but_c(self):
        assert keep_record({"tier": "A"}, None)
        assert keep_record({"tier": "B"}, None)
        assert not keep_record({"tier": "C"}, None)

    def test_keep_tiers_filters(self):
        assert keep_record({"tier": "A"}, ["A"])
        assert not keep_record({"tier": "B"}, ["A"])
        assert not keep_record({"tier": "C"}, ["A", "B", "C"])  # C không bao giờ xuất

    def test_split_stable_and_speaker_level(self):
        s1 = assign_split("spkA", 0.1, 0.1, 42)
        assert s1 == assign_split("spkA", 0.1, 0.1, 42)
        splits = {assign_split(f"spk{i}", 0.1, 0.1, 42) for i in range(200)}
        assert splits == {"train", "val", "test"}
