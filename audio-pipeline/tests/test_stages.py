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
from pipeline.stages.s8_package import assign_split, assign_tier, keep_record, normalize_source_path

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


class TestVerifyBatched:
    def _asr(self, fail_on=None, oom_above=None):
        calls = []
        def asr(paths, batch_size):
            calls.append(len(paths))
            if oom_above and len(paths) > oom_above:
                raise RuntimeError("CUDA out of memory")
            if fail_on and fail_on in paths:
                raise ValueError("file hỏng")
            return [{"text": f" {p} "} for p in paths]
        return asr, calls

    def test_batches(self):
        from pipeline.stages.s5_verify_worker import transcribe_batched
        asr, calls = self._asr()
        items = {f"s{i}": f"p{i}" for i in range(10)}
        out = transcribe_batched(asr, items, 4)
        assert calls == [4, 4, 2] and out["s9"] == "p9" and len(out) == 10

    def test_oom_halves_batch(self):
        from pipeline.stages.s5_verify_worker import transcribe_batched
        asr, calls = self._asr(oom_above=2)
        out = transcribe_batched(asr, {f"s{i}": f"p{i}" for i in range(4)}, 8)
        assert calls == [4, 2, 2] and len(out) == 4

    def test_bad_file_only_loses_itself(self):
        from pipeline.stages.s5_verify_worker import transcribe_batched
        asr, _ = self._asr(fail_on="p1")
        out = transcribe_batched(asr, {f"s{i}": f"p{i}" for i in range(3)}, 3)
        assert out == {"s0": "p0", "s1": None, "s2": "p2"}


class TestNormalizeSourcePath:
    def test_staging_name_to_feed_path(self):
        assert normalize_source_path("vi-podcast__vnexpress.net_rss_podcast_ban-on-khong.rss__Tập_1.mp3") \
            == "vnexpress.net_rss_podcast_ban-on-khong.rss/Tập_1.mp3"
        assert normalize_source_path("vi-podcast__Tập_1.mp3") == "Tập_1.mp3"  # trước khi có thư mục theo feed

    def test_stream_path_unchanged(self):
        assert normalize_source_path("vnexpress.net_rss_podcast_ban-on-khong.rss/Tập_1.mp3") \
            == "vnexpress.net_rss_podcast_ban-on-khong.rss/Tập_1.mp3"
        assert normalize_source_path("a/b__c.mp3") == "a/b__c.mp3"
        assert normalize_source_path(None) is None


class TestSeparateChunks:
    def test_long_file_is_chunked_and_concatenated(self, monkeypatch, tmp_path):
        """Giả lập demucs: copy đầu vào làm vocals. File 10s, chunk 4s -> 3 khúc, nối lại đủ 10s."""
        import shutil
        import numpy as np
        from pipeline import audio_utils
        from pipeline.stages import s1_separate

        src = tmp_path / "in.wav"
        audio_utils.save_wav(str(src), np.random.rand(24000 * 10).astype("float32") * 0.1, 24000)
        calls = []
        def fake_demucs(piece, out_dir, model, dev):
            calls.append(round(audio_utils.duration_seconds(piece), 1))
            os.makedirs(out_dir, exist_ok=True)
            out = os.path.join(out_dir, "vocals.wav"); shutil.copy(piece, out)
            return out
        monkeypatch.setattr(s1_separate, "_run_demucs", fake_demucs)
        dst = tmp_path / "out.wav"
        s1_separate.separate_vocals(str(src), str(dst), "htdemucs", "cpu", 24000, 4.0)
        assert calls == [4.0, 4.0, 2.0]
        assert abs(audio_utils.duration_seconds(str(dst)) - 10.0) < 0.01


class TestPackageEmpty:
    def test_s8_with_all_tier_c_writes_empty_dataset(self, tmp_path):
        import json
        from pipeline.stages import s8_package
        from pipeline import manifest
        wd = str(tmp_path)
        with manifest.ManifestWriter(manifest.manifest_path(wd, "s7_loudnorm")) as w:
            w.write({"id": "x", "file_id": "f", "audio_path": "/khong/co.wav", "duration": 3.0,
                     "text": "a", "cer": None, "snr_db": 30, "clipping": 0})
        cfg = {"package": {"tiers": {"A": {"max_cer": 0.02, "min_snr_db": 20, "min_dnsmos": 3, "max_clipping": 0.001,
                                            "min_seconds": 2, "max_seconds": 13, "allow_multi_speaker": False}},
                           "keep_tiers": None, "split": {"val_ratio": 0.02, "test_ratio": 0.02, "seed": 1},
                           "parquet": {"enabled": False}}}
        s8_package.run(cfg, wd)
        assert (tmp_path / "s8_package" / "dataset" / "metadata.csv").read_text().count("\n") == 1  # chỉ header


class TestPackageIncremental:
    """`pipeline package`: gói phần s7 mới, nối vào dataset đang có."""

    CFG = {"package": {"tiers": TIERS, "keep_tiers": None,
                       "split": {"val_ratio": 0.02, "test_ratio": 0.02, "seed": 1},
                       "parquet": {"enabled": True, "shard_max_mb": 500}}}

    @staticmethod
    def _wav(path):
        import soundfile as sf
        os.makedirs(os.path.dirname(path), exist_ok=True)
        sf.write(path, np.zeros(2400, dtype="float32"), 24000)

    def _add_s7(self, wd, ids, cer=0.01):
        """Thêm segment vào manifest s7 kèm wav; trả về danh sách record."""
        from pipeline import manifest
        recs = []
        with manifest.ManifestWriter(manifest.manifest_path(wd, "s7_loudnorm")) as w:
            for i in ids:
                p = os.path.join(wd, "s7_loudnorm", "audio", f"{i}.wav")
                self._wav(p)
                r = rec(id=i, file_id=i.split("_")[0], speaker_id=i.split("_")[0] + "_spk0", audio_path=p, cer=cer)
                w.write(r)
                recs.append(r)
        return recs

    @staticmethod
    def _csv_ids(wd):
        import csv
        p = os.path.join(wd, "s8_package", "dataset", "metadata.csv")
        if not os.path.exists(p):
            return []
        with open(p, newline="", encoding="utf-8") as f:
            return [os.path.splitext(os.path.basename(r["file_name"]))[0] for r in csv.DictReader(f)]

    @staticmethod
    def _parquet_ids(wd):
        import glob
        import pyarrow.parquet as pq
        out = []
        for p in glob.glob(os.path.join(wd, "s8_package", "dataset", "parquet", "*.parquet")):
            out += [os.path.splitext(x["path"])[0] for x in pq.read_table(p).column("audio").to_pylist()]
        return out

    def test_second_run_packs_only_new(self, tmp_path):
        from pipeline import manifest
        from pipeline.stages.s8_package import package_incremental
        wd = str(tmp_path)
        self._add_s7(wd, ["f1_a", "f1_b", "f2_a"])
        st = package_incremental(self.CFG, wd)
        assert st["new"] == 3 and st["kept"] == 3
        self._add_s7(wd, ["f3_a", "f3_b"])
        st = package_incremental(self.CFG, wd)
        assert st["new"] == 2
        done = manifest.done_ids(manifest.manifest_path(wd, "s8_package"))
        assert done == {"f1_a", "f1_b", "f2_a", "f3_a", "f3_b"}
        assert sorted(self._csv_ids(wd)) == sorted(done)
        assert sorted(self._parquet_ids(wd)) == sorted(done)
        assert (tmp_path / "s8_package" / "report.html").exists()
        assert (tmp_path / "s8_package" / "config_hash").exists()
        # hardlink: cùng inode với wav s7
        src = tmp_path / "s7_loudnorm" / "audio" / "f1_a.wav"
        assert os.stat(src).st_nlink == 2
        st = package_incremental(self.CFG, wd)
        assert st["new"] == 0

    def test_tier_c_not_in_dataset_but_in_manifest(self, tmp_path):
        from pipeline import manifest
        from pipeline.stages.s8_package import package_incremental
        wd = str(tmp_path)
        self._add_s7(wd, ["f1_a"], cer=0.9)
        package_incremental(self.CFG, wd, cleanup=True)
        recs = manifest.read_records(manifest.manifest_path(wd, "s8_package"))
        assert recs[0]["tier"] == "C"
        assert self._csv_ids(wd) == []
        assert not os.path.exists(os.path.join(wd, "s7_loudnorm", "audio", "f1_a.wav"))  # cleanup xóa tier C

    def test_deleted_s7_wav_of_packed_segments_is_fine(self, tmp_path):
        from pipeline.stages.s8_package import package_incremental
        wd = str(tmp_path)
        self._add_s7(wd, ["f1_a", "f1_b"])
        package_incremental(self.CFG, wd, copy=True)
        for i in ("f1_a", "f1_b"):
            os.remove(os.path.join(wd, "s7_loudnorm", "audio", f"{i}.wav"))
        self._add_s7(wd, ["f2_a"])
        st = package_incremental(self.CFG, wd, copy=True)
        assert st["new"] == 1 and st["skipped_missing"] == 0
        assert sorted(self._csv_ids(wd)) == ["f1_a", "f1_b", "f2_a"]

    def test_missing_wav_of_new_kept_segment_is_retried_later(self, tmp_path):
        from pipeline import manifest
        from pipeline.stages.s8_package import package_incremental
        wd = str(tmp_path)
        self._add_s7(wd, ["f1_a", "f1_b"])
        os.remove(os.path.join(wd, "s7_loudnorm", "audio", "f1_b.wav"))
        st = package_incremental(self.CFG, wd)
        assert st["skipped_missing"] == 1
        assert manifest.done_ids(manifest.manifest_path(wd, "s8_package")) == {"f1_a"}
        self._wav(os.path.join(wd, "s7_loudnorm", "audio", "f1_b.wav"))
        st = package_incremental(self.CFG, wd)
        assert st["new"] == 1 and st["skipped_missing"] == 0

    def test_crash_before_manifest_leaves_no_duplicates(self, tmp_path, monkeypatch):
        from pipeline import manifest
        from pipeline.stages.s8_package import package_incremental
        wd = str(tmp_path)
        self._add_s7(wd, ["f1_a", "f1_b", "f1_c"])
        real = manifest.ManifestWriter.write_many

        def boom(self, recs):
            raise RuntimeError("kill giữa chừng")

        monkeypatch.setattr(manifest.ManifestWriter, "write_many", boom)
        with pytest.raises(RuntimeError):
            package_incremental(self.CFG, wd)
        # csv + parquet đã ghi, manifest chưa -> chạy lại phải dọn csv mồ côi và ghi đè parquet
        assert len(self._csv_ids(wd)) == 3
        monkeypatch.setattr(manifest.ManifestWriter, "write_many", real)
        st = package_incremental(self.CFG, wd)
        assert st["csv_removed"] == 3 and st["new"] == 3
        assert sorted(self._csv_ids(wd)) == ["f1_a", "f1_b", "f1_c"]
        assert sorted(self._parquet_ids(wd)) == ["f1_a", "f1_b", "f1_c"]

    def test_chunks_split_parquet_by_batch(self, tmp_path):
        from pipeline.stages.s8_package import package_incremental
        wd = str(tmp_path)
        self._add_s7(wd, [f"f{i}_a" for i in range(5)])
        package_incremental(self.CFG, wd, chunk=2)
        import glob
        shards = glob.glob(os.path.join(wd, "s8_package", "dataset", "parquet", "*-inc-*.parquet"))
        assert len(shards) == 3
        assert sorted(self._parquet_ids(wd)) == sorted(f"f{i}_a" for i in range(5))

    def test_config_change_is_refused(self, tmp_path):
        import copy
        from pipeline.stages.s8_package import package_incremental
        wd = str(tmp_path)
        self._add_s7(wd, ["f1_a"])
        package_incremental(self.CFG, wd)
        cfg = copy.deepcopy(self.CFG)
        cfg["package"]["tiers"]["A"]["max_cer"] = 0.001
        with pytest.raises(SystemExit, match="config tier/split đã đổi"):
            package_incremental(cfg, wd)

    def test_after_full_run_incremental_continues(self, tmp_path):
        from pipeline import manifest
        from pipeline.stages import s8_package
        wd = str(tmp_path)
        self._add_s7(wd, ["f1_a", "f1_b"])
        s8_package.run(self.CFG, wd)
        self._add_s7(wd, ["f2_a"])
        st = s8_package.package_incremental(self.CFG, wd)
        assert st["new"] == 1
        assert manifest.done_ids(manifest.manifest_path(wd, "s8_package")) == {"f1_a", "f1_b", "f2_a"}
        assert sorted(self._csv_ids(wd)) == ["f1_a", "f1_b", "f2_a"]
        assert sorted(self._parquet_ids(wd)) == ["f1_a", "f1_b", "f2_a"]

    def test_dry_run_writes_nothing(self, tmp_path):
        from pipeline.stages.s8_package import package_incremental
        wd = str(tmp_path)
        self._add_s7(wd, ["f1_a"])
        st = package_incremental(self.CFG, wd, dry_run=True)
        assert st["new"] == 1
        assert not os.path.exists(os.path.join(wd, "s8_package", "manifest.jsonl"))
        assert not os.path.exists(os.path.join(wd, "s8_package", "dataset"))

    def test_drop_wav_keeps_only_parquet(self, tmp_path):
        from pipeline.stages.s8_package import package_incremental
        wd = str(tmp_path)
        self._add_s7(wd, ["f1_a", "f1_b"])
        self._add_s7(wd, ["f2_a"], cer=0.9)  # tier C
        st = package_incremental(self.CFG, wd, cleanup=True, drop_wav=True)
        assert st["dropped"] == 2
        assert sorted(self._parquet_ids(wd)) == ["f1_a", "f1_b"]
        assert sorted(self._csv_ids(wd)) == ["f1_a", "f1_b"]
        assert not os.path.exists(os.path.join(wd, "s8_package", "dataset", "wav"))
        assert os.listdir(os.path.join(wd, "s7_loudnorm", "audio")) == []  # A/B đã vào parquet, C bị cleanup
        # parquet giữ nguyên bytes wav
        import glob
        import pyarrow.parquet as pq
        rows = pq.read_table(glob.glob(os.path.join(wd, "s8_package", "dataset", "parquet", "*.parquet"))[0]) \
            .column("audio").to_pylist()
        assert rows[0]["bytes"][:4] == b"RIFF"
        # lần sau chỉ gói phần mới, wav cũ đã mất không sao
        self._add_s7(wd, ["f3_a"])
        st = package_incremental(self.CFG, wd, cleanup=True, drop_wav=True)
        assert st["new"] == 1 and st["dropped"] == 1

    def test_drop_wav_requires_parquet(self, tmp_path):
        import copy
        from pipeline.stages.s8_package import package_incremental
        cfg = copy.deepcopy(self.CFG)
        cfg["package"]["parquet"]["enabled"] = False
        with pytest.raises(SystemExit, match="drop-wav"):
            package_incremental(cfg, str(tmp_path), drop_wav=True)
