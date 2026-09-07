"""Test phần tail ledger / staging của tools/crawl_and_process.py — không chạy pipeline thật."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

import crawl_and_process as cap


def _write_ledger(d, recs):
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "ledger.jsonl", "a", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


class TestLedgerTail:
    def test_only_new_lines_each_poll(self, tmp_path):
        src = tmp_path / "raw" / "s1"
        _write_ledger(src, [{"id": "a", "key": "ka", "path": str(src / "a.mp3"), "source": "s1"}])
        tail = cap.LedgerTail(str(tmp_path / "raw"))
        first = tail.poll()
        assert [r["id"] for r in first] == ["a"] and os.path.isabs(first[0]["path"])
        assert tail.poll() == []
        _write_ledger(src, [{"id": "b", "key": "kb", "path": str(src / "b.mp3"), "source": "s1"}])
        assert [r["id"] for r in tail.poll()] == ["b"]

    def test_partial_line_waits(self, tmp_path):
        src = tmp_path / "raw" / "s1"; src.mkdir(parents=True)
        with open(src / "ledger.jsonl", "w") as f:
            f.write('{"id": "a", "key": "k", "path": "x.mp3", "sou')  # đang ghi dở
        tail = cap.LedgerTail(str(tmp_path / "raw"))
        assert tail.poll() == []
        with open(src / "ledger.jsonl", "a") as f:
            f.write('rce": "s1"}\n')
        assert [r["id"] for r in tail.poll()] == ["a"]

    def test_multiple_sources(self, tmp_path):
        for s in ("s1", "s2"):
            _write_ledger(tmp_path / "raw" / s, [{"id": s, "key": s, "path": f"{s}.mp3", "source": s}])
        assert sorted(r["source"] for r in cap.LedgerTail(str(tmp_path / "raw")).poll()) == ["s1", "s2"]

    def test_tail_root_ledger(self, tmp_path):  # source khai dir: "." -> ledger nằm ngay <out>/
        _write_ledger(tmp_path / "raw", [{"id": "r", "key": "r", "path": "r.mp3", "source": "root"}])
        _write_ledger(tmp_path / "raw" / "s1", [{"id": "s1", "key": "s1", "path": "s1.mp3", "source": "s1"}])
        assert sorted(r["source"] for r in cap.LedgerTail(str(tmp_path / "raw")).poll()) == ["root", "s1"]


class TestStaging:
    def test_symlinks_audio_and_sidecar_with_source_prefix(self, tmp_path):
        raw = tmp_path / "raw" / "kenh"; raw.mkdir(parents=True)
        (raw / "Tập_1.m4a").write_bytes(b"x"); (raw / "Tập_1.json").write_text("{}")
        (raw / "Tập_2.mp3").write_bytes(b"y")  # không có sidecar
        recs = [{"path": str(raw / "Tập_1.m4a"), "source": "kenh"},
                {"path": str(raw / "Tập_2.mp3"), "source": "kenh"}]
        d = cap.stage_batch(recs, str(tmp_path / "work"))
        names = sorted(os.listdir(d))
        assert names == ["kenh__Tập_1.json", "kenh__Tập_1.m4a", "kenh__Tập_2.mp3"]
        assert os.path.islink(os.path.join(d, "kenh__Tập_1.m4a"))
        assert open(os.path.join(d, "kenh__Tập_1.m4a"), "rb").read() == b"x"

    def test_subdir_in_staging_name(self, tmp_path):
        # 2 feed cùng nguồn, cùng tên tập -> tên staging khác nhau nhờ subdir
        recs = []
        for feed in ("feedA", "feedB"):
            d = tmp_path / "raw" / "kenh" / feed; d.mkdir(parents=True)
            (d / "Tập_1.mp3").write_bytes(b"x")
            recs.append({"path": str(d / "Tập_1.mp3"), "source": "kenh", "subdir": feed})
        names = sorted(os.listdir(cap.stage_batch(recs, str(tmp_path / "work"))))
        assert names == ["kenh__feedA__Tập_1.mp3", "kenh__feedB__Tập_1.mp3"]

    def test_processed_resume(self, tmp_path):
        p = cap.Processed(str(tmp_path / "work"))
        p.add([{"path": "/a.mp3", "source": "s", "key": "k"}])
        assert cap.Processed(str(tmp_path / "work")).done == {"/a.mp3"}


class TestCleanup:
    def test_cleanup_batch_removes_raw_and_intermediate(self, tmp_path):
        raw = tmp_path / "raw" / "k"; raw.mkdir(parents=True)
        (raw / "a.m4a").write_bytes(b"x"); (raw / "a.json").write_text("{}")
        w = tmp_path / "work"
        for st in ("s0_ingest", "s1_separate", "s2_segment"):
            (w / st / "audio").mkdir(parents=True)
        (w / "s0_ingest" / "manifest.jsonl").write_text(
            json.dumps({"id": "id1", "source_path": "k__a.m4a"}) + "\n"
            + json.dumps({"id": "id2", "source_path": "k__khac.m4a"}) + "\n")
        for f in ("s0_ingest/audio/id1.wav", "s1_separate/audio/id1.wav", "s2_segment/audio/id1_0_1.wav",
                  "s2_segment/audio/id1_1_2.wav", "s0_ingest/audio/id2.wav"):
            (w / f).write_bytes(b"w")
        n = cap.cleanup_batch(str(w), {"k__a.m4a"}, [str(raw / "a.m4a")])
        assert n == {"raw": 1, "work": 4}
        assert not (raw / "a.m4a").exists() and (raw / "a.json").exists()      # giữ sidecar
        assert (w / "s0_ingest/audio/id2.wav").exists()                        # file khác lô không đụng

    def test_cleanup_tier_c_only_in_s7(self, tmp_path):
        w = tmp_path / "work"; (w / "s7_loudnorm" / "audio").mkdir(parents=True); (w / "s8_package").mkdir()
        for f in ("c1.wav", "a1.wav"):
            (w / "s7_loudnorm" / "audio" / f).write_bytes(b"w")
        outside = tmp_path / "ngoai.wav"; outside.write_bytes(b"w")
        (w / "s8_package" / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in [
            {"tier": "C", "audio_path": str(w / "s7_loudnorm/audio/c1.wav")},
            {"tier": "A", "audio_path": str(w / "s7_loudnorm/audio/a1.wav")},
            {"tier": "C", "audio_path": str(outside)},   # ngoài s7 -> không xóa
        ]) + "\n")
        assert cap.cleanup_tier_c(str(w)) == 1
        assert not (w / "s7_loudnorm/audio/c1.wav").exists()
        assert (w / "s7_loudnorm/audio/a1.wav").exists() and outside.exists()
