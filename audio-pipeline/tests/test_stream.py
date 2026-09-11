"""Test chế độ stream — không cần model/mạng."""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline import manifest, stream
from pipeline.stages.s2_segment import done_files
from pipeline.stages.s4_speaker import split_ready


class TestManifestTail:
    def test_partial_line_waits(self, tmp_path):
        p = tmp_path / "m.jsonl"
        t = manifest.ManifestTail(str(p))
        assert t.poll() == []  # chưa có file
        with open(p, "w") as f:
            f.write(json.dumps({"id": "a"}) + "\n" + '{"id": "b"')  # dòng 2 ghi dở
        assert [r["id"] for r in t.poll()] == ["a"]
        with open(p, "a") as f:
            f.write("}\n" + json.dumps({"id": "c"}) + "\n")
        assert [r["id"] for r in t.poll()] == ["b", "c"]
        assert t.poll() == []

    def test_utf8_offsets(self, tmp_path):
        p = tmp_path / "m.jsonl"
        t = manifest.ManifestTail(str(p))
        with open(p, "w", encoding="utf-8") as f:
            f.write(json.dumps({"id": "Tập 1 ơi"}, ensure_ascii=False) + "\n")
        assert t.poll()[0]["id"] == "Tập 1 ơi"
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps({"id": "x"}) + "\n")
        assert [r["id"] for r in t.poll()] == ["x"]


class TestResumeMarkers:
    def test_s2_done_files_needs_all_segments(self, tmp_path):
        p = tmp_path / "m.jsonl"
        with open(p, "w") as f:
            for i in range(3):
                f.write(json.dumps({"id": f"f1_{i}", "file_id": "f1", "n_segments": 3}) + "\n")
            f.write(json.dumps({"id": "f2_0", "file_id": "f2", "n_segments": 2}) + "\n")  # thiếu 1
            f.write(json.dumps({"id": "f3_0", "file_id": "f3"}) + "\n")  # manifest cũ
        assert done_files(str(p)) == {"f1", "f3"}

    def test_s4_split_ready_by_file(self):
        from collections import Counter
        pend = [{"id": "a1", "file_id": "a", "n_segments": 2}, {"id": "a2", "file_id": "a", "n_segments": 2},
                {"id": "b1", "file_id": "b", "n_segments": 3},
                {"id": "c1", "file_id": "c", "n_segments": 2}]
        ready, rest = split_ready(pend, Counter({"c": 1}), upstream_done=False)  # c: 1 đã ghi + 1 chờ = đủ
        assert sorted(r["id"] for r in ready) == ["a1", "a2", "c1"]
        assert [r["id"] for r in rest] == ["b1"]
        ready, rest = split_ready(pend, Counter(), upstream_done=True)
        assert len(ready) == 4 and rest == []


class FakeWorker:
    """Ghi record vào manifest của stage mình; gom batch_n để kiểm tra logic gom/flush."""
    batch_n = 3
    flush_s = 0.5

    def __init__(self, cfg, workdir):
        self.w = manifest.ManifestWriter(manifest.manifest_path(workdir, cfg["_stage"]))
        self.calls = []

    def is_done(self, rec):
        return self.w.is_done(rec["id"])

    def ready(self, pending, up):
        return pending, []

    def process(self, recs):
        self.calls.append(len(recs))
        for r in recs:
            self.w.write({**r, "seen": True})

    def close(self):
        self.w.close()


class TestStageLoop:
    def _run_loop(self, monkeypatch, workdir, stage, upstream, records_by_time):
        """records_by_time: [(delay_s, [rec...])] ghi vào manifest upstream theo thời gian, rồi tạo DONE."""
        import threading
        import types

        holder = {}

        def fake_import(name):
            mod = types.SimpleNamespace()
            def W(cfg, wd):
                holder["w"] = FakeWorker({**cfg, "_stage": stage}, wd)
                return holder["w"]
            mod.Worker = W
            return mod
        monkeypatch.setattr(stream.importlib, "import_module", fake_import)

        up_path = manifest.manifest_path(workdir, upstream)
        def feeder():
            for delay, recs in records_by_time:
                time.sleep(delay)
                with manifest.ManifestWriter(up_path) as w:
                    for r in recs:
                        w.write(r)
            time.sleep(0.2)
            stream.touch(stream.done_flag(workdir, upstream))
        th = threading.Thread(target=feeder); th.start()
        stream.stage_loop(stage, {"stream": {"poll_seconds": 0.1}}, workdir)
        th.join()
        return holder["w"]

    def test_batches_then_flush_then_done(self, monkeypatch, tmp_path):
        wd = str(tmp_path)
        recs = [{"id": f"r{i}"} for i in range(5)]
        w = self._run_loop(monkeypatch, wd, "s3_quality", "s2_segment",
                           [(0, recs[:3]), (0.1, recs[3:4]), (1.0, recs[4:])])
        # 3 đủ batch ngay; r3 chờ hết flush_s (0.5s) chạy lẻ; r4 chạy khi upstream DONE
        assert w.calls == [3, 1, 1]
        assert os.path.exists(stream.done_flag(wd, "s3_quality"))
        assert sorted(manifest.done_ids(manifest.manifest_path(wd, "s3_quality"))) == [f"r{i}" for i in range(5)]

    def test_resume_skips_done(self, monkeypatch, tmp_path):
        wd = str(tmp_path)
        with manifest.ManifestWriter(manifest.manifest_path(wd, "s3_quality")) as w:
            w.write({"id": "r0", "seen": True})  # lần chạy trước đã làm r0
        w = self._run_loop(monkeypatch, wd, "s3_quality", "s2_segment",
                           [(0, [{"id": "r0"}, {"id": "r1"}])])
        assert w.calls == [1]
        assert len(manifest.read_records(manifest.manifest_path(wd, "s3_quality"))) == 2


class TestVerifyClient:
    def test_persistent_worker_roundtrip_and_restart(self, monkeypatch, tmp_path):
        """Giả lập worker --serve bằng script nhỏ: lần 1 trả ok, kill giữa chừng -> client tự khởi động lại."""
        import subprocess as sp
        from pipeline.stages.s5_transcribe import VerifyClient

        fake = tmp_path / "fake_worker.py"
        fake.write_text(
            "import sys, json\n"
            "print('ready', flush=True)\n"
            "for line in sys.stdin:\n"
            "    req = json.loads(line)\n"
            "    items = json.load(open(req['in']))\n"
            "    json.dump({k: 'text ' + k for k in items}, open(req['out'], 'w'))\n"
            "    print('ok', flush=True)\n"
        )
        real_popen = sp.Popen
        def fake_popen(cmd, **kw):
            return real_popen([sys.executable, str(fake)], **kw)
        monkeypatch.setattr(sp, "Popen", fake_popen)

        c = VerifyClient("m", "cpu", 4)
        out = c.run([{"id": "a", "audio_path": "/x/a.wav"}, {"id": "b", "audio_path": "/x/b.wav"}])
        assert out == {"a": "text a", "b": "text b"}
        pid1 = c.proc.pid
        c.proc.kill(); c.proc.wait()
        out = c.run([{"id": "c", "audio_path": "/x/c.wav"}])  # worker chết -> khởi động lại
        assert out == {"c": "text c"} and c.proc.pid != pid1
        c.close()
        assert c.proc.poll() is not None


class TestGpuLock:
    def test_lock_is_exclusive_across_processes(self, tmp_path):
        import subprocess as sp
        from pipeline.gpulock import gpu_lock

        wd = str(tmp_path)
        script = (
            "import sys, time; sys.path.insert(0, %r)\n"
            "from pipeline.gpulock import gpu_lock\n"
            "with gpu_lock(%r): time.sleep(1.0)\n"
            "print('released')\n"
        ) % (os.path.join(os.path.dirname(__file__), ".."), wd)
        t0 = time.time()
        with gpu_lock(wd):
            p = sp.Popen([sys.executable, "-c", script], stdout=sp.PIPE, text=True)
            time.sleep(0.5)
            assert p.poll() is None  # tiến trình kia đang chờ khóa
        p.wait(timeout=10)
        assert time.time() - t0 >= 1.4  # 0.5 giữ khóa + 1.0 bên kia


class TestCleanupCommand:
    def test_removes_only_finished_intermediates(self, tmp_path):
        from pipeline import cleanup

        wd, raw = tmp_path / "work", tmp_path / "raw"
        (raw / "f").mkdir(parents=True)
        def touch(p, n=1024):
            p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b"x" * n)
        touch(raw / "f" / "a.mp3"); touch(raw / "f" / "a.json"); touch(raw / "f" / "b.mp3")
        with manifest.ManifestWriter(manifest.manifest_path(str(wd), "s0_ingest")) as w:
            w.write({"id": "A", "source_path": "f/a.mp3"})  # b.mp3 chưa qua s0
        touch(wd / "s0_ingest" / "audio" / "A.wav"); touch(wd / "s0_ingest" / "audio" / "B.wav")
        with manifest.ManifestWriter(manifest.manifest_path(str(wd), "s2_segment")) as w:
            w.write({"id": "A_1", "file_id": "A", "n_segments": 1})
            w.write({"id": "B_1", "file_id": "B", "n_segments": 2})  # B chưa đủ segment
        touch(wd / "s2_segment" / "audio" / "A_1.wav"); touch(wd / "s2_segment" / "audio" / "B_1.wav")
        with manifest.ManifestWriter(manifest.manifest_path(str(wd), "s7_loudnorm")) as w:
            w.write({"id": "A_1", "audio_path": str(wd / "s7_loudnorm" / "audio" / "A_1.wav"), "loudnorm": "lufs"})
        touch(wd / "s7_loudnorm" / "audio" / "A_1.wav")
        with manifest.ManifestWriter(manifest.manifest_path(str(wd), "s8_package")) as w:
            w.write({"id": "A_1", "audio_path": str(wd / "s7_loudnorm" / "audio" / "A_1.wav"), "tier": "C"})

        freed = cleanup.run(str(wd), str(raw), dry_run=True)
        assert (raw / "f" / "a.mp3").exists() and sum(freed.values()) == 4 * 1024
        cleanup.run(str(wd), str(raw))
        assert not (raw / "f" / "a.mp3").exists() and (raw / "f" / "a.json").exists()
        assert (raw / "f" / "b.mp3").exists()                      # chưa qua s0
        assert not (wd / "s0_ingest" / "audio" / "A.wav").exists()
        assert (wd / "s0_ingest" / "audio" / "B.wav").exists()     # B chưa đủ segment ở s2
        assert not (wd / "s2_segment" / "audio" / "A_1.wav").exists()
        assert (wd / "s2_segment" / "audio" / "B_1.wav").exists()  # chưa qua s7
        assert not (wd / "s7_loudnorm" / "audio" / "A_1.wav").exists()  # tier C


class TestStageDiskGuard:
    def test_stage_waits_while_disk_low(self, monkeypatch, tmp_path):
        """s0/s1 không xử lý khi đĩa thấp; đĩa trống lại -> xử lý rồi DONE."""
        import threading
        import types
        from collections import namedtuple
        DU = namedtuple("DU", "total used free")
        free = {"v": 5 << 30}
        monkeypatch.setattr(stream.shutil, "disk_usage", lambda p: DU(0, 0, free["v"]))
        wd = str(tmp_path)
        holder = {}
        def fake_import(name):
            mod = types.SimpleNamespace()
            def W(cfg, w):
                holder["w"] = FakeWorker({**cfg, "_stage": "s1_separate"}, w); return holder["w"]
            mod.Worker = W
            return mod
        monkeypatch.setattr(stream.importlib, "import_module", fake_import)
        with manifest.ManifestWriter(manifest.manifest_path(wd, "s0_ingest")) as w:
            w.write({"id": "f1"})
        stream.touch(stream.done_flag(wd, "s0_ingest"))
        def raise_disk():
            time.sleep(0.5); assert holder["w"].calls == []  # vẫn chưa xử lý vì đĩa thấp
            free["v"] = 40 << 30
        threading.Thread(target=raise_disk).start()
        stream.stage_loop("s1_separate", {"stream": {"poll_seconds": 0.1, "min_free_gb": 20}}, wd)
        assert holder["w"].calls == [1] and os.path.exists(stream.done_flag(wd, "s1_separate"))
