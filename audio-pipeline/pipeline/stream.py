"""Chế độ stream: 8 stage chạy song song, mỗi stage một tiến trình.

    python -m pipeline serve --raw-dir raw --workdir work [--input-done] [--cleanup]

Cơ chế:
  * Stage sN tail manifest của sN-1 (ManifestTail: theo offset, bỏ qua dòng ghi dở),
    lọc record chưa có trong manifest của mình, gom rồi Worker.process(). Manifest ghi
    append + flush từng dòng chính là dấu "đã làm": kill giữa chừng, chạy lại là tiếp.
  * s0 quét thư mục raw thay vì tail manifest.
  * Cờ xong lan xuống: <workdir>/INPUT_DONE (không còn file raw mới; wrapper crawler
    hoặc --input-done tạo) -> s0 cạn hàng đợi thì tạo s0_ingest/DONE -> s1 ... -> s7.
    s7 DONE -> supervisor chạy s8 một lần rồi thoát.
  * Stage chết (exit != 0) -> supervisor khởi động lại (tối đa stream.max_restarts).
  * Ctrl-C: tắt mọi stage; chạy lại lệnh cũ là resume.
"""

import importlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time

from . import manifest

STAGES = ["s0_ingest", "s1_separate", "s2_segment", "s3_quality",
          "s4_speaker", "s5_transcribe", "s6_textnorm", "s7_loudnorm"]
FINAL = "s8_package"
INPUT_DONE = "INPUT_DONE"


def done_flag(workdir: str, stage: str | None) -> str:
    """Đường dẫn cờ DONE của stage; stage None = đầu vào (INPUT_DONE)."""
    return os.path.join(workdir, INPUT_DONE) if stage is None else os.path.join(workdir, stage, "DONE")


def touch(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")


class _Prefix:
    """Gắn tiền tố '[s3] ' vào đầu mỗi dòng stdout để log 8 stage chung một luồng đọc được."""

    def __init__(self, stream, prefix: str):
        self.stream, self.prefix, self._nl = stream, prefix, True

    def write(self, s: str) -> None:
        for part in s.splitlines(True):
            if self._nl:
                self.stream.write(self.prefix)
            self.stream.write(part)
            self._nl = part.endswith("\n")

    def flush(self) -> None:
        self.stream.flush()


# ------------------------------------------------------------------ hãm theo dung lượng audio của stage
GB = 1 << 30


def dir_size(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


class DiskGuard:
    """Stage sinh nhiều dữ liệu (s0: mp3 -> wav gấp 3) ngừng nhận việc khi <workdir>/<stage>/audio
    vượt max_gb; giảm xuống dưới resume_gb (stage sau xử lý xong và xóa, cần cleanup) thì tiếp.
    Tùy chọn thêm: đĩa trống dưới min_free_gb cũng dừng. Đo lại tối đa mỗi RECHECK_S giây."""

    RECHECK_S = 10.0

    def __init__(self, workdir: str, stage: str, max_gb: float, resume_gb: float, min_free_gb: float = 0.0):
        self.dir = os.path.join(workdir, stage, "audio")
        self.workdir, self.stage = workdir, stage
        self.max_b, self.resume_b, self.min_free = max_gb * GB, resume_gb * GB, min_free_gb * GB
        self.paused = False
        self._t = 0.0

    def touch(self) -> None:
        """Vừa xử lý xong một lượt -> lần hỏi kế tiếp phải đo lại ngay (bỏ throttle)."""
        self._t = 0.0

    def blocked(self) -> bool:
        if self.max_b <= 0 and self.min_free <= 0:
            return False
        if time.time() - self._t < self.RECHECK_S:
            return self.paused
        self._t = time.time()
        size = dir_size(self.dir) if self.max_b > 0 else 0
        free = shutil.disk_usage(self.workdir).free if self.min_free > 0 else None
        too_big = self.max_b > 0 and size > self.max_b
        too_little = free is not None and free < self.min_free
        if not self.paused and (too_big or too_little):
            self.paused = True
            why = f"audio {size / GB:.2f} GB > {self.max_b / GB:g} GB" if too_big else f"đĩa trống {free / GB:.1f} GB"
            print(f"[stream] {self.stage} tạm dừng: {why} — chờ stage sau xử lý xong "
                  f"(giảm dưới {self.resume_b / GB:g} GB)", flush=True)
        elif self.paused and (self.max_b <= 0 or size < self.resume_b) and \
                (free is None or free > self.min_free * 1.5):
            self.paused = False
            print(f"[stream] {self.stage} tiếp tục: audio {size / GB:.2f} GB", flush=True)
        return self.paused


# ------------------------------------------------------------------ vòng lặp một stage
def stage_loop(stage: str, cfg: dict, workdir: str) -> None:
    idx = STAGES.index(stage)
    upstream = STAGES[idx - 1] if idx > 0 else None
    poll_s = float(cfg["stream"].get("poll_seconds", 2))
    mod = importlib.import_module(f"pipeline.stages.{stage}")
    worker = mod.Worker(cfg, workdir)
    tail = None if upstream is None else manifest.ManifestTail(manifest.manifest_path(workdir, upstream))
    my_done, up_done_path = done_flag(workdir, stage), done_flag(workdir, upstream)
    if os.path.exists(my_done):
        os.remove(my_done)

    def poll() -> list:
        items = worker.scan() if tail is None else tail.poll()
        return [it for it in items if not worker.is_done(it)]

    guard = None
    st = cfg["stream"]
    if stage in st.get("disk_guard_stages", ["s0_ingest", "s2_segment"]):
        guard = DiskGuard(workdir, stage, float(st.get("max_dir_gb", 20)), float(st.get("resume_dir_gb", 5)),
                          float(st.get("min_free_gb", 0)))
        if guard.max_b > 0 and not st.get("cleanup"):
            print(f"[stream] {stage} CẢNH BÁO: max_dir_gb chỉ có ý nghĩa khi cleanup bật "
                  f"(stage sau xóa wav đã dùng xong); không thì thư mục không giảm và stage dừng mãi", flush=True)
    print(f"[stream] {stage} sẵn sàng (gom {worker.batch_n} item mỗi lượt xử lý, "
          f"chờ gom tối đa {worker.flush_s:g}s)")
    try:
        _loop(stage, worker, poll, up_done_path, my_done, poll_s, guard)
    finally:
        worker.close()


def _key(item) -> str:
    return item if isinstance(item, str) else item["id"]


def _loop(stage, worker, poll, up_done_path, my_done, poll_s, guard=None) -> None:
    pending: list = []
    first_pending_t: float | None = None
    n_done = 0
    while True:
        # s0 quét lại thư mục mỗi vòng -> file đang chờ (bị hãm / chưa đủ file) sẽ xuất hiện lại: bỏ trùng
        have = {_key(it) for it in pending}
        new = [it for it in poll() if _key(it) not in have]
        if new:
            pending += new
            first_pending_t = first_pending_t or time.time()
        up_done = os.path.exists(up_done_path)
        ready, pending = worker.ready(pending, up_done)
        waited = (time.time() - first_pending_t) if first_pending_t else 0.0
        if ready and (len(ready) >= worker.batch_n or up_done or waited >= worker.flush_s):
            if guard is None:
                worker.process(ready)
                n_done += len(ready)
                first_pending_t = time.time() if pending else None
                continue  # có thể còn hàng: đọc tiếp ngay, không ngủ
            # stage bị hãm theo dung lượng: từng item một, đo lại trước mỗi item
            did = 0
            while ready and not guard.blocked():
                worker.process([ready.pop(0)])
                guard.touch()
                did += 1
            n_done += did
            pending = ready + pending  # phần bị hãm giữa chừng giữ lại
            first_pending_t = time.time() if pending else None
            if did and not pending:
                continue
        elif ready:  # chưa đủ batch, chưa hết giờ chờ -> giữ lại
            pending = ready + pending
        if up_done and not pending and not new:
            if not poll():  # đọc lại lần cuối cho chắc rồi mới báo xong
                touch(my_done)
                print(f"[stream] {stage} XONG ({n_done} item trong phiên này)")
                break
        time.sleep(poll_s)


# ------------------------------------------------------------------ supervisor
def counts(workdir: str) -> dict:
    out = {}
    for st in STAGES + [FINAL]:
        p = manifest.manifest_path(workdir, st)
        n = 0
        if os.path.exists(p):
            with open(p, "rb") as f:
                n = sum(1 for line in f if line.strip())
        out[st] = (n, os.path.exists(done_flag(workdir, st)))
    return out


def status_line(workdir: str) -> str:
    c = counts(workdir)
    parts = [("IN✓" if os.path.exists(done_flag(workdir, None)) else "IN…")]
    for st in STAGES + [FINAL]:
        n, d = c[st]
        parts.append(f"{st.split('_')[0]}={n}{'✓' if d else ''}")
    return " ".join(parts)


def serve(cfg: dict, workdir: str, input_done: bool = False, run_final: bool = True) -> int:
    """run_final=False: không chạy s8 khi s7 xong (dataset gói dần bằng `pipeline package`)."""
    os.makedirs(workdir, exist_ok=True)
    cfg = dict(cfg)
    cfg["stream"] = {**cfg.get("stream", {}), "_active": True}
    for st in STAGES + [FINAL]:  # cờ của lần chạy trước không còn giá trị
        if os.path.exists(done_flag(workdir, st)):
            os.remove(done_flag(workdir, st))
    if input_done:
        touch(done_flag(workdir, None))
    elif os.path.exists(done_flag(workdir, None)):
        os.remove(done_flag(workdir, None))

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(cfg, f, ensure_ascii=False)
        cfg_path = f.name
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    max_restarts = int(cfg["stream"].get("max_restarts", 5))
    status_every = float(cfg["stream"].get("status_seconds", 60))

    def spawn(stage: str) -> subprocess.Popen:
        return subprocess.Popen([sys.executable, "-m", "pipeline.stream", "stage", stage, cfg_path, workdir],
                                env=env)

    procs = {st: spawn(st) for st in STAGES}
    restarts = {st: 0 for st in STAGES}
    print(f"[serve] {len(procs)} stage đã chạy, workdir={workdir}; "
          f"{'đầu vào cố định' if input_done else f'chờ {INPUT_DONE} khi hết file mới'}", flush=True)

    stopping = False

    def on_sigint(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, on_sigint)
    signal.signal(signal.SIGTERM, on_sigint)

    last_status = 0.0
    rc = 0
    try:
        while True:
            if stopping:
                print("\n[serve] dừng — tắt các stage; chạy lại lệnh cũ để tiếp tục", flush=True)
                for p in procs.values():
                    if p.poll() is None:
                        p.terminate()
                for p in procs.values():
                    p.wait()
                return 130
            for st, p in procs.items():
                code = p.poll()
                if code is None or os.path.exists(done_flag(workdir, st)):
                    continue
                if restarts[st] >= max_restarts:
                    print(f"[serve] {st} lỗi {restarts[st]} lần (mã {code}) — bỏ cuộc", flush=True)
                    stopping, rc = True, 1
                    break
                restarts[st] += 1
                print(f"[serve] {st} thoát mã {code} — khởi động lại ({restarts[st]}/{max_restarts})", flush=True)
                time.sleep(5)
                procs[st] = spawn(st)
            if time.time() - last_status >= status_every:
                print(f"[serve] {time.strftime('%H:%M:%S')} {status_line(workdir)}", flush=True)
                last_status = time.time()
            if os.path.exists(done_flag(workdir, STAGES[-1])):
                break
            time.sleep(2)
        for p in procs.values():
            p.wait()
        if rc:
            return rc
        if not run_final:
            print(f"[serve] s0-s7 xong: {status_line(workdir)} — bỏ qua {FINAL} (--no-s8); "
                  f"gói bằng: python -m pipeline package --workdir {workdir}", flush=True)
            return 0
        print(f"[serve] s0-s7 xong: {status_line(workdir)} — chạy {FINAL}", flush=True)
        r = subprocess.run([sys.executable, "-m", "pipeline.stage_runner", FINAL, cfg_path, workdir, "-"])
        if r.returncode != 0:
            print(f"[serve] {FINAL} lỗi (mã {r.returncode}); chạy lại: python -m pipeline run --stages s8", flush=True)
            return r.returncode
        touch(done_flag(workdir, FINAL))
        print(f"[serve] hoàn tất: {status_line(workdir)}", flush=True)
        return 0
    finally:
        try:
            os.remove(cfg_path)
        except OSError:
            pass


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["stage"]:
        stage, cfg_path, workdir = argv[1:4]
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        sys.stdout = _Prefix(sys.stdout, f"[{stage.split('_')[0]}] ")
        # Ctrl-C ở terminal gửi SIGINT cho cả nhóm: stage bỏ qua, chờ supervisor SIGTERM
        # -> SystemExit -> finally: worker.close() (đóng manifest, tắt worker PhoWhisper)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
        stage_loop(stage, cfg, workdir)
        return
    raise SystemExit("dùng: python -m pipeline serve ... (xem pipeline.cli)")


if __name__ == "__main__":
    main()
