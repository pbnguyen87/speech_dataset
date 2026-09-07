"""Test hàm thuần — không cần mạng."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import shutil
import subprocess

import pytest

from crawler import downloader
from crawler.downloader import ext_from_url, safe_filename
from crawler.sources.html.adapter import DEFAULT_PATTERN, audio_links, extract_links
from crawler.sources.rss.adapter import parse_duration, parse_feed
from crawler.cli import needs_probe
from crawler.sources.archive_org.adapter import to_identifier
from crawler import targets
from crawler.sources.youtube.adapter import build_cmd, finalize, info_to_sidecar
from crawler.state import Ledger
from crawler import sources

RSS_FIXTURE = """<?xml version="1.0"?>
<rss version="2.0"><channel><title>Podcast Test</title>
<item><title>Tập 1: Xin chào</title>
  <enclosure url="https://cdn.example.com/ep1.mp3" type="audio/mpeg" length="123"/>
  <pubDate>Mon, 01 Jan 2026 00:00:00 GMT</pubDate>
  <description>Mô tả tập 1</description></item>
<item><title>Tập không có audio</title></item>
<item><title>Tập 2</title>
  <enclosure url="https://cdn.example.com/ep2.m4a" type="audio/mp4"/></item>
</channel></rss>"""
RSS_FIXTURE = RSS_FIXTURE.replace('<rss version="2.0">',
    '<rss version="2.0" xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd">').replace(
    "<description>Mô tả tập 1</description>", "<description>Mô tả tập 1</description><itunes:duration>01:02:03</itunes:duration>")


class TestState:
    def test_ledger_resume(self, tmp_path):
        d = str(tmp_path)
        with Ledger(d) as lg:
            assert not lg.has("http://x/a.mp3")
            lg.add("http://x/a.mp3", "a.mp3", "s1")
        with Ledger(d) as lg:
            assert lg.has("http://x/a.mp3")
            assert not lg.has("http://x/b.mp3")


class TestDownloader:
    def test_safe_filename(self):
        assert safe_filename("Tập 1: Xin chào / thế giới!") == "Tập_1_Xin_chào_thế_giới"
        assert safe_filename("") == "untitled"
        assert len(safe_filename("x" * 500)) == 120

    def test_ext_from_url(self):
        assert ext_from_url("https://a.com/f.MP3?x=1") == ".mp3"
        assert ext_from_url("https://a.com/stream") == ".mp3"
        assert ext_from_url("https://a.com/f.m4a") == ".m4a"


class TestRSS:
    def test_parse_feed(self):
        eps = parse_feed(RSS_FIXTURE)
        assert len(eps) == 2  # item không có enclosure bị bỏ
        assert eps[0]["title"] == "Tập 1: Xin chào"
        assert eps[0]["audio_url"].endswith("ep1.mp3")
        assert eps[1]["audio_url"].endswith("ep2.m4a")
        assert eps[0]["duration"] == 3723 and eps[1]["duration"] is None

    def test_parse_duration(self):
        assert parse_duration("01:02:03") == 3723
        assert parse_duration("12:30") == 750
        assert parse_duration("3600") == 3600
        assert parse_duration("") is None and parse_duration("abc") is None


class TestHtmlListing:
    HTML = '''<a href="/audio/ep1.mp3">1</a> <a HREF="https://c.dn/ep2.MP3?t=9">2</a>
              <a href="/page/about.html">x</a>'''

    def test_extract_and_filter(self):
        links = extract_links(self.HTML, "https://site.vn/list")
        assert "https://site.vn/audio/ep1.mp3" in links
        auds = audio_links(links, DEFAULT_PATTERN)
        assert len(auds) == 2
        assert all(".mp3" in u.lower() for u in auds)


class TestRegistry:
    def test_all_sources_discovered(self):
        assert set(sources.REGISTRY) == {"archive_org", "rss", "html", "youtube"}
        for mod in sources.REGISTRY.values():
            assert callable(getattr(mod, "list_items", None) or getattr(mod, "crawl", None))

    def test_unknown_type(self):
        try:
            sources.get("tiktok")
        except KeyError as e:
            assert "tiktok" in e.args[0]
        else:
            raise AssertionError("phải raise KeyError")


class TestYouTube:
    def test_build_cmd(self):
        cmd = build_cmd({"name": "k", "url": "https://youtube.com/@x/videos",
                         "max_items": 5, "extra_args": ["--proxy", "p"]},
                        "/out", 3.0, None)
        assert cmd[-1] == "https://youtube.com/@x/videos"
        assert "--playlist-end" in cmd and cmd[cmd.index("--playlist-end") + 1] == "5"
        assert cmd[cmd.index("--sleep-interval") + 1] == "3.0"
        assert "--proxy" in cmd
        # --limit của CLI thắng max_items
        cmd = build_cmd({"name": "k", "url": "u", "max_items": 5}, "/out", 1, 2)
        assert cmd[cmd.index("--playlist-end") + 1] == "2"

    def test_finalize_writes_sidecar_and_ledger(self, tmp_path):
        import json
        d = str(tmp_path)
        (tmp_path / "abc.m4a").write_bytes(b"x")
        (tmp_path / "abc.info.json").write_text(json.dumps({
            "id": "abc", "title": "Tập 1", "webpage_url": "https://youtu.be/abc",
            "channel": "Kênh", "duration": 100, "description": "d" * 1000}))
        (tmp_path / "zzz.info.json").write_text(json.dumps({"id": "zzz"}))  # không có audio
        assert finalize(d, "src") == 1
        side = json.loads((tmp_path / "abc.json").read_text())
        assert side["title"] == "Tập 1" and side["source"] == "src"
        assert side["youtube_id"] == "abc" and len(side["description"]) == 500
        assert not (tmp_path / "abc.info.json").exists()
        assert (tmp_path / "zzz.info.json").exists()  # giữ lại để lần sau xử lý
        assert finalize(d, "src") == 0  # chạy lại không ghi trùng
        with Ledger(d) as lg:
            assert lg.has("youtube/abc")

    def test_info_to_sidecar_fallback_url(self):
        s = info_to_sidecar({"id": "q1"}, "s")
        assert s["url"].endswith("v=q1") and s["title"] == "q1"


class TestTargets:
    def test_single_list_file_merged_dedup(self, tmp_path):
        (tmp_path / "links.txt").write_text(
            "# comment\nhttps://c\n\n  https://a  \nhttps://d\n")
        cfg = {"name": "x", "url": "https://a", "urls": ["https://b", "https://c"],
               "urls_file": "links.txt", "_config_dir": str(tmp_path)}
        assert targets.resolve(cfg, "url") == ["https://a", "https://b", "https://c", "https://d"]

    def test_file_only_relative_to_config_dir(self, tmp_path):
        sub = tmp_path / "cfg"; sub.mkdir()
        (sub / "ids.txt").write_text("id1\nid2\n")
        cfg = {"name": "x", "identifiers_file": "ids.txt", "_config_dir": str(sub)}
        assert targets.resolve(cfg, "identifier") == ["id1", "id2"]

    def test_missing_all_raises(self):
        try:
            targets.resolve({"name": "x"}, "url")
        except SystemExit as e:
            assert "url | urls | urls_file" in str(e)
        else:
            raise AssertionError("phải raise")

    def test_youtube_cmd_uses_all_urls(self, tmp_path):
        (tmp_path / "yt.txt").write_text("https://youtu.be/1\nhttps://youtu.be/2\n")
        cmd = build_cmd({"name": "k", "urls_file": "yt.txt", "_config_dir": str(tmp_path)},
                        "/out", 1.0, None)
        assert cmd[-2:] == ["https://youtu.be/1", "https://youtu.be/2"]


class TestArchiveIdentifier:
    def test_to_identifier(self):
        assert to_identifier("Uynngaowww.truyenaudio.net") == "Uynngaowww.truyenaudio.net"
        assert to_identifier("https://archive.org/details/abc_123/") == "abc_123"
        assert to_identifier("https://archive.org/details/abc?x=1#y") == "abc"
        assert to_identifier("https://archive.org/download/abc/f.mp3") == "abc"

    def test_dedup_after_normalize(self, monkeypatch):
        from crawler.sources.archive_org import adapter
        calls = []
        monkeypatch.setattr(adapter, "_list_item", lambda i: calls.append(i) or [])
        adapter.list_items({"name": "x", "identifiers": [
            "abc", "https://archive.org/details/abc", "xyz"]})
        assert calls == ["abc", "xyz"]


class TestTrim:
    def test_ext_from_format(self):
        assert downloader.ext_from_format("mov,mp4,m4a,3gp,3g2,mj2") == ".m4a"
        assert downloader.ext_from_format("mp3") == ".mp3"
        assert downloader.ext_from_format("ogg") == ".ogg"
        assert downloader.ext_from_format("weird") == ".mp3"

    def test_trim_cmd(self):
        cmd = downloader.trim_cmd("https://x/a.m4a", "/out/a.m4a.part", 36000)
        assert cmd[0] == "ffmpeg" and "-c" in cmd and cmd[cmd.index("-c") + 1] == "copy"
        assert cmd[cmd.index("-t") + 1] == "36000"
        assert cmd[cmd.index("-f") + 1] == "ipod"   # muxer theo đuôi dest
        assert cmd[-1] == "/out/a.m4a.part"

    @pytest.mark.skipif(not shutil.which("ffmpeg"), reason="cần ffmpeg")
    def test_download_trimmed_local_file(self, tmp_path):
        src = str(tmp_path / "long.m4a")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=30",
                        "-c:a", "aac", "-b:a", "32k", src], check=True)
        info = downloader.probe(src)
        assert info and 29 < info["duration"] < 31 and info["ext"] == ".m4a"
        dest = str(tmp_path / "short.m4a")
        assert downloader.download_trimmed(src, dest, max_seconds=10, delay_s=0)
        out = downloader.probe(dest)
        assert 9.5 < out["duration"] < 11
        assert not (tmp_path / "short.m4a.part").exists()

    def test_probe_returns_none_on_bad_input(self, tmp_path):
        assert downloader.probe(str(tmp_path / "khong-ton-tai.mp3")) is None


class TestNeedsProbe:
    def test_skip_when_known_and_short(self):
        assert needs_probe(1800, 36000) is False
    def test_probe_when_unknown_or_long(self):
        assert needs_probe(None, 36000) is True
        assert needs_probe(139143, 36000) is True


class TestCrawlSourceTrim:
    """max_hours: không bao giờ tải nguyên file dài khi ffprobe lỗi hoặc thiếu ffmpeg."""

    def _run(self, monkeypatch, tmp_path, probe_result, feed_duration):
        from crawler import cli, sources as srcs
        calls = []
        monkeypatch.setattr(shutil, "which", lambda n: "/usr/bin/" + n)
        monkeypatch.setattr(srcs, "get", lambda t: type("A", (), {"list_items": staticmethod(
            lambda cfg: [{"url": "https://x/a.mp3", "title": "a", "duration": feed_duration}])}))
        monkeypatch.setattr(downloader, "probe", lambda url: probe_result)
        monkeypatch.setattr(downloader, "download", lambda *a, **k: calls.append("full") or True)
        monkeypatch.setattr(downloader, "download_trimmed",
                            lambda url, dest, s, **k: calls.append(("trim", s)) or open(dest, "wb").close() or True)
        cli.crawl_source({"name": "t", "type": "x", "max_hours": 0.1, "delay_s": 0}, str(tmp_path), None)
        return calls

    def test_probe_fails_still_trims(self, monkeypatch, tmp_path):
        assert self._run(monkeypatch, tmp_path, None, None) == [("trim", 360.0)]

    def test_probe_fails_feed_long_trims(self, monkeypatch, tmp_path):
        assert self._run(monkeypatch, tmp_path, None, 139143) == [("trim", 360.0)]

    def test_short_file_downloads_full(self, monkeypatch, tmp_path):
        assert self._run(monkeypatch, tmp_path, {"duration": 100, "ext": ".mp3"}, None) == ["full"]

    def test_missing_ffmpeg_aborts(self, monkeypatch, tmp_path):
        from crawler import cli, sources as srcs
        monkeypatch.setattr(shutil, "which", lambda n: None)
        monkeypatch.setattr(srcs, "get", lambda t: type("A", (), {"list_items": staticmethod(lambda cfg: [])}))
        with pytest.raises(SystemExit, match="ffmpeg"):
            cli.crawl_source({"name": "t", "type": "x", "max_hours": 0.1}, str(tmp_path), None)


class TestFeedSubdir:
    """rss: mỗi feed một thư mục con raw/<source>/<feed_slug>/."""

    def test_feed_slug(self):
        from crawler.sources.rss.adapter import feed_slug
        assert feed_slug("https://vnexpress.net/rss/podcast/ban-on-khong.rss") == "vnexpress.net_rss_podcast_ban-on-khong.rss"
        assert feed_slug("https://anchor.fm/s/ccc8a36c/podcast/rss") == "anchor.fm_s_ccc8a36c_podcast_rss"
        assert feed_slug("https://x.com/") == "x.com"

    def test_cli_puts_file_in_subdir(self, monkeypatch, tmp_path):
        from crawler import cli, sources as srcs
        monkeypatch.setattr(srcs, "get", lambda t: type("A", (), {"list_items": staticmethod(
            lambda cfg: [{"url": "https://x/a.mp3", "title": "a", "subdir": "feed1"}])}))
        monkeypatch.setattr(downloader, "download", lambda url, dest, **k: open(dest, "wb").close() or True)
        cli.crawl_source({"name": "src", "type": "x", "delay_s": 0}, str(tmp_path), None)
        assert (tmp_path / "src" / "feed1" / "a.mp3").exists()
        assert (tmp_path / "src" / "feed1" / "a.json").exists()
        rec = [l for l in (tmp_path / "src" / "ledger.jsonl").read_text().splitlines() if l][0]
        import json
        assert json.loads(rec)["subdir"] == "feed1"

    def test_dir_dot_writes_into_out_root(self, monkeypatch, tmp_path):
        from crawler import cli, sources as srcs
        monkeypatch.setattr(srcs, "get", lambda t: type("A", (), {"list_items": staticmethod(
            lambda cfg: [{"url": "https://x/a.mp3", "title": "a", "subdir": "feed1"}])}))
        monkeypatch.setattr(downloader, "download", lambda url, dest, **k: open(dest, "wb").close() or True)
        cli.crawl_source({"name": "src", "type": "x", "dir": ".", "delay_s": 0}, str(tmp_path), None)
        assert (tmp_path / "feed1" / "a.mp3").exists()
        assert (tmp_path / "ledger.jsonl").exists()
        assert not (tmp_path / "src").exists()


class TestFeedByFeed:
    def test_iter_items_yields_before_next_feed_is_fetched(self, monkeypatch):
        """Tập của feed 1 phải ra trước khi feed 2 được đọc."""
        from crawler.sources.rss import adapter
        fetched = []

        class R:
            text = RSS_FIXTURE
            def raise_for_status(self): pass
        def fake_get(url, **k):
            fetched.append(url); return R()
        monkeypatch.setattr(adapter.requests, "get", fake_get)
        gen = adapter.iter_items({"name": "x", "urls": ["https://f1/rss", "https://f2/rss"]})
        first = next(gen)
        assert fetched == ["https://f1/rss"] and first["subdir"] == "f1_rss"
        rest = list(gen)
        assert fetched == ["https://f1/rss", "https://f2/rss"] and rest[-1]["subdir"] == "f2_rss"

    def test_cli_limit_on_stream(self, monkeypatch, tmp_path):
        from crawler import cli, sources as srcs
        def gen(cfg):
            for i in range(10):
                yield {"url": f"https://x/{i}.mp3", "title": str(i)}
        monkeypatch.setattr(srcs, "get", lambda t: type("A", (), {"iter_items": staticmethod(gen)}))
        monkeypatch.setattr(downloader, "download", lambda url, dest, **k: open(dest, "wb").close() or True)
        cli.crawl_source({"name": "s", "type": "x", "delay_s": 0}, str(tmp_path), 3)
        assert sorted(p.name for p in (tmp_path / "s").glob("*.mp3")) == ["0.mp3", "1.mp3", "2.mp3"]

    def test_rss_package_exports_iter_items(self):
        assert callable(getattr(sources.get("rss"), "iter_items", None))
