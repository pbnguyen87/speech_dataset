"""Test hàm thuần — không cần mạng."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from crawler.downloader import ext_from_url, safe_filename
from crawler.sources.html_listing import DEFAULT_PATTERN, audio_links, extract_links
from crawler.sources.rss import parse_feed
from crawler.state import Ledger

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


class TestHtmlListing:
    HTML = '''<a href="/audio/ep1.mp3">1</a> <a HREF="https://c.dn/ep2.MP3?t=9">2</a>
              <a href="/page/about.html">x</a>'''

    def test_extract_and_filter(self):
        links = extract_links(self.HTML, "https://site.vn/list")
        assert "https://site.vn/audio/ep1.mp3" in links
        auds = audio_links(links, DEFAULT_PATTERN)
        assert len(auds) == 2
        assert all(".mp3" in u.lower() for u in auds)
