"""1001Tracklists integration for getting timestamped tracklists of DJ sets."""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup, Tag

logger = logging.getLogger(__name__)

BASE_URL = "https://www.1001tracklists.com"
SEARCH_URL = f"{BASE_URL}/search/result.php"

# Rate limiting: minimum seconds between requests
REQUEST_DELAY = 1.5

# Common timestamp formats found on 1001Tracklists
_TS_PATTERNS = [
    # "1:23:45" -> h:mm:ss
    re.compile(r"^(\d+):(\d{2}):(\d{2})$"),
    # "23:45" -> mm:ss
    re.compile(r"^(\d{1,2}):(\d{2})$"),
]


@dataclass
class TracklistEntry:
    """A single track in a tracklist with timing info."""

    position: int
    title: str
    artist: str
    timestamp_sec: float | None = None  # seconds into the set
    musicbrainz_id: str = ""


def _parse_timestamp(ts_text: str) -> float | None:
    """Parse a timestamp string into seconds.

    Handles "H:MM:SS", "MM:SS", and bare second values.
    """
    text = ts_text.strip()
    if not text:
        return None

    # Try h:mm:ss
    m = _TS_PATTERNS[0].match(text)
    if m:
        h, mm, ss = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return h * 3600.0 + mm * 60.0 + ss

    # Try mm:ss
    m = _TS_PATTERNS[1].match(text)
    if m:
        mm, ss = int(m.group(1)), int(m.group(2))
        return mm * 60.0 + ss

    # Try bare number (seconds)
    try:
        return float(text)
    except ValueError:
        return None


class TracklistScraper:
    """Scrape timestamped tracklists from 1001Tracklists."""

    def __init__(
        self,
        request_delay: float = REQUEST_DELAY,
        timeout: float = 15.0,
    ) -> None:
        self._delay = request_delay
        self._last_request: float = 0.0
        self._timeout = timeout
        self._headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }

    async def _rate_limit(self) -> None:
        """Enforce minimum delay between requests."""
        now = asyncio.get_event_loop().time()
        elapsed = now - self._last_request
        if elapsed < self._delay:
            await asyncio.sleep(self._delay - elapsed)
        self._last_request = asyncio.get_event_loop().time()

    async def _fetch(self, url: str, params: dict | None = None) -> str:
        """Fetch a page with rate limiting."""
        await self._rate_limit()
        async with httpx.AsyncClient(
            headers=self._headers,
            timeout=self._timeout,
            follow_redirects=True,
        ) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            return resp.text

    async def search(self, dj_name: str, set_title: str = "") -> list[dict]:
        """Search 1001Tracklists for matching sets.

        Returns a list of dicts with keys: title, url, dj_name.
        """
        query = f"{dj_name} {set_title}".strip()
        html = await self._fetch(SEARCH_URL, params={"search": query})
        soup = BeautifulSoup(html, "html.parser")

        results: list[dict] = []
        # Search results are in divs/links pointing to /tracklist/ pages
        for link in soup.find_all("a", href=True):
            href = link["href"]
            if "/tracklist/" not in href:
                continue
            title_text = link.get_text(strip=True)
            if not title_text:
                continue
            full_url = urljoin(BASE_URL, href) if href.startswith("/") else href
            results.append(
                {
                    "title": title_text,
                    "url": full_url,
                    "dj_name": dj_name,
                }
            )

        # Deduplicate by URL
        seen: set[str] = set()
        deduped: list[dict] = []
        for r in results:
            if r["url"] not in seen:
                seen.add(r["url"])
                deduped.append(r)
                logger.debug("Found tracklist: %s -> %s", r["title"], r["url"])

        return deduped

    async def get_tracklist(self, tracklist_url: str) -> list[TracklistEntry]:
        """Fetch and parse a tracklist page.

        Extracts track entries with position, artist, title, and timestamp
        from a 1001Tracklists tracklist page.
        """
        html = await self._fetch(tracklist_url)
        soup = BeautifulSoup(html, "html.parser")
        return self._parse_tracklist(soup)

    def _parse_tracklist(self, soup: BeautifulSoup) -> list[TracklistEntry]:
        """Parse track entries from a 1001Tracklists page.

        The site uses various markup patterns. We look for:
        1. Track items with class containing "tlpItem" or in ordered track containers
        2. Timestamp elements (cue/time spans)
        3. Artist/title from meta spans or link text
        """
        entries: list[TracklistEntry] = []

        # Primary strategy: look for track item containers
        track_items = soup.select(".tlpItem, .tlpTog, [id^='tlp']")
        if not track_items:
            # Fallback: look for numbered track rows in any table or list
            track_items = soup.select("tr.or, tr.ev, .tl_track, .trackRow")

        for idx, item in enumerate(track_items):
            if not isinstance(item, Tag):
                continue

            entry = self._parse_track_item(item, idx + 1)
            if entry is not None:
                entries.append(entry)

        # If we found no tracks via CSS selectors, try a broader heuristic
        if not entries:
            entries = self._parse_tracklist_fallback(soup)

        return entries

    def _parse_track_item(self, item: Tag, position: int) -> TracklistEntry | None:
        """Extract a single track entry from a DOM element."""
        # Try to find artist and title
        artist = ""
        title = ""

        # Look for meta elements with artist/title info
        meta_artist = item.select_one(".trackValue, .tp_artist, [itemprop='byArtist']")
        meta_title = item.select_one(".trackName, .tp_title, [itemprop='name']")

        if meta_artist:
            artist = meta_artist.get_text(strip=True)
        if meta_title:
            title = meta_title.get_text(strip=True)

        # Fallback: parse "Artist - Title" from the full text
        if not artist and not title:
            full_text = item.get_text(" ", strip=True)
            # Strip position numbers and timestamps from the front
            full_text = re.sub(r"^\d+[\.\)]\s*", "", full_text)
            full_text = re.sub(r"^\d{1,2}:\d{2}(:\d{2})?\s*", "", full_text)
            if " - " in full_text:
                parts = full_text.split(" - ", 1)
                artist = parts[0].strip()
                title = parts[1].strip()
            elif full_text:
                title = full_text

        # Must have at least a title
        if not title:
            return None

        # Extract timestamp
        timestamp = self._extract_timestamp(item)

        return TracklistEntry(
            position=position,
            title=title,
            artist=artist,
            timestamp_sec=timestamp,
        )

    def _extract_timestamp(self, item: Tag) -> float | None:
        """Extract a timestamp from a track item element."""
        # Look for explicit cue/time elements
        time_el = item.select_one(
            ".cueValueField, .tlpCue, .cue, .timestamp, [class*='time'], [class*='cue']"
        )
        if time_el:
            ts = _parse_timestamp(time_el.get_text(strip=True))
            if ts is not None:
                return ts

        # Look for data attributes with timing
        for attr in ("data-cue", "data-time", "data-start"):
            val = item.get(attr)
            if val:
                ts = _parse_timestamp(str(val))
                if ts is not None:
                    return ts

        # Scan all text nodes for timestamp patterns
        text = item.get_text(" ", strip=True)
        for pattern in _TS_PATTERNS:
            m = pattern.search(text)
            if m:
                ts = _parse_timestamp(m.group(0))
                if ts is not None:
                    return ts

        return None

    def _parse_tracklist_fallback(self, soup: BeautifulSoup) -> list[TracklistEntry]:
        """Fallback parser: look for numbered lines with 'Artist - Title' patterns."""
        entries: list[TracklistEntry] = []
        # Look in the full page text for numbered track lines
        text = soup.get_text("\n")
        pattern = re.compile(
            r"(\d+)[\.\)]\s+"  # position number
            r"(?:\[?(\d{1,2}:\d{2}(?::\d{2})?)\]?\s+)?"  # optional timestamp
            r"(.+?)\s+-\s+(.+)"  # artist - title
        )
        for m in pattern.finditer(text):
            pos = int(m.group(1))
            ts = _parse_timestamp(m.group(2)) if m.group(2) else None
            artist = m.group(3).strip()
            title = m.group(4).strip()
            entries.append(
                TracklistEntry(
                    position=pos,
                    title=title,
                    artist=artist,
                    timestamp_sec=ts,
                )
            )
        return entries
