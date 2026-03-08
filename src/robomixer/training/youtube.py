"""YouTube DJ set scraping via yt-dlp."""

from __future__ import annotations

from pathlib import Path

from robomixer.config import settings
from robomixer.models.training import DJSet


# Well-known DJ set channels
DJ_SET_CHANNELS = [
    "Boiler Room",
    "Cercle",
    "Mixmag",
    "HATE",
    "Anjunadeep",
    "Drumcode",
    "Resident Advisor",
]


class YouTubeScraper:
    """Downloads and catalogs DJ sets from YouTube."""

    def __init__(self, download_dir: Path | None = None) -> None:
        self.download_dir = download_dir or settings.youtube_download_dir

    def download_set(self, url: str) -> DJSet:
        """Download audio from a YouTube DJ set URL.

        Returns a DJSet record with audio_path populated.
        """
        import yt_dlp

        output_template = str(self.download_dir / "%(id)s.%(ext)s")
        ydl_opts = {
            "format": "bestaudio/best",
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "320",
                }
            ],
            "outtmpl": output_template,
            "quiet": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)

        audio_path = str(self.download_dir / f"{info['id']}.mp3")

        return DJSet(
            source_url=url,
            dj_name=info.get("uploader", ""),
            title=info.get("title", ""),
            duration_sec=info.get("duration", 0),
            audio_path=audio_path,
        )

    def search_sets(self, query: str, max_results: int = 20) -> list[dict]:
        """Search YouTube for DJ sets matching a query."""
        import yt_dlp

        ydl_opts = {
            "quiet": True,
            "extract_flat": True,
            "default_search": "ytsearch",
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            results = ydl.extract_info(f"ytsearch{max_results}:{query}", download=False)

        return [
            {
                "url": f"https://youtube.com/watch?v={entry['id']}",
                "title": entry.get("title", ""),
                "duration": entry.get("duration"),
                "uploader": entry.get("uploader", ""),
            }
            for entry in results.get("entries", [])
        ]
