"""Example plugin: weather query via AMAP (Gaode) API.

Demonstrates a plugin that declares `__deps__` and calls an external HTTP
API. The API key is read from `.env` (see the project root).

Copy to `plugins/` to use, or point PluginManager at this directory.
"""

__deps__ = ["python-dotenv"]

import json
import logging
import os
import urllib.parse
import urllib.request

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))


def get_weather(city: str) -> str:
    """Get the current weather for a city (AMAP/Gaode).

    Args:
        city: The city name or adcode, e.g. "北京" or "110101"

    Returns:
        Formatted weather info string.
    """
    api_key = os.environ.get("AMAP_API_KEY", "")
    if not api_key:
        return "[error] AMAP_API_KEY not set in .env"

    query = {
        "city": city,
        "key": api_key,
        "extensions": "base",
        "output": "JSON",
    }
    url = "https://restapi.amap.com/v3/weather/weatherInfo?" + urllib.parse.urlencode(query)
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method="GET"), timeout=8) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        logging.error("AMAP weather request failed: %s", e)
        return f"[error] AMAP weather request failed: {e}"

    if str(payload.get("status")) != "1":
        return f"[error] AMAP returned failure: {payload.get('info')}"

    lives = payload.get("lives", [])
    if not lives:
        return "[error] no live weather data"
    live = lives[0]
    return (
        f"{live.get('province', '')}{live.get('city', '')} weather: {live.get('weather', '')}, "
        f"temp {live.get('temperature', '')}C, wind {live.get('winddirection', '')} "
        f"{live.get('windpower', '')}, humidity {live.get('humidity', '')}%"
    )


__tools__ = [get_weather]
