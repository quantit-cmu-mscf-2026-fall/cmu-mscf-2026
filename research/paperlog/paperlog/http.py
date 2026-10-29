import time

import requests

USER_AGENT = "paperlog/0.1 (academic literature tracker)"


def get(url, params=None, headers=None, retries=5, timeout=40):
    h = {"User-Agent": USER_AGENT, **(headers or {})}
    delay = 3.0
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=h, timeout=timeout)
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(delay)
            delay *= 2
            continue
        if r.status_code in (429, 500, 502, 503, 504) and attempt < retries - 1:
            wait = float(r.headers.get("Retry-After", delay))
            time.sleep(wait)
            delay *= 2
            continue
        r.raise_for_status()
        return r
    raise RuntimeError(f"GET failed after {retries} attempts: {url}")
