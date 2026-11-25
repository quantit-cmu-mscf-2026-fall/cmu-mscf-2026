"""HTTP pieces shared by the scripts that call the Jev (TypeSafe) system-one endpoint."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping

JEV_URL = "https://api.typesafe.ai/v1/systemone"
RETRYABLE = (429, 500, 502, 503, 504, 529)  # 529: overloaded


def http_post(url: str, headers: dict, body: dict, timeout: float = 120.0) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers=headers, method="POST"
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read().decode())


def post_with_retry(
    post: Callable[[str, dict, dict], dict],
    headers: dict,
    body: dict,
    sleep: Callable[[float], None] = time.sleep,
    retries: int = 3,
    url: str = JEV_URL,
) -> dict:
    """POST, retrying retryable HTTP errors with exponential backoff; others raise."""
    for attempt in range(retries + 1):
        try:
            return post(url, headers, body)
        except urllib.error.HTTPError as exc:
            if exc.code not in RETRYABLE or attempt == retries:
                raise
            sleep(2.0**attempt)
    raise AssertionError("unreachable")


def jev_key(environ: Mapping[str, str] | None = None) -> str:
    """TYPESAFE_API_KEY from the environment, else the Windows user environment."""
    env = os.environ if environ is None else environ
    if env.get("TYPESAFE_API_KEY"):
        return env["TYPESAFE_API_KEY"]
    try:  # pragma: no cover - Windows only
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as handle:
            return winreg.QueryValueEx(handle, "TYPESAFE_API_KEY")[0]
    except (ImportError, OSError):
        raise SystemExit("TYPESAFE_API_KEY is not set") from None
