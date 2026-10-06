"""Where config and data live. Override with environment variables so the
package works from anywhere inside a larger repo."""

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent


def config_path() -> str:
    if os.environ.get("PAPERLOG_CONFIG"):
        return os.environ["PAPERLOG_CONFIG"]
    if Path("config.yaml").exists() and Path("paperlog").is_dir():
        return "config.yaml"  # running from the paperlog folder itself
    return str(PACKAGE_ROOT / "config.yaml")


def db_path() -> str:
    return os.environ.get("PAPERLOG_DB", "papers.db")
