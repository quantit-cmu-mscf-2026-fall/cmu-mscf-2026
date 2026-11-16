"""Export a commit as a byte-reproducible .tar.gz with a RELEASE stamp.

    python scripts/export_release.py --out release.tar.gz [--commit REF]

The archive is `git archive <commit>` (so .gitattributes line endings apply) under the
prefix `cmu-mscf-2026-<sha12>/`, plus `RELEASE`: `{commit, tag, date, files}` with the
exact-match tag (or null), the commit date and the sha256 of every exported file. An
extracted tree has no .git; the stamp is what names its commit and lets a reader detect
that an exported file changed. Archive bytes depend only on the commit (gzip mtime 0,
member times from git). The script only reads the repository and writes `--out`: it
pushes, uploads and registers nothing, and the stamp holds no local paths.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def git(*args, repo=REPO):
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, check=True).stdout.decode(
        "utf-8"
    )


def export(out, commit="HEAD", repo=REPO):
    sha = git("rev-parse", "--verify", f"{commit}^{{commit}}", repo=repo).strip()
    try:
        tag = git("describe", "--tags", "--exact-match", sha, repo=repo).strip() or None
    except subprocess.CalledProcessError:
        tag = None
    date, epoch = git("show", "-s", "--format=%cI %ct", sha, repo=repo).split()
    prefix = f"cmu-mscf-2026-{sha[:12]}/"
    raw = subprocess.run(
        # autocrlf off: a user's line-ending setting must not change the archive bytes
        ["git", "-c", "core.autocrlf=false", "archive", "--format=tar", f"--prefix={prefix}", sha],
        cwd=repo,
        capture_output=True,
        check=True,
    ).stdout
    files = {}
    with tarfile.open(fileobj=io.BytesIO(raw)) as source:
        members = [(m, source.extractfile(m).read() if m.isfile() else None) for m in source]
    for member, data in members:
        if data is not None:
            files[member.name.removeprefix(prefix)] = hashlib.sha256(data).hexdigest()
    stamp = {"commit": sha, "tag": tag, "date": date, "files": files}
    text = (json.dumps(stamp, indent=2, sort_keys=True) + "\n").encode("utf-8")
    info = tarfile.TarInfo(prefix + "RELEASE")
    info.size, info.mtime, info.mode = len(text), int(epoch), 0o644
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as handle, gzip.GzipFile("", "wb", fileobj=handle, mtime=0) as packed:
        with tarfile.open(fileobj=packed, mode="w", format=tarfile.PAX_FORMAT) as target:
            for member, data in members:
                target.addfile(member, None if data is None else io.BytesIO(data))
            target.addfile(info, io.BytesIO(text))
    return {"archive": str(out), "commit": sha, "tag": tag, "date": date, "files": len(files)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--commit", default="HEAD")
    args = parser.parse_args(argv)
    print(json.dumps(export(args.out, args.commit), sort_keys=True))


if __name__ == "__main__":
    main()
