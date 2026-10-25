"""Download the Yambda track metadata (artist and album of every track) with size and SHA256 checks.

    python -m strongfeat.download_meta [--root PROJECT_ROOT]

Files go to data/data_yambda/raw/meta/. A file is first written as .part and renamed only after the
check passes; an existing file is accepted only if its hash matches.
"""
from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

from .common import EXPERIMENT, file_sha256

BASE = "https://huggingface.co/datasets/yandex/yambda/resolve/main/"
FILES = {
    "artist_item_mapping.parquet": (38_274_621, "cf1110aa2413ed70008ec9232d395660b2181ffda731002fb74da54e87b6f25d"),
    "album_item_mapping.parquet": (54_528_894, "559b6576063d96280414574b32292eee7b47ee0b9f2952cb3ae670a5d19125e4"),
}


def download(root: Path) -> None:
    target = root / "data" / "data_yambda" / "raw" / "meta"
    target.mkdir(parents=True, exist_ok=True)
    for name, (size, sha) in FILES.items():
        path = target / name
        if path.exists():
            if path.stat().st_size == size and file_sha256(path) == sha:
                print(f"{name}: already present and verified")
                continue
            raise ValueError(f"{path} exists but does not match the expected size and SHA256")
        part = path.with_suffix(path.suffix + ".part")
        digest = hashlib.sha256()
        with urllib.request.urlopen(BASE + name) as response, open(part, "wb") as out:
            while chunk := response.read(1 << 22):
                digest.update(chunk)
                out.write(chunk)
        if part.stat().st_size != size or digest.hexdigest() != sha:
            raise ValueError(f"{name}: downloaded file does not match the expected size and SHA256 (kept as {part})")
        part.replace(path)
        print(f"{name}: downloaded and verified")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(EXPERIMENT.parents[1]))
    download(Path(parser.parse_args().root))


if __name__ == "__main__":
    main()
