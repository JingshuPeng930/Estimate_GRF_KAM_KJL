#!/usr/bin/env python3
"""Unpack the generated AB03 subject-dependent KJL dataset archive."""

from __future__ import annotations

import argparse
import os
import tarfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = (
    Path(__file__).resolve().parent
    / "data_archive"
    / "kjl_subject_dependent_noncascaded_ab03_data.tar.gz"
)


def _safe_extract(tf: tarfile.TarFile, target_dir: Path) -> None:
    target_dir = target_dir.resolve()
    for member in tf.getmembers():
        member_path = (target_dir / member.name).resolve()
        if os.path.commonpath([target_dir, member_path]) != str(target_dir):
            raise ValueError(f"Unsafe path in archive: {member.name}")
    tf.extractall(target_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=DEFAULT_ARCHIVE,
        help=f"Dataset archive path. Default: {DEFAULT_ARCHIVE}",
    )
    args = parser.parse_args()

    archive = args.archive.resolve()
    if not archive.exists():
        raise FileNotFoundError(f"Missing dataset archive: {archive}")

    with tarfile.open(archive, "r:gz") as tf:
        _safe_extract(tf, REPO_ROOT)

    data_root = REPO_ROOT / "kjl_subject_dependent_noncascaded" / "data" / "kjl_ab03_dep"
    print(f"Unpacked generated data to: {data_root}")


if __name__ == "__main__":
    main()
