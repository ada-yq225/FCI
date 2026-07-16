"""Download the public Tennessee STAR files used by this case study."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import shutil
import tempfile
import urllib.request
from pathlib import Path

import pandas as pd

DATASET_DOI = "https://doi.org/10.7910/DVN/SIWH9F"
STUDENT_FILE_URL = "https://dataverse.harvard.edu/api/access/datafile/666716"
USER_GUIDE_URL = "https://dataverse.harvard.edu/api/access/datafile/666705"
EXPECTED_ROWS = 11_601
EXPECTED_COLUMNS = 379

HERE = Path(__file__).resolve().parent
RAW_DIR = HERE / "data" / "raw"
STUDENT_PATH = RAW_DIR / "STAR_Students.tab.gz"
GUIDE_PATH = RAW_DIR / "starUsersGuide.pdf"


def sha256(path: Path) -> str:
    """Return the SHA-256 digest for a local file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(force: bool = False) -> tuple[Path, Path]:
    """Download and validate the Dataverse tab export and official user guide."""

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if force or not STUDENT_PATH.exists():
        with tempfile.TemporaryDirectory(prefix="star-download-") as temp_dir:
            tab_path = Path(temp_dir) / "STAR_Students.tab"
            urllib.request.urlretrieve(STUDENT_FILE_URL, tab_path)
            frame = pd.read_csv(tab_path, sep="\t", low_memory=False)
            if frame.shape != (EXPECTED_ROWS, EXPECTED_COLUMNS):
                raise RuntimeError(
                    "Unexpected STAR student table shape: "
                    f"{frame.shape}, expected "
                    f"{(EXPECTED_ROWS, EXPECTED_COLUMNS)}."
                )
            with tab_path.open("rb") as source, STUDENT_PATH.open("wb") as raw_out:
                with gzip.GzipFile(
                    filename="",
                    mode="wb",
                    fileobj=raw_out,
                    compresslevel=9,
                    mtime=0,
                ) as compressed:
                    shutil.copyfileobj(source, compressed)

    if force or not GUIDE_PATH.exists():
        urllib.request.urlretrieve(USER_GUIDE_URL, GUIDE_PATH)

    return STUDENT_PATH, GUIDE_PATH


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    student_path, guide_path = download(force=args.force)
    print(f"Dataset DOI: {DATASET_DOI}")
    print(f"{student_path}: {sha256(student_path)}")
    print(f"{guide_path}: {sha256(guide_path)}")


if __name__ == "__main__":
    main()
