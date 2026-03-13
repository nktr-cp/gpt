"""Download the small names dataset used for early GPT experiments."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen

DATASET_URL = "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt"
DATASET_SHA256 = "0a30b5557f192f32ab962680889aac5f6fda0f4cecf40a6d0b5694f58ea8cc4d"
DATASET_FILENAME = "names.txt"


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def raw_data_dir() -> Path:
    return project_root() / "data" / "raw"


def dataset_path(filename: str = DATASET_FILENAME) -> Path:
    return raw_data_dir() / filename


def sha256_digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download_names_dataset(destination: Path | None = None) -> Path:
    target_path = destination or dataset_path()
    target_path.parent.mkdir(parents=True, exist_ok=True)

    with urlopen(DATASET_URL) as response:
        payload = response.read()

    digest = sha256_digest(payload)
    if digest != DATASET_SHA256:
        msg = f"dataset checksum mismatch: expected {DATASET_SHA256}, got {digest}"
        raise ValueError(msg)

    target_path.write_bytes(payload)
    return target_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m gpt.data.download",
        description="Download the small names dataset used for initial GPT experiments.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=dataset_path(),
        help="Where to store the downloaded dataset.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    path = download_names_dataset(destination=args.output)
    print(path)


if __name__ == "__main__":
    main()
