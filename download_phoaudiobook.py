"""Download the thivux/phoaudiobook dataset from Hugging Face.

Usage:
  python download_phoaudiobook.py                            # download everything
  python download_phoaudiobook.py --out ./data/phoaudiobook  # custom output dir
  python download_phoaudiobook.py --workers 8                # parallel downloads
"""

import argparse
import os

from huggingface_hub import snapshot_download

REPO_ID = "thivux/phoaudiobook"


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Download {REPO_ID} dataset")
    parser.add_argument(
        "--out",
        default="./data/phoaudiobook",
        help="Output directory (default: ./data/phoaudiobook)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel download workers (default: 4)",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face access token (optional, default: HF_TOKEN env var)",
    )
    parser.add_argument(
        "--allow-patterns",
        nargs="*",
        default=None,
        help='Only download files matching these glob patterns, e.g. "data/train-000*"',
    )
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"Downloading {REPO_ID} -> {args.out}")

    path = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=args.out,
        token=args.token,
        max_workers=args.workers,
        allow_patterns=args.allow_patterns,
    )

    print(f"Done. Dataset saved at: {path}")


if __name__ == "__main__":
    main()
