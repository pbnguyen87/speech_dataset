"""Download the capleaf/viVoice dataset from Hugging Face.

NOTE: viVoice is a *gated* dataset. Before running:
  1. Visit https://huggingface.co/datasets/capleaf/viVoice and accept the terms.
  2. Login with `huggingface-cli login` or set the HF_TOKEN env var.

Usage:
  python download_vivoice.py                       # download everything
  python download_vivoice.py --out ./data/viVoice  # custom output dir
  python download_vivoice.py --workers 8           # parallel downloads
"""

import argparse
import os
import sys

from huggingface_hub import snapshot_download

REPO_ID = "capleaf/viVoice"


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Download {REPO_ID} dataset")
    parser.add_argument(
        "--out",
        default="./data/viVoice",
        help="Output directory (default: ./data/viVoice)",
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
        help="Hugging Face access token (default: HF_TOKEN env var / cached login)",
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

    try:
        path = snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            local_dir=args.out,
            token=args.token,
            max_workers=args.workers,
            allow_patterns=args.allow_patterns,
        )
    except Exception as e:
        msg = str(e)
        if "401" in msg or "403" in msg or "gated" in msg.lower():
            print(
                "\nERROR: viVoice is a gated dataset. Accept the terms at\n"
                f"  https://huggingface.co/datasets/{REPO_ID}\n"
                "then login (`huggingface-cli login`) or pass --token / set HF_TOKEN.",
                file=sys.stderr,
            )
            sys.exit(1)
        raise

    print(f"Done. Dataset saved at: {path}")


if __name__ == "__main__":
    main()
