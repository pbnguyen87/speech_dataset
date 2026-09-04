"""Extract audio from downloaded parquet shards into individual .wav files.

Works with both datasets (capleaf/viVoice and thivux/phoaudiobook): it reads
every parquet file, finds the audio column (a struct with `bytes`), decodes it
and writes one .wav per row, plus a metadata.csv mapping wav filename -> text.

Requirements:
    pip install pyarrow soundfile numpy

Usage:
  # extract everything from a downloaded dataset
  python extract_audio.py --input ./data/viVoice --out ./wav/viVoice
  python extract_audio.py --input ./data/phoaudiobook --out ./wav/phoaudiobook

  # quick test: only the first 20 rows of the first shard
  python extract_audio.py --input ./data/viVoice --out ./wav_test --limit 20

  # keep the original encoding (no re-encode to wav, faster, e.g. .mp3 stays .mp3)
  python extract_audio.py --input ./data/viVoice --out ./raw/viVoice --raw
"""

import argparse
import csv
import glob
import io
import os
import sys

import pyarrow.parquet as pq

# Column names commonly used for the transcript, in priority order.
TEXT_COLUMNS = ["text", "transcription", "transcript", "sentence", "caption"]

# Magic bytes -> file extension, used in --raw mode.
MAGIC = [
    (b"RIFF", ".wav"),
    (b"fLaC", ".flac"),
    (b"OggS", ".ogg"),
    (b"ID3", ".mp3"),
    (b"\xff\xfb", ".mp3"),
    (b"\xff\xf3", ".mp3"),
    (b"\xff\xf2", ".mp3"),
]


def detect_ext(data: bytes) -> str:
    for magic, ext in MAGIC:
        if data.startswith(magic):
            return ext
    return ".bin"


def find_audio_column(schema) -> str:
    """Find the struct column that contains audio bytes."""
    import pyarrow as pa

    for field in schema:
        if pa.types.is_struct(field.type):
            names = [f.name for f in field.type]
            if "bytes" in names:
                return field.name
    raise SystemExit(
        f"ERROR: no audio column (struct with 'bytes') found. Columns: {schema.names}"
    )


def find_text_column(schema) -> str | None:
    for name in TEXT_COLUMNS:
        if name in schema.names:
            return name
    return None


def safe_name(raw: str | None, shard_idx: int, row_idx: int) -> str:
    """Build a filesystem-safe base name for one row."""
    if raw:
        base = os.path.splitext(os.path.basename(raw))[0]
        base = "".join(c if c.isalnum() or c in "-_." else "_" for c in base)
        if base:
            return base
    return f"shard{shard_idx:05d}_row{row_idx:06d}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract wav files from parquet shards")
    parser.add_argument("--input", required=True, help="Directory containing the downloaded dataset (searched recursively for *.parquet)")
    parser.add_argument("--out", required=True, help="Output directory for audio files + metadata.csv")
    parser.add_argument("--limit", type=int, default=None, help="Stop after N rows (for testing)")
    parser.add_argument("--raw", action="store_true", help="Dump original encoded bytes as-is instead of re-encoding to wav (faster)")
    args = parser.parse_args()

    parquet_files = sorted(glob.glob(os.path.join(args.input, "**", "*.parquet"), recursive=True))
    if not parquet_files:
        sys.exit(f"ERROR: no .parquet files found under {args.input}")
    print(f"Found {len(parquet_files)} parquet file(s) under {args.input}")

    if not args.raw:
        try:
            import soundfile as sf  # noqa: F401
        except ImportError:
            sys.exit("ERROR: soundfile is required for wav export. Run: pip install soundfile")

    os.makedirs(args.out, exist_ok=True)
    meta_path = os.path.join(args.out, "metadata.csv")
    meta_file = open(meta_path, "w", newline="", encoding="utf-8")
    meta = csv.writer(meta_file)
    meta.writerow(["file_name", "text"])

    total = 0
    errors = 0
    done = False

    for shard_idx, pf_path in enumerate(parquet_files):
        if done:
            break
        pf = pq.ParquetFile(pf_path)
        audio_col = find_audio_column(pf.schema_arrow)
        text_col = find_text_column(pf.schema_arrow)
        cols = [audio_col] + ([text_col] if text_col else [])

        for batch in pf.iter_batches(batch_size=64, columns=cols):
            rows = batch.to_pylist()
            for row_idx, row in enumerate(rows):
                if args.limit is not None and total >= args.limit:
                    done = True
                    break

                audio = row[audio_col] or {}
                data = audio.get("bytes")
                if not data:
                    errors += 1
                    continue

                base = safe_name(audio.get("path"), shard_idx, total)
                text = (row.get(text_col) or "") if text_col else ""

                try:
                    if args.raw:
                        out_name = base + detect_ext(data)
                        with open(os.path.join(args.out, out_name), "wb") as f:
                            f.write(data)
                    else:
                        import soundfile as sf

                        samples, sr = sf.read(io.BytesIO(data))
                        out_name = base + ".wav"
                        sf.write(os.path.join(args.out, out_name), samples, sr)
                except Exception as e:
                    errors += 1
                    print(f"  skip row {total} ({base}): {e}", file=sys.stderr)
                    continue

                meta.writerow([out_name, text])
                total += 1
                if total % 500 == 0:
                    print(f"  {total} files extracted...")
            if done:
                break
        print(f"[{shard_idx + 1}/{len(parquet_files)}] {os.path.basename(pf_path)} done (total so far: {total})")

    meta_file.close()
    print(f"\nFinished: {total} audio files written to {args.out}")
    print(f"Metadata: {meta_path}")
    if errors:
        print(f"WARNING: {errors} row(s) skipped due to errors")


if __name__ == "__main__":
    main()
