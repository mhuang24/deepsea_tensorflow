#!/usr/bin/env bash
set -euo pipefail

RAW_DIR="${1:-data/raw}"
URL="https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz"

mkdir -p "$RAW_DIR"

if [[ -f "$RAW_DIR/hg38.fa" ]]; then
  echo "hg38.fa already exists at $RAW_DIR/hg38.fa"
  exit 0
fi

echo "Downloading hg38..."
curl -L "$URL" -o "$RAW_DIR/hg38.fa.gz"

echo "Decompressing hg38..."
gunzip "$RAW_DIR/hg38.fa.gz"

echo "Download complete: $RAW_DIR/hg38.fa"
echo "pyfaidx will generate the .fai index automatically when the FASTA is opened."
