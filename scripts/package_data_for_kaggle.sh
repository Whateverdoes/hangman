#!/usr/bin/env bash
# Bundle the inputs a Kaggle TRAINING run needs, so the GPU session does not
# have to regenerate labels on its (slower, fewer-core) CPUs.
#
#   bash scripts/package_data_for_kaggle.sh
#   -> dist/hangman-data.zip
#
# Upload as a PRIVATE dataset named "hangman-data". Contains:
#   train.txt   — needed for the held-out validation split
#   labels.tsv  — the leave-one-out teacher labels (~22 min of CPU work)
#
# Deliberately EXCLUDES test.txt. Training must never see it, and the leak
# guards in build_labels.py / train.py abort if it ever appears.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"

[ -f data/train.txt ]         || { echo "missing data/train.txt"; exit 1; }
[ -f experiments/labels.tsv ] || { echo "missing experiments/labels.tsv — run build_labels.py"; exit 1; }

rm -rf dist/hangman-data dist/hangman-data.zip
mkdir -p dist/hangman-data
cp data/train.txt          dist/hangman-data/train.txt
cp experiments/labels.tsv  dist/hangman-data/labels.tsv

if [ -f data/test.txt ]; then
  echo "note: data/test.txt exists and is deliberately NOT bundled"
fi

cd dist/hangman-data && zip -qr ../hangman-data.zip . && cd ../..
echo "dist/hangman-data.zip  ($(du -h dist/hangman-data.zip | cut -f1))"
echo
echo "Upload as a private dataset named 'hangman-data', then attach it"
echo "alongside 'hangman-src' in the training notebook."
