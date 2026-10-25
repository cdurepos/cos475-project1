#!/usr/bin/env bash
set -euo pipefail

BASE_URL="https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2026/2026-04-07"
DATA_DIR="$(cd "$(dirname "$0")/.." && pwd)/data"

mkdir -p "$DATA_DIR"

for file in repairs.csv repairs_text.csv; do
    curl -fsSL "$BASE_URL/$file" -o "$DATA_DIR/$file"
    echo "Downloaded $file"
done
