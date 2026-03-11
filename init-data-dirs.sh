#!/usr/bin/env bash
# Create bind-mount directories before docker compose up.
# Without this, Docker creates them as root on Linux.
set -euo pipefail

DATA_DIR="${FIELDNOTES_DATA:-$HOME/.fieldnotes/data}"

mkdir -p "$DATA_DIR/neo4j"
mkdir -p "$DATA_DIR/qdrant"
chmod 700 "$DATA_DIR" "$DATA_DIR/neo4j" "$DATA_DIR/qdrant"

echo "Data directories ready: $DATA_DIR/{neo4j,qdrant}"
