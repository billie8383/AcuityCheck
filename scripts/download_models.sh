#!/usr/bin/env bash
set -euo pipefail

# Download YuNet face detector model (ONNX) into models/onnx by default.

ROOT_DIR="$(cd "$(dirname "$0")"/.. && pwd)"
DEST_DIR="${1:-$ROOT_DIR/models/onnx}"
mkdir -p "$DEST_DIR"

# OpenCV Zoo canonical raw URL.
echo "Downloading YuNet face detector (ONNX)…"
curl -fsSL -o "$DEST_DIR/face_detection_yunet_2023mar.onnx" \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
echo "✓ YuNet saved to $DEST_DIR/face_detection_yunet_2023mar.onnx"

echo "Done. Files placed in: $DEST_DIR"

