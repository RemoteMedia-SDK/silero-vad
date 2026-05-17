#!/usr/bin/env bash
# Build the silero-vad-loadable cdylib + host binary, then run the
# host against the freshly-built .so. Optional env:
#
#   SILERO_DEMO_WAV  path to a 16 kHz mono WAV (optional;
#                    falls back to a 220 Hz tone)
#
# First run downloads the Silero VAD ONNX model (~1.7 MB) into
# the current working directory as `silero_vad.onnx`.
set -euo pipefail
cd "$(dirname "$0")"

echo "[run] building plugin (cdylib)..."
cargo build

echo "[run] building host..."
( cd host && cargo build )

PLUGIN_BASENAME="libsilero_vad_loadable_plugin.so"
case "$(uname)" in
  Darwin)  PLUGIN_BASENAME="libsilero_vad_loadable_plugin.dylib" ;;
  MINGW*|MSYS*|CYGWIN*) PLUGIN_BASENAME="silero_vad_loadable_plugin.dll" ;;
esac
PLUGIN="$(pwd)/target/debug/${PLUGIN_BASENAME}"

echo "[run] plugin: ${PLUGIN}"
ls -lh "${PLUGIN}"

# Pre-download the silero VAD ONNX model into the working dir so the
# plugin doesn't try to fetch it from the polling thread the FFI
# future runs on (no Tokio reactor lives there, so reqwest panics).
# `silero_vad.onnx` is read from the cwd by `SileroVADNode`.
if [ ! -f "silero_vad.onnx" ]; then
  echo "[run] downloading silero_vad.onnx (~2 MB)..."
  curl -sL -o silero_vad.onnx \
    "https://huggingface.co/onnx-community/silero-vad/resolve/main/onnx/model.onnx"
fi
echo "[run] model: $(pwd)/silero_vad.onnx ($(du -h silero_vad.onnx | cut -f1))"

SILERO_DEMO_WAV="${SILERO_DEMO_WAV:-}" \
  ./host/target/debug/silero-vad-loadable-host "${PLUGIN}"
