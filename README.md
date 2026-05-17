# silero-vad — Silero VAD as a RemoteMedia SDK loadable plugin

Single-file Rust cdylib that registers `SileroVADNode` into the
[RemoteMedia SDK](https://github.com/matbeedotcom/remotemedia-sdk)
streaming pipeline registry via the
[`#[node(loadable_export)]`](https://github.com/matbeedotcom/remotemedia-sdk/blob/main/docs/CUSTOM_NODE_REGISTRATION.md)
dual-emit path (Path 2 / 4).

## Use from a manifest

```json
{
  "version": "v1",
  "plugins": ["silero-vad@v0.1.0"],
  "nodes": [
    {
      "id": "vad",
      "node_type": "SileroVADNode",
      "params": { "model_path": "./silero_vad.onnx" }
    }
  ]
}
```

The SDK resolver expands `silero-vad@v0.1.0` to
`github.com/RemoteMedia-SDK/silero-vad`, fetches `plugin.toml`, then
falls through to `release-manifest.json` for the platform-specific
prebuilt `.so` / `.dylib` / `.dll` asset.

> **Status:** plugin.toml + source published. **Prebuilt release
> binaries are not yet uploaded** — the matrix-build CI workflow is
> pending. Until then, consumers should either build the cdylib
> themselves (see below) or use a local-path plugin entry.

## Build the cdylib locally

```bash
git clone https://github.com/RemoteMedia-SDK/silero-vad
cd silero-vad
cargo build --release
# → target/release/libsilero_vad_loadable_plugin.so
```

Then reference it from your manifest:

```json
{ "plugins": ["./path/to/libsilero_vad_loadable_plugin.so"] }
```

## What it exports

| Node type      | Input                  | Output           |
|----------------|------------------------|------------------|
| `SileroVADNode` | Audio (any sample rate, mono) | JSON VAD events  |

**Caveat — single-output FFI:** the in-tree `SileroVADNodeWrapper`
emits TWO outputs per chunk (a JSON VAD event AND a passthrough copy
of the original audio). The FFI ABI (`FfiNode::process` in
`loadable-node-abi`) is single-output, so the loadable path emits ONLY
the JSON VAD event. Hosts that need the audio passthrough must use the
in-tree factory registration in the monorepo instead.

## What's in the repo

```
silero-vad/
├── plugin.toml                ← metadata (resolver fetches this first)
├── Cargo.toml                 ← git-deps the SDK at a pinned rev
├── src/lib.rs                 ← arg-less `plugin_export!()` macro call
├── silero_vad.onnx            ← bundled model (handy for smoke tests)
├── run.sh                     ← local smoke-test driver
└── README.md
```

## License

See `LICENSE.md`. This plugin reuses RemoteMedia SDK source and is
governed by the same RemoteMedia SDK Community License 1.0.
