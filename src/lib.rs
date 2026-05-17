//! SileroVAD as a loadable plugin (Phase B / Path 2 — dual-emit).
//!
//! `SileroVADNode` is defined in `remotemedia-core` (gated by the
//! `silero-vad` feature) using the `#[node(...)]` attribute macro.
//! Adding `loadable-export` to this crate's dep-features list flips
//! the macro's conditional emission, which:
//!
//! * compiles a `SileroVADNodeLoadableFactory` unit struct implementing
//!   `FfiNodeFactory`;
//! * registers a `LoadableFactoryEntry { make: ... }` via `inventory`;
//! * keeps everything else (the in-tree `AsyncStreamingNode` impl, the
//!   `SileroVADNodeFactory` consumed by `core_provider`) intact.
//!
//! All this crate has to do is call the arg-less `plugin_export!()`
//! macro — it walks the inventory at startup and emits the abi_stable
//! root module the host loads via `dlopen`.
//!
//! ## Caveat: single-output FFI
//!
//! The in-tree `SileroVADNodeWrapper` emits TWO outputs per chunk
//! (a JSON VAD event AND a passthrough copy of the original audio).
//! The FFI ABI (`FfiNode::process` in `loadable-node-abi`) is
//! single-output, so the loadable path emits ONLY the JSON VAD event.
//! Hosts that need the audio passthrough (e.g. accumulator-based
//! pipelines) must use the in-tree factory registration instead.

// Force the linker to keep the upstream `remotemedia-core` rlib's
// object files (and therefore the
// `inventory::submit!{ LoadableFactoryEntry { ... } }` static the
// `#[node]` macro emitted for `SileroVADNode`). Without a hard
// reference into the rlib, the linker prunes it as dead code and
// the inventory comes up empty at runtime — hence
// `plugin exposes: []` from the host.
//
// Touching any pub symbol from core's silero_vad module is enough; we
// reach for the most-stable thing on the surface (the in-tree
// factory's default constructor).
#[allow(dead_code)]
fn _force_link_silero_vad_factory() {
    let _f = remotemedia_core::nodes::silero_vad::SileroVADNodeFactory::default();
}

remotemedia_plugin_sdk::plugin_export!();
