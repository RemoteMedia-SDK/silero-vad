fn main() {
    println!("cargo:rerun-if-changed=src/android_libcxx_streams.cc");

    let target = std::env::var("TARGET").unwrap_or_default();
    if target.contains("android") {
        cc::Build::new()
            .cpp(true)
            .file("src/android_libcxx_streams.cc")
            .compile("silero_android_libcxx_streams");
    } else {
        // The `silero_android_force_libcxx_streams` symbol is referenced
        // unconditionally from src/lib.rs (SileroVADNode::new), but the .cc
        // that defines it was only compiled for android — so a Linux/macOS
        // cdylib ends up with an undefined symbol and fails to dlopen. Emit
        // a tiny no-op definition of the symbol for non-android targets so
        // the loadable plugin links + loads cleanly off-device.
        let out_dir = std::env::var("OUT_DIR").unwrap();
        let shim = std::path::Path::new(&out_dir).join("silero_libcxx_streams_stub.c");
        std::fs::write(&shim, "void silero_android_force_libcxx_streams(void) {}\n")
            .expect("write silero libcxx stub");
        cc::Build::new()
            .file(&shim)
            .compile("silero_android_libcxx_streams");
    }
}
