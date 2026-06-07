fn main() {
    println!("cargo:rerun-if-changed=src/android_libcxx_streams.cc");

    let target = std::env::var("TARGET").unwrap_or_default();
    if target.contains("android") {
        cc::Build::new()
            .cpp(true)
            .file("src/android_libcxx_streams.cc")
            .compile("silero_android_libcxx_streams");
    }
}
