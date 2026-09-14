# Native metadata interoperability

`main.cpp` extracts raw, decompressed Exif/XMP/JUMBF box payloads using libjxl's public decoder
API. It does not decode image pixels, rewrite Exif orientation, strip the four-byte TIFF offset,
or interpret the opaque payloads. It limits collected native output to 1 MiB per box and requires
successful decoder completion. This is development-only code.

The Rust test builds it in a unique temporary directory with `clang++ -std=c++17 -O2 -Wall
-Wextra -Werror` and `pkg-config --cflags --libs libjxl`. Input reuses the unchanged checked-in
`basic.jxl.hex` codestream with three deterministic opaque metadata documents. Each plain and
Brotli-compressed payload must extract byte-for-byte. Temporary files are removed after the test.
JUMBF is treated as opaque bytes; this is not a JUMBF document validator.

The separate Google `brotli` CLI test independently compresses and decompresses deterministic
inputs for all quality/window combinations. It does not require checked-in generated files.

```sh
JXL_REQUIRE_NATIVE_ORACLES=1 cargo test --locked -p jxl_gpu_bitstream --lib metadata:: -- --test-threads=1 --nocapture
cargo test --locked -p jxl_wgpu_decode --test container_metadata -- --test-threads=1 --nocapture
```

Native tools are optional for ordinary unit-test users; `JXL_REQUIRE_NATIVE_ORACLES=1` makes
missing tools fail the interoperability gate. The executed checkpoint used Google Brotli 1.2.0,
libjxl/djxl 0.12.0, Apple clang 17.0.0 and Rust 1.98.1. The source algorithm and fixtures are in
`src/metadata/tests/native.rs`; the GPU precedence tests are in the decoder's normal
`tests/container_metadata` module tree.
