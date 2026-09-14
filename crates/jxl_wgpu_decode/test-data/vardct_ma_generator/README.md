# VarDCT previous-channel MA conformance

This offline generator links libjxl 0.12.0 at
[`a7a9c787341cf703dede03c2009fa460cae5e5df`](https://github.com/libjxl/libjxl/tree/a7a9c787341cf703dede03c2009fa460cae5e5df).
It uses the unmodified native encoder and `PrecomputeReferences`; it is never a production
dependency or a GPU codec fallback. CMake verifies the source commit.

The 74 generated files in [`../vardct_ma`](../vardct_ma) comprise:

- 72 raw codestreams in hexadecimal: 8×8, odd 129×73 and sectioned 272×32 images, each using
  properties 16–27 with a zero threshold. The greater leaf uses Gradient or Weighted; the other
  uses Zero with offset −3. The native encoder performs both entropy coding and image coding.
- `manifest.txt`, identifying each image's extent, property and predictor family.
- `references.json`, containing native signed channel samples and four properties for each of
  four preceding-reference ranks at every sample. Its 64 LF cases combine 4×4, 2×4, 4×2 and 2×2
  component dimensions with the zero Modular shifts used by VarDCT LF. Six HF cases exercise
  shifted correlation maps, matching strategy/sharpness dimensions, missing references and
  capacity-strided metadata. Signed endpoints, negative values, row edges and zero are included.

Native references require matching width, height, horizontal shift and vertical shift, scanning
backwards over eligible channels. The properties are absolute sample, signed sample, absolute
clamped-gradient residual and signed residual; missing references are zero. LF component sampling
changes dimensions without changing Modular shifts. In HF metadata, correlation shifts are `(3, 3)`
while strategy/quantizer and sharpness shifts are `(0, 0)`.

With CMake, a C++17 compiler, and system Highway, Brotli and LCMS2 available, set `JXL_SOURCE` to a
clean checkout of that exact commit with its submodules, then run from the repository root:

```sh
cmake -S crates/jxl_wgpu_decode/test-data/vardct_ma_generator \
  -B /tmp/jxl-vardct-ma-build -DJXL_SOURCE="$JXL_SOURCE" -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/jxl-vardct-ma-build --target generate_vardct_ma -j 4
/tmp/jxl-vardct-ma-build/generate_vardct_ma /tmp/jxl-vardct-ma-output
diff -r crates/jxl_wgpu_decode/test-data/vardct_ma /tmp/jxl-vardct-ma-output
cargo test -p jxl_wgpu_decode --test vardct_ma --locked -- --test-threads=1 --nocapture
```

The output directory must not exist. Regeneration compares every file byte for byte. Tests load
the checked corpus directly and do not invoke the generator.

The GPU probe compares all 41,616 native property values exactly. Public decoding validates every
fixture's retained MA split and Weighted requirement, checks whole input against seven-byte
fragments and a 40-byte GPU input cap, proves continuation submissions occur for larger images,
and requires byte-identical F32 output when only the MA tree changes. Completion releases both
GPU and incremental-input reservations. Independent Rust `jxl` and optional live libjxl references
check every image; a missing adapter or native oracle is not evidence of conformance.

On Apple M5, the observed F32 maxima were `0.000028353184` against Rust `jxl` and `0.00029148534`
against libjxl's native inverse-color/transfer implementation. The respective regression bounds
are `1e-4` and `1/1024`, matching existing F32 comparisons. Integer MA references and whole/bounded
GPU output have no tolerance. Forced previous-channel trees with independently local packet
descriptors, transformed side images and broader frame combinations remain conformance work.
