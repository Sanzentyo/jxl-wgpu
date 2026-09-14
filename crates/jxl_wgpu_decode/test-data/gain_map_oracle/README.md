# Native JPEG XL gain-map references

This offline helper links pristine reference sources. Neither CPU image codec is a production
dependency or fallback. `CMakeLists.txt` checks these exact source revisions:

- libjxl `a7a9c787341cf703dede03c2009fa460cae5e5df` (0.12.0), including
  `lib/extras/gain_map.cc` for `JxlGainMapReadBundle` / `JxlGainMapWriteBundle`.
- libultrahdr `6929c2b087e74120e6de52f361e77b06f07b1441` (2.0.2), for
  `uhdr_gainmap_metadata_frac`, `bt709ToBt2100` and `applyGain`.

Provide CMake, a C++17 compiler and system Highway, Brotli, LittleCMS2 and JPEG development
libraries. With those checkouts in `reference/libjxl` and `reference/libultrahdr`:

```sh
cmake -S crates/jxl_wgpu_decode/test-data/gain_map_oracle -B target/gain-map-oracle \
  -DJXL_SOURCE="$PWD/reference/libjxl" -DUHDR_SOURCE="$PWD/reference/libultrahdr" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build target/gain-map-oracle --target gain_map_oracle --parallel 4
target/gain-map-oracle/gain_map_oracle generate target/gain-map-generated
JXL_REQUIRE_NATIVE_ORACLES=1 \
JXL_GAIN_MAP_ORACLE="$PWD/target/gain-map-oracle/gain_map_oracle" \
  cargo test --locked -p jxl_wgpu_decode --test gain_map -- --test-threads=1 --nocapture
```

Generate outside the source tree under test, compare all output bytes, then import deliberately.
Do not modify sources/fixtures while any compiler, generator or test run is using them.
`JXL_REQUIRE_NATIVE_ORACLES=1` makes an absent native executable an error; ordinary runs without
the executable report that only the live native roundtrip test was skipped. Stored native image
references remain mandatory. GPU tests require an actual adapter.

`generate` creates 64 streams and four reference planes per stream, plus `cases.txt` (321 files):

| Suffix | Meaning |
|---|---|
| `.jxl` | Native-encoded baseline and auxiliary streams, native `jhgm` bundle, ordinary container |
| `.base.f32` | libjxl baseline, unassociated RGBA, un-oriented, linear BT.709 |
| `.working.f32` | Baseline after libultrahdr's explicit application-primary conversion |
| `.gain.f32` | libjxl auxiliary original Gray/RGB values, expanded to RGBA with opaque alpha |
| `.expected.f32` | libultrahdr gain application to working pixels; unchanged primary alpha |

F32 planes are interleaved little-endian RGBA without a file header. Baselines are 17×9, maps have
the dimensions recorded in `cases.txt`. Manifest columns are name, baseline mode, auxiliary mode,
gray map, wide application color, map width, map height and primary EXIF orientation. Modes 0–3
are original Modular, original VarDCT, Modular XYB and VarDCT XYB. Mode 0 is lossless; others use
distance 0.5 and effort 3. Restoration/noise/patches/splines/progressive DC are disabled in this
focused corpus. The baseline has independent 8-bit alpha and intensity target 203 nits.

The native helper independently implements edge-aligned bilinear map interpolation, then calls
the unmodified libultrahdr gain primitive. A separate Rust F64 oracle checks interpolation, exact
fraction math and chromaticity-derived primary conversion. Native working pixels isolate the
reference library's rounded primary coefficients from the gain-formula tolerance. Native baseline
decode keeps source primaries to preserve below-black behavior before explicit linear conversion.
See [the full contract and limits](../../../../docs/GAIN_MAP.md).

`iso INPUT OUTPUT` uses native ISO metadata decode/encode. `bundle INPUT OUTPUT` uses the native
JPEG XL bundle reader/writer with exact consumed/written sizes. The Rust interop test invokes
these on Rust-serialized data and checks preserved bytes and fractions in both directions.
