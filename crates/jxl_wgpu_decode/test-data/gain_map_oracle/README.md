# Native JPEG XL gain-map references

This offline helper links pristine reference sources. Neither CPU image codec is a production
dependency or fallback. `CMakeLists.txt` checks these exact source revisions:

- libjxl `a7a9c787341cf703dede03c2009fa460cae5e5df` (0.12.0), including
  `lib/extras/gain_map.cc` for `JxlGainMapReadBundle` / `JxlGainMapWriteBundle`.
- libultrahdr `6929c2b087e74120e6de52f361e77b06f07b1441` (2.0.2), for
  `bt709ToBt2100` and `applyGain` only. Its obsolete draft ISO fraction syntax is not used.
- libavif `b994fe4601c62d6f98dbff295bd5c251940789b0` (1.4.2), for the ISO 21496-1
  reader/writer, exact rational metadata validation and display-headroom weight selection.
- Little CMS 2.19, checked at runtime by the ICC command, for native target-profile evaluation.

Provide CMake 3.22+, C11/C++17 compilers and system Highway, Brotli, Little CMS 2.19 and JPEG development
libraries. With those checkouts in `reference/libjxl`, `reference/libultrahdr` and `reference/libavif`:

```sh
cmake -S crates/jxl_wgpu_decode/test-data/gain_map_oracle -B target/gain-map-oracle \
  -DJXL_SOURCE="$PWD/reference/libjxl" -DUHDR_SOURCE="$PWD/reference/libultrahdr" \
  -DAVIF_SOURCE="$PWD/reference/libavif" \
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
the executable report that live native roundtrips, weighted applications and ICC comparisons were skipped. Stored native image
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

`iso INPUT OUTPUT` uses libavif's ISO metadata decode/encode. `iso_read.c` and `iso_write.c` compile
the complete, pristine reference translation units and expose their private ISO entry points.
These objects supply the corresponding symbols instead of pulling the same objects from the
static library. The reader receives a single zero wrapper byte for its native `tmap` envelope.
No AV1 codec, native image conversion, copied parser or modified reference source is needed.
Compatible future-writer records are accepted by the native reader and rewritten as version zero;
Rust retains the original writer version and opaque extension bytes. Reserved flags are checked
strictly by Rust; the native reader ignores them, so shared rejection tests cover version/fraction/
trailing-data errors rather than asserting identical reserved-bit policy.

`apply ISO BASE_F32 MAP_F32 BASE_WIDTH BASE_HEIGHT MAP_WIDTH MAP_HEIGHT HEADROOM OUTPUT_F32`
reads ISO metadata with the same native parser. `iso_weight.c` compiles pristine `src/gainmap.c`
and exposes its private signed display-headroom weight selection. The helper bilinearly samples
the map and passes that weight to unmodified libultrahdr `applyGain`; a zero weight copies the
baseline, without applying offsets. Headroom is in log2 stops. Caller-supplied base RGBA is linear
in the selected application primaries and gain reference-white units; gain RGBA contains original
component values. Output retains that linear unit and baseline alpha. This command does not
perform HDR transfer conversion, choose reference white or decode JPEG XL images.

The Rust tests supply independent F64 working pixels, lowered to F32 at this native boundary.
Eighty forward/reverse selections and 384 HDR-baseline selections compare the native application
at the existing `3e-6` normalized bound. The HDR inputs reuse all 48 still streams in the native
HDR corpus; tests keep original-color and XYB reconstruction uncertainty explicit through the
gain and output transfer equations. No original fixture needs regeneration for this extension.

`icc PROFILE_ROOT TARGET INTENT PCS_F64 OUTPUT` evaluates a declared target profile from the
resident ICC corpus. Each input pixel contains three little-endian F64 `(center, radius)` PCS
components. The helper checks the exact profile bytes against the shared independent recipe,
evaluates its interval equations, and separately invokes Little CMS on the center. Each output
component is the shared 28-byte record: six little-endian F32 values (native, independent center,
lower/upper propagated bounds and lower/upper native-model bounds) plus U32 model flags. Little
CMS ink-space percentages are divided by 100 to obtain unit device components. The native-model
bounds account for documented CMM behavior independently of the GPU comparison bounds.

The ICC tests reuse all original gain-map/HDR pixels and the existing 13 selected profile files.
They perform 640 live native profile comparisons and 2,560 GPU outputs, preserving the established
gain, PCS and profile uncertainty contracts. The RGB/Gray and multi-component C++ reference code
lives in `tools/jxl_test_support/native/icc`; production Rust/WGSL is not included. Changes to that
shared evaluator must also reproduce the stored RGB-to-ICC and HDR-to-ICC corpora unchanged.

`bundle INPUT OUTPUT` uses the native
JPEG XL bundle reader/writer with exact consumed/written sizes. The Rust interop test invokes
these on Rust-serialized data and checks preserved bytes and fractions with either headroom order.
The ISO-format correction changes 21 of the original metadata payloads; all 128 image codestreams,
256 pixel planes and the manifest are unchanged. The complete regenerated corpus is reproducible.
