# Spline conformance references

`animation_spline.npy.gz` is the unchanged float32 reference from the official
[JPEG XL conformance corpus](https://github.com/libjxl/conformance/tree/b1d0f990b03e57bf6d137c365cd5dc8b470b9191/testcases/animation_spline),
compressed with `gzip -n`. Its uncompressed SHA-256 is
`a571c5cbba58affeeb43c44c13f81e2b1962727eb9d4e017e4f25d95c7388f10`.
The test verifies that digest before comparing any pixels.

The original 60 × 320 × 320 × 3 little-endian float32 array uses the sRGB
encoding described by `animation_spline.icc` (SHA-256
`80a1d9ea2892c89ab10a05fcbd1d752069557768fac3159ecd91c33be0d74a19`).
The corresponding input is the existing `fixtures/animation_spline.jxl` at
the workspace root, byte-identical to the corpus input.

For **every** frame, the corpus requires the maximum per-channel RMSE to be
at most 0.0001 and the absolute peak error to be at most 0.004. The test uses
these limits directly, without clipping negative or extended-range samples.
Whole and fragmented bounded-window execution must also produce identical
float32 bits.

The upstream files are distributed under the adjacent `LICENSE`.
To fetch the original objects, use the content-addressed URLs
`https://storage.googleapis.com/jxl-conformance/objects/<sha256>` with the
digests above. Run `gzip -n -c reference_image.npy > animation_spline.npy.gz`
to recreate the compressed array.

## Generated feature and progression streams

`features/` contains 68 explicit scenarios from
`jxl_test_support::fixtures::splines::cases()`: both coding modes, original/XYB color, positive and
negative quantization adjustments, curved control points leaving/re-entering the frame, signed
coefficients/thickness, patches and reference overwrites, noise, standard/custom upsampling,
single/nested LF producers and consumers, independent extras and subsampled JPEG restoration.
Ten scenarios cover unequal 2/8 color/extra factors in ordinary images and LF roots, including
integer/float extras and custom weights. `progressive/` contains four two-frame patch/spline
chains plus eight spline-only streams with 2/4, 2/8 and 4/8 factors, signed float depth and
standard/custom kernels. Native snapshots retain every pass and final output. Empty leading
Modular prefixes have native reference snapshots but produce no GPU update until image samples
have been validated; the manifest states the first emitted pass explicitly.

Each `.jxl.hex` freezes the complete input. A feature `.f32.hex` contains final interleaved linear
RGBA followed by one planar scalar array per declared extra. A progressive reference repeats
that layout for every native pass prefix and final frame; its color is linear for XYB sources
and unchanged sRGB for original-color sources. All references preserve alpha and orientation.
Run `cargo run -p jxl_wgpu_decode --example regenerate_splines` with the native libjxl oracle
configured as described in `docs/CONFORMANCE_CORPUS.md` to regenerate these files.
Generate the unequal progressive seeds first:

```sh
cargo run -p jxl_wgpu_decode --example regenerate_lf_extra_channels -- \
  crates/jxl_wgpu_decode/test-data/frame_resampling --resampling
```

Reference selection is fixed in the scenario manifest. Original-color feature streams use
unconverted native sRGB and the analytic signed f64 EOTF because libjxl 0.12's CMS approximates
extended-range values. The two `jpeg_440`/`jpeg_420` restoration cases use pinned jxl-oxide 0.12.6
linear output; the independent scalar audit of native vertical-subsampling restoration is
documented in the conformance corpus. Native libjxl must still accept every generated input.
Tests use normalized color error `abs(actual-reference)/(1+abs(reference)) <= 1/1024` and scalar
extra error at most `2e-6*(1+abs(reference))`; these are separate from the stricter official
animation limits above. Whole and bounded fragmented execution must agree bit-for-bit.
