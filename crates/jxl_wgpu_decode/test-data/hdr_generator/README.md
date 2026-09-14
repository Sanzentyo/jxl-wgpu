# Enumerated HDR corpus

The public libjxl **0.12.0** encoder/decoder APIs generate these fixtures. The audited source is
tag v0.12.0, commit `a7a9c787341cf703dede03c2009fa460cae5e5df`. The native codec is an offline
test dependency; production entropy, reconstruction, transfers, composition and packing use GPU
shaders. Existing corpora are not rewritten.

```sh
c++ -std=c++17 -O2 -Wall -Wextra -Werror main.cpp references.cpp \
  $(pkg-config --cflags --libs libjxl libjxl_cms) -o generate-hdr
./generate-hdr ../hdr
```

From the workspace root:

```sh
cargo test -p jxl_wgpu_decode --test hdr -- --test-threads=1 --nocapture
```

`manifest.txt` explicitly records name, width, height, Modular flag, XYB flag, primaries/gray,
transfer, intensity target in nits, sequence flag and F32 flag. Tests parse these columns and
assert actual image/frame metadata; filenames do not decide the coding mode. The 161 data files
contain 56 streams, 56 native original-color references, 48 native linear references and the
manifest. The 56 streams present 80 complete images:

- Modular/VarDCT × original RGB/XYB × PQ/HLG × 100/255/1000/4000 nits: 32 stills at 17×9.
- All eight mode/transfer combinations: six physical frames, four presentations at 37×19 and
  255 nits, including hidden/cropped layers, reused references, all five blend modes and separate
  alpha Replace.
- Original Modular and VarDCT XYB × PQ/HLG × BT.709/Display-P3/gray: 12 stills at 1000 nits.
- Those two modes × PQ/HLG: four 257×17 group-boundary stills.

Color is 16-bit, except original Modular BT.2020 stills at 100/1000 nits, which preserve F32
dyadic samples. Alpha has independent 10-bit precision. Source values include zero, saturated
colors and the HLG breakpoint. Lossy reconstruction produces negative RGB and negative luminance;
those samples remain in the corpus. VarDCT enables progressive AC. Filters, patches and noise are
disabled to isolate color and composition; their combinations remain separate conformance work.

## Luminance and reference contracts

Linear RGB is relative to the image's `intensity_target`: unit white represents that many nits.
PQ uses the ST.2084 EOTF/OETF with the explicit 10000/intensity scaling. HLG applies its coupled
display OOTF using the declared primary luminances and
`gamma = 1.2 * 1.111 ^ log2(intensity / 1000)`. The codec rendering-stage threshold is
`abs(gamma - 1) > 0.01`, or `abs(1/gamma - 1) > 0.01` for encoding. The conversion neither changes
the display peak nor performs tone/gamut mapping. Generic backend output without image intensity
keeps its existing absolute-normalized PQ and scene-linear HLG contract.

The native linear files use the public decoder with its CMS. They are direct reconstruction
references for XYB stills. Original RGB linear output instead uses independent F64 EOTF/OOTF
equations: the native CMS approximates PQ black and normalizes HLG saturated highlights at low
intensity. Those CMS policies are not a neutral unbounded conversion. Native XYB sequences cannot
request linear output: `render_pipeline/stage_blending.cc` requires the original encoding.
Sequence conversion therefore starts from the native original-color reference in F64.

HLG outside the nonnegative luminance domain needs an explicit arithmetic extension. Native
`HlgOOTF::Apply` calls `base/fast_math-inl.h::FastLog2f`, whose negative domain is documented as
undefined; its signed integer range reduction nevertheless affects actual decoded XYB pixels.
The GPU retains that pinned reconstruction behavior using wrapping integer operations, a positive
mantissa logarithm and a capped factor, without evaluating a negative fractional power. The F64
test expresses the exponent wrap independently. This is an interoperability contract, not a claim
that negative light is part of nominal BT.2100. Broader HDR/gamut policies remain open.

## Precision and validation

Native original RGB uses normalized error `abs(actual-reference)/(1+abs(reference))` of `1e-5`
for Modular and `1/1024` for VarDCT. XYB sequences use `1/1024` in the original encoding. XYB
stills check native linear reconstruction with `1/1024` and propagate that interval through the
coupled OOTF/OETF for original and converted outputs. PQ greatly amplifies tiny linear differences
around black; a fixed encoded epsilon would confuse reconstruction and transfer accuracy.
The interval is derived before observing GPU output, includes the signed primary matrix, and
adds a separate `5e-5*(1+abs(reference))` transfer allowance. Alpha has its own `2e-6` bound.
Native U16 adds one quantization code and clamps only at final integer packing.

All 48 stills also use jxl-oxide **0.12.6**. Original samples are requested in their declared
encoding. XYB is requested as raw components, then reconstructed with independent F64 opsin and
primary matrices; this avoids that decoder's automatic Rec.2408 mapping for >255-nit linear
targets. GPU linear output has a stricter `1e-4` normalized budget against this second decoder.
The shared helper used for SDR primary calibration is also reused here. Sequence references
remain native because of the extra-channel source-selector limitation recorded by the original
color corpus.

The actual-GPU transfer test independently checks 1728 RGB triples across both directions,
PQ↔HLG, primary conversion, identity, black/negative samples and nine intensities, including the
HLG near-unity threshold. Its F64 budget is `5e-5`; same-encoding values retain their F32 bits.
Full streams check whole input against 256-byte GPU windows with 43-byte fragments, retained
progressive outputs, final-only equality, original numeric color/alpha, and zero reservations
after release. HDR↔ICC still requires a defined luminance connection and is rejected before
submission. This corpus does not close the full color/HDR or ISO conformance roadmap gates.
