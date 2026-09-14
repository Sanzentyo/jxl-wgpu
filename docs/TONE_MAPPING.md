# Explicit display luminance

`GpuOutputRequest::with_tone_mapping(LuminanceRange)` opts color presentation into a GPU
luminance mapping. The source header supplies `intensity_target`, `min_nits`, `linear_below`
and `relative_to_max_display`. Source linear 1.0 denotes the image intensity; target linear 1.0
denotes the requested display white. `LuminanceRange::new(black_nits, white_nits)` requires
finite `0 <= black <= white` and positive white. The default request performs its existing color
conversion without this mapping. Numeric channels, private reference surfaces, and native samples
with Default/Undefined color specifications retain their original domain.

```rust
use jxl_gpu_protocol::LuminanceRange;
use jxl_wgpu_decode::{GpuOutputRequest, vardct_rgb8_format};

let display = LuminanceRange::new(0.0, 100.0).unwrap();
let request = GpuOutputRequest::color(vardct_rgb8_format())?
    .with_tone_mapping(display);
# Ok::<(), jxl_wgpu_decode::Error>(())
```

## Luminance policy

The base curve follows [ITU-R BT.2408-8 (2024), Annex 5](https://www.itu.int/dms_pub/itu-r/opb/rep/R-REP-BT.2408-8-2024-PDF-E.pdf):
normalize PQ between source black and white, connect the highlight knee to target white with
a cubic Hermite segment, and apply the fourth-power shadow lift for target black. Multiplying
all three linear components by the luminance ratio retains their chromaticity. At nonpositive
or very small luminance (`<= 1e-6` nit), the base curve uses a neutral cap. RGB neutral is
(1,1,1); ICC PCS neutral is D50 (0.9642,1,0.8249). PQ-space clipping precedes the inverse to
avoid evaluating ST 2084 beyond its asymptote. F32 output retains chromatic components outside
[0,1]; integer output quantizes and clips normally. Optional [RGB gamut mapping](GAMUT_MAPPING.md)
follows this operation and gives priority to its protected light, including global protection
when the threshold reaches either white.

[ISO/IEC 18181-1:2024, E.3](https://previewnorm.com/iso/ISO%20IEC%2018181-1-2024%20PDF.pdf)
requires luminance below `linear_below` to remain unchanged. Absolute thresholds are nits;
relative thresholds are multiplied by the requested target white in F64 metadata, retaining the
exact product of the binary16 header fraction and F32 display white. The following choices make
that constraint explicit, including requests which cannot fit all protected light on the display:

- Below a positive threshold, preserve absolute light by multiplying unit values by
  source-white/target-white. Compare against the exclusive F32 upper neighbor of the exact
  threshold/source-white quotient, so a preceding sample cannot round onto the threshold during
  a multiplication by source white.
- Above that threshold, connect source and target curves at the same protected luminance.
  This protected connection supersedes black remapping. Clamp the normalized knee to zero and
  limit the start slope to three times the shoulder secant, retaining a continuous monotonic
  highlight shoulder even with little remaining headroom. The threshold may lie below source black.
- If the threshold reaches either white, preserve absolute light throughout. It may exceed
  target white in F32 output; a decreasing shoulder would violate the protected constraint.
- If the source already fits and both black points agree, preserve absolute light throughout.
  Equal ranges, including equal degenerate ranges, retain identity behavior.
- Otherwise a target range with equal endpoints returns neutral target white; a source range
  with equal endpoints normalizes positive unit luminance to white, with a neutral cap at unit
  luminance `<= 1e-6`. Protected samples are still handled first. These degenerate conventions
  avoid undefined normalization and do not claim to reconstruct missing source dynamic range.

The decoder applies this explicit policy after frame composition, reference storage and spot
rendering, before output transfer, alpha association and quantization. It uses the source intensity
for PQ/HLG input and target intensity for output PQ/HLG. This is an explicit BT.2408 policy for
linear light from any admitted encoding; it does not reproduce libjxl's automatic HLG display
adaptation policy, which uses a different OOTF decision. The display-texture API remains separate.

## ICC and GPU contract

Enumerated output maps in target-primary linear RGB. ICC output maps D50 PCS Y after the selected
rendering-intent connection (including any black-point detection) and before target device curves.
`IccTransform::with_tone_mapping(ToneMapping)` reconstructs this stage order and sets source/target
RGB endpoint intensities. Repeated calls replace the mapping. Same-profile output selects a real
conversion when mapping is requested; it cannot bypass PCS. Generated CMYK K remains a device
component separate from the original Black extra channel. ICC-to-enumerated output uses the final
RGB packer for mapping, exactly once. ICC `lumi` tags do not select a physical display range.

Production host code evaluates only metadata endpoints and coefficients. Pixel operations share
`tone_mapping.wgsl` across image output and resident ICC. A 48-byte `ToneMappingParams` record
contains source/target white, protected comparison bound, mode, PQ endpoints and knee coefficients.
It occupies bytes 240–287 of the 304-byte image-output uniform. The separate codec-source uniform
remains 160 bytes, for 464 bytes total. Resident ICC opcode 11 stores the same 48-byte payload with
one 16-byte stage descriptor; its dispatch storage remains 320 bytes. No additional image buffer,
GPU dispatch or readback is introduced by the mapping itself. An existing color conversion may
still require its established intermediate image. Admission and cancellation charge and retain
the complete program and output resources.

## Evidence and limits

The [native recipe](../crates/jxl_wgpu/test-data/tone_mapping_generator/README.md) calls pinned
libjxl 0.12.0 directly for 4,986 XYZ samples across 18 source/target ranges. All 44,874 GPU components
across Scalar/Lanes32/Tile16x16 meet the independent F64 reference. That oracle uses the Bernstein
form of the cubic instead of the shader's Hermite polynomial. Dense neutral ramps, strict threshold
neighbors, raised source black, fitted source ranges and degenerate ranges test preservation and
monotonic shoulders independently of native automatic rendering policy.

All 56 existing HDR streams retain their physical frame bytes. Edited tone headers cover nonzero
minimum light, raised display black and absolute/relative thresholds. Header/ICC initialization also agrees with
jxl-oxide; this metadata check does not ask it to decode frame entropy. Their 80 presentations
produce 960 mapped outputs and 1,681,344 checked color components through P3 PQ/HLG and identity
PCS targets, both layouts and whole/256-byte-window fragmented input. Progressive images remain
immutable, final words agree, alpha words match unmapped output, and input/GPU budgets return to
zero. Eight embedded RGB/Gray ICC × Modular/VarDCT × original/XYB cases add 32 outputs against
existing independent scalar/native linear references, with exact alpha. Source errors propagate
through signed matrices, luminance-ratio intervals and output transfer; fixed arithmetic allowances
are declared before GPU execution, never fitted from GPU differences.

Full JPEG XL remains incomplete. Requested-profile tone-mapping pixel evidence currently covers
identity PCS; existing profile/intent processing is independently tested, and admission tests also
exercise tone programs with dynamic black detection and CMYK/15-channel output. Broader combined
ICC target/intent pixel oracles, wider gamut/profile combinations, automatic display policy, extreme numeric ranges,
HDR features/LF combinations, display textures and the remaining codec/conformance/encoder gates
remain in the [full roadmap](FULL_JPEG_XL_ROADMAP.md).
