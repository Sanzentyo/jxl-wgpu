# Resident ICC color processing

The metadata and resident GPU execution layers support RGB/Gray matrix/TRC, integer LUT methods
and floating-point MPE programs with XYZ or Lab PCS.
JPEG XL decoding now admits embedded ICC for unfiltered original Modular numeric samples and
independent extra-channel output in the supported single-frame paths through both codecs. Codec
reconstruction and LF configuration are independent of color conversion. The common decoder now
also handles original and XYB ICC RGB/Gray color surfaces, including YCbCr reconstruction,
original-domain references and composition, all four matrix/TRC intents and U8/F32 requested output.
Requested RGB MPE output and embedded/requested RGB/Gray LUTs are also exercised through the public decoder.
Original CMYK sources now connect to RGB/Gray through the independently composed Black extra,
including YCbCr reconstruction and source-domain spot presentation.
Enumerated SDR sources also target ICC through original/XYB/YCbCr reconstruction and composition.
Spot presentation runs before ICC connections in the actual source domain, after reference storage.
Broader ICC XYB conformance, other ICC methods
and HDR mapping remain open.
This checkpoint does not change the full JPEG XL support claim.

ICC spot presentation retains the existing
[reference-stage ordering](https://github.com/libjxl/libjxl/blob/v0.12.0/lib/jxl/dec_cache.cc).
Unreferenced XYB uses linear RGB; original and composed images use their tagged original domain.
The [ink equation](https://github.com/libjxl/libjxl/blob/v0.12.0/lib/jxl/render_pipeline/stage_spot.cc)
applies each declaration in order, including signed or above-one coverage. A Gray ICC connection
consumes its single first device component, matching native Gray CMS channel consumption.
Linear/RGB sources retain all three colored components before a requested Gray conversion.
The ink offsets come from the actual extra-plane layouts, including the different Gray/linear
color-plane counts. Numeric requests keep the base samples and never render inks.

The ICC presenter records the ink copy, optional color connection and final packing in one
submission. Its private copy leaves saved references and independent extras unchanged. Its
storage and 32-byte-per-ink metadata are admitted together with the existing ICC dispatch and
packing resources, and remain owned through completion or cancellation. Preserve omits this
copy and metadata. The [32-source recipe](../crates/jxl_wgpu_decode/test-data/icc_spots_generator/README.md)
separates native reconstruction, independent spot/composition equations, ICC curve intervals
and native CMM comparisons. Same-profile F32 output bypasses curves and preserves extended
rendered values; CMM references for an actual connection apply the specified unit device range
at the native float API boundary.

Original CMYK surfaces explicitly own their ICC profile and Black extra-channel index. They store
three complemented CMY planes and borrow the actual Black plane for a four-channel ICC view;
Black is never duplicated into color storage. Separate blend modes and reference slots therefore
remain authoritative. YCbCr inversion packs these three components into non-color storage with
`ImageOutputParams::for_components`; only the image-owned domain gives them CMYK meaning.
The resident ICC dispatch maps image samples through `ResidentIccSampleEncoding::Complement`
on input/output as selected. Its 320-byte storage record adds the image encoding flags at byte
304 and retains preparation status at byte 300. Profile black-point probes remain in the ICC
program's device domain. All three GPU kernel variants check both sample conventions and guards.

The official `cmyk_layers` corpus checks all five original CMY/Black/Alpha components against
unchanged upstream bounds. The [CMYK recipe](../crates/jxl_wgpu_decode/test-data/cmyk_generator/README.md)
adds 18 three-frame sequences with three LUT formats, both PCS domains, non-leading Black,
independent Black references, F32 extras, both codecs and YCbCr. Native reconstruction and
independent F64/native ICC references check RGB/Gray presentation under four intents, spot
Render/Preserve, both layouts, bounded input, held frames and memory release. Missing or ambiguous
Black declarations remain typed unsupported profiles.
XYB-to-original CMYK reconstruction, wider sampling/range/profile combinations and full JPEG XL
conformance remain open.

YCbCr reconstruction carries the actual original device profile through the inverse codec matrix;
it does not select an ICC transform or introduce an enumerated RGB carrier. The resulting surface
has one Gray plane, three RGB planes, or three CMY planes with a separate Black extra before
reference storage and composition. Same-profile RGB/Gray requests
therefore work independently of the requested CMS intent. A 146-source corpus checks both codecs,
sampling, precision, filtering, resampling and retained compositions. Five sources also check
linear/sRGB and other-profile conversion with independent error propagation from each existing
codec bound. [Generator and interval derivation](../crates/jxl_wgpu_decode/test-data/embedded_icc_ycbcr_generator/README.md).

XYB reconstruction keeps direct presentations in unbounded linear D65 BT.709. References saved
after the color transform and frame blends first enter the original ICC device domain. The
original profile's header intent selects reconstruction; the requested intent selects presentation.
Only required connections are selected and uploaded. Both stages use the same image-owned,
budgeted ICC program abstraction, including intermediate storage retained through GPU completion.
Gray reconstruction changes three linear planes into one device plane and keeps extra-channel
offsets consistent. Four stills, four additive sequences and seven LF/patch substitutions cover
this boundary. Forty-eight additional native sequences cover straight/associated F32 alpha and all five
color blend modes against independent device-domain equations and propagated bounds. Identical
selected connections share one image-owned GPU program across reconstruction and presentation,
including mixed original/linear sequences and exact byte-budget cancellation. [XYB generator and precision](../crates/jxl_wgpu_decode/test-data/embedded_icc_xyb_generator/README.md).

`ColorSpecification::Icc(IccProfile)` now carries the exact profile through owned pixel formats
and layouts. The color specification is `Clone`, not `Copy`; clones share the original bytes and
tag directory. `EmbeddedIccInventory::profile` likewise shares reconstructed bytes across image,
preview and physical-frame inventories. Profile interpretation remains separate from inventory
reconstruction. `ColorModel::Gray` describes gray X with optional alpha W, with explicit U8/F32
planar/interleaved classification. ICC RGB/Gray/XYZ signatures must match their pixel color model;
an ICC profile cannot relabel numeric or YCbCr storage. Enumerated packers/display currently reject
unexecuted ICC targets and gray color outputs. `ImageOutputParams::for_icc_device` is a separate
packing entry point: it verifies the exact source/target profile and packs already converted
device values without assigning an enumerated RGB meaning or evaluating any curves.

The [embedded numeric corpus](../crates/jxl_wgpu_decode/test-data/embedded_icc_generator/README.md)
adds eight native RGB/Gray original/XYB streams and metadata-only substitutions into existing wide
integer and IEEE-754 fixtures. It checks exact sample storage, complete/fragmented input, both
standalone producers and the common decoder, and byte-budget release. It does not use an ICC
profile-to-itself conversion for passthrough: such a conversion would evaluate curves and lose
out-of-range values or original bit patterns.

The common decoder resolves the original profile once per selected image. Its component producer
contract uses three non-color F32 planes with explicit private tagging, followed by checked GPU
copies into the original profile's one Gray or three RGB planes. References keep that profile;
blending derives alpha/extra offsets from the actual color count. Original-device F32 packing with
preserved alpha association retains Modular IEEE words, including nonfinite values. ICC conversion
uses the finite device-value contract described below; its unit-domain curve rules apply only
when an actual profile conversion is requested.

Requested color conversion uses a selected immutable program shared across physical frames. Program
upload is lazy, budgeted and retryable; each dispatch retains its uploaded program through GPU
completion even if the image session is dropped. Intermediate color surfaces, unchanged extra
planes, output words, the 320-byte ICC dispatch storage, optional four-byte validation word
and the 208-byte RGB/Gray or 320-byte device packing uniform are all accounted.
No frame readback or CPU pixel CMS is involved. `GpuOutputRequest::with_icc_rendering_intent`
defaults to relative colorimetric; non-Bradford conversion and unsupported selected methods return errors.
Exact same-profile packing does not select a CMS method and therefore does not require an
executable matrix/TRC or inverse curve.

## Profile component output

`PixelFormat::icc_device(profile, sample, storage, alpha)` creates explicit profile component
storage for `GpuOutputRequest::color`. `ColorModel::IccDevice`, `Channel::Device(index)` and
`Channel::Alpha` distinguish each profile component from opacity. `Swizzle::Device` reads these
semantic fields directly; physical component/plane order may differ. Canonical constructors put
alpha last. `Swizzle::Xyzw` retains the existing four-component mapping for other formats.
The format supports one through fifteen device components and an independent alpha plane,
U8/F32 samples, and interleaved/planar storage. Unsupported selected profile methods still return
typed errors; recognizing a device signature does not establish executable color conversion.

F32 uses the selected ICC program's device units. CMYK and multicolor ink values use unit amounts,
with zero meaning no ink and one meaning full ink. Original JPEG XL CMYK complements are converted
only at the output boundary. U8 clamps each component to [0,1] and rounds `255 * value` with ties
upward. Alpha association applies after color conversion, independently to all device components.
Same-profile output selects no CMS program and preserves extended F32 samples. Numeric requests
retain their original codec-component convention.

`DeviceOutputParams` validates actual input plane offsets/strides, exact profile identity, output
packing, rotated extent and bounded byte addressing. Its 320-byte uniform carries separate source
and target mappings. Each shader invocation owns one output word and gathers its bytes, so odd
row pitches, unaligned F32 planes and an incomplete final word need no overlapping writes. Padding
bytes are zero. The selected color connection and output packer share the existing completion-owned
budget, retry and cancellation path. Display consumers require an explicit RGB conversion.

The [device output recipe](../crates/jxl_wgpu_decode/test-data/device_output_generator/README.md)
redecodes 42 unchanged JPEG XL sources and reproduces their frozen native sample words. Independent
F64 and Little CMS references cover 403,920 components, all four intents and both PCS domains.
CMYK inputs exercise each 1/2/3/4/5/15-component target through original Modular, original VarDCT
and YCbCr. Public decoder tests compare 3,231,360 color components in 4,224 presentations, with
independent alpha, spot policies, U8/F32, both layouts and whole/bounded input. Same-profile tests
add 624 presentations and 323,136 original components. Twenty enumerated RGB/Gray sources add
800 presentations and 1,417,248 independently bounded components across all reconstruction modes.

The standalone packer checks 1,152 guarded dispatches and 931,040 bytes, including all eight
orientations, component permutations, varied plane pitches, missing alpha and alpha conversion.
Exact-budget tests include fifteen-component targets, dynamic black preparation, same-profile
CMYK, failed admission, retry, concurrent submissions and cancellation. These checks do not
complete XYB-to-original CMYK reconstruction, uncommon device-space image conformance, HDR,
full profile range/conditioning or the remaining full JPEG XL requirements.

## Model and supported scope

`jxl_gpu_protocol::icc::IccProfile` owns the original profile bytes in an `Arc<[u8]>` and a checked
tag directory. Parsing accepts v2 and v4 through 4.4. It checks declared size, header signature,
version/intent/PCS illuminant fields, reserved bytes, tag count, element ranges, alignment,
duplicate signatures, partial overlaps and padding. Complete shared tag elements are allowed.
V4.4 tag elements must be contiguous. Unknown tags remain available without interpretation.
This validates the structures used for execution; it is not a validator for every private or
descriptive tag's internal semantics.

Default bounds are 16 MiB per profile, 4,096 tags and 1,048,576 samples per selected curve.
`IccError` distinguishes malformed structures, resource limits, unsupported methods and undefined
curve inverses. No source pixels are passed to this crate. A fixed number of metadata endpoint
evaluations checks the mathematical domain of parametric curves and monotonicity for inversion.

`IccProfile::select` selects a direction and explicit intent and returns an `IccProfileProgram`.
Its `IccProgram` is a checked sequence of typed stages with explicit input/output channel counts.
DToB/BToD MPE tags take priority. Unknown processing-element signatures discard that MPE method
according to ICC.1 section 10.16.1; malformed supported elements return an error. Legacy AToB/BToA
selection uses the requested intent, then intent zero if absent. Absolute uses the relative LUT
and the PCS media-white connection. `matrix_trc` remains restricted metadata inspection of a
selected matrix/TRC method; LUT/MPE callers use the general selected program.

Matrix/TRC covers RGB input/display and monochrome input/display/output classes with XYZ PCS.
LUT/MPE supports input/display/output profiles, recognized device channel counts and XYZ/Lab PCS;
this resident metadata support is broader than JPEG XL decoder admission, which still uses
RGB/Gray and original CMYK source surfaces. Requested device outputs support up to fifteen color components. XYB-to-original CMYK
reconstruction, other profile classes,
full floating-point-range conformance and broader gamut/HDR policies remain open.

Relative intent connects the profiles in media-relative PCS. Absolute intent uses fully adapted
media-white scaling, `source_white / target_white`, between the original PCS matrices. V2 display
profiles use PCS D50 as their effective media white, as required for their legacy display policy;
other profiles retain their exact `wtpt`. Perceptual and saturation matrix-shaper intent use black
compensation when the target is v4. Their PCS affine connection preserves D50 and maps source
black to target black: `scale = (D50 - target_black) / (D50 - source_black)` and
`offset = target_black - scale * source_black`. V2 targets do not enable this automatic compensation.
This is a declared matrix-shaper CMM policy, not a substitute for profile-supplied gamut mapping.

Matrix/TRC black is obtained by evaluating at most three curve endpoints and the profile's colorant matrix.
The CMM darker-colorant policy clips Lab L* to 0–50, resets L* above 95 to zero, and retains a*/b*.
The policy uses decimal PCS D50 `(0.9642, 1, 0.8249)`; encoded colorants and RGB connection geometry
remain unchanged. No CPU image pixels are evaluated. Connection matrices and offsets compose in host f64 before upload.

Matrix columns retain the exact signed fixed-point values, represented losslessly in host f64.
Media white and chromatic adaptation retain their original signed integer records. ICC colorants
are already relative to PCS D50; the resident transform does not apply `chad` a second time or
approximate the colorants by recognised RGB primaries. Gray scales its one curve by PCS D50 on
input and uses PCS Y on output. Identical matrices cancel exactly while both curves still run.
A singular matrix can be used in the forward direction; an inverse request rejects it.

Independent channel curves support identity, u8Fixed8 gamma, sampled u16 curves with linear
interpolation, and all five ICC parametric functions. Forward evaluation permits sampled curves
that cannot be inverted; target curves must have a defined monotone inverse. Increasing and
decreasing sampled curves, interior/final plateaus, parametric branch gaps and clipping are
explicitly handled. Parametric inversion solves the equation directly; it does not search a
rounded forward function and invent a numerical plateau near black.
Parametric target inverses currently require increasing branches, or a single decreasing linear
branch. Other decreasing parametric combinations return a typed unsupported inverse error.

Sampled interpolation computes the exact product of the input F32 significand and the integer
interval count with portable u32 arithmetic, then rounds only the fractional weight. This avoids
losing the interpolation coordinate in large or sharply varying tables. GPU tests include 1,001
and 1,000,003 irregular samples, subnormal/near-zero coordinates and exact endpoint guards.

## Enumerated RGB connections

`IccTransform::to_rgb` and `from_rgb` connect the selected profile to a declared
`RgbColorEncoding`. `IccTransformEndpoint::Rgb` owns both its geometry and transfer;
`to_linear_rgb` and `from_linear_rgb` are convenience constructors with a linear transfer.
Relative colorimetric intent uses Bradford
between the RGB reference white and ICC's exact encoded PCS D50. Colorants are connected directly
in f64; no synthetic quantized profile or approximation by recognized primaries is introduced.
All intents treat this RGB endpoint as an ideal, fully adapted v4 endpoint with PCS D50 white
and zero black. The same media-white and black-compensation rules therefore work in both directions.

`ColorMatrix` in the backend-neutral protocol owns the shared CIE geometry and white adaptation
calculation. Existing RGB output/display lower this same implementation to F32. ICC connections
combine the profile colorants and RGB/PCS matrix in f64 before a single F32 lowering. The host
calculates metadata matrices only. The shader evaluates all pixel curves and matrix products.

The linear endpoint has no ICC curve descriptor and applies no unit clipping. A negative input
or input above one reaches the PCS matrix unchanged. A linear output can remain negative or
above one after conversion from a wider gamut. The profile endpoint still follows the bounded
ICC curve contract. This does not extend arbitrary ICC device curves to unbounded/HDR domains.

Nonlinear RGB adds one `IccStage::RgbTransfer` before the input PCS matrix or after the output
PCS matrix. Its WGSL functions are shared with image output, with explicit transfer selector,
gamma exponent and direction. Linear, sRGB, BT.709, BT.2020, PQ, HLG, Gamma and DCI retain their
declared extended-range rules. This resident normalized-transfer API does not admit PQ/HLG JPEG XL
metadata or establish HDR luminance mapping. The decoder still admits its existing SDR declarations.
No surrogate ICC profile, CPU pixel conversion or additional linearization image is introduced.

Requested ICC output enters the common compositor even for a plain unreferenced sRGB stream.
Original and composed surfaces use their declared RGB transfer; direct XYB surfaces use linear
RGB in the original primaries. Plans retain actual source encodings and deduplicate equal ones:
an originally linear image cannot accidentally select an absent original-history slot. Alpha
and extra planes remain outside the ICC program and are copied to the target's actual color count.

The [RGB-to-ICC corpus](../crates/jxl_wgpu_decode/test-data/rgb_icc_generator/README.md) pairs all
228 existing enumerated streams with nine matrix/TRC, LUT and identity-MPE targets, using all
four intents. Its 4,049,280 independent/native reference components are checked 16,197,120 times
in 9,120 decoder presentations. Whole and fragmented input, planar/interleaved F32, retained
progressive images, final-only equality, exact alpha and final memory release are covered.
The independent input bound includes codec reconstruction before propagating through target
curves, matrices, Lab and CLUTs. Little CMS's rounded reference-black Z has a separate native
model; it never enlarges the primary GPU interval. Standalone resident tests additionally cover
both directions, all eight transfers, five RGB geometries and three kernel variants.

## GPU contract

`ResidentIccMemoryPlan::new` checks capability and program bounds before allocation.
`ResidentIccProgram::new` uploads immutable stage metadata, shared curves and shared CLUTs once.
`ResidentIccPipeline::encode` records conversion between up to sixteen planar F32 color channels
in distinct resident storage buffers. Offsets are scalar indices relative to their binding;
each plane has its own row stride. Alignment, usage, extents, channel counts, storage capacity,
non-overlapping output ranges, u32 addressing and workgroup counts are checked before dispatch.
Padding, alpha and extra planes remain outside the color views and are not written.

The caller admits the reported program and dispatch allocations and retains their handles through
GPU completion, as with the other resident codec primitives. This layer neither submits nor maps
image buffers. Recorded work can be abandoned; the same program can be reused across frame extents,
pitches and later submissions. Queue/session integration must keep the existing memory permits
until the last submitted consumer completes.

Inputs must be finite. Legacy ICC device curves clamp their domain and range to [0,1]. MPE
formulas and matrices do not impose that clipping; a CLUT clamps only its input coordinates.
Uploaded coefficients use F32. Pixel arithmetic uses F32 with extended exponents and compensated
affine bases where needed. The backend checks finite matrix lowering
and the existing legacy parametric-power bound. Full-range overflow/conditioning coverage for
arbitrary MPE formulas remains a conformance gate, not an established HDR guarantee.

## Integer LUT methods

`mft1`/`mft2` profiles execute their matrix, input curves, CLUT and output curves in order.
`mAB`/`mBA` use named offsets and directional A/CLUT/M/matrix/B stages, including all four
permitted combinations. Embedded curves reuse the validated unit-domain curve representation;
sampled payloads shared by complete or suffix curve sets retain one allocation. Physical ordering
is independent of execution order. Unsupported selected types, invalid combinations, overlap,
truncation, reserved bytes and resource excess return errors before GPU execution. No malformed
selected method falls back to another tag.

The table formats accept v2/v4; A/B formats require v4. `mft1` has 256 entries per input/output
curve; `mft2` has 2–4,096. CLUTs have at least two grid points per dimension and use 8-/16-bit
samples. Counts and byte ranges are checked before payload allocation, using the same channel,
sample and CLUT resource limits as MPE. The table matrix must be identity unless its input is
PCSXYZ. A device-space `XYZ ` signature does not make input device data PCS.

Explicit stages convert normalized LUT samples to physical PCS. XYZ uses `65535/32768`;
`mft1` XYZ is implementation-defined and follows this same Little CMS convention. General
Lab uses L* 0–100 and a*/b* −128–127. `mft2` retains the legacy Lab encoding even in v4:
L*=100 at `0xff00`, neutral a*/b* at `0x8000`, with valid a*/b* values up to 127.99609375.
The physical PCS connection preserves that range; L* is clipped to 0–100. Normalization, matrix
and curve clipping boundaries remain explicit and cannot be removed by affine composition.

Interpolation is a typed property of the CLUT. Legacy Lab-indexed output LUTs use multilinear
interpolation; other LUTs use the existing tetrahedral/leading-axis-linear policy. This follows
Little CMS 2.19's legacy LUT selection without changing floating MPE interpolation. GPU payload
addresses and shared payload storage retain checked layouts. The dispatch record is now
320-byte writable storage so a preparation pass can publish its connection coefficients.

The v4 selected-method black policy also applies to LUTs. Perceptual/saturation conversion
from a v2 LUT to a v4 or virtual linear endpoint embeds its selected source program and a
normalized darker-colorant endpoint in `IccStage::BlackPointConnection`. A one-invocation GPU
metadata pass executes that source program before image conversion. Gray/RGB use device zero,
CMY/CMYK use device one, and device Lab uses `(0, 128/255, 128/255)`. Recognized device spaces
without a CMM darker-colorant estimate use zero PCS black; they need no probe. Relative,
absolute and v2-target connections retain their static policy.

The GPU probe applies the same Lab L* 0–50 / above-95 reset policy while retaining a*/b*.
Unchanged lightness preserves the evaluated XYZ directly. It builds the PCS scale/offset in
per-dispatch storage; shared profile metadata remains immutable across concurrent submissions.
Nonfinite black or connection coefficients set an error status and suppress image writes.
`ResidentIccDispatch::validation_buffer` exposes a four-byte map whose status must pass
`validate_status` before output becomes authoritative. Decoder wait/poll performs this check
and preserves `ResidentIccError::Precision`; completion and cancellation release all admitted
resources. No pixel data is read back for black detection or validation.

The `black` corpus adds 20 v2 profiles: 8-/16-bit LUTs, XYZ/Lab PCS, Gray/RGB/CMYK/5CLR,
zero-floor black and above-95 lightness cases. Independent/native source-black references check
120 components; six target domains and all four intents check 318,240 color components,
repeated on three GPU kernels for 954,720 comparisons and 576 validated preparations.
Twenty-four embedded-profile Modular/VarDCT streams add 384 decoder presentations and
117,504 color checks with exact alpha and bounded transport. The independent connection
interval encloses the input/black error-box corners of its rational equation and rejects any
box containing a singular denominator. Every native and GPU interval remains checked.
CMY/device-Lab endpoint metadata is tested, while native image conformance for those spaces
and other uncommon device spaces remains open.

The `lut` corpus contains 436 files for 41 profiles and 202,436 independently evaluated/native
components, checked 607,308 times on all three GPU kernels. It covers RGB, Gray, CMYK, 2/5/15
channels, XYZ/Lab, table precision, every A/B combination, shared offsets, curve branches and all
implemented directional/intent connections; the dedicated `black` corpus covers automatic v2 connections.
Another 145 files contain 24 libjxl original-color streams and 96 LUT-to-LUT references.
Their 29,376 components are checked through 384 public decoder presentations (117,504 GPU
components), with whole/fragmented input, planar/interleaved output, exact alpha, held-image
rereads and final memory release. Lossless source words are exact; VarDCT source uncertainty
remains 2e-5 and is propagated through every LUT stage.

Primary intervals use independent f64 curve equations, branch extrema, matrix magnitudes,
CLUT gradients and Lab corner bounds. Native intervals separately account for integer
interpolation and the CMM's unclipped matrices/analytical curves. Native mask bit 5 marks
propagated departures from the unit-domain model, including the native offset curve's
zero-clamped branch threshold. Every record retains the native value, and every component
checks both its primary GPU interval and its separately derived native interval; no mask skips
an assertion. Neither intervals nor profile choices are fitted to GPU output. See the
[generator and reproducibility instructions](../crates/jxl_wgpu/test-data/icc_generator/README.md).

## Floating-point multi-process elements

The ordered stage representation replaces the fixed source-curves/matrix/target-curves ABI.
Matrix/TRC, profile-to-profile and linear RGB connections use this same interpreter. Adjacent
unclipped affine operations combine in host f64; exact identity matrices disappear, preserving
subnormal inputs before a discontinuous curve. Curve breakpoints use ordered IEEE bit keys so
GPU subnormal flushing cannot choose the wrong segment at zero. Positive and negative zero
compare equal, and the first segment containing a repeated breakpoint owns that input.

The MPE parser covers `matf`, `cvst` with all three formula forms and sampled segments, `clut`,
and the required `bACS`/`eACS` pass-through elements. It checks position-table ranges, alignment,
zero padding, whole-element sharing, execution order, channel continuity, valid stored numbers,
curve domains, sample counts and table products before payload allocation. Sampled segments
reconstruct their implicit initial metadata value from the preceding segment. No CPU image
samples or pixel CMS are used. Shared curve/element ranges retain shared immutable storage.
Stored `float32Number` values reject subnormals as well as infinities/NaNs (ICC.1 section 4.3);
computed subnormals remain valid inputs to the resident interpreter. An empty sampled segment
is never selected, but retains its final stored sample for the following segment's implicit
endpoint. Logarithmic metadata endpoints do not materialize a potentially overflowing power.
Power metadata endpoints retain the affine sum's f64 remainder through `log1p` and use `expm1`
when an outer constant cancels the unit term. This also covers minimum-normal increments
amplified by maximum F32 exponents; ordinary f64 evaluation would round the base to one.

MPE curves keep an F32 significand and separate integer exponent for intermediate arithmetic.
This avoids overflow in affine power bases, exponential multipliers and sampled interval widths.
Logarithmic curves combine signed terms in the logarithmic domain; small `log1p` increments
are retained before the outer scale. Near unity, a bounded atanh series avoids subtracting
rounded logarithms; the complementary exponential also uses a bounded series near cancellation.
Final conversion rounds subnormals with integer bits, and zero formula scales
avoid evaluating irrelevant powers. Curve evaluation remains entirely on the GPU.

Power curves preserve two scaled terms for their affine base. Integer significand multiplication
and addition retain product and sum remainders before a nonlinear exponent can amplify them;
this does not depend on [`fma` being fused](https://www.w3.org/TR/WGSL/#fma-builtin), which WGSL
does not guarantee. The near-unit logarithm uses the compensated difference, and a scaled
`exp2` increment retains small results after unit-offset cancellation. A tiny exponent uses
this increment even when its base is far from one. The program ABI and upload size are unchanged.

Default MPE limits are 4,096 processing elements, sixteen processing channels, 4,096 segments
per curve and 4,194,304 CLUT component values, in addition to the profile/tag/sample limits above.
The channel limit is a configurable metadata resource policy; it is not a normative limit on
all MPE matrices. The current resident backend separately admits at most sixteen live channels.
One-/two-dimensional CLUTs use linear/bilinear interpolation. Higher dimensions use tetrahedral
interpolation on the last three axes and linear interpolation on the preceding axes, matching
the native comparison policy. Float CLUT outputs and formula outputs retain signed and above-one values.

Physical Lab/XYZ stages connect differing PCS declarations. DToB3/BToD3 already use absolute
PCS: their media white is not applied again. Mixed absolute/relative endpoints apply only the
remaining conversion. Selected v4 LUT/MPE perceptual/saturation methods use the PCS reference black
(0.00336, 0.0034731, 0.0028646); unused TRC tags do not alter this selected-method policy. This
is explicit CMM policy, not a claim of parity with every CMM's heuristics for hybrid profiles.

The `mpe` corpus contains nine processing profiles and an identity connection profile. All
72 directional/intent connections have 135,864 independent f64 and Little CMS 2.19 components.
It covers all formula forms, sampled segments, repeated breakpoints, signed zeros/subnormals,
reversed physical storage, anisotropic 1–5D CLUTs, Lab PCS and fifteen intermediate channels.
Native Little CMS rejects a sixteen-channel intermediate; that native limit is not used as
an ICC format limit. Per-stage uncertainty includes affine coefficient magnitudes, curve branch
bounds, CLUT gradients and the operands before Lab cancellation; no output tolerance is fitted
to GPU results. All original matrix/TRC acceptance intervals remain unchanged.

Another 48 native/scalar connections consume the existing original RGB/Gray Modular/VarDCT
reference pixels and three MPE targets. Source uncertainty stays zero for original Modular and
2e-5 for original VarDCT. Their 22,032 components are checked through 192 decoder presentations
covering planar/interleaved output and whole/fragmented input, with exact alpha, held-output
rereads and final budget release. Native CMM fixed-PCS uncertainty is propagated separately;
it never enlarges the primary GPU interval. This native/scalar portion contains 132 files.

A separate 28-file `range` corpus adds thirteen profiles and 10,959 independent scalar components,
checked 65,754 times across both directions and all kernel variants. It includes full F32
extrema, computed subnormals, large intermediate powers/products, logarithmic cancellation and
small increments with large outer scales, zero scales, wide/narrow sampled intervals and empty
sampled segments. Identity values and rounded halves are exact; other intervals are derived
before GPU execution with explicit cancellation conditioning. Little CMS's finite substitutes
for infinite segment endpoints prevent using it as a full-range oracle. See the
[range equations and bounds](../crates/jxl_wgpu/test-data/icc_generator/README.md).

The 46-file `power` corpus adds twenty-two profiles and 72,864 independent scalar components,
checked 437,184 times across both directions and all kernel variants. It covers affine increments
amplified by large positive/negative exponents, exact-product remainders, signed integer powers,
tiny exponents, offset cancellation and four implicit sampled endpoints. Its radii are relative
to the result plus half a minimum-subnormal ULP, so loss of a small representable result cannot
hide inside an absolute unit-scale tolerance. All 160 earlier MPE reference files are unchanged.
This improves execution of representable results; it does not prove that every admitted formula
has a finite result across its entire segment. Complete formula-range validation, arbitrary
ill-conditioning, and full-range matrix/CLUT/Lab arithmetic remain open.

## Evidence and precision

The [offline generator](../crates/jxl_wgpu/test-data/icc_generator/README.md) creates ten profiles
and every ordered source/target pair: 100 transforms, each with 629 pixels, or 176,120 output
components. It includes v2/v4, independent RGB gammas/tables, all five parametric functions,
gray, alternate colorants/white, and inputs around transfer boundaries. Every oracle reopens the
exact serialized profile, including its fixed-point rounding. GPU readback also verifies binding
prefixes, tails and per-row/inter-plane guards.

Little CMS 2.19 supplies native references. A separate C++ f64 implementation uses Little CMS's
tag decoder, pivoted matrix elimination and analytical roots/exhaustive sample-segment inversion
to apply ICC.1:2022. It shares no parser, matrix inversion or GPU binary-search implementation
with the production code. Every component is checked against this independent reference.

Before the inverse curve, the F32 uncertainty is `4e-7 * (1 + magnitude + coefficient_sum)`, where
`magnitude = abs(offset) + sum(abs(matrix[c] * linear[c]))` and `coefficient_sum = sum(abs(matrix[c]))`.
The independently evaluated inverse at both ends of this interval, plus `2e-7` output rounding,
forms the acceptance interval. This expresses steep/flat curve conditioning without pretending
that a fixed output-code error is meaningful everywhere. In the current corpus the largest
absolute error is 0.0001 for offset→offset near black, where F32 cannot retain a tiny power added
to the offset. Other pairwise maxima are below 1.8e-6. These are F32 results, not exact encoded-word
preservation; original numeric decoder output must retain its existing exact bypass.

175,851 native components also satisfy an independently propagated native precision interval.
The additional native allowance is `3/65535 * coefficient_sum` in the intermediate domain,
plus `2/4095` at output for sampled inverse tables (otherwise `2e-7`). It accounts for Little CMS's
16-bit sampled curves and 4,096-entry reverse approximation. The GPU uses the narrower F32
interval above. Native values for the other 269 components are retained and marked by cause:
56 lose the nonzero offset at a zero power base; 213 clamp a negative value before an offset
inverse to device zero. The scalar/GPU result still has to meet ICC's boundary rules. No marked
component is excluded from the primary GPU assertion.

Additional GPU tests cover both monotone directions, exact plateau endpoint rules, all parametric
inverse branches, clipped plateaus/gaps, metadata reuse after abandoned commands, Scalar/Lanes32/
Tile16x16 dispatches, multiple extents/pitches, exact program limits and invalid bindings.
The shader is Naga-validated without optional capabilities and its 320-byte storage record is checked
against the parsed WGSL layout.

The `linear` subcorpus adds 100 connections between the ten original profiles and five linear
spaces (BT.709, BT.2020, Display P3, equal-white BT.709 and the native JPEG XL calibrated RGB
coordinates). It checks all 182,410 output components using independent f64 CIE/Bradford geometry,
with primary normalization and pivoted elimination distinct from production. Native ICC execution
uses Little CMS's XYZ double interface, with the independent matrix on the linear side, so no
fixed-point surrogate profile changes the requested endpoint. Inputs include mixed signed and
above-one RGB values. Primary uncertainty remains `4e-7 * (1 + magnitude + coefficient_sum)`
before the target inverse (or linear identity), plus `2e-7` at output.

On Metal the linear corpus has maximum absolute error below 8.6e-7. It retains 1,228 negative
and 294 above-one output components outside a 1e-4 boundary margin. Native semantics agree for
180,494 components; 30 zero-base offsets and 1,886 negative offset inverses carry the previously
documented masks and still pass every primary scalar/GPU assertion. The original 121 corpus
files remain byte-identical. The complete corpus now has 223 files and 358,530 checked components.

## Matrix/TRC rendering-intent evidence

The `intents` corpus adds 26 RGB/Gray v2/v4 profiles and 2,704 ordered profile/intent connections,
with 827,424 output components. It covers tinted input media whites, legacy v2 display white,
sampled black levels including Lab lightness above 50 and 95, chromatic black, and parametric
offsets/clipped plateaus. A further 1,040 bidirectional connections to all five linear spaces check
397,800 components, including signed and above-one RGB. Every GPU component uses the same
independent pre-inverse precision interval; native precision never relaxes it.

In addition to the original two native boundary masks, bit 2 marks the native inverse's
extrapolation past a unit-range endpoint (including its existing native uncertainty); bit 3 marks
native forward evaluation omitting unit-range clipping; bit 4 marks black compensation affected
by Little CMS dropping the type-3 offset at a zero power base. Native values are retained in every
record. Profile pairs have 767,812 unmarked native components; linear pairs have 378,914. Every
marked component still undergoes the full primary scalar/GPU check. ICC.1:2022 section 10.18
requires parametric function domain and range clipping to [0,1].

The independent analytical inverse scores a valid root by its solved value rather than a rounded
forward re-evaluation. Otherwise a one-ULP residual could incorrectly prefer x=1 over the first
point of a clipped final plateau. Generator regressions cover 31 offsets. All 1,015 pre-existing
resident/embedded/YCbCr/XYB/alpha reference files remain unchanged under their documented build flags.

Decoder references add 192 original/XYB-to-ICC connections from the eight existing native streams
to six targets with different media white, nonzero/chromatic black and v2 policy. The old source
bounds are propagated through all source-box corners: original Modular is exact, original VarDCT
uses 2e-5, and native XYB uses `(1 + abs(linear)) / 1024`. Tests check 768 presentations across both
layouts and whole/fragmented input, unchanged alpha, held outputs and final budget release. The
independent intervals also distinguish each non-relative intent from relative in applicable cases.
Selected nonfinite MPE rejection and unused-original-method bypass tests retain their original protection.

The normative references are [ICC.1:2022](https://www.color.org/specification/ICC.1-2022-05.pdf),
sections 7, 8.10, 10.6, 10.16, 10.18 and Annex F. The observed native boundaries follow Little CMS 2.19
[`cmsgamma.c`](https://github.com/mm2/Little-CMS/blob/lcms2.19/src/cmsgamma.c), cases 3 and -2.
The declared matrix-shaper CMM connection policy is independently checked against Little CMS 2.19
[`cmscnvrt.c`](https://github.com/mm2/Little-CMS/blob/lcms2.19/src/cmscnvrt.c),
[`cmssamp.c`](https://github.com/mm2/Little-CMS/blob/lcms2.19/src/cmssamp.c) and
[`cmsio1.c`](https://github.com/mm2/Little-CMS/blob/lcms2.19/src/cmsio1.c).
