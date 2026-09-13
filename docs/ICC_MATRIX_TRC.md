# Resident ICC matrix/TRC conversion

The metadata and resident GPU execution layers support RGB matrix/TRC and XYZ gray profiles.
JPEG XL decoding now admits embedded ICC for unfiltered original Modular numeric samples and
independent extra-channel output in the supported single-frame paths through both codecs. Codec
reconstruction and LF configuration are independent of color conversion. The common decoder now
also handles original and XYB ICC RGB/Gray color surfaces, including YCbCr reconstruction,
original-domain references and composition, relative matrix/TRC conversion and U8/F32 requested output.
Broader ICC XYB conformance, enumerated-source-to-ICC conversion, spot rendering, full intents
and HDR mapping remain open.
This checkpoint does not change the full JPEG XL support claim.

YCbCr reconstruction carries the actual original device profile through the inverse codec matrix;
it does not select an ICC transform or introduce an enumerated RGB carrier. The resulting surface
has one Gray or three RGB planes before reference storage and composition. Same-profile requests
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
this boundary. [XYB generator and precision](../crates/jxl_wgpu_decode/test-data/embedded_icc_xyb_generator/README.md).

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
planes, output words, the 80-byte ICC uniform and 208-byte packing uniform are all accounted.
No frame readback or CPU pixel CMS is involved. `GpuOutputRequest::with_icc_rendering_intent`
defaults to relative colorimetric; non-Bradford conversion and unimplemented intents return errors.
Exact same-profile packing does not select a CMS method and therefore does not require an
executable matrix/TRC or inverse curve.

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

`IccProfile::matrix_trc` selects a direction and explicit intent. Higher-priority DToB/BToD or
AToB/BToA tags produce a typed unsupported error, including the perceptual-LUT fallback when the
requested LUT is absent. They are never silently discarded in favour of colorants/TRCs.
Only media-relative colorimetric intent is executed at this checkpoint. RGB input/display and
monochrome input/display/output classes with XYZ PCS are supported. Lab, CMYK, other profile
classes, LUTs, absolute/perceptual/saturation policies and black-point compensation remain open.

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

## Linear RGB connections

`IccTransform::to_linear_rgb` and `from_linear_rgb` connect the selected profile to a declared
`RgbColorSpace`. The endpoint model distinguishes a profile's one/three device channels and
curves from three unbounded linear RGB components. Relative colorimetric intent uses Bradford
between the RGB reference white and ICC's exact encoded PCS D50. Colorants are connected directly
in f64; no synthetic quantized profile or approximation by recognized primaries is introduced.

`ColorMatrix` in the backend-neutral protocol owns the shared CIE geometry and white adaptation
calculation. Existing RGB output/display lower this same implementation to F32. ICC connections
combine the profile colorants and RGB/PCS matrix in f64 before a single F32 lowering. The host
calculates metadata matrices only. The shader evaluates all pixel curves and matrix products.

The linear endpoint has no ICC curve descriptor and applies no unit clipping. A negative input
or input above one reaches the PCS matrix unchanged. A linear output can remain negative or
above one after conversion from a wider gamut. The profile endpoint still follows the bounded
ICC curve contract. This does not extend arbitrary ICC device curves to unbounded/HDR domains.

## GPU contract

`ResidentIccMemoryPlan::new` checks capability and program bounds before allocation.
`ResidentIccProgram::new` uploads immutable, deduplicated curve metadata once.
`ResidentIccPipeline::encode` records conversion between one or three planar F32 color channels
in distinct resident storage buffers. Offsets are scalar indices relative to their binding;
each plane has its own row stride. Alignment, usage, extents, channel counts, storage capacity,
non-overlapping output ranges, u32 addressing and workgroup counts are checked before dispatch.
Padding, alpha and extra planes remain outside the color views and are not written.

The caller admits the reported program and dispatch allocations and retains their handles through
GPU completion, as with the other resident codec primitives. This layer neither submits nor maps
image buffers. Recorded work can be abandoned; the same program can be reused across frame extents,
pitches and later submissions. Queue/session integration must keep the existing memory permits
until the last submitted consumer completes.

Inputs must be finite. ICC device-domain and curve-range values are clamped to [0,1]; matrix
intermediates preserve signed XYZ-derived values before inverse-curve domain clipping. This is an
explicit bounded ICC contract, separate from the existing unbounded enumerated SDR conversion.
No HDR or unbounded ICC behaviour is claimed. Matrix lowering and pixel arithmetic use F32.
A legal parametric power that exceeds finite F32 evaluation returns `ResidentIccError::Precision`.

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
`magnitude = sum(abs(matrix[c] * linear[c]))` and `coefficient_sum = sum(abs(matrix[c]))`.
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
The shader is Naga-validated without optional capabilities and its 80-byte uniform is checked
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

The normative references are [ICC.1:2022](https://www.color.org/specification/ICC.1-2022-05.pdf),
sections 7, 8.10, 10.6, 10.18 and Annex F. The observed native boundaries follow Little CMS 2.19
[`cmsgamma.c`](https://github.com/mm2/Little-CMS/blob/lcms2.19/src/cmsgamma.c), cases 3 and -2.
