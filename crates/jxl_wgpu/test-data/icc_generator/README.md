# ICC matrix/TRC references

`main.cpp` uses Little CMS **2.19**. It serializes ten generated RGB/gray profiles, fixes their
timestamp and clears their optional profile ID, reopens the exact bytes, and converts all 100
ordered pairs with relative colorimetric intent, `NOOPTIMIZE | NOCACHE`, and no black-point
compensation. Float native output is clipped to the declared unit output range.

`tools/jxl_test_support/native/icc/scalar.hpp` independently evaluates the ICC.1:2022 matrix/TRC equations in f64, using Little CMS
only to read the profile's exact colorant/curve metadata. It uses pivoted elimination and
analytical roots or exhaustive sampled-segment search. Production Rust/WGSL code is not called.
See [the execution contract](../../../../docs/ICC_COLOR.md) for scope, precision intervals
and the two documented Little CMS boundary differences.

The deterministic corpus lives in `../icc`:

- `manifest.json`: dimensions (37×17), ordered profile names and channel counts.
- `<name>.icc`: exact ICC profile bytes; `gamma_v2` is v2.4, the others v4.3.
- `<name>_input.f32le`: interleaved input samples in row order.
- `<source>_to_<target>.reference`: one 28-byte little-endian record per interleaved output
  component: six F32 values (`native`, `exact`, `lower`, `upper`, `native_lower`, `native_upper`),
  then one u32 native-semantics mask. Bit 0 marks the dropped offset at a zero power base;
  bit 1 marks negative input to the offset inverse. Zero means native semantics agree.

The original native values remain in every record, including the marked boundary differences.
All components are always asserted against the independent scalar interval. Neither oracle nor
Little CMS is a production dependency. The unit-domain contract is not an embedded-ICC decoder
admission claim.

From the workspace root, build with a C++17 compiler and the pinned `lcms2` development package:

```sh
mkdir -p .git/icc-regenerate
c++ -std=c++17 -Wall -Wextra -Werror \
  -Itools/jxl_test_support/native \
  crates/jxl_wgpu/test-data/icc_generator/main.cpp \
  $(pkg-config --cflags --libs lcms2) -o .git/icc-regenerate/generator
.git/icc-regenerate/generator .git/icc-regenerate/corpus
diff -rq -x intents crates/jxl_wgpu/test-data/icc .git/icc-regenerate/corpus
cargo test -p jxl_wgpu --test icc -- --test-threads=1
```

Generation is offline. `cargo test` consumes the checked-in bytes and does not spawn an oracle.

The shared `icc/linear.hpp` adds the `linear` subdirectory with 100 bidirectional connections between these ten
profiles and five linear RGB spaces. Its CIE/Bradford calculation uses normalized primaries and
pivoted elimination, independently of production's homogeneous geometry. Native ICC execution
uses `TYPE_XYZ_DBL` on the PCS side plus the independent matrix on the linear side, without
serializing a surrogate linear RGB profile. Linear input/output is unbounded; only ICC device
curves use the unit-domain contract.

- `linear/manifest.json` records all five exact white/primary declarations.
- `linear/input.f32le` contains 629 three-channel inputs including signed/above-one values.
- `linear/<profile>_to_<space>.reference` and the reverse name use the same 28-byte records.
  Forward inputs reuse the original `<profile>_input.f32le` files.

All 182,410 additional components have primary scalar intervals and native references with the
same precision and semantics-mask policy. The generator still reproduces every original file
byte for byte; the complete corpus contains 223 files. The build/regeneration command above
reproduces both parts together.

## All matrix/TRC rendering intents

`intents.cpp` uses the same pinned Little CMS 2.19, with independent metadata equations in
`tools/jxl_test_support/native/icc/intents.hpp`. Its 26 profiles cover RGB/Gray, v2.4/v4.4,
tinted input media whites, legacy display white, nonzero sampled/chromatic black, high-Lab
black and parametric clipped/offset boundaries. All 2,704 profile/intent pairs and 1,040
bidirectional linear RGB connections use the existing independent precision intervals.
The five linear-space declarations remain in `../icc/linear/manifest.json`.

The additional 192 decoder references read the existing embedded native device/alpha and XYB
linear files. Six requested profiles exercise white scaling, v4 black compensation and v2
policy through all four intents. Source precision is propagated through every error-box corner;
no new fixed output tolerance is introduced. The generator never reads GPU-produced pixels.

From the workspace root:

```sh
c++ -std=c++17 -Wall -Wextra -Werror -ffp-contract=off \
  -Itools/jxl_test_support/native \
  crates/jxl_wgpu/test-data/icc_generator/intents.cpp \
  $(pkg-config --cflags --libs lcms2) -o .git/icc-regenerate/intents
.git/icc-regenerate/intents crates/jxl_wgpu_decode/test-data/embedded_icc \
  .git/icc-regenerate/intents-corpus
diff -rq crates/jxl_wgpu/test-data/icc/intents .git/icc-regenerate/intents-corpus
```

The output directory must be new. Its 3,990 files contain 26 profiles, 26 profile inputs, a
manifest, 2,704 profile-pair records, 1,040 linear-connection records and one linear input,
plus 192 decoder records. Every record retains the original seven-field layout. Additional
native masks are bit 2 (inverse endpoint extrapolation), bit 3 (forward range clipping) and
bit 4 (the known zero-base offset error affecting black metadata). Every GPU component is
still checked against its primary independent interval. See the execution contract for details.

The analytical inverse's clipped-endpoint regression covers 31 exact offset curves. Re-evaluating
an analytical root in floating point must not discard it in favour of a distant endpoint.
All 223 original files and all 792 earlier embedded/YCbCr/XYB/alpha files reproduce unchanged
under their respective documented compiler options. In particular, retain the original generator's
compiler contraction setting: `linear/input.f32le` itself contains multiply/subtract probes.

## Floating-point MPE programs

`mpe.cpp` independently constructs ICC.1:2022 processing elements, including reversed physical
storage and repeated curve breakpoints, and evaluates their mathematical stages in C++ f64.
Little CMS 2.19 reopens each exact profile and executes every native connection with
`NOOPTIMIZE | NOCACHE`. No production parser, GPU shader or GPU output participates in generation.

```sh
c++ -std=c++17 -Wall -Wextra -Werror -ffp-contract=off \
  -Itools/jxl_test_support/native \
  crates/jxl_wgpu/test-data/icc_generator/mpe.cpp \
  $(pkg-config --cflags --libs lcms2) -o .git/icc-regenerate/mpe
.git/icc-regenerate/mpe crates/jxl_wgpu_decode/test-data/embedded_icc \
  .git/icc-regenerate/mpe-corpus
diff -rq crates/jxl_wgpu/test-data/icc/mpe .git/icc-regenerate/mpe-corpus
```

The native/scalar portion contains 132 files: ten profiles, one 629-pixel RGB input, a manifest,
72 native/scalar directional/intent references and 48 decoder references. Records retain the
same seven-field layout; all native semantics masks are zero. The nine processing profiles cover
matrices, all three MPE formula forms, implicit sampled endpoints, repeated breakpoints,
1–5D anisotropic CLUTs, Lab PCS and fifteen intermediate channels. Both signs of zero and
subnormal boundary probes are retained. A sixteen-channel native intermediate is outside
Little CMS 2.19's admitted range, independently of the resident backend's sixteen-channel limit.

Primary intervals propagate F32 arithmetic from each stage: affine magnitudes and input bounds,
curve branch endpoints, CLUT gradients, and Lab operands before cancellation. Large extended-Lab
values therefore have magnitude-dependent absolute errors. Decoder source bounds remain zero
for original Modular and 2e-5 for original VarDCT. The separate native CMM interval includes
`3/65535 * (1 + abs(PCS))` before MPE evaluation for its fixed-PCS/white/curve uncertainty; this
never widens a primary GPU interval. [The execution contract](../../../../docs/ICC_COLOR.md)
defines the supported scope and remaining conformance gates.

The `range` subdirectory adds 28 files (160 total): thirteen processing profiles, a 281-pixel
RGB input, a manifest and thirteen independent scalar references. Each reference record stores
two little-endian f64 values: the mathematical result and its absolute error radius. These
probes deliberately exceed Little CMS 2.19's `-1e22`/`+1e22` substitutes for unbounded curve
endpoints in [`cmstypes.c`](https://github.com/mm2/Little-CMS/blob/lcms2.19/src/cmstypes.c).
They have no native-CMM comparison claim; the earlier 132 files and their native assertions
remain unchanged. The scalar equations use f64 `log1p`, `sqrt` and `exp2`, independently of
the production GPU's separated significand/exponent arithmetic.

The 10,959 scalar components cover signed zero, computed subnormals, normal exponent bins,
both finite extrema, large powers/products, logarithm increments/differences with large outer
scales, zero scales, sampled domains spanning both extrema or only the minimum normals, and
empty sampled segments. Identity has zero radius; halving allows half the minimum subnormal
and separately checks rounding to even. Other radii are
`16 * 4e-7 * (1 + max(abs(result), conditioning_magnitude))`, a fixed F32 arithmetic budget
for these equations, including transcendental lowering and interpolation. The wide sampled curve uses
`MAX_F32` as its conditioning magnitude, accounting for cancellation of large endpoint values;
it does not promise small relative output error near the midpoint. All radii are calculated
before GPU execution. Tests require finite results in both directions and all three kernel
variants, checking 65,754 components and the existing buffer guards.

The `power` subdirectory adds 46 files (206 MPE files total): twenty-two profiles, a 1,104-pixel
RGB input, a manifest and twenty-two scalar references using the same two-f64 record layout.
The inputs retain every earlier range probe and add dense segment interiors and neighboring
F32 values around cancellation points. Reduced equations use independent f64 `log1p`, `expm1`
and `exp`; products of two F32 inputs are exact in f64. In particular,
`(1 + 2^-23) * (1 - 2^-23) = 1 - 2^-46` raised to `2^46` approaches `exp(-1)`.
The ordinary rounded F32 product would incorrectly produce one.

Other cases amplify a minimum-normal affine increment with either sign of the maximum F32
exponent, preserve positive/negative-base odd/even/reciprocal powers, cancel the final unit
offset, and retain tiny exponents on bases spanning the positive F32 range. Separate affine
and outer constants test cancellation after an exact product. Four sampled curves test implicit
metadata endpoints after the same operations. Their independent initial samples round to F32
before interpolation, matching the stored resident sample format.

Every new radius is `16 * 4e-7 * abs(result) + MIN_SUBNORMAL / 2`, fixed before GPU execution.
It scales with the result, rather than the large operands that cancel, so returning zero for
a small representable result fails. No older radius or reference is changed. The 72,864 scalar
components are checked 437,184 times across both directions and all three kernel variants,
including buffer guards. These are scalar comparisons; they do not claim native CMM coverage
of these extreme profiles or arbitrary ill-conditioned power/offset combinations.

## Legacy integer LUT programs

`lut.cpp` constructs `mft1`, `mft2`, `mAB` and `mBA` bytes and independently evaluates their
ordered stages in f64. `lut/profile.hpp` defines profile/tag layout; `curve.hpp` and `stages.hpp`
define scalar equations and propagated intervals. The ordinary headers share no production
Rust/WGSL implementation. Little CMS **2.19** reopens each exact profile and executes native
conversions with `NOOPTIMIZE | NOCACHE`.

The 436 resident files contain 41 profiles (33 v4, eight v2), both 221-pixel input directions,
a manifest and 312 directional/intent references. All four A/B stage combinations, reverse
physical ordering, shared curve sets/suffixes, table precisions and XYZ/Lab PCS are represented.
Device spaces include Gray, RGB, CMYK, 2CLR, 5CLR and FCLR. Little CMS's CMYK/5–15-channel float
formatters use percentages; corpus device inputs remain normalized unit values. V2 source LUT
perceptual/saturation connections to v4 are covered by the dedicated source-black corpus below;
the original resident files retain their previous directional/intent coverage unchanged.

`lut_decoder.cpp` uses libjxl **0.12.0** to encode/decode 24 original-color 17×9 images with the
exact resident LUT profiles. Both source and data profile queries must preserve the ICC bytes.
Twelve RGB/Gray × XYZ/Lab × LUT-format cases run through Modular and VarDCT, then request the
next LUT format and the opposite RGB/Gray/PCS space with all four intents. Original Modular words
and all alpha words must equal input bits. Native color conversion consumes decoded original
samples; an output-profile label is never used as a color oracle. The 145 decoder files contain
24 `.jxl` streams, 24 native `.f32le` images, 96 references and a manifest.

The A/B fixture matrix keeps primary Y nonzero so libjxl can represent the extracted
primary chromaticities. Negative X, clipped matrix/curve
intermediates and distinct intent curves remain exercised by independent/native/GPU checks.

```sh
c++ -std=c++17 -Wall -Wextra -Werror -ffp-contract=off \
  crates/jxl_wgpu/test-data/icc_generator/lut.cpp \
  $(pkg-config --cflags --libs lcms2) -o .git/icc-regenerate/lut
.git/icc-regenerate/lut crates/jxl_wgpu/test-data/icc/mpe/identity.icc \
  .git/icc-regenerate/lut-corpus
c++ -std=c++17 -Wall -Wextra -Werror -ffp-contract=off \
  crates/jxl_wgpu/test-data/icc_generator/lut_decoder.cpp \
  $(pkg-config --cflags --libs libjxl lcms2) -o .git/icc-regenerate/lut-decoder
.git/icc-regenerate/lut-decoder .git/icc-regenerate/lut-corpus \
  .git/icc-regenerate/lut-corpus/decoder
diff -rq crates/jxl_wgpu/test-data/icc/lut .git/icc-regenerate/lut-corpus
```

Every `.reference` retains the existing six-F32/u32 layout. Primary scalar intervals use an
F32 arithmetic budget `epsilon = 4e-7`: affine coefficient/input magnitudes, curve endpoint
and breakpoint extrema with `8*epsilon*(1+abs(value))`, CLUT per-axis gradients plus
`16*epsilon*dimensions`, and every Lab error-box corner plus cancellation operand magnitudes.
The decoder adds only the existing source uncertainty (zero for Modular, 2e-5 for VarDCT).

Native scalar evaluation separately models the unclipped matrices and analytical curves in
Little CMS [`cmslut.c`](https://github.com/mm2/Little-CMS/blob/lcms2.19/src/cmslut.c) and
[`cmsgamma.c`](https://github.com/mm2/Little-CMS/blob/lcms2.19/src/cmsgamma.c), including the
offset curve's zero-clamped threshold. Native intervals add half a 16-bit quantization step
at sampled-curve input/output and CLUT input, plus `dimensions/(2*65535)` for nested integer
CLUT interpolation. Mask bit 5 propagates any departure from the primary clipped equations.
Both native and GPU intervals are asserted for every component, including marked records.
Original native values are retained, no GPU pixels enter the generator, and no primary interval
is widened to cover a different CMM policy. The 202,436 resident and 29,376 decoder components
produce 607,308 and 117,504 GPU comparisons respectively. All earlier corpora remain unchanged.

## V2 LUT source-black detection

`lut_black.cpp` adds 561 resident files: 20 exact v2 profiles, 20 input planes, 40 source-black
references, 480 color references and one manifest. `lut/black.hpp` independently evaluates the
selected LUT at its device endpoint, applies the darker-colorant Lab policy and connects the
result to target black. Gray/RGB, CMYK and unavailable 5CLR estimates cover both table precisions
and both PCS domains; low-floor and above-95-lightness profiles exercise clipping/reset behavior.
Five unbounded linear RGB spaces and a v4 identity MPE target run through all four intents.
Native Little CMS **2.19** directly checks 120 detected-black components and 318,240 converted
color components with `NOOPTIMIZE | NOCACHE`.

The policy follows [`cmssamp.c`](https://github.com/mm2/Little-CMS/blob/lcms2.19/src/cmssamp.c)
and [`cmscnvrt.c`](https://github.com/mm2/Little-CMS/blob/lcms2.19/src/cmscnvrt.c).
Independent primary/native intervals retain their separate stage rules. Black lightness bounds
enclose both branches if they cross the strict L*>95 reset. For each component, connection
bounds evaluate every corner of `target + (D50-target)*(input-black)/(D50-black)`; an interval
containing a singular denominator is rejected. F32 operation uncertainty adds
`16*epsilon*(1+abs(value)+abs(scale*input)+abs(scale*black))`, with the existing `epsilon=4e-7`.
No GPU samples determine these intervals, and mask bit 5 never suppresses either comparison.

`lut_black_decoder.cpp` adds 157 files under `black/decoder`: 24 exact-profile original RGB/Gray
Modular/VarDCT streams, 24 decoded native F32 images, 96 conversion references, twelve v4 target
profiles and a manifest. Targets use the opposite color count and PCS. Both codecs preserve
exact alpha; Modular source words are bit-exact, and VarDCT uses the existing independently
propagated 2e-5 source uncertainty. Native libjxl is **0.12.0**.

```sh
c++ -std=c++17 -Wall -Wextra -Werror -ffp-contract=off \
  -Itools/jxl_test_support/native \
  crates/jxl_wgpu/test-data/icc_generator/lut_black.cpp \
  $(pkg-config --cflags --libs lcms2) -o .git/icc-regenerate/lut-black
.git/icc-regenerate/lut-black crates/jxl_wgpu/test-data/icc/mpe/identity.icc \
  .git/icc-regenerate/black-corpus
c++ -std=c++17 -Wall -Wextra -Werror -ffp-contract=off \
  crates/jxl_wgpu/test-data/icc_generator/lut_black_decoder.cpp \
  $(pkg-config --cflags --libs libjxl lcms2) -o .git/icc-regenerate/lut-black-decoder
.git/icc-regenerate/lut-black-decoder .git/icc-regenerate/black-corpus \
  .git/icc-regenerate/black-corpus/decoder
diff -rq crates/jxl_wgpu/test-data/icc/black .git/icc-regenerate/black-corpus
```

Two clean regenerations reproduce all 718 files byte-for-byte. The shared recipe's default
parameters also reproduce all 581 original LUT files unchanged. Resident tests compare 954,720
components through three kernel variants and validate 576 actual GPU preparation statuses.
Decoder tests compare 117,504 components through 384 presentations, with whole/fragmented
transport, both layouts, retained frames and final memory release. Separate GPU tests reject a
singular source-black connection without image writes; completion tests retain typed errors
and cover shared program reuse, exact admission, failed poll/wait and cancellation.
