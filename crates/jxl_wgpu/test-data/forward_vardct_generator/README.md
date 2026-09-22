# Native forward VarDCT references

This offline generator calls libjxl's `TransformFromPixels`, `DCFromLowestFrequencies`,
`ComputeNaturalCoeffOrder` and `DequantMatrices::Matrix` directly. It requires libjxl 0.12.0
commit `a7a9c787341cf703dede03c2009fa460cae5e5df`. Production crates never link this generator
or read these fixtures.

The three committed outputs are:

| File | SHA-256 |
|---|---|
| [`forward_vardct.bin`](../forward_vardct.bin) | `191c0f4d4be583bfcd3c4c781b71884ce66246831c485d8dc73d91b58109ad51` |
| [`vardct_metadata.bin`](../../../jxl_gpu_protocol/test-data/vardct_metadata.bin) | `696d7dc023349024af2f97d3210bd2a3729c4cd0b9df5be706d1d2a7829ff0c9` |
| [`parametric_matrices.bin`](../../../jxl_gpu_protocol/test-data/parametric_matrices.bin) | `f900ca5cf7f270866ae88014547006aff8047b770e67b348d8315b3460db5deb` |

## Reproduction

From the repository root, with CMake, a C++17 compiler and system Highway, Brotli and LCMS2:
reuse an existing checkout at the pinned revision and its build directory when available.
The paths below describe a first setup.

```sh
mkdir -p .git/codex-validation/forward-native
git clone https://github.com/libjxl/libjxl.git .git/codex-validation/forward-native/libjxl
git -C .git/codex-validation/forward-native/libjxl checkout --detach a7a9c787341cf703dede03c2009fa460cae5e5df
cmake -S crates/jxl_wgpu/test-data/forward_vardct_generator \
  -B .git/codex-validation/forward-native/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DJXL_SOURCE="$PWD/.git/codex-validation/forward-native/libjxl"
cmake --build .git/codex-validation/forward-native/build \
  --target generate_forward_vardct generate_parametric_matrices -j 4
.git/codex-validation/forward-native/build/generate_forward_vardct \
  .git/codex-validation/forward-native/forward_vardct.bin \
  .git/codex-validation/forward-native/vardct_metadata.bin
.git/codex-validation/forward-native/build/generate_parametric_matrices \
  .git/codex-validation/forward-native/parametric_matrices.bin
cmp crates/jxl_wgpu/test-data/forward_vardct.bin .git/codex-validation/forward-native/forward_vardct.bin
cmp crates/jxl_gpu_protocol/test-data/vardct_metadata.bin .git/codex-validation/forward-native/vardct_metadata.bin
cmp crates/jxl_gpu_protocol/test-data/parametric_matrices.bin .git/codex-validation/forward-native/parametric_matrices.bin
```

Homebrew builds additionally use `-DCMAKE_PREFIX_PATH=/opt/homebrew`. CMake enforces the source
commit and compiles the generator with `-Wall -Wextra -Werror -ffp-contract=off`. Outputs must not
already exist. The recorded hashes were generated on Apple silicon with libjxl's native SIMD
dispatch; another compiler/architecture may differ in floating-point rounding and must be audited
before replacing references. Regeneration on the recorded environment must be byte-exact.

## Binary schemas

All words are little-endian u32; f32 values use their IEEE-754 bit representation. There is no
implicit alignment or record padding. Strategy IDs are in standard codestream order, 0 through 26.

`forward_vardct.bin` begins with eight bytes `JXLFWD01`, then record count 667. Each record has
`strategy, test, width, height`, then three `width*height` coefficient planes in X/Y/B order,
then three `(width/8)*(height/8)` spatial LF planes. Coefficients use native canonical transform
buffer order; regular square/tall DCTs transpose spatial-frequency coordinates into that order.

Every strategy has test 0: X is `((37*x + 101*y + 3*x*y) % 509 - 254) / 256`, Y is constant
0.375, and B has one -0.75 impulse at `(width-1, height/2)`. All ten strategies with an 8×8
footprint additionally have tests 1 through 64. For `p = test-1`, their only nonzero samples are
X +1 at p, Y -0.5 at `(p+17)%64`, and B +0.25 at `63-p`. Thus the special transforms and DCT8
have complete independently generated basis coverage, including asymmetric inputs in all channels.

`vardct_metadata.bin` begins with `JXLQNT01`, then count 27. Each record has `strategy, area`,
`area` natural-order indices, then three `area` default-dequantization f32 planes in X/Y/B order.
Native unused LLF matrix entries are preserved verbatim, including the DCT2 DC sentinel. Only AC
entries are compared as quantization multipliers; LLF uses the separate LF quantizer.

`parametric_matrices.bin` begins with `JXLPQM01`, then count 4. Each record has
`mode, variant, parameters_per_channel`, then X/Y/B parameter words containing binary16 bits,
then three 64-entry dequantization f32 planes. Modes 1/2 have 3/6 parameters per channel.
Variant 0 uses `0x3c00` (one); variant 1 uses `0x4000 + ((channel + index) % 3) * 0x400`
(two, four or eight). The generator writes these parameters literally into family 0, leaves
the other sixteen families at default, and calls `DequantMatrices::Decode` and `EnsureComputed`.
These four records occupy 3,348 bytes. They independently establish the ×64 scale applied to
Hornuss/DCT2 wire parameters; the older `jxl-vardct` parser omits this scale. The unused DC value
is preserved, including the native DCT2 sentinel, and is excluded from AC comparisons.

## Validation and limits

```sh
cargo test -p jxl_gpu_protocol vardct:: -- --test-threads=2 --nocapture
cargo test -p jxl_wgpu forward_vardct:: -- --test-threads=2 --nocapture
cargo test -p jxl_wgpu_encode --lib vardct_encoder::tests::single:: -- --test-threads=2 --nocapture
```

The latter two commands require an actual GPU; the encoder test also requires `djxl` in PATH.
Run GPU test executables sequentially with two libtest threads each. The shared forward primitive checks all 667 records under Scalar,
32/64/128/256 lanes, demanding byte-identical variants and coefficient error at most
`2e-6*(1+abs(reference))`, LF error at most `2e-5*(1+abs(reference))`. The recorded peak absolute
errors are 1.1920929e-7 and 8.9386594e-7. Offset bindings, independent row strides and scalar
origins, poisoned padding, output guards, allocation sizes and rejected overlapping/invalid
bindings are also exercised. Actual devices requested with only four storage bindings or a
64-byte uniform-binding cap must return typed errors before pipeline compilation. A scalar DCT8
requires a 2-D dispatch: a four-workgroup axis cap is rejected before recording, while eight
workgroups per axis execute the full 64-sample transform successfully.

All 27 natural orders match exactly. Default AC matrix relative error is bounded by 3e-6;
the observed peak is 2.3667124e-6. This comparison caught incorrect Y/B base constants for the
128×256 family in the previous decoder dependency. The shared metadata now uses the native
values 22389.441 and 11679.847 for those channels in both orientations.

The encoder tests use textured RGB8 inputs for every strategy with default and explicit LF/HF
correlation, natural/custom orders, and parametric/raw matrices: 162 streams. Independent f64 color
conversion and cosine sums (or the native impulse basis for 8×8 transforms), native default
matrices/orders, independent parametric matrices and scalar raw sample/denominator products check every compressed AC coefficient
within one integer quantizer step. Rust `jxl`, native `djxl` and the stock GPU decoder agree within
one RGB8 code; all five workgroup variants emit identical bytes. A dense maximum-range 256×256
fragment fills its final allocated word, and a 500 KiB device binding limit rejects oversized
matrix/order storage before submission.

Parametric modes 1/2 compare every AC entry exactly against the native records above; modes
3–6 compare bit for bit against the independent `jxl-vardct` parser, including 1/3/16 bands,
all compatible families and both rectangular orientations. Equivalent constant mode-1/2/6
streams also check native, Rust and GPU pixels to prevent encoder/decoder agreement from hiding
a shared scaling error. The [procedural corpus](../../../../docs/CONFORMANCE_CORPUS.md#procedural-vardct-encoder-matrix)
records the mixed-map, tiled, malformed-parameter and ownership cases.

These are regression and interoperability bounds fixed before running the tests. They do not
establish ISO precision, perceptual distance, rate control, content-adaptive strategy/matrix
selection, adaptive entropy or progressive encoding. Raw-matrix encoding has separate GPU entropy,
malformed-input and ownership evidence in the procedural corpus. Spectral/quantized AC encoding has
separate coefficient, intermediate-image and ownership gates in the
[progressive encoder corpus](../../../../docs/CONFORMANCE_CORPUS.md#progressive-vardct-encoding).
The full JPEG XL goal remains open.
