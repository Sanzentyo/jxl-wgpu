# WGSL ABI and memory audit

This document is the cross-crate audit record for every WGSL module shipped by this workspace. It
covers the Rust/WGSL ABI, buffer-copy alignment, binding and dispatch bounds, workgroup-local
storage, and explicit per-job/concurrent memory accounting. The audited implementation uses
`wgpu 30.0.1`.

The word *bytes* below means explicit application-visible buffer bytes. Pipeline objects, shader
compiler allocations, texture tiling/compression selected by the driver, command buffers, and
other driver-private allocations cannot be measured portably and are not included.

## ABI rules

- Every Rust value copied into a uniform or structured storage buffer is `#[repr(C)]`, derives
  `bytemuck::Pod` and `Zeroable`, and has compile-time size/alignment assertions plus a field-order
  test. Uploads use `bytemuck::bytes_of` or `cast_slice`; fixed readback records use
  `try_cast_slice`. Scalar-only WGSL records have natural alignment 4; their total sizes are
  deliberately multiples of 16 for uniform bindings.
- `Dct8Uniform` ends in `vec4<f32>`. Its WGSL natural alignment is therefore 16 and Rust uses
  `#[repr(C, align(16))]`.
- Storage arrays use the same element stride on both sides: `GpuTask`/`Task` is 28 bytes and
  `GpuResourceVector`/`vec4<f32>` is 16 bytes with 16-byte alignment.
- Modular transform planning uses two 32-byte, four-byte-aligned `repr(C)`/`Pod` records.
  `ModularChannelGeometry` carries width, height, signed horizontal/vertical shifts, bit depth, and
  three zero words while meta-applying the transform stack. `GpuModularChannelLayout` replaces the
  padding with packed word offset and row stride for storage-buffer execution. All offsets and the
  final sample end are checked against WGSL `u32` before allocation. The descriptor-specialized
  Modular kernel binds these layouts for both reusable pass-group lanes and a frame-resident arena.
  Reverse planning retains two such channel vectors at a time; it does not allocate a topology per
  transform. A separate cumulative topology-work limit rejects legal-looking metadata that would
  otherwise force quadratic host planning before GPU admission.
- Generalized Modular entropy adds a separate 32-byte `GpuModularEntropyChannel` storage record:
  `word_offset,row_stride,width,height,decoded_start,decoded_end,reference_offset,reference_count`.
  Records and their flattened `u32` reference indices are appended to the immutable MA/entropy
  metadata. Reference offsets are rebased to absolute metadata words with checked arithmetic.
  Reference-list construction is keyed by exact geometry and shifts and emits at most 60 prior
  channels per descriptor, avoiding adversarial quadratic scans.
- `ModularSqueezeParams` is one 64-byte, 16-byte-aligned `Pod` uniform containing three
  `width,height,row_stride,word_offset` views and one direction/reserved record. All views address a
  single read-write storage binding and their complete row footprints must be pairwise disjoint.
  This avoids binding the same physical `wgpu::Buffer` simultaneously as separate read-only and
  read-write resources. The kernel uses no workgroup storage; each invocation serially owns one row
  or column and therefore needs no inter-invocation barrier. Smooth tendency uses two-word integer
  temporaries because portable WGSL has no `i64`.
- Entropy and Palette reconstruction share `modular_predict.wgsl`. Its signed 64-bit predictions,
  averages, corrections, and weighted products use the low/high `u32` pair in
  `modular_int64.wgsl`; those temporaries add no storage binding or allocation. Committed true
  errors remain `i32`, and absolute subprediction errors remain `u32`, including their specified
  narrowing and wrapping sums. The five-word-per-column weighted row state and existing resume
  tails are unchanged. Implicit Palette scaling also uses the shared wide product through a
  32-bit working depth, while negative delta scaling still caps at 24 bits.
- `ModularRctParams` is another 64-byte, 16-byte-aligned `Pod` uniform with three
  `width,height,row_stride,word_offset` records and an RCT type plus three reserved words. The three
  footprints are pairwise non-overlapping views of one read-write arena binding. Every invocation
  loads all three words before storing the in-place permutation, uses no workgroup storage, and
  preserves signed wrapping through `u32` bit-pattern arithmetic.
- The Modular inverse planner uses one word-addressed arena rather than one buffer per intermediate
  channel. Initial entropy planes occupy a checked packed prefix. RCT jobs reference three current
  spans without allocation. For Squeeze, a best-fit free list indexed by size and offset allocates
  each destination before dispatch, then releases its two source ranges and coalesces neighbors in
  logarithmic time. This ordering proves that a job never aliases its own inputs while allowing
  later jobs to overwrite dead entropy or intermediate data. The nested 9×5 Squeeze plan has 45 live
  entropy words and a 90-word peak; an RCT/Squeeze/RCT plan preserves five-job order over three
  planes; the real 1024×128 progressive-DC root stays within twice its 393,216-word entropy storage.
- All buffer offsets and sizes are computed with checked integer arithmetic. Host-side sizes use
  `u64`; values consumed as WGSL indices are rejected unless they fit `u32`.
- Uniforms are bound at offset zero. Resident/storage suballocations use an explicit binding size
  and an offset aligned to `max(4, min_storage_buffer_offset_alignment)`.

### Uniform and structured-storage table

The field-order column is authoritative from first byte to last byte. Rust padding arrays map to
the individual WGSL padding fields shown here. `layout` maps to the semantically equivalent WGSL
name shown in parentheses.

| Crate / shader | Rust type / WGSL type | Field order | Size | Natural alignment | Address space |
|---|---|---|---:|---:|---|
| `jxl_wgpu/copy.wgsl` | `CopyParams` / `Params` | `width, height, input_stride, output_stride` | 16 | 4 | uniform |
| `jxl_wgpu/modular_to_f32.wgsl` | `ModularParams` / `Params` | `width, height, input_stride, output_stride, multiplier, bias, _pad0, _pad1` | 32 | 4 | uniform |
| `jxl_wgpu/chroma_upsample.wgsl` | `ChromaUpsampleUniform` or `ResidentChromaUpsampleParams` / `Params` | `input_width, input_height, output_width, output_height, input_stride, output_stride, axis, _pad0` | 32 | 4 / 16 | uniform |
| `jxl_wgpu/chroma_2d.wgsl` | `Chroma2dUniform` or `ResidentChromaUpsampleParams` / `Params` | `input_width, input_height, output_width, output_height, input_stride, output_stride, _pad0, _pad1` | 32 | 4 / 16 | uniform |
| `jxl_wgpu/gaborish.wgsl` | `GaborishUniform` / `Params` | `width, height, input_stride, output_stride, weight0, weight1, weight2, _pad0` | 32 | 4 | uniform |
| `jxl_wgpu/gaborish_rgb.wgsl` | `GaborishRgbUniform` or `ResidentGaborishParams` / `Params` | dimensions/6 strides, then four values for each of X, Y and B: `weight0, weight1, weight2, pad` | 80 | 4 / 16 | uniform |
| `jxl_wgpu/epf.wgsl` | `EpfUniform` or `ResidentEpfUniform` / `Params` | dimensions/6 image strides, sigma dimensions/stride/kind, 6 filter floats, constant inverse sigma, one pad | 80 | 4 / 16 | uniform |
| `jxl_wgpu_decode/vardct_epf.wgsl` | `EpfSigmaUniform` / `Params` | LF-group block/task/sharpness geometry, full-image block-grid extent plus group destination origin, artifact status/task offsets, global scale, quant multiplier, two four-value sharpness LUT rows | 80 | 16 | uniform |
| `jxl_wgpu_decode/modular_squeeze` | `ModularSqueezeParams` / `Params` | average, residual, and output `width,height,row_stride,word_offset` records, then direction and 3 reserved words | 64 | 16 | uniform |
| `jxl_wgpu_decode/modular_rct` | `ModularRctParams` / `Params` | three in-place plane `width,height,row_stride,word_offset` records, then RCT type and 3 reserved words | 64 | 16 | uniform |
| `jxl_wgpu_decode/modular_scalar_output.wgsl` | `ScalarParams` | source width/height/word stride/offset; destination width/height/byte stride/offset; packed source precision/component bytes/F32-output flag/orientation; logical bytes/output words/dispatch width/source domain | 64 | 16 | uniform binding 2; a separate four-byte atomic status at binding 3 records unrepresentable native samples |
| `jxl_wgpu_decode/modular_render.wgsl` | `NormalizeParams` | source width/height/word stride/offset; packed source precision/output stride/two zero words | 32 | 16 | uniform binding 2; encoded input words at binding 0 and distinct decoded binary32 words at binding 1; 16×16 workgroups |
| `jxl_wgpu/upsample.wgsl` | `UpsampleUniform` or resident `UpsampleParams` / `Params` | `input_width, input_height, output_width, output_height, input_stride, output_stride, factor, _pad0` | 32 | 4 / 16 | uniform |
| `jxl_wgpu/ycbcr_to_rgb.wgsl` | `YcbcrUniform` / `Params` | `width, height, cb_stride, y_stride, cr_stride, output_stride, component, _pad0` | 32 | 4 | uniform |
| `jxl_wgpu/xyb_to_rgb.wgsl` | `XybUniform` / `Params` | dimensions/6 strides, three padded inverse-opsin rows, padded cube-root bias, padded scaled bias, `intensity_scale`, 3 pads | 128 | 4 | uniform |
| `jxl_wgpu/transfer_function.wgsl` | `TransferUniform` / `Params` | dimensions/6 strides, `transfer, gamma, intensity_target, min_nits, luminance_rgb` | 64 | 4 | uniform |
| `jxl_wgpu/blend.wgsl` | `BlendUniform` / `Params` | dimensions/5 value strides, 2 alpha strides, `mode, component, clamp, alpha_associated, has_alpha` | 48 | 4 | uniform |
| `jxl_wgpu/premultiply_alpha.wgsl` | `PremultiplyUniform` / `Params` | `width, height, color_stride, alpha_stride, output_stride, _pad0, _pad1, _pad2` | 32 | 4 | uniform |
| `jxl_wgpu/extend.wgsl` | `ExtendUniform` / `Params` | image/frame dimensions, 3 strides, signed origin, `has_reference`, 2 pads | 48 | 4 | uniform |
| `jxl_wgpu/save.wgsl` | `SaveUniform` / `Params` | `width, height, source_stride, channels, channel, layout (output_layout), orientation, _pad0` | 32 | 4 | uniform |
| `jxl_wgpu/image_output.wgsl` | `ImageOutputParams` / `Params` | dimensions/3 source strides, format fields, 4 plane offset/stride pairs, `logical_size, dispatch_width, orientation, source_transfer, target_transfer, identity_color_transform`, three padded primary-matrix rows, alpha conversion and three pads | 192 | 4 | uniform |
| `jxl_wgpu/display_rgb.wgsl` | `DisplayRgbParams` / `DisplayRgbParams` | `width, height, channels, sample_type, layout (storage_layout), logical_samples, _padding0, _padding1` | 32 | 4 | uniform |
| `jxl_wgpu/display_numeric.wgsl` | `DisplayNumericParams` / `NumericParams` | dimensions/type/depth/components, plane offset/stride, visualization/non-finite/transfer/clamp, reserved word, `scale, bias`, 2 pads | 64 | 4 | uniform |
| `jxl_wgpu/display_image.wgsl` | `DisplayImageParams` / `Params` | dimensions/format fields, 4 plane offset/stride pairs, `chroma_width, chroma_height, transfer`, three padded source-linear-to-BT.709 matrix rows | 144 | 4 | uniform |
| `jxl_wgpu/vardct_dct8.wgsl` | `GpuTask` / `Task` | `coefficient_offset, destination_x, destination_y, quant_index, matrix_index, correlation_index, lf_index` | 28 | 4 | storage element |
| `jxl_wgpu/vardct_dct8.wgsl` | `GpuResourceVector` / `vec4<f32>` | four `f32` lanes | 16 | 16 | storage element |
| `jxl_wgpu/vardct_dct8.wgsl` | `Dct8Uniform` / `Params` | `task_count`, output dimensions/3 strides, 4 resource offsets, 2 pads, `quant_biases[4]`/`vec4<f32>` | 64 | 16 | uniform |
| `jxl_wgpu/vardct_general.wgsl`, `vardct_special.wgsl` | `ResidentVarDctParams` / `Params` | task range, transform/LF dimensions, resource offsets, 3 output dimension/stride tuples, transform/correlation geometry, artifact task/bucket offsets, X LF stride, 3 pads, `quant_biases[4]`, then Y/B LF bases and strides | 144 | 16 | uniform |
| `jxl_wgpu_decode/vardct_resource.wgsl` | `VarDctResourceParams` / `Params` | geometry, three source channel extents/bases, three destination stride/origin/base records, X/Y/B LF scales plus extra-precision multiplier, final LF X/B chroma-correlation slopes and 2 pads | 144 | 16 | uniform |
| `jxl_wgpu_decode/color_output.wgsl` | `ColorSourceParams` | 3 component stride/extent/shift records; alpha offset/stride/maximum/enabled; 3 padded inverse-matrix rows; padded cube-root/scaled biases; intensity scale, transform mode, 2 pads | 160 | 16 | uniform binding 5; shared 192-byte output parameters occupy binding 4 |
| `jxl_wgpu_decode/vardct_artifact.wgsl` | `HfMetadataLoweringParams` / `Params` | six `vec4<u32>` records for dimensions/capacities/image/artifact/metadata/source offsets, three channel shift/LF-base/stride records, seven `vec4<u32>` records containing all 27 strategy matrix offsets, X/Y/B dequantization scale multipliers, then base X/B correlation and reciprocal colour factor | 288 | 16 | uniform |
| `jxl_wgpu_decode/vardct_packet.wgsl` | `VarDctPacketControl` / `PacketControl` | eight `vec4<u32>` records for section ranges, geometry, physical metadata offsets/capacities, expectations, quantization, streams, and scratch | 128 | 16 | uniform |
| `jxl_wgpu_decode/vardct_packet.wgsl` | `VarDctModularParams` / `Params` | 12-byte entropy prefix; logical/upload window starts, stream/yield ends, flags, state offset, stream base; 49 consumer words; one pad | 240 | Rust 16 / WGSL 4 | read-only storage |
| `jxl_wgpu_decode/vardct_pass_group.wgsl` | `HfCoefficientPassParams` / `Params` | 92-byte stream/geometry prefix, 48-byte block-context locations, component shifts, metadata/order bases, spatial group index, one pad | 160 | Rust 16 / WGSL 4 | read-only storage |
| `jxl_wgpu_decode/vardct_packet.wgsl` | `GenericPacketExecutionState` / reconstruction words | common entropy/LZ/consumer state, active LF/HF phase, decoded LF/HF counts, first-block count, extra precision, previous gradient, two pads | 64 | 16 | storage subrecord |
| `jxl_wgpu_decode/vardct_packet.wgsl` | `WeightedPacketExecutionState` / reconstruction words | generic prefix plus four true errors, twelve subprediction-error accumulators, two pads | 128 | 16 | storage subrecord |
| `jxl_wgpu_decode/progressive_dc.wgsl` | `ProgressiveDcPackParams` / `PackParams` | extent/count, three input strides, LF vec4 offset/stride and two reserved words | 48 | 16 | uniform |
| `jxl_wgpu_encode/vardct_encoder.wgsl` | `VarDctKernelParams` / `Params` | source/block geometry, strategy/global/LF quantization, separate 19-entry DC and HF prefix tables, X/Y/B LF quantization, LF/HF X/B correlation, X/Y/B HF quantization, and explicit padding | 512 | 4 | read-only storage |
| `jxl_wgpu_encode/vardct_encoder.wgsl` | `VarDctKernelArtifact` / `Artifact` | 16-entry strategy map, 48 DC samples/tokens/extras, 64-word DC fragment, DC histogram, 256-word AC fragment, AC histogram, and 3×1024 forward/quantized XYB coefficient words | 26,880 | 4 | storage/readback record |
| `jxl_wgpu_encode/vardct_large_encoder.wgsl` | `ScalableVarDctKernelParams` / `Params` | source/block geometry, strategy/quantization, 19 two-word prefix entries, five artifact offsets/capacities, topology, LF-fragment descriptor offset/length and LF-grid dimensions, X/Y/B inverse LF-dequantization factors, and X/B LF-correlation slopes | 256 | 4 | read-only storage |
| `jxl_wgpu_encode/vardct_large_encoder.wgsl` | `ScalableVarDctArtifactHeader` / header words | status/live counts, section offsets/lengths, total fragment bits, source/block geometry, topology, 19-bin histogram, LF-fragment descriptor offset/length/grid/count, 18 pads | 256 | 4 | storage/readback record |
| `jxl_wgpu_encode/vardct_large_encoder.wgsl` | `ScalableDcFragmentDescriptor` / two words | `bit_offset, bit_len` for one row-major LF group | 8 | 4 | storage/readback element |
| `jxl_wgpu_encode/lossless_gray8.wgsl` | `Gray8Params` / `Params` | `width, height, row_stride, byte_offset` | 16 | 4 | uniform |
| `jxl_wgpu_encode/lossless_gray8.wgsl` | `Gray8ArtifactHeader` / `output_words[0..53]` | `event_count, raw_counts[19], lz77_counts[33]` | 212 | 4 | storage/readback record |
| `jxl_wgpu_encode/lossless_gray8.wgsl` | `Gray8Event` / four-word event | `kind, token, extra_bit_count, extra_bits` | 16 | 4 | storage/readback element |
| `jxl_wgpu_decode/lossless_gray8.wgsl` | `ShaderParams` / `Params` | entropy prefix/window, group geometry, sample/channel counts, channel-layout offset, output kind/transfer/range, channels/order/depth, 4 plane offset/stride pairs, chroma geometry/size/mapping, status/stream/fixed-leaf/weighted-predictor fields; canvas width/height and orientation | 256 | 4 | read-only storage element |
| `jxl_wgpu_decode/lossless_gray8.wgsl` | `DecodeStatus` / `status[0..4]` | `code, decoded_samples, cursor, expected_cursor` | 16 | 4 | storage/readback record |
| `jxl_wgpu_decode/vardct_raw_matrix.wgsl` | `RawMatrixParams` / `RawMatrixParams` | denominator, raster width/height, target count, then padded four-lane source offsets, source strides, and resident resource target offsets | 64 | 16 | uniform |
| `jxl_wgpu_decode/codec_engine/composition/blend.wgsl` | `BlendParams` / `Params` | canvas, intersection, source, dispatch, four reference geometry/presence records | 128 | 16 | uniform |
| `jxl_wgpu_decode/codec_engine/composition/blend.wgsl` | `BlendChannel` / `Channel` | mode, background slot, alpha plane, clamp/association flags, alpha-background slot, three pads | 32 | 16 | read-only storage element |
| `jxl_wgpu_decode/codec_engine/composition/native.wgsl` | `NativeParams` / `Params` | extent, format, output, source (plane words, first alpha, scalar plane, flags: F32/linear RGB) | 64 | 16 | uniform |
| `jxl_wgpu_decode/codec_engine/composition/spot.wgsl` | `SpotColor` | absolute plane word offset and three pads; declared RGBA | 32 | 16 | read-only storage element, presentation binding 7 |

The Modular finalizer has its own 176-byte, 16-byte-aligned `ModularFinalizeParams` uniform at
binding 2. Eleven `vec4<u32>` records contain the source extent, region/status, four source offsets,
four source strides, four source encodings, output/format fields, two plane offset/stride pairs,
logical/chroma bounds, and the complete unrotated canvas width/height, oriented output width,
and orientation. Source encodings start at byte 64 and the canvas record starts at byte 160.
The region's final word at byte 28 identifies original encoded words (`0`) or decoded F32 (`1`)
source words. Scalar packing uses the same values in its final uniform word at byte 60.
VarDCT alpha's fourth geometry word is `0` for absent, `1` for encoded and `2` for
decoded F32; its third word stores packed precision. Its 160-byte source uniform is unchanged. Native resampled output rounds once
at the destination depth and rejects nonrepresentable values; F32 retains interpolation fractions.
Ordinary entropy parameter records are 256 bytes; their final canvas width, canvas height,
and orientation words are at offsets 244, 248, and 252. Compile-time
Rust sizes/alignment, byte-order assertions, Naga validation, and actual-device output cover both.
Exact uniform/parameter sizes feed the existing transient-budget plans; no rotated image arena is
allocated. Mirrored/transposed groups select atomic byte updates. Packed 4:2:2 assigns two bytes
to each source pixel, with the odd final pixel owning the tail pair; no group overwrites another
group's luma word. Native and converted output use the oriented layout's bounds.

Output-channel selection binds up to four views from the complete inverse result, or one view for
a selected extra channel. Gray+alpha can bind the gray view three times without copying samples.
Per-view `ModularSampleEncoding` values retain independently declared precision: total bits occupy
the low byte and exponent bits the next byte (zero denotes integers). Host validation admits only
representable declarations. `ModularOutputPlane` pairs this value with geometry without replacing
the working depth or using reserved descriptor words. Alpha converts independently for F32 output;
native color output rounds it to the destination precision. Scalar F32 output uses numeric mapping 4:
integer samples divide by their own maximum, while floating samples are widened by integer bit
assembly. A shared `modular_sample.wgsl` fragment serves scalar output, finalization, normalization
and VarDCT alpha. It preserves zeros, subnormals, infinities and NaN payloads without floating
arithmetic during representation conversion. Decoded scalar words and unchanged-transfer,
unchanged-association F32 RGB words are copied directly. Filtering, color and composition opt into
F32 arithmetic. Native unsigned range checks remain separate. The unchanged uniform is charged through
`size_of::<ModularFinalizeParams>()` for each existing finalizer, adding no buffer or submission.
All entropy channels, inverse arenas and jobs retain their existing budget/lifetime ownership even
when only one plane is selected. Public declaration/name vectors remain host metadata, outside
the GPU byte budget. Invalid selection is rejected before admission; retry, cancellation and
caller-held output clones have actual-GPU lifetime coverage.

### Values that intentionally are not `Pod`

`Pod` describes an in-memory Rust/WGSL ABI, not every sequence of bytes handled by the workspace:

- JPEG XL codestreams, container boxes, and the `jwgp` acceleration index are wire formats. Their
  serializers use explicit little-endian fields and packed offsets. In particular, a serialized
  prefix entry is three bytes (`u8` plus `u16`), while Rust's naturally aligned
  `PrefixCodeEntry` occupies four bytes. Deriving `Pod` would serialize host padding and produce a
  different, invalid wire format. Incremental transport therefore uses `StreamSlice` ranges over
  caller-owned `Arc<[u8]>`, not a fake fixed ABI. Apart from the two-byte codestream signature
  reconstructed inline across arbitrary chunk boundaries, ordered raw/`jxlc`/`jxlp` and auxiliary
  payloads share those allocations. Only out-of-order version-1 fragments are copied and arbitrary
  input chunks are coalesced into one retained payload buffer per future fragment; logical retained
  bytes and their peak are reported and hard-limited independently. Allocator capacity and
  collection metadata are outside those logical counters. Incremental codestream inventory copies
  only the current image or frame-header/TOC probe into a contiguous `Vec`; independent image/frame
  prefix limits apply before growth, and live/peak plus cumulative copied logical bytes are
  reported. `FrameStart` shares one `Arc<FrameInventory>` with the scanner while section events are
  active. Section payload after a bounded probe overshoot retains its existing `StreamSlice`
  backing and is never assembled into a complete host codestream.
- Raw image planes and mapped packed-image byte ranges have runtime-selected formats, plane counts,
  row strides, and lengths. `ImageLayout` validates their byte ranges; there is no single fixed
  Rust record that could safely represent their contents.
- Codestream storage and the decoder prefix lookup are variable arrays, not fixed records. The
  lookup is retained session-locally as `Arc<[u32]>` and uploaded with `cast_slice`; the raw
  codestream remains bytes so its bit/byte offsets and explicit four-byte sentinel are preserved.
  Both decoder engines retain a checked `Arc`-backed logical span table. Its exact range iterator
  and copier cross physical chunk boundaries, and both codec modes use its span-native bit reader.
  VarDCT stream planning reads only the logical length; scalar headers, block-context maps, custom
  coefficient orders, and local-HF continuations do not assemble a host slice; bounded windows copy
  exact span ranges; and its still-required whole-codestream GPU buffer is initialized while mapped
  directly from those spans. Neither engine constructs a second full-size host `Vec`.
  `GpuDecoder::stream` retains transport slices under one growable incremental-input permit, moves
  that permit with the source into the selected engine, and releases it after the last source-using
  submission or on cancellation. This host-retention budget is deliberately distinct from the GPU
  allocation budget because source and destination coexist while an upload is populated.

Manual endian-aware serialization is therefore retained only for bitstream/container/file
formats. Fixed GPU records do not use hand-written byte flattening.

## Bindings and dispatch bounds

`RO`, `RW`, `U`, and `T` mean read-only storage, read-write storage, uniform, and write-only storage
texture. All two-dimensional kernels return before accessing memory when `gid.x >= width` or
`gid.y >= height`.

Tier A size-agnostic entry points declare WGSL override constants (`override wg_x: u32 = ...; override wg_y: u32 = ...;`)
and are parameterizable at pipeline creation via `KernelPolicy` and `KernelVariant` (`Tile16x16`, `Tile16x8`, `Tile8x8`,
`Tile32x4`, `Lanes256`, `Lanes128`, `Lanes64`, `Lanes32`, `Scalar`). The planner, decoder, and display pipelines validate
the selected workgroup dimensions and invocation counts against device limits prior to pipeline creation and dispatch
recording. Tier B kernels (such as `vardct_dct8`, `vardct_special`, `vardct_artifact`, `vardct_packet`, and encoder
control/modular passes) are structurally fixed to their algorithm-defined workgroup dimensions and reject non-default
variants. Tier C kernels contain algorithm-specific reductions or tiling. The two encoder VarDCT data passes have
generalized their lane assignment and accept every linear `KernelVariant`; `vardct_lf` and `vardct_general` remain
fixed until their reduction or tiling structures are generalized.

The table below states the default workgroup configuration for each entry point:

| Shader / entry points | Bindings in order | Default workgroup | Parameterization | Dispatch and address bound |
|---|---|---:|---|---|
| `copy` | input RO, output RW, U | 16x16 | Tier A (`KernelVariant` 2-D) | checked 2-D extent and strides |
| `modular_to_f32` | input RO, output RW, U | 16x16 | Tier A (`KernelVariant` 2-D) | checked 2-D extent and strides |
| `chroma_upsample` | input RO, output RW, U | 16x16 | Tier A (`KernelVariant` 2-D) | checked input/output extents, axis and strides |
| `chroma_2d` | input RO, output RW, U | 16x16 | Tier A (`KernelVariant` 2-D) | checked input/output extents and strides |
| `gaborish` | input RO, output RW, U | 16x16 | Tier A (`KernelVariant` 2-D) | checked extent/strides; clamped neighbor reads |
| `gaborish_rgb` | X/Y/B RO, X/Y/B RW, U | 16x16 | Tier A (`KernelVariant` 2-D) | checked common extent and per-plane strides |
| `epf0`, `epf1`, `epf2` | X/Y/B/sigma RO, X/Y/B RW, U | 16x16 | Tier A (`KernelVariant` 2-D) | checked common extent, sigma shape and all strides |
| `upsample` | input/weights RO, output RW, U | 16x16 | Tier A (`KernelVariant` 2-D) | checked factor, output extent, weights and strides |
| `vardct_frame_upsample` (decoder, `upsample`) | restored component/phase weights RO, full output component RW, U | 16x16 | Tier A (`KernelVariant` 2-D) | one immutable expanded kernel shared by three channel dispatches; exact encoded/output extents, mirrored 5×5 neighborhoods, range clamping, and odd-edge cropping; planes, weights, and 32-byte uniforms share the frame budget |
| `ycbcr_to_rgb` | Cb/Y/Cr RO, output RW, U | 16x16 | Tier A (`KernelVariant` 2-D) | one checked dispatch per output component |
| `xyb_to_rgb` | X/Y/B RO, R/G/B RW, U | 16x16 | Tier A (`KernelVariant` 2-D) | one dispatch; checked common F32 extent, per-plane strides, finite inverse-opsin parameters and positive intensity target |
| `transfer_function` | R/G/B RO, R/G/B RW, U | 16x16 | Tier A (`KernelVariant` 2-D) | one dispatch; checked common F32 extent and Linear/sRGB/BT.709/Gamma/PQ/HLG parameters |
| `blend` | base/source/base-alpha/source-alpha RO, output RW, U | 16x16 | Tier A (`KernelVariant` 2-D) | one scalar-channel dispatch; two or four equal F32 inputs keep the shader within portable storage-binding limits |
| `premultiply_alpha` | color/alpha RO, output RW, U | 16x16 | Tier A (`KernelVariant` 2-D) | one checked dispatch per color component |
| `extend` | frame/reference RO, full-canvas output RW, U | 16x16 | Tier A (`KernelVariant` 2-D) | exact u32 word copy for I32/F32; checked signed origin, crop, target extent and optional reference canvas |
| `save` | source RO, packed output RW, U | 16x16 | Tier A (`KernelVariant` 2-D) | checked orientation and exact packed allocation |
| `rgb_to_image` | R/G/B RO, packed output RW, U | 256x1 | Tier A (`KernelVariant` 1-D) | checked linear word count is split into a legal 2-D dispatch; shader checks `logical_size` before stores |
| `modular_scalar_output::pack` (decoder) | signed arena RO, packed output/status RW, U | 256x1 | Tier A (`KernelVariant` linear) | one invocation owns four output bytes; checked source view, oriented rows, binding offsets, padding and 2-D dispatch bounds; native sample-range failure is sticky and joins final frame validation |
| `display_rgb` | source RO, destination T, U | 16x16 | Tier A (`KernelVariant` 2-D) | source must have `STORAGE`; logical samples and final source address fit the bound range/WGSL `u32` |
| `display_numeric` | source words RO, destination T, U; native-F64 variant also binds the same source as F64 RO | 16x16 | Tier A (`KernelVariant` 2-D) | exact pitch-linear plane range/stride and WGSL `u32` addresses; explicit sample kind, affine mapping, non-finite handling, clamp, transfer, and channel visualization |
| `display_image` | source RO, RGBA8 or RGBA16F destination T, U | 16x16 | Tier A (`KernelVariant` 2-D) | source must have `STORAGE`; each pitch-linear plane and its final address is bounded; wide-gamut/HDR requires float output |
| `vardct_resource` (decoder) | LF-group table RO, full-image dequantized-LF atlas RW, U | 64x1 | Tier A (`KernelVariant` 1-D) | checked per-component extents/source bases and global atlas base/stride/origin; coalesced XYB can apply chroma-from-luma, while JPEG sampling writes compact component grids directly; one 1D workgroup per LF-group block batch |
| `vardct_raw_matrix` (decoder) | inverse-transformed side-image arena RO, resident resource table and shared decode status RW, U | 64x1 default, autotuned linear lanes | Tier A (`KernelVariant` 1-D) | one invocation per canonical matrix sample; checked plane offsets/strides and one, two, or four aliased resource targets; non-positive or oversized weights set a typed sticky status before AC/render |
| `vardct_packet` (decoder) | whole or reusable-window codestream/MA metadata RO, reconstruction/raw metadata/coefficients/status RW, control U, Modular params RW | 1x1 | Tier B (fixed) | combined/global-tree and split LF/HF local-tree entry points use logical channel widths with explicit physical strides; all packet forms resume across ordered windows using one 64/128-byte aligned state and reusable upload, while local trees and single-entry TOCs map LF cursors; single-entry TOCs also map the HF-global boundary before the final authoritative map |
| `progressive_dc::pack_lf` (decoder) | X/Y/B planes RO, VarDCT resources RW, U | 64x1 | Tier A (`KernelVariant` linear) | checked common extents/strides, plane binding ranges, destination resource vec4 range, storage limits and WGSL-u32 addresses; four versioned LF slots retain tracked plane leases through the last consumer; scratch is released after producer validation |
| `vardct_artifact` (decoder) | LF-group raw metadata RO, artifact/occupancy plus full-image resources RW, U | 1x1 | Tier B (fixed) | validates non-overlapping mixed varblocks, global LF/correlation strides and aligned destination origin, derives per-channel task masks/destinations/LF offsets from JPEG shifts, compacts all 27 strategy buckets, and emits three bounded indirect records per strategy plus exact coefficient ranges |
| resident `vardct_general` (decoder) | coefficients/artifact/resources RO, two global scratch buffers and X/Y/B output RW, U | 64x1 | Tier B (fixed) | one indirect dequantize/horizontal/vertical sequence per populated regular strategy bucket; task/artifact/resource/per-channel-LF/output ranges are host-validated |
| resident `vardct_special` (decoder) | coefficients/artifact/resources RO, X/Y/B output RW, U | 8x8 | Tier B (fixed) | one indirect dispatch per populated special strategy bucket; 2,304-byte workgroup storage and raster coefficient/matrix layout are fixed by the transform contract |
| `vardct_large_encoder::quantize_blocks` | source/params RO, artifact RW | 64x1 | Tier C (`KernelVariant` linear) | one 2-D workgroup per 8x8 block; checked block-grid axes and source/artifact ranges; 1,024 bytes workgroup storage |
| `vardct_large_encoder::serialize_control` | params RO, artifact RW | 1x1 | Tier B (fixed) | one bounded scalar dispatch serializes LF groups row-major, resets prediction at each 256x256-block boundary, and writes checked contiguous fragment descriptors |
| `color_output` (decoder) | X/Y/B, Cb/Y/Cr or R/G/B planes and opacity RO, output RW, 2 U | 256x1 | Tier A (`KernelVariant` 1-D) | shared 192-byte output uniform plus 160-byte codec-source uniform; checked word count is linearized across 2-D workgroups; each invocation writes one packed u32 after full-precision inverse opsin or encoded BT.601 reconstruction, requested color conversion and packing; normative JPEG 2× component interpolation is fused when restoration did not already expand the planes |
| `vardct_chroma_upsample` (decoder, `chroma_upsample`/`chroma_2d`) | compact component RO, distinct full-resolution component RW, U | 16x16 | Tier A (`KernelVariant` 2-D) | one-axis or fused two-axis quarter/three-quarter interpolation before restoration; checked logical extents, padded strides, storage usage/alignment/binding limits, dispatch counts, and replicated odd borders; the decoder allocates distinct destinations |
| `vardct_gaborish` (decoder, `gaborish_rgb`) | resident X/Y/B RO, distinct resident X/Y/B RW, U | 16x16 | Tier A (`KernelVariant` 2-D) | checked actual image extent, padded per-plane stride/range, storage usage/alignment/binding limits, finite normalized weights and dispatch counts |
| `vardct_epf_sigma` (decoder) | LF-group raw metadata/artifact RO, full-image inverse-sigma atlas RW, U | 64x1 | Tier A (`KernelVariant` 1-D) | one invocation per validated transform task; artifact status/task count gate writes, while local block extent, global destination rectangle, and sharpness are bounded before addressing |
| `vardct_epf` (decoder, `epf0`/`epf1`/`epf2`) | resident X/Y/B/sigma RO, distinct resident X/Y/B RW, U | 16x16 | Tier A (`KernelVariant` 2-D) | checked actual extent, padded plane strides/ranges, sigma block-grid coverage, finite parameters, binding/device limits, and mirrored whole-image neighbors |
| decoder `lossless_gray8` | codestream/prefix RO, reconstructed/output/status RW, 256-byte parameter records RO, 16-byte dispatch U | 64x1 | Tier A (`KernelVariant` 1-D) | bounded `jwgp` index, aligned token words plus sentinel, per-group MA metadata base, channel-layout tables, four planes/final addresses, packed-row alignment, sample/output ranges and status allocation are prevalidated; one invocation per group lane; channel-fixed Gradient groups may resume through 16-byte-overlapped stream segments using one aligned 32-byte state record per lane |
| encoder `vardct_encode_bounded` | source RO, parameters RO, artifact RW | 256x1 | Tier C (`KernelVariant` 1-D) | one workgroup cooperatively loads and transforms at most 1,024 pixels; fixed 16 KiB workgroup storage is validated before pipeline creation |
| encoder `vardct_encode_quantize` | source RO, parameters RO, artifact RW | 64x1 | Tier C (`KernelVariant` 1-D) | one workgroup per checked 8x8 block; lanes stride over exactly 64 samples and use fixed 1 KiB workgroup storage |
| encoder VarDCT `serialize_control` | parameters RO, artifact RW | 1x1 | Tier B (fixed) | a separate pass establishes global visibility; one invocation performs sequential DC prediction and bit-offset serialization |
| decoder `vardct_pass_group` | bounded stream/entropy bundle RO, quantized-LF plus disjoint LZ/state slices/status RW, 160-byte pass params RO; artifact/order RO, coefficients RW, sink U | 1x1 | Tier B (fixed) | one serial invocation per pass-group window; eight storage bindings meet the portable stage limit; a 464-byte aligned state retains common entropy, nested coefficient progress, sink failure and the 96-word nonzero grid; the 48-byte block-context table ABI addresses QF and signed X/Y/B LF thresholds |
| `vardct_dct8` | coefficients/tasks/resources RO, X/Y/B RW, U | 8x8 | Tier B (fixed) | exactly one workgroup per validated task; task count and all upload bindings are device-bounded |
| encoder `lossless_gray8` | source words RO, artifact RW, U | 1x1 | Tier B (fixed) | profile dimensions are 2..=256; source subrange/alignment/u32 address and artifact capacity are prevalidated |

The decoder entropy shaders share a nested host/WGSL ABI rather than duplicating an untyped word
prefix. `EntropyStreamParams` is a 12-byte, four-byte-aligned `repr(C)`/`Pod` record of three `u32`
values: token start/end bounds and the LZ77 ring mask. It begins the 256-byte Modular
`ShaderParams` storage record and the 240-byte, 16-byte-aligned VarDCT packet parameter record. Consumers supply
storage access and LZ scratch-base functions; their
geometry, prediction, output, and coefficient suffixes are not forced into one binding layout.
Compile-time Rust size/alignment checks, full-record word casts, Naga parsing of every composed
shader variant, and actual GPU Modular/VarDCT entropy tests validate this contract; shader source
text is not used as a semantic test oracle.

The Modular record places six `u32` window values immediately after that prefix: logical segment
start, physical upload start, full logical token end, yield end, flags, and the per-lane state
offset. The Rust window planner that produces those values is consumer-neutral. Modular, VarDCT AC,
combined/global-tree packets, and staged local-tree VarDCT LF/HF packets use the same checked
overlap/batching values without sharing their WGSL state layout.
The later Modular `metadata_base` word selects a global or rebased/deduplicated local MA and
entropy descriptor for that group; its config/tree/table offsets remain absolute within the shared
metadata binding. Channel-layout tables have a separate absolute base and may be deduplicated
independently of MA and weighted-predictor choices. Each split segment is four-byte
padded and followed by a zero sentinel word. A middle
segment includes 16 bytes before and after its core range; this exceeds the largest complete
Prefix/ANS + hybrid/LZ77 output token, so the shader yields only after a complete value and can map
the overshoot in the next upload. The common lane tail is rounded to 16 bytes and stores eight
`u32` values (32 bytes): bit cursor, ANS state, LZ copy remaining/position, entropy decoded count,
last value, consumer decoded count, and sticky error. Generic MA adds its previous-gradient state
and explicit padding for a 48-byte tail. Weighted/SelfCorrecting MA uses a 112-byte tail that also
persists four true errors and twelve subprediction-error accumulators; its existing five-row-width
workspace remains immediately before the tail. Channel-local predictor state is reset rather than
imported when a resumed consumer starts a new channel. The descriptor-sized LZ history ring remains
in the preceding scratch region and is not duplicated per segment. Rust `repr(C, align(16))`/`Pod`
records and compile-time size/alignment assertions pin the largest layout.

The VarDCT packet record adds an explicit bounded-mode bit independent of FIRST/FINAL, so a middle
segment with neither boundary bit cannot fall back to whole-range addressing. Its per-group state is
64 bytes for generic MA and 128 bytes for SelfCorrecting MA, both `repr(C, align(16))`/`Pod`. The
first LF command clears reconstruction and state, intermediate commands only rewrite the reusable
stream and parameter buffers after prior queue use, and the final LF command copies every 64-byte
group status into one staging map. Host-packed HF descriptors then produce exact ranges over the
same upload and state allocation. HF resumes independently across two correlation maps, the
capacity-strided strategy/quantizer plane, and sharpness; its final segment performs ANS and fixed
packet-tail validation, clears coefficient storage, and shares the first downstream submission.
Combined/global-tree packets use the same five state words for the active LF/HF phase, retained LF
and HF decoded counts, first-block count, and extra precision. Their range begins before the LF local
header, needs no intermediate map, and only the final window validates ANS/padding plus the fixed
packet tail before sharing the first downstream submission.

When an external LF image removes coefficient entropy but LF-group extras remain, the initial
submission only clears/copies frame arenas. Its map fences that setup and is never interpreted as
an LF-success status. The common Modular subimage executor supplies each group's validated extra
end cursor before HF descriptors are parsed. Admission reserves conservative HF LZ history and
128-byte packet-state capacity, plus one bounded upload if an enclosing LF packet exceeds the
stream limit; no initial LF/HF entropy metadata buffer is allocated. Exact late HF descriptor bytes
reserve a separate permit. Known HF-only entries use their actual generic/weighted state capacity
consistently in allocation, accounting and device validation. Source XYB leases, extra arenas and
all discovered tables remain owned by the frame/callback until completion or canceled-map retirement.

Display source buffers are now also checked for the usage needed by the operation: `STORAGE` for
shader conversion and `COPY_SRC` for direct RGBA8 buffer-to-texture copies. A multi-row direct
buffer-to-texture copy requires `bytes_per_row` to be a multiple of 256.

## Workgroup-local storage

The following shaders declare `var<workgroup>` memory:

| Shader | Declaration | Bytes per workgroup | Host validation |
|---|---|---:|---|
| `vardct_dct8` | two `array<f32, 192>` scratch arrays | 1,536 | reject when `max_compute_workgroup_storage_size < 1,536` |
| `vardct_special` | three `array<f32, 192>` scratch arrays | 2,304 | below the portable 16 KiB minimum; wgpu validates pipeline creation |
| decoder `vardct_lf` | `array<vec4<f32>, 324>` tile | 5,184 | checked as `ADAPTIVE_LF_WORKGROUP_BYTES` before submission |
| encoder `vardct_encode_bounded` | `array<vec3<f32>, 1024>` XYB block | 16,384 | selected variant and bytes checked before pipeline creation |
| encoder `vardct_encode_quantize` | `array<vec3<f32>, 64>` XYB block | 1,024 | selected variant and bytes checked before pipeline creation |

Other parameterized image kernels use zero explicit workgroup-local bytes. Default invocation counts are 256
(`16x16` tiled 2D kernels and `256x1` linear kernels `rgb_to_image` / `color_output`), 64
(`vardct_resource`, decoder `lossless_gray8`, encoder scalable VarDCT quantization at `64x1`, and fixed VarDCT DCT8 at `8x8`), and one (encoder control/modular passes).
Planner and pipeline creation validate selected `KernelVariant` dimensions and invocation counts
against device limits prior to pipeline compilation and dispatch recording.

## Four-byte copy and mapping invariant

- `aligned_buffer_size` rounds every buffer-copy/readback allocation up to a nonzero multiple of
  `wgpu::COPY_BUFFER_ALIGNMENT` (4). Logical output lengths remain separate, so padding is never
  returned as image data.
- Core packed output and CPU staging copies use the same padded byte count. VarDCT upload element
  sizes (4, 16, and 28) are already multiples of 4.
- Generic `ImageReadbackPipeline::submit_frames` independently pads every source copy and aggregate
  staging offset across all supplied frames to 4 bytes, validates `COPY_SRC` and each source
  allocation, records one command buffer/queue submission, and maps one bounded aggregate staging
  buffer. Returned frames retain their original output ranges and exclude all copy padding.
- Encoder artifact storage and mapped readback have identical checked, 4-byte-aligned sizes. The
  16-byte `Gray8Params` uniform and both artifact buffers are leased as one exact-size set, so a
  buffer cannot be reused by another submission until the mapped artifacts have been consumed and
  the readback buffer has been unmapped. If the future is abandoned, its callback-owned lifetime
  performs that unmap and return only after mapping resolves.
  Its 212-byte header and 16-byte events are parsed as checked `Pod` records. Decoder codestream
  storage is padded to a word and includes a four-byte sentinel; its 16-byte status is parsed as a
  checked `DecodeStatus` record.
- Gray8 decoder output allocation is rounded to four bytes while `logical_size` remains explicit.
  RGBA/BGRA pixels and odd-width YUYV/UYVY pairs use aligned whole-word stores; byte and 16-bit
  plane writers bounds-check each addressed byte against `logical_size`.
- Buffer-to-texture paths apply WebGPU's separate 256-byte multi-row pitch rule. Texture-to-buffer
  tests likewise use 256-byte row padding.

## Memory accounting and concurrency

| Path | Per-job accounting | Concurrent accounting / admission | Deliberate exclusions or remaining gap |
|---|---|---|---|
| Core render session | `WgpuSubmissionStats` reports physical resident-plane bytes and exact explicit transient bytes: uniforms, uploads, packed outputs and staging. `max_transient_bytes` is enforced per submission. | `WgpuFrameSession::pending_transient_bytes()` checked-adds submitted jobs and checked-subtracts them on all wait paths. | The aggregate is observable, not a second admission limit. Queue-ordered reusable resident allocations make it conservative. Caller-owned GPU outputs can outlive `wait`, so the session cannot track them afterward. |
| Core resident arena | Planner accounts physical slots once, respects simultaneous lifetimes, validates every slot against `max_buffer_size` and every bound plane against `max_storage_buffer_binding_size`. | Buffer pool has a configured hard byte limit and never leases one buffer concurrently. | Pipeline/driver memory excluded. |
| VarDCT transform kernels | Exact coefficient, compact task/artifact, all normative default matrix/AFV resources, two global scratch buffers, and 27 aligned 144-byte uniform records are included in the core transient total. Every binding is checked against `min(max_buffer_size, max_storage_buffer_binding_size)`. | Included in core pending total. | Regular buckets use global ping-pong scratch; special buckets use fixed 2,304-byte workgroup storage. Empty indirect records perform no work. |
| Incremental decoder input | `GpuDecodeStreamStats` reports the inventory scanner counters, exact retained logical codestream bytes, physical span count, completed frames, authoritative-end state, and the shared input-budget snapshot. | One growable permit per stream reserves each nonempty `CodestreamChunk` before scanner mutation. Exhaustion is typed and retryable with the borrowed event. The permit moves into `GpuCodestream`; Modular drops it after upload submission, staged local-tree VarDCT retains it through HF planning/submission, and every error/cancellation path drops it. Concurrent streams share a caller-replaceable `IncrementalInputBudget`. | Allocator capacity, span-table metadata, auxiliary boxes, and caller allocations outside the retained logical slice are excluded. Host bytes are not charged to the GPU `MemoryBudget`, avoiding a false deadlock while source and upload coexist. |
| Decoder frame sequence | Bounded inventories and `FrameExecutionPlan` describe physical producers, four versioned references, and presentation ranges. Independent Replace uses existing producer ABI. Composed frames use one planar F32 allocation for RGB plus all extras, a 128-byte geometry uniform, 32 bytes of blend metadata per channel, and a 64-byte native/scalar or 192-byte common output uniform; rendered spot color adds a 32-byte record per ink at final packing. | Four post-transform slots retain all planes through shared output leases; each complete aligned surface, operation table and uniform reserves exact bytes from the backend budget. Source/ref/output leases remain in submission callbacks until completion. Initial admission failures retain the producer; dependent prefetch reports `FrameDependency`. Blocking and async advance the same ordered stages and release references on final output/cancellation. | Pre-transform patch surfaces and complete original color domains remain. Mid-presentation allocation failure is terminal, as for staged VarDCT. The common executor validates every physical LF/color/extra producer exactly once, including unused LF versions. LF planes move from exclusive transient admission into tracked leases via `MemoryPermit::split_off`; no bytes are re-admitted. Four slots retain exact producer versions until their last consumer, including across presentations. Actual GPU tests check exact retained plane bytes, last-use release and cancellation. Independent Replace releases each overwritten output before the next admission, shares immutable input/inventory across pending presentations, and hands off only the final native output; 129 Gray31 layers fit the first producer's GPU footprint, with exact submission totals and staged-cancellation evidence. Pipeline/driver and host descriptor capacity exclusions remain unchanged. |
| Bounded VarDCT decoder | `VarDctDecodeMemoryStats` accounts codestream, entropy metadata including HF block-context thresholds, every LF group's parameters/LZ/status plus the selected 64/128-byte packet-state capacity, 464 bytes of AC resume state per pass group, the resolved four-byte-aligned stream cap, one reusable packet stream peak and initial batch count, reusable AC stream peak and batch count, per-pass 39-descriptor/all-order coordinate tables, packet/artifact records, occupancy, reconstruction, capacity-strided raw metadata, coefficients, shared full-image LF/correlation/default-or-parametric-matrix/AFV resources, exact per-component resident transform planes (physically compact under JPEG subsampling), packed output storage, one optional three-plane restoration scratch set, restoration uniforms, and aggregate validation staging. Packet state is included once inside reconstruction bytes and also exposed as an audit subtotal. Staged LF groups reserve 128 bytes because the HF tree is discovered only after LF completion. Every staged single-entry frame reserves worst-case AC LZ, 464-byte execution state, status, parameters, and sink uniforms before its HF-global cursor is known. | One shared backend byte reservation covers all LF-group transient buffers until final validation and output bytes until the last lease clone. The caller/device stream cap is an upper bound; deterministic total-capacity planning searches four-byte-aligned layouts down to the 40-byte overlap/sentinel minimum, records the resolved cap, and returns typed `MemoryBudgetTooSmall` before submission if that layout cannot fit. Live concurrency remains typed non-blocking admission rather than changing an opened session's plan. Sectioned global-tree packets use one known-range packet plan and no intermediate map; their final window is co-submitted with downstream work. Windowed staged LF and host-discovered HF reuse one upload/state sequentially; only LF maps cursors, while final HF is co-submitted with downstream work and one final map validates all status. `hf_packet_stream_batch_count()` and the exact local-tree submission count become available after that plan is installed; a larger local-HF metadata peak separately admits its exact difference. A single-entry frame maps status 31, admits only exact late entropy/order/window buffers from the same budget, uploads any parametric matrix into the existing resource region, and retains both physical lifetimes until the logical final frame completes. Cancellation leaves callback-owned permits and buffers alive until mapping completes. Gaborish and EPF share one ping-pong scratch set. | Pipeline/driver-private allocations are excluded. The retained whole-codestream GPU buffer is mapped at creation and filled directly from checked source spans, with no full-size host `Vec`; it remains part of this decoder layout until every whole-range kernel is windowed. All bounded host metadata readers consume those spans directly and never decode image entropy or own an intermediate image. |
| VarDCT encoder | `VarDctMemoryPlan` charges either the bounded 512-byte parameter plus 26,880-byte artifact pair, or the 256-byte scalable parameter plus variable artifact, and an equal-size mapped readback. The bounded artifact retains its GPU-generated DC and AC fragments, histograms, and forward/quantized coefficients. The scalable artifact is a 64-word header, two words per LF group, then independently 64-word-aligned strategy (`N`), DC/token/extra (`3N` each), and checked maximum entropy-fragment sections. The source binding is reported but caller-owned. | The context's backend-wide byte budget reserves parameter + artifact + readback through map validation. The host validates fragment lengths, token counts, histograms, and zero padding before appending GPU-owned bits; it does not rescan image coefficients. Tiled quantization uses a 2-D block dispatch, so the device workgroup limit is checked per axis rather than against `blocks_x * blocks_y`; storage binding and buffer limits independently cap the complete artifact. | Bounded DCT8 AC serialization is sequential inside its control pass and capped at 256 words. Scalable/tiled and non-DCT8 paths remain zero-AC. Full 16K-square allocation is profile-valid but adapter/budget-dependent; actual-adapter coverage exercises 16Kx1 and 1x16K without claiming every adapter admits a 16K square. |
| Lossless Modular encoder | `LosslessModularMemoryPlan` reports source binding ranges (full and peak), 256-byte-aligned parameter storage, peak artifact storage, mapped readback, diagnostic total artifact bytes, batch count, exact GPU submission count, streaming mode, valid bits, component storage bytes, channel count, format, group grid, owned bytes/job and addressed bytes/job. `EncoderBufferPoolStats` separately reports exact idle bytes, three-buffer set counts, hits, misses and evictions. | Every submit non-blockingly reserves `owned_bytes_per_job` from the context's shared `MemoryBudget`. The exclusive buffer lease and permit survive until mapped artifacts are consumed. If the future is abandoned, its callback-owned lifetime unmaps and returns the set only after mapping resolves; the mapped artifact buffer is parsed in place instead of being duplicated into a host `Vec`. A bounded poll slot is reserved before `Queue::submit`, so poll saturation returns both memory and buffers without orphaning GPU work. The idle pool uses exact artifact-size matching and has an independent 32 MiB default hard limit, configurable down to zero, plus a 256-set object-count cap for tiny workloads. | Caller-owned source bindings are sampled directly: they are reported as addressed, are neither copied nor pooled, and are not charged as encoder-owned. Queue/driver-private command metadata excluded. Physical caller-visible allocation is bounded by live admitted bytes plus the separately reported idle-pool bytes. |
| Gray8 decoder | `WgpuDecodeMemoryStats` splits complete `per_frame_bytes` into `output_lease_bytes + transient_bytes`, then reports `modular_metadata_bytes`, local-stream and unique-config counts, LF-group stream count, `max_frame_slots`, `max_frame_window_bytes`, the actual stream peak, submissions, lane stride, Prefix/ANS/Mixed representation, and the selected 32/48/112-byte execution-state tail per lane. `WgpuDecodeBufferPoolStats` separately reports exact idle/leased bytes and objects, hits, misses, recycling, evictions, limits, and clear generation. | Output and transient portions use the backend-wide transient `MemoryBudget` by default, shared with encode and generic readback; an explicit cloneable budget can define another intentional sharing group. Prefetch keeps each permit from queue submission through the ordered pending frame and then the returned frame lease. Rebased global/deduplicated-local entropy metadata, reconstruction/state scratch sized for the maximum stream contract, status, mapped status staging, 256-byte POD parameter records, and the 16-byte dispatch uniform (plus a native-F64 dummy when used) have exclusive exact-size/usage/alignment leases. One stream allocation is reused in queue order across every LF/pass subimage/segment batch. The map callback owns those leases through completion; abandonment still unmaps staging before return. Output leases retain their reservation beyond session drop. Memory and bounded-poller saturation are explicit prefetch backpressure, with poll capacity reserved before source consumption and queue submission; the count limiter remains independent. | Explicit stream caps below 40 bytes are rejected. Every accepted stock Modular global/local MA profile can span bounded windows; VarDCT AC and combined/global-tree plus staged local-tree LF/HF packets do so through the separate engine. Requested Modular window exposure above 64 MiB is rejected. Idle decoder retention is bounded independently at 32 MiB, 256 buffers total, and 32 per exact key by default; all limits can be reduced to zero. Clear invalidates outstanding generations without disrupting submitted work. Raw codestream and caller-owned output buffers are never pooled. Active logical bytes and idle physical bytes are reported separately rather than double-counted. Driver-private allocations excluded. |
| Generic image readback | `ImageReadbackStats` reports frame/output counts, logical bytes, exact aggregate staging bytes, and padding bytes. One `submit_frames` call uses one staging allocation, command buffer, queue submission, map callback, and completion future/wait across all supplied frames; `ImageReadbackLimits::max_transient_bytes` and device `max_buffer_size` are enforced on that aggregate. | `max_in_flight_bytes` is a hard byte-weighted budget shared by pipeline clones (or backend clones when created from a backend). The complete staging allocation is admitted atomically. A permit and every source lease remain attached through mapping/consumption; an abandoned future leaves them owned by the callback until GPU completion, and exhaustion is a typed non-blocking error. | Codec dispatches are not coalesced by this transport API. Driver-private mapping/command metadata excluded. |
| Display textures | Pitch-linear source buffers are fully range/usage bounded. RGB, numeric, and color-image dispatches use exact 32, 64, and 144-byte Pod uniforms respectively. Color images produce RGBA8 SDR or RGBA16F wide-gamut/HDR linear BT.709 textures. | No texture-memory reservation API. | Portable `wgpu` cannot report driver-selected texture tiling/compression size; texture backing, short-lived uniform allocation internals, command metadata, and display-pipeline objects are intentionally excluded. |
| Video readback | Each frame pads and bounds its own staging copy. | Animation/session in-flight limits bound decode work. | It does not expose a separate aggregate staging-byte statistic. |

Ordinary VarDCT 2×/4×/8× frame resampling uses the scheduler's existing `upsample.wgsl` through
`ResidentUpsamplePipeline`. Its 32-byte, 16-byte-aligned `Pod` stores input/output dimensions at
offsets 0/8, strides at 16/20, factor at 24, and padding at 28. Each of three dispatches retains
one uniform. `frame_upsample_bytes` charges exactly `output_width * output_height * 12` for three
F32 planes; `frame_upsample_weight_bytes` charges `factor * factor * 25 * 4` for the single shared
phase-major kernel; `frame_upsample_uniform_bytes` is 96. The frontend expands only the bounded
15/55/210 scalar header weights. Restoration uses the encoded extent; upsampling mirrors full-frame
borders and crops right/bottom output edges before color conversion. These allocations remain in
the pending job until final validation, and are checked against device limits and the same
adaptive frame budget before allocation. Single-entry TOCs always use LF and HF-global cursor maps
and retain conservative HF storage; image dimensions no longer imply a transform strategy.

VarDCT and the render graph share `ImageOutputParams` and the `image_output.wgsl` fragment.
Its 192-byte uniform at binding 4 retains output/input dimensions at offsets 0/8, source strides
at 16, target layout fields from 28, logical bytes at 104, dispatch width at 108, zero-based
orientation at 112, source/target transfer at 116/120, identity-color flag at 124, and padded
primary-matrix rows at 128/144/160. Alpha conversion is at byte 176, followed by three reserved
words. The identity flag bypasses a redundant EOTF/OETF round trip when both transfer and primaries
match.
The codec source fragment adds a 160-byte, 16-byte-aligned `ColorSourceParams` at binding 5:
component geometry starts at 0, alpha offset/stride/maximum/enabled at 48, inverse matrix rows at
64/80/96, cube-root/scaled biases at 112/128, intensity scale at 144, and transform mode at 148.
The two uniform bindings are individually limit-checked; `output_uniform_bytes` and transient
admission charge their full 352 bytes. Read-only storage binding 6 supplies optional signed i32
alpha. Planning and pipeline creation require five storage bindings. Opaque output reuses the first
read-only color binding with the alpha flag disabled, adding no allocation. Source geometry,
offsets, depth and last addressed word are checked before encoding.
Alpha conversion divides the signed sample by its declared maximum; it never masks/wraps negative
values or overshoot. F32 preserves these values, while integer output clamps at final quantization.

Each output word resolves its samples back to codestream coordinates before XYB/JPEG reconstruction.
Target chroma subsampling averages only valid oriented pixels, then packing quantizes once into
8/10/12/16-bit codes, or writes IEEE 754 F32 RGB components without quantization. RGB storage kinds
0/1 use `bits = storage_bits = 32` for F32 and 8 for U8. Every invocation still owns one output word;
byte extraction supports unaligned float plane starts and row pitches. A source fragment supplies
`source_rgb_at` and linear `source_alpha_at` in output coordinates: unclipped linear BT.709 for XYB or encoded sRGB
for JPEG; no intermediate RGB allocation or queue submission is added. The requested `ImageLayout`
defines output lease bytes, plane gaps, row pitches, and four-byte final storage rounding. Range
checks stop at the last row payload instead of treating unused row-tail capacity as part of a plane,
and dispatch padding exits before multiplying a word index into a byte index. The GPU test covers
both one-pixel axes in all orientations with interleaved and padded RGB/RGBA planes, unaligned
plane starts, opaque and signed independently normalized alpha, zero internal/tail padding, and unchanged guard bytes outside storage.
Grayscale luminance stays folded into inverse-matrix metadata. Progressive-DC planes retain their
unoriented three-channel shape even for gray presentation or a non-RGB output request.

Modular's ordinary and inverse-stage writers use output kinds 5/6 with 32-bit samples for F32 RGB.
They normalize the 1–16-bit integer planes after inverse reconstruction, preserve alpha separately,
and store complete word-aligned float components. Ordinary records remain 256 bytes; finalizer
uniforms are now 176 bytes with independent selected-channel masks. Output planning charges the
complete 3/4-component F32 layout, validates all four-byte alignments and source/target plane bounds,
and includes sample width in
group-isolation proofs. `OrientationPolicy::Keep` lowers output orientation to identity while
retaining the checked source extent. It adds no buffer or submission and does not change LF storage.

Staged HF metadata and AC traversal use the checked MCU-padded block grid for subsampled JPEG
edges. Late host uploads of default/parametric matrices use coalesced vector ranges that exclude
GPU-reconstructed raw matrices and every aliased transform. They cannot overwrite LF resources
or the AFV basis. Raw side-image scratch can therefore be released after its mapped status without
retaining a second matrix copy or repeating image entropy.

Spectral/refinement AC uses one immutable entropy bundle and one order bundle for all passes.
Each entropy descriptor rebases its internal configuration/tree/table offsets when appended; each
160-byte invocation selects that descriptor and its complete order table. Spatial task identity is
independent of the logical validation index (`pass * spatial_group_count + group`). Disjoint LZ77
and 464-byte resume spans isolate every pass/group, including simultaneous whole-range dispatches.
Atomic signed-integer addition accumulates into one coefficient buffer; dequantization and
restoration run only after all AC work. The final aggregate map validates every pass. Deferred
HF-global admission multiplies status, parameter, and worst-case history/state capacity by the
declared pass count. Intermediate pass output is not published.

The final word (byte offset 156) of each 160-byte AC parameter record is `stream_end`: zero
requires the packet's exact padded end, one selects an entropy cursor continuation. The latter
validates the terminal ANS state and leaves the cursor unaligned for the next consumer; a stream
that finishes in a nonfinal input window can return success immediately. The 32-byte status and
464-byte resume ABI remain unchanged. Host continuation validation checks group identity and
both bounds against its packet plan, including the reported token end. Changing a logical pass
group's mode updates both whole-range and window-resume parameter records. Public distributed
extra execution selects continuation only for AC groups with nonempty Modular subimages, while
all other groups retain exact packet termination. Host validation rebases bounded cursors against
the original packet range; no input buffer contents are treated as authoritative bounds.

A Modular frame containing only DC-global image samples uses its admitted frame arena and zero
subimage lanes. The otherwise-unused reconstruction binding retains a counted four-byte placeholder.
The final global batch shares the same inverse, progressive-DC conversion/output, and aggregate
status-map function used by the final LF/pass batch, without an extra submission. This closes the
global-only root in the checked recursive DC-plus-AC fixture.

Raw mode-7 VarDCT side images use a dynamic, exact shared-budget reservation because later matrices
are discovered only after the preceding GPU entropy cursor is mapped. The reservation covers
packed MA/channel metadata, the transformed/inverse arena plus entropy state and LZ/predictor
scratch, dummy output, decode status and its staging copy, the 256-byte entropy parameters,
16-byte dispatch control, 64-byte overlay uniform, and every Palette/RCT/Squeeze inverse uniform.
It is acquired before allocation and remains attached to the pending stage and its map callback
until status validation and matrix overlay complete. The callback also retains the frame lifetime,
so cancellation cannot release either reservation while a submission still uses the resource table.
The large frame-resident resource table is already covered by
`VarDctDecodeMemoryStats` and is not charged again. The decode binding is one reusable four-byte-
aligned upload copied directly from shared codestream spans. Lazy window geometry carries 16-byte
overlap and a four-byte sentinel; its bytes are included in the same permit. The caller/device cap
can shrink against currently available budget to the 40-byte minimum. An unaffordable minimum
returns typed `MemoryBackpressure` before recording that image. Mapped cursors are rebased to
absolute codestream bits and checked against the current window; a valid terminal sample count and
ANS state stop before the enclosing HF-global upper bound. The frame's existing whole-codestream
GPU buffer remains separately accounted for other consumers.

The common `ModularSideImagePlan` separates image geometry, transformed meta-channel count,
MA/channel descriptors, original plane views, inverse jobs and entropy bounds from the raw-matrix
denominator and targets. `wgpu_engine::side_image::modular` records entropy into an initial encoder
and retains inverse jobs and downstream consumers in a completion encoder when input is windowed.
The raw wrapper appends overlay to that completion encoder even when there are no inverse jobs.
Each entropy window maps a 16-byte status. Only validated completion submits inverse/overlay work
and its status copy; whole input keeps them in the initial submission. The wrapper retains and
charges exactly its 64-byte uniform. The common arena also permits GPU copies for
downstream plane delivery. Its contents stay integer words, without sample normalization or color
conversion. The 256-byte entropy, 16-byte status and 64-byte overlay ABIs and workgroup memory stay
unchanged. Submission accounting includes every raw window and any deferred finalization.

The global VarDCT extra-channel stage uses the common executor directly from encoded host spans,
copying consumed windows into one reusable GPU stream buffer with 16-byte overlap on each side
and a four-byte zero sentinel. No initial whole-codestream GPU allocation is needed. The shared
`EntropyStreamWindows` geometry computes each range on demand in constant host space; it also
drives oversized ordinary group streams. The caller/device cap and total budget capacity resolve
the actual binding, down to the 40-byte minimum. One 256-byte storage parameter buffer is rewritten
in queue order; the arena's 48/112-byte MA/predictor/ANS/LZ77 execution tail survives each upload.
A 16-byte status map either yields to the next window or validates an early ending cursor. Once
complete, a bounded stream submits its retained inverse command/uniforms and maps the same status
before advancing to color; a whole stream keeps entropy and inverses in one submission. Zero-bit
single-symbol Prefix streams use only a four-byte sentinel at byte-aligned endpoints.
Arena and transient bytes have separate permits, both acquired before allocation. The map
callback retains the job and both reservations even when the pending frame is abandoned. After
status/entropy/cursor validation, transient buffers retire and the first-alpha or explicitly
selected extra plane retains the arena lease through the frame job. An unused arena retires
immediately. Frame preflight subtracts the
retained arena from its available per-frame limit; every subsequently discovered LF/HF/raw-table
allocation still uses the same shared budget. Only the 16-byte status crosses to the host.

Public `VarDctDecodeSession::memory_stats()` is the optional frame-stage plan, published after
cursor validation. It excludes the separately owned global arena; `global_modular_memory_stats()`
reports initial-stage buffer bytes and `in_flight_memory_stats()` is authoritative for live bytes.
Seven public fixtures cover single/multi-entry TOCs, multiple AC passes, independent alpha depths,
Apply/Keep and fragmented input. Entropy failure never exposes a color frame; cancellation,
initial admission/retry, middle-window cancellation, budget-driven upload reduction and undersized
window/budget rejection have actual-adapter tests. Internal
plane readback remains test-only. Distributed LF/AC extras now use the same executor.

Scalar VarDCT output retains complete LF/HF/AC validation and its coefficient/metadata resources,
but allocates no resident color planes, inverse-transform scratch, restoration or color-resampling
resources. Its 64-byte packing uniform and four-byte status are transient; the four-byte status
tail joins the existing aggregate map, and packed output uses the normal output lease. Native
range failures become `ModularScalarOutputError::SampleOutOfRange` before validated delivery.
The packer reads the retained integer view directly, writes each output word once (including zero
padding), and uses shared orientation helpers. No extra submission or intermediate color image
is introduced. The same seven public fixtures select all 32 extra planes in both output modes.

Distributed VarDCT extras reserve their full transformed/inverse arena separately from the remaining
frame transient buffers. `extra_arena_bytes` and `extra_inverse_uniform_bytes` are included in
`total_frame_bytes`; a retained nonempty global-prefix arena is separate and shares the same budget.
Each LF/AC subimage is admitted after its descriptor is known. Its reusable input can shrink to
available capacity; all image workspace, metadata, uniforms and 16-byte status/staging buffers
remain charged until its callback releases them. Dynamic subimage bytes appear in the live budget,
not the immutable base-frame plan. The callback also retains the frame lifetime.

After validated local entropy, any deferred local inverse precedes checked GPU row copies into
the frame arena. A status copy after those copies fences their completion before releasing the
local reservation. LF completion resumes HF metadata; AC completion checks padding to the TOC
boundary. Once all required subimages finish, a retained command buffer runs the global inverse
and output. The frame exposes no unvalidated output before this tail is submitted. All stages
use the existing entropy/status/parameter ABIs; no pixel or coefficient readback is introduced.

Selected resampled integer and floating channels use `ModularRenderPlan` in both coding-mode producers.
It allocates one aligned F32 destination arena, one reusable low-resolution normalization plane
large enough for the largest selected resampled input, and one weight table per distinct factor.
Destination view offsets satisfy the adapter's storage alignment. Each selected source has a
32-byte normalization uniform; each factor above one adds the existing 32-byte upsampling uniform.
The source representation is decoded before interpolation, so packing performs no intermediate integer
quantization. Only bounded scalar weight expansion runs on the host.

`modular_render_bytes` (Modular) and `extra_render_bytes` (VarDCT) include every destination,
scratch, weight and uniform byte in the base admission plan. Modular resampling forces frame-wide
assembly when multiple groups contribute, preserving filter neighbors across group boundaries.
VarDCT keeps its existing extra arena lease and renders only the first alpha or selected scalar
plane. Render buffers and uniforms remain in the frame lifetime through cancellation and the
final validation callback. The reconstruction/packing dispatches join the existing output tail;
they add neither a submission nor a status readback.

For cross-group DC-global Palette/Squeeze, the Gray8 decoder additionally charges one
`frame_modular_arena_bytes` allocation containing transformed samples plus its optional LZ77,
Weighted-predictor, and aligned execution-state tail. Pass-group lanes copy only validated row
ranges into disjoint views, after which frame-wide inverse and finalizer uniform bytes remain covered
by the same transient permit. `global_reconstruction_sample_words` exposes the DC-global decoded
prefix separately. Channels with both shifts at least three use LF-group streams; channels with
either shift below three use pass-group streams, including asymmetric shifts. The LF streams are
scheduled first and share the same scratch lanes, bounded uploads, metadata inventory, and aggregate
status map with the pass streams. One through three declared passes assign channels through the
header's shift brackets, schedule only nonempty subimages, and expose `progressive_pass_count`; empty
physical sections are zero-validated without allocating a lane. Recursive progressive-DC uses one
logical pending state. A single-entry intermediate first runs GPU HF metadata to status 31, maps
only the HF-global cursor, then admits exact entropy/order/window bytes and submits general AC plus
resident reconstruction before the next dependency. Its worst-case LZ ring, 464-byte resume state
per pass group, status, parameters, and sink uniforms are included in the initial reservation;
late buffers use the same budget and all producer lifetimes remain retained. Parametric custom
matrix modes overwrite the already-accounted resource-table matrix region and add no GPU
allocation. Sectioned raw matrices add only their exact temporary reservation above; local-tree
packets validate every LF and bounded HF-local metadata stage before entering it. Raw matrices with
their own local MA tree and local LF/HF packet trees have whole/windowed coverage. Final validation
records the HF-metadata stop selected when those packet commands are built, including late raw
continuations. Larger/transformed raw images, the remaining Global/LF/HF streams, and intermediate
presentation still require broader scheduling.

## Shader write bounds fixed by this audit

The Gray8 encoder artifact consists of 53 header words followed by four-word events. The WGSL
`append_event` function now derives capacity from `arrayLength(&output_words)`, checks the event
index before every record write, and emits an overflow sentinel instead of indexing out of bounds.
An exhaustive host mirror checks every zero/nonzero residual stream through 16 samples, plus
maximum-size adversarial patterns; it proves the final event word remains inside the allocated
artifact. The checked fixture remains byte-for-byte stable.

The display image validator now checks the final addressed byte, not only the host `u64` range, so
a valid large host buffer cannot wrap a WGSL `u32` byte index. VarDCT uploads are now bounded by
both relevant device limits, and DCT8 rejects insufficient workgroup storage before encoding the
pass.

The Gray8 decoder classifies output storage through `classify_pixel_format`, checks all four plane
offset/stride/end values and the complete logical allocation against WGSL's `u32` address space,
and separately enforces four-byte row alignment for whole-word RGBA, packed-4:2:2, 32-bit numeric,
and 64-bit numeric writes. Numeric U8/S8/U16/S16 use bounds-checked byte stores; 2S16, U32/S32,
F32/2F32, and F64 use aligned whole words. The F64 template is validated in both portable
exact-F32-widening and `FLOAT64`-capable native forms; the native pipeline is compiled lazily only
for a resolved native-F64 request. `ShaderF64Policy::Auto` requests `SHADER_F64` when the adapter
advertises it, `Disabled` omits it, and `Require` returns a typed error when unavailable. The Naga
regression test also proves that the native WGSL is rejected without the `FLOAT64` validator
capability, so a source containing `array<f64>` cannot leak into the portable pipeline.

## Regression coverage

The ABI tests pin every Rust size, natural alignment, and field order, including the 16-byte
VarDCT alignment, the color/transfer/blend uniforms, and both Gray8 readback schemas.
GPU tests compile and execute every portable core shader, all display formats, every VarDCT
strategy, the deterministic encoder fixture, and the bounded decoder. Dedicated tests cover:

- storage-binding device-limit selection;
- checked resident aliasing and exact transient estimates;
- pending transient accumulation and release for multiple submissions;
- 4-byte and 256-byte copy-pitch rules;
- display final-address rejection above WGSL's `u32` space;
- host negotiation and exact GPU readback for all 30 VPI pitch-linear formats (20 color-bearing
  and 10 explicitly normalized numeric formats),
  including odd extents, Y16, four-plane alpha, and packed-4:2:2 tail duplication;
- explicit same-queue numeric display and texture readback for all 10 VPI numeric formats,
  including signed endpoints, two-component visualization, NaN/infinity policy, unit clamp,
  Linear/sRGB transfer, and reported native/portable F64 precision;
- actual-GPU RGBA16F display of BT.2020 PQ/OETF, Display-P3 HLG, and BT.2020
  constant-luminance YCbCr against independent scalar oracles, plus typed rejection of implicit
  wide/HDR conversion into RGBA8;
- scalar-oracle actual-GPU conversion across D65 BT.709/BT.2020/Display-P3 primaries,
  Linear/sRGB/BT.709/PQ/HLG/BT.2020 transfer contracts, and both BT.2020 NCL and
  constant-luminance YCbCr, plus pre-dispatch typed rejection of mismatched, undefined, or
  numerically incomplete contracts;
- typed rejection of missing/mismatched numeric mappings and native-required F64 on devices without
  enabled `SHADER_F64`, plus an explicitly skipped native-F64 GPU test on unsupported adapters;
- the 1,536-byte DCT8 and 2,304-byte special-VarDCT workgroup-storage requirements, all 27 inverse
  transforms, mixed strategy buckets, and raster versus transposed coefficient placement;
- custom LF dequantization and LF/HF chroma-correlation parameters through Naga validation and
  actual-GPU numeric buffer readback, including the 144-byte component-aware LF uniform and
  288-byte HF-lowering uniform;
- stream-defined inverse XYB, all six standard transfer curves, and all eight JPEG XL patch/frame
  blend modes, including straight and associated alpha; and
- the encoder's worst-case event capacity.

Any new WGSL host record must extend the ABI table and size/alignment tests. Any new buffer must be
added to both the per-job estimate and, where jobs can overlap, the corresponding reservation or
observable in-flight total before the shader is advertised as supported.

## Decoder crop/blend composition ABI and lifetime

`jxl_wgpu_decode::codec_engine::composition` retains RGB with an explicit sRGB or linear transfer
tag, followed by every extra plane in one F32 allocation. Blending and post-transform references
require sRGB; unreferenced XYB presentations can keep linear RGB until final packing.
A plane has `width * height` samples and its starting byte
offset is aligned to `max(4, min_storage_buffer_offset_alignment)`. The complete allocation is
`(3 + extra_count) * aligned_plane_bytes`. `FrameSurfaceLayout` validates this total against
buffer/binding/u32 limits before allocating extra views. Output 0 describes only the three color
planes; outputs 1 onward describe independently normalized scalar extras with their real global
offsets and logical ends. Every output aliases the same tracked `GpuBufferLease`, so one allocation
is charged once. Import validates layout, extent, output IDs and buffer identity. Only the private
physical-producer/compositor boundary exposes these all-channel views.

Modular uses one 176-byte color finalizer plus a 176-byte scalar finalizer per extra, for every
existing group/frame-finalizer batch. The shared allocation and all uniforms enter admission
before GPU submission. Resampled sources pass all selected planes through the common resident
normalizer/filter, without an RGBA channel-count cap. VarDCT retains a vector of original extra
views over one global/distributed arena lease, normalizes/resamples them together, then copies the
F32 planes to their aligned final offsets after color packing. The render output therefore has
`STORAGE | COPY_SRC` usage. Shared arena reservations are subtracted once at stage transitions,
regardless of how many views retain them. All outputs and changed regions are published only after
physical entropy/inverse validation.

The 128-byte, 16-byte-aligned `BlendParams` contains canvas geometry (width, height, plane words,
channel count), crop intersection, foreground geometry, dispatch geometry, and four reference
geometry/presence records starting at byte 64. A separate readonly table has one 32-byte
`BlendChannel` per color/extra plane: mode, background slot, absolute alpha plane, clamp/association
bits; then alpha-background slot and three reserved words. The shader binds foreground at 0,
four references at 1–4, output at 5, channel metadata at 6 and the uniform at 7. Its seven storage
bindings fit the portable limit. Missing slots bind foreground but disable reads; no image-sized
zero buffer is allocated. Metadata selectors and device binding/workgroup limits are checked
before submission. Reference clones share their allocation charge and remain immutable until
replacement is ready.

Negative/oversized crops are intersected using host `i64` and lowered to in-bounds unsigned
coordinates. One 64-lane invocation owns one channel sample. Each channel uses its own background
slot and alpha selector; the alpha background comes from the selected alpha's own slot. Replace,
Add, straight/associated source-over, alpha-weighted Add and Multiply run without quantization.
Color source-over updates its selected alpha; alpha's own weighted Add keeps its background.
Multiply clamps only foreground. Absent alpha declarations make color Blend replace and weighted
Add add, with virtual opaque alpha at presentation. References keep the image-header association;
the physical producer requests Preserve, and only final output may change association.

Final packing uses either the shared 192-byte `ImageOutputParams` with planar source overrides or
the 64-byte, 16-byte-aligned `NativeParams`: output/source extents, channel/depth/row format,
byte-size/dispatch/orientation/alpha-conversion, then source plane stride, first-alpha plane,
selected scalar plane and flags at byte 48 (bit 0: F32 output; bit 1: linear RGB source).
The common packer selects one 192-byte parameter set for the actual source domain; the native
packer encodes linear RGB to sRGB before association/quantization, leaving extras unchanged.
Native color and selected extras clamp and round
once at presentation; scalar F32 preserves extended normalized values without color/alpha
conversion. One invocation owns one output word including tail padding. Both shaders support 2-D
linear dispatch and checked u32 byte addressing, buffer/binding sizes and arithmetic overflow.

When Render applies to color with spot declarations, either packer also binds a readonly ink table
at binding 7. Each 32-byte, 16-byte-aligned `SpotColor` stores the absolute plane word offset at
byte 0, three reserved words, and declared RGBA at byte 16. Declarations are visited in order;
`mix = solidity * normalized_sample` is not clipped, and only RGB changes. Mixing follows reference
storage and precedes target color conversion, association and chroma filtering. Numeric requests
and Preserve compile a no-op presentation helper and allocate/bind no ink table. The complete table
is checked against storage limits and charged with the output uniform for the submission lifetime.
No additional image-sized spot buffer or readback is introduced.

Each job reserves a native poll slot before submission, takes GPU access guards on every input,
and retains source/reference/output leases, operation-table bytes and uniforms through completion
or error callbacks. `composition::submission` owns this common lifetime; `composition::blend`
lowers the per-channel metadata, while `composition::gpu` handles surfaces and output pipelines.
Cancellation drops unsubmitted input immediately; submitted reservations survive until callbacks
release them. Presentation completes only after every physical status check and final packing.
Unvalidated output is unavailable until packing is submitted. Initial admission can be retried
without changing frame order; later allocation failure is terminal. One composed presentation
runs at a time while caller-held previous output leases remain independent.

Native completion uses the command encoder's work-done callback. On WebGPU those callbacks
require `Send`, while browser buffer handles are local to the event loop. A separately accounted
4-byte `MAP_READ | COPY_DST` fence is cleared after the composition commands and mapped with a
browser-local callback. Its contents are never read. That callback retains all buffer leases,
unmaps the fence and signals completion without a runtime dependency or CPU image processing.

## Alpha association at output

Integer sample delivery covers valid depths 1–31 with canonical 8/16/32-bit storage. The existing
Modular finalizer (176 bytes), scalar packer (64 bytes), and composition native packer (64 bytes)
retain their layouts. Their existing component-storage field determines one, two, or four output
bytes; valid precision never substitutes for storage stride. Native high padding bits remain zero.

`modular_sample.wgsl` shares exact integer alpha rescaling and final F32 quantization. Multiplication
uses two `u32` limbs; alpha division uses a bounded 32-step unsigned quotient with a 31-bit divisor.
Quantization decomposes the binary32 significand, multiplies by the exact requested maximum and
rounds the rational product to the nearest integer (half up). This avoids F32 multiply rounding
and 31-bit endpoint overflow without requiring `SHADER_F64`. Intermediates are invocation-local;
no new workgroup memory, bindings, uniforms, scratch allocation or readback is required.

High-precision Modular sources select the descriptor-based entropy/inverse/finalizer path so the
legacy 8/16-bit direct packers cannot truncate them. Converted wide color output uses the existing
accounted all-channel F32 presentation surface, shared with floating sources and composition.
That surface's allocation and lifetime continue to follow the common frame byte budget.

The shared `ImageOutputParams` is 192 bytes: `alpha[0]` at byte 176 selects Preserve (0),
Unpremultiply (1) or Premultiply (2), followed by three zero padding words. Existing field offsets
are unchanged. `AlphaConversion` is a typed host enum and `ALPHA_OUTPUT_SHADER` is shared by the
render graph, VarDCT, Modular and composition native packers. Conversion follows the target RGB
transfer/primary transform and precedes quantization and chroma subsampling; constant-luminance
YUV re-linearizes the converted encoded RGB for its luma calculation. The alpha component itself
remains unchanged. Packed-4:2:2 odd tails clamp alpha in the same oriented coordinates as replicated luma. The finite floor is exactly `2^-26` for both division and multiplication.

The Modular finalizer retains its 176-byte ABI: `bounds.w` at byte 156 now carries the conversion.
A converting RGB request retains its first-alpha source view even if the destination omits alpha.
It uses the generalized finalizer instead of a direct integer kernel; no extra copy or buffer is
needed. Numeric requests always select Preserve. The composition native packer uses `output.w`
at byte 44 for the same conversion; its 64-byte ABI adds planar source addressing at byte 48
and preserves output word ownership.
All references keep the source association, independent of the caller's output policy. Enlarged
common-output uniforms are charged through the existing size-derived admission and lifetime paths.

### Modular color and restoration integration

`modular_render/color.wgsl` consumes the inverse-Modular arena and three decoded F32 output
planes with one 80-byte aligned `NormalizeColorParams` uniform. Its three source records retain
width, height, stride and word offset; a fourth vector carries sample encodings and the XYB flag;
a fifth carries LF multipliers. The 16×16 dispatch has checked extents and storage ranges. XYB
working-word addition precedes conversion to F32; source precision metadata does not normalize XYB.

Color normalization owns three coded-resolution planes. Gaborish/EPF add one reusable three-plane
ping-pong set; color resampling adds three presentation-resolution planes only when needed. Extras
reuse one separately sized normalization scratch buffer. The common `color_output` packer writes
into the existing aligned all-channel render arena; its 192/160-byte uniforms and each 80-byte
Gaborish/EPF or 32-byte upsampling uniform are included exactly once. A 37×17, factor-2 color plan
with Gaborish and two EPF passes accounts 11,652 working-plane bytes and 768 uniform bytes,
separately from the final render arena and shared weights.

`ResidentEpfSigma::Constant` stores the negative inverse sigma in byte 72 of the existing 80-byte
EPF uniform, with mode 2 and one remaining padding word. It binds an existing read-only source
for the unused sigma binding and allocates no sigma image. Modes 0/1 retain the generic scalar
buffer and VarDCT block-plane contracts. The scheduler and resident Rust records share the WGSL
field order; ABI reflection and actual GPU filtering validate this boundary. Invalid constants,
Modular sigma below 1e-8, mismatched grids and device limits are typed errors. Every temporary
allocation is included in `ModularRenderPlan::total_bytes` and the shared decoder reservation.

Modular LF producers share `modular_render/color.wgsl` normalization (80-byte `NormalizeColorParams`),
Gaborish, constant-sigma EPF and frame upsampling with presentation frames. They stop before the
color packer. `modular_render_bytes` includes every reconstruction allocation, weight and uniform;
`progressive_dc_plane_bytes` and `progressive_dc_uniform_bytes` report subsets of that total.
The final buffer set is upsampled storage when present, otherwise the restoration destination
selected by pass parity. Each final plane splits its exact reservation from the exclusive
transient permit before submission. Releasing producer scratch preserves only those three plane
reservations; cloning/deleting LF slot versions does not duplicate/release a live reservation.
The allocation test covers 32 combinations of factors 1/2/4/8, Gaborish and EPF0/1/2/3 with
37×17 final geometry and padded input rows. Both lane selection and frame admission include this
complete footprint. No separate Modular-to-LF conversion shader or uniform remains.
