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
| `jxl_wgpu/chroma_upsample.wgsl` | `ChromaUpsampleUniform` / `Params` | `input_width, input_height, output_width, output_height, input_stride, output_stride, axis, _pad0` | 32 | 4 | uniform |
| `jxl_wgpu/chroma_2d.wgsl` | `Chroma2dUniform` / `Params` | `input_width, input_height, output_width, output_height, input_stride, output_stride, _pad0, _pad1` | 32 | 4 | uniform |
| `jxl_wgpu/gaborish.wgsl` | `GaborishUniform` / `Params` | `width, height, input_stride, output_stride, weight0, weight1, weight2, _pad0` | 32 | 4 | uniform |
| `jxl_wgpu/gaborish_rgb.wgsl` | `GaborishRgbUniform` or `ResidentGaborishParams` / `Params` | dimensions/6 strides, then four values for each of X, Y and B: `weight0, weight1, weight2, pad` | 80 | 4 / 16 | uniform |
| `jxl_wgpu/epf.wgsl` | `EpfUniform` or `ResidentEpfUniform` / `Params` | dimensions/6 image strides, sigma dimensions/stride/kind, 6 filter floats, `_pad0, _pad1` | 80 | 4 / 16 | uniform |
| `jxl_wgpu_decode/vardct_epf.wgsl` | `EpfSigmaUniform` / `Params` | LF-group block/task/sharpness geometry, full-image block-grid extent plus group destination origin, artifact status/task offsets, global scale, quant multiplier, two four-value sharpness LUT rows | 80 | 16 | uniform |
| `jxl_wgpu/upsample.wgsl` | `UpsampleUniform` / `Params` | `input_width, input_height, output_width, output_height, input_stride, output_stride, factor, _pad0` | 32 | 4 | uniform |
| `jxl_wgpu/ycbcr_to_rgb.wgsl` | `YcbcrUniform` / `Params` | `width, height, cb_stride, y_stride, cr_stride, output_stride, component, _pad0` | 32 | 4 | uniform |
| `jxl_wgpu/xyb_to_rgb.wgsl` | `XybUniform` / `Params` | dimensions/6 strides, three padded inverse-opsin rows, padded cube-root bias, padded scaled bias, `intensity_scale`, 3 pads | 128 | 4 | uniform |
| `jxl_wgpu/transfer_function.wgsl` | `TransferUniform` / `Params` | dimensions/6 strides, `transfer, gamma, intensity_target, min_nits, luminance_rgb` | 64 | 4 | uniform |
| `jxl_wgpu/blend.wgsl` | `BlendUniform` / `Params` | dimensions/5 value strides, 2 alpha strides, `mode, component, clamp, alpha_associated, has_alpha` | 48 | 4 | uniform |
| `jxl_wgpu/premultiply_alpha.wgsl` | `PremultiplyUniform` / `Params` | `width, height, color_stride, alpha_stride, output_stride, _pad0, _pad1, _pad2` | 32 | 4 | uniform |
| `jxl_wgpu/extend.wgsl` | `ExtendUniform` / `Params` | image/frame dimensions, 3 strides, signed origin, `has_reference`, 2 pads | 48 | 4 | uniform |
| `jxl_wgpu/save.wgsl` | `SaveUniform` / `Params` | `width, height, source_stride, channels, channel, layout (output_layout), orientation, _pad0` | 32 | 4 | uniform |
| `jxl_wgpu/rgb_to_image.wgsl` | `ImageOutputUniform` / `Params` | dimensions/3 source strides, format fields, 4 plane offset/stride pairs, `logical_size, dispatch_width, orientation, source_transfer, target_transfer`, 1 pad | 128 | 4 | uniform |
| `jxl_wgpu/display_rgb.wgsl` | `DisplayRgbParams` / `DisplayRgbParams` | `width, height, channels, sample_type, layout (storage_layout), logical_samples, _padding0, _padding1` | 32 | 4 | uniform |
| `jxl_wgpu/display_numeric.wgsl` | `DisplayNumericParams` / `NumericParams` | dimensions/type/depth/components, plane offset/stride, visualization/non-finite/transfer/clamp, reserved word, `scale, bias`, 2 pads | 64 | 4 | uniform |
| `jxl_wgpu/display_image.wgsl` | `DisplayImageParams` / `Params` | dimensions/format fields, 4 plane offset/stride pairs, `chroma_width, chroma_height, transfer` | 96 | 4 | uniform |
| `jxl_wgpu/vardct_dct8.wgsl` | `GpuTask` / `Task` | `coefficient_offset, destination_x, destination_y, quant_index, matrix_index, correlation_index, lf_index` | 28 | 4 | storage element |
| `jxl_wgpu/vardct_dct8.wgsl` | `GpuResourceVector` / `vec4<f32>` | four `f32` lanes | 16 | 16 | storage element |
| `jxl_wgpu/vardct_dct8.wgsl` | `Dct8Uniform` / `Params` | `task_count`, output dimensions/3 strides, 4 resource offsets, 2 pads, `quant_biases[4]`/`vec4<f32>` | 64 | 16 | uniform |
| `jxl_wgpu/vardct_general.wgsl`, `vardct_special.wgsl` | `ResidentVarDctParams` / `Params` | task range, transform/LF dimensions, resource offsets, 3 output dimension/stride tuples, transform/correlation geometry, artifact task/bucket offsets, LF stride, 3 pads, `quant_biases[4]` | 128 | 16 | uniform |
| `jxl_wgpu_decode/vardct_resource.wgsl` | `VarDctResourceParams` / `Params` | geometry and source/output offsets, X/Y/B LF scales plus extra-precision multiplier, final LF X/B chroma-correlation slopes and 2 pads | 64 | 16 | uniform |
| `jxl_wgpu_decode/vardct_artifact.wgsl` | `HfMetadataLoweringParams` / `Params` | six `vec4<u32>` records for dimensions/capacities/image/artifact/metadata/source offsets, seven `vec4<u32>` records containing all 27 strategy matrix offsets, X/Y/B dequantization scale multipliers, then base X/B correlation and reciprocal colour factor | 240 | 16 | uniform |
| `jxl_wgpu_decode/vardct_packet.wgsl` | `VarDctPacketControl` / `PacketControl` | eight `vec4<u32>` records for section ranges, geometry, physical metadata offsets/capacities, expectations, quantization, streams, and scratch | 128 | 16 | uniform |
| `jxl_wgpu_decode/vardct_packet.wgsl` | `VarDctModularParams` / `Params` | 12-byte entropy prefix; logical/upload window starts, stream/yield ends, flags, state offset, stream base; 49 consumer words; one pad | 240 | Rust 16 / WGSL 4 | read-only storage |
| `jxl_wgpu_decode/vardct_packet.wgsl` | `GenericPacketExecutionState` / reconstruction words | common entropy/LZ/consumer state, five reserved LF/HF phase words, previous gradient, two pads | 64 | 16 | storage subrecord |
| `jxl_wgpu_decode/vardct_packet.wgsl` | `WeightedPacketExecutionState` / reconstruction words | generic prefix plus four true errors, twelve subprediction-error accumulators, two pads | 128 | 16 | storage subrecord |
| `jxl_wgpu_encode/vardct_encoder.wgsl` | `VarDctKernelParams` / `Params` | source/block geometry, strategy/quantization, 19 two-word prefix entries, X/Y/B inverse LF-dequantization factors, X/B LF-correlation slopes, and 12 pads | 256 | 4 | read-only storage |
| `jxl_wgpu_encode/vardct_large_encoder.wgsl` | `ScalableVarDctKernelParams` / `Params` | source/block geometry, strategy/quantization, 19 two-word prefix entries, five artifact offsets/capacities, topology, LF-fragment descriptor offset/length and LF-grid dimensions, X/Y/B inverse LF-dequantization factors, and X/B LF-correlation slopes | 256 | 4 | read-only storage |
| `jxl_wgpu_encode/vardct_large_encoder.wgsl` | `ScalableVarDctArtifactHeader` / header words | status/live counts, section offsets/lengths, total fragment bits, source/block geometry, topology, 19-bin histogram, LF-fragment descriptor offset/length/grid/count, 18 pads | 256 | 4 | storage/readback record |
| `jxl_wgpu_encode/vardct_large_encoder.wgsl` | `ScalableDcFragmentDescriptor` / two words | `bit_offset, bit_len` for one row-major LF group | 8 | 4 | storage/readback element |
| `jxl_wgpu_encode/lossless_gray8.wgsl` | `Gray8Params` / `Params` | `width, height, row_stride, byte_offset` | 16 | 4 | uniform |
| `jxl_wgpu_encode/lossless_gray8.wgsl` | `Gray8ArtifactHeader` / `output_words[0..53]` | `event_count, raw_counts[19], lz77_counts[33]` | 212 | 4 | storage/readback record |
| `jxl_wgpu_encode/lossless_gray8.wgsl` | `Gray8Event` / four-word event | `kind, token, extra_bit_count, extra_bits` | 16 | 4 | storage/readback element |
| `jxl_wgpu_decode/lossless_gray8.wgsl` | `ShaderParams` / `Params` | token range, dimensions/sample count, output kind/transfer/range, channels/order/depth, 4 plane offset/stride pairs, chroma dimensions, logical size, numeric mapping | 96 | 4 | uniform |
| `jxl_wgpu_decode/lossless_gray8.wgsl` | `DecodeStatus` / `status[0..4]` | `code, decoded_samples, cursor, expected_cursor` | 16 | 4 | storage/readback record |

### Values that intentionally are not `Pod`

`Pod` describes an in-memory Rust/WGSL ABI, not every sequence of bytes handled by the workspace:

- JPEG XL codestreams, container boxes, and the `jwgp` acceleration index are wire formats. Their
  serializers use explicit little-endian fields and packed offsets. In particular, a serialized
  prefix entry is three bytes (`u8` plus `u16`), while Rust's naturally aligned
  `PrefixCodeEntry` occupies four bytes. Deriving `Pod` would serialize host padding and produce a
  different, invalid wire format.
- Raw image planes and mapped packed-image byte ranges have runtime-selected formats, plane counts,
  row strides, and lengths. `ImageLayout` validates their byte ranges; there is no single fixed
  Rust record that could safely represent their contents.
- Codestream storage and the decoder prefix lookup are variable arrays, not fixed records. The
  lookup is retained session-locally as `Arc<[u32]>` and uploaded with `cast_slice`; the raw
  codestream remains bytes so its bit/byte offsets and explicit four-byte sentinel are preserved.
  Aligned codestream spans are uploaded directly from shared input storage, without constructing
  a second full-size host `Vec`.

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
| `ycbcr_to_rgb` | Cb/Y/Cr RO, output RW, U | 16x16 | Tier A (`KernelVariant` 2-D) | one checked dispatch per output component |
| `xyb_to_rgb` | X/Y/B RO, R/G/B RW, U | 16x16 | Tier A (`KernelVariant` 2-D) | one dispatch; checked common F32 extent, per-plane strides, finite inverse-opsin parameters and positive intensity target |
| `transfer_function` | R/G/B RO, R/G/B RW, U | 16x16 | Tier A (`KernelVariant` 2-D) | one dispatch; checked common F32 extent and Linear/sRGB/BT.709/Gamma/PQ/HLG parameters |
| `blend` | base/source/base-alpha/source-alpha RO, output RW, U | 16x16 | Tier A (`KernelVariant` 2-D) | one scalar-channel dispatch; two or four equal F32 inputs keep the shader within portable storage-binding limits |
| `premultiply_alpha` | color/alpha RO, output RW, U | 16x16 | Tier A (`KernelVariant` 2-D) | one checked dispatch per color component |
| `extend` | frame/reference RO, full-canvas output RW, U | 16x16 | Tier A (`KernelVariant` 2-D) | exact u32 word copy for I32/F32; checked signed origin, crop, target extent and optional reference canvas |
| `save` | source RO, packed output RW, U | 16x16 | Tier A (`KernelVariant` 2-D) | checked orientation and exact packed allocation |
| `rgb_to_image` | R/G/B RO, packed output RW, U | 256x1 | Tier A (`KernelVariant` 1-D) | checked linear word count is split into a legal 2-D dispatch; shader checks `logical_size` before stores |
| `display_rgb` | source RO, destination T, U | 16x16 | Tier A (`KernelVariant` 2-D) | source must have `STORAGE`; logical samples and final source address fit the bound range/WGSL `u32` |
| `display_numeric` | source words RO, destination T, U; native-F64 variant also binds the same source as F64 RO | 16x16 | Tier A (`KernelVariant` 2-D) | exact pitch-linear plane range/stride and WGSL `u32` addresses; explicit sample kind, affine mapping, non-finite handling, clamp, transfer, and channel visualization |
| `display_image` | source RO, destination T, U | 16x16 | Tier A (`KernelVariant` 2-D) | source must have `STORAGE`; each pitch-linear plane and its final address is bounded |
| `vardct_resource` (decoder) | LF-group table RO, full-image dequantized-LF atlas RW, U | 64x1 | Tier A (`KernelVariant` 1-D) | checked local block count, global atlas stride/origin, and output range; one 1D workgroup per LF-group block batch |
| `vardct_packet` (decoder) | whole or reusable-window codestream/MA metadata RO, reconstruction/raw metadata/coefficients/status RW, control U, Modular params RW | 1x1 | Tier B (fixed) | combined global-tree and split LF/HF local-tree entry points use logical channel widths with explicit physical strides; both staged local-tree streams resume across ordered windows using one 64/128-byte aligned state and shared upload, preserve LF storage, map one aggregate LF cursor set, and defer authoritative status to the final downstream map |
| `vardct_artifact` (decoder) | LF-group raw metadata RO, artifact/occupancy plus full-image resources RW, U | 1x1 | Tier B (fixed) | validates non-overlapping mixed varblocks, global LF/correlation strides and aligned destination origin, compacts all 27 strategy buckets, and emits three bounded indirect records per strategy plus exact coefficient ranges |
| resident `vardct_general` (decoder) | coefficients/artifact/resources RO, two global scratch buffers and X/Y/B output RW, U | 64x1 | Tier B (fixed) | one indirect dequantize/horizontal/vertical sequence per populated regular strategy bucket; task/artifact/resource/LF-stride/output ranges are host-validated |
| resident `vardct_special` (decoder) | coefficients/artifact/resources RO, X/Y/B output RW, U | 8x8 | Tier B (fixed) | one indirect dispatch per populated special strategy bucket; 2,304-byte workgroup storage and raster coefficient/matrix layout are fixed by the transform contract |
| `vardct_large_encoder::quantize_blocks` | source/params RO, artifact RW | 64x1 | Tier C (`KernelVariant` linear) | one 2-D workgroup per 8x8 block; checked block-grid axes and source/artifact ranges; 1,024 bytes workgroup storage |
| `vardct_large_encoder::serialize_control` | params RO, artifact RW | 1x1 | Tier B (fixed) | one bounded scalar dispatch serializes LF groups row-major, resets prediction at each 256x256-block boundary, and writes checked contiguous fragment descriptors |
| `vardct_output` (decoder) | X/Y/B planes RO, output RW, U | 256x1 | Tier A (`KernelVariant` 1-D) | checked word count linearized across 2-D workgroups; each invocation writes one packed u32 |
| `vardct_gaborish` (decoder, `gaborish_rgb`) | resident X/Y/B RO, distinct resident X/Y/B RW, U | 16x16 | Tier A (`KernelVariant` 2-D) | checked actual image extent, padded per-plane stride/range, storage usage/alignment/binding limits, finite normalized weights and dispatch counts |
| `vardct_epf_sigma` (decoder) | LF-group raw metadata/artifact RO, full-image inverse-sigma atlas RW, U | 64x1 | Tier A (`KernelVariant` 1-D) | one invocation per validated transform task; artifact status/task count gate writes, while local block extent, global destination rectangle, and sharpness are bounded before addressing |
| `vardct_epf` (decoder, `epf0`/`epf1`/`epf2`) | resident X/Y/B/sigma RO, distinct resident X/Y/B RW, U | 16x16 | Tier A (`KernelVariant` 2-D) | checked actual extent, padded plane strides/ranges, sigma block-grid coverage, finite parameters, binding/device limits, and mirrored whole-image neighbors |
| decoder `lossless_gray8` | codestream/prefix RO, reconstructed/output/status RW, 236-byte parameter records RO, 16-byte dispatch U | 64x1 | Tier A (`KernelVariant` 1-D) | bounded `jwgp` index, aligned token words plus sentinel, prefix table, four planes/final addresses, packed-row alignment, sample/output ranges and status allocation are prevalidated; one invocation per group lane; channel-fixed Gradient groups may resume through 16-byte-overlapped stream segments using one aligned 32-byte state record per lane |
| encoder `vardct_encode_bounded` | source RO, parameters RO, artifact RW | 256x1 | Tier C (`KernelVariant` 1-D) | one workgroup cooperatively loads and transforms at most 1,024 pixels; fixed 16 KiB workgroup storage is validated before pipeline creation |
| encoder `vardct_encode_quantize` | source RO, parameters RO, artifact RW | 64x1 | Tier C (`KernelVariant` 1-D) | one workgroup per checked 8x8 block; lanes stride over exactly 64 samples and use fixed 1 KiB workgroup storage |
| encoder VarDCT `serialize_control` | parameters RO, artifact RW | 1x1 | Tier B (fixed) | a separate pass establishes global visibility; one invocation performs sequential DC prediction and bit-offset serialization |
| decoder `vardct_pass_group` | bounded stream/entropy bundle RO, quantized-LF plus disjoint LZ/state slices/status RW, 144-byte pass params RO; artifact/order RO, coefficients RW, sink U | 1x1 | Tier B (fixed) | one serial invocation per pass-group window; eight storage bindings meet the portable stage limit; a 464-byte aligned state retains common entropy, nested coefficient progress, sink failure and the 96-word nonzero grid; the 48-byte block-context table ABI addresses QF and signed X/Y/B LF thresholds |
| `vardct_dct8` | coefficients/tasks/resources RO, X/Y/B RW, U | 8x8 | Tier B (fixed) | exactly one workgroup per validated task; task count and all upload bindings are device-bounded |
| encoder `lossless_gray8` | source words RO, artifact RW, U | 1x1 | Tier B (fixed) | profile dimensions are 2..=256; source subrange/alignment/u32 address and artifact capacity are prevalidated |

The decoder entropy shaders share a nested host/WGSL ABI rather than duplicating an untyped word
prefix. `EntropyStreamParams` is a 12-byte, four-byte-aligned `repr(C)`/`Pod` record of three `u32`
values: token start/end bounds and the LZ77 ring mask. It begins the 236-byte Modular
`ShaderParams` storage record and the 240-byte, 16-byte-aligned VarDCT packet parameter record. Consumers supply
storage access and LZ scratch-base functions; their
geometry, prediction, output, and coefficient suffixes are not forced into one binding layout.
Compile-time Rust size/alignment checks, full-record word casts, Naga parsing of every composed
shader variant, and actual GPU Modular/VarDCT entropy tests validate this contract; shader source
text is not used as a semantic test oracle.

The Modular record places six `u32` window values immediately after that prefix: logical segment
start, physical upload start, full logical token end, yield end, flags, and the per-lane state
offset. The Rust window planner that produces those values is consumer-neutral. Modular and VarDCT
AC and the staged local-tree VarDCT LF/HF packet paths use the same checked overlap/batching values
without sharing their WGSL state layout; combined/global packet adoption remains unfinished. Each split segment is four-byte
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
Five words are reserved in the stable state layout for upcoming combined LF/HF phase progress; they
are not reused for diagnostics.

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
(`16x16` tiled 2D kernels and `256x1` linear kernels `rgb_to_image` / `vardct_output`), 64
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
| VarDCT transform kernels | Exact coefficient, compact task/artifact, all normative default matrix/AFV resources, two global scratch buffers, and 27 aligned 128-byte uniform records are included in the core transient total. Every binding is checked against `min(max_buffer_size, max_storage_buffer_binding_size)`. | Included in core pending total. | Regular buckets use global ping-pong scratch; special buckets use fixed 2,304-byte workgroup storage. Empty indirect records perform no work. |
| Bounded VarDCT decoder | `VarDctDecodeMemoryStats` accounts codestream, entropy metadata including HF block-context thresholds, every LF group's parameters/LZ/status plus the selected 64/128-byte packet-state capacity, 464 bytes of AC resume state per pass group, one reusable local-tree packet stream peak, LF packet batch count, reusable AC stream peak and batch count, the 39-descriptor/all-order coordinate table, packet/artifact records, occupancy, reconstruction, capacity-strided raw metadata, coefficients, shared full-image LF/correlation/default-matrix/AFV resources, XYB/transform/output storage, one optional three-plane restoration scratch set, restoration uniforms, and aggregate validation staging. Packet state is included once inside reconstruction bytes and also exposed as an audit subtotal. Local-tree groups currently reserve 128 bytes because the HF tree is discovered only after LF completion. | One shared backend byte reservation covers all LF-group transient buffers until final validation and output bytes until the last lease clone. Windowed staged LF and host-discovered HF reuse one packet upload/state sequentially; only LF maps cursors, while final HF is co-submitted with the downstream prefix and one final map validates all status. `hf_packet_stream_batch_count()` and the exact dynamic submission count become available after that plan is installed. Cancellation leaves callback-owned permits and buffers alive until mapping completes. Gaborish and EPF share one ping-pong scratch set. | Combined/global-tree packet entropy still binds whole ranges, and the VarDCT cap is not yet reduced dynamically by the byte budget. Pipeline/driver-private allocations are excluded. The host parses cursor-dependent HF descriptors but never decodes image entropy or owns an intermediate image. |
| VarDCT encoder | `VarDctMemoryPlan` charges one 256-byte parameter record, one scalable artifact, and one equal-size mapped readback. The artifact is a 64-word header, two words per LF group, then independently 64-word-aligned strategy (`N`), DC/token/extra (`3N` each), and checked maximum entropy-fragment sections. The source binding is reported but caller-owned. | The context's backend-wide byte budget reserves parameter + artifact + readback through map validation. Tiled quantization uses a 2-D block dispatch, so the device workgroup limit is checked per axis rather than against `blocks_x * blocks_y`; storage binding and buffer limits independently cap the complete artifact. | The scalar entropy serializer remains sequential. Full 16K-square allocation is profile-valid but adapter/budget-dependent; actual-adapter coverage exercises 16Kx1 and 1x16K without claiming every adapter admits a 16K square. |
| Lossless Modular encoder | `LosslessModularMemoryPlan` reports source binding ranges (full and peak), 256-byte-aligned parameter storage, peak artifact storage, mapped readback, diagnostic total artifact bytes, batch count, exact GPU submission count, streaming mode, valid bits, component storage bytes, channel count, format, group grid, owned bytes/job and addressed bytes/job. `EncoderBufferPoolStats` separately reports exact idle bytes, three-buffer set counts, hits, misses and evictions. | Every submit non-blockingly reserves `owned_bytes_per_job` from the context's shared `MemoryBudget`. The exclusive buffer lease and permit survive until mapped artifacts are consumed. If the future is abandoned, its callback-owned lifetime unmaps and returns the set only after mapping resolves; the mapped artifact buffer is parsed in place instead of being duplicated into a host `Vec`. A bounded poll slot is reserved before `Queue::submit`, so poll saturation returns both memory and buffers without orphaning GPU work. The idle pool uses exact artifact-size matching and has an independent 32 MiB default hard limit, configurable down to zero, plus a 256-set object-count cap for tiny workloads. | Caller-owned source bindings are sampled directly: they are reported as addressed, are neither copied nor pooled, and are not charged as encoder-owned. Queue/driver-private command metadata excluded. Physical caller-visible allocation is bounded by live admitted bytes plus the separately reported idle-pool bytes. |
| Gray8 decoder | `WgpuDecodeMemoryStats` splits complete `per_frame_bytes` into `output_lease_bytes + transient_bytes`, then reports `max_frame_slots`, `max_frame_window_bytes`, the actual stream peak, submissions, lane stride, parsed Prefix/ANS representation, and the selected 32/48/112-byte execution-state tail per lane. `WgpuDecodeBufferPoolStats` separately reports exact idle/leased bytes and objects, hits, misses, recycling, evictions, limits, and clear generation. | Output and transient portions use the backend-wide transient `MemoryBudget` by default, shared with encode and generic readback; an explicit cloneable budget can define another intentional sharing group. Prefetch keeps each permit from queue submission through the ordered pending frame and then the returned frame lease. Entropy metadata, reconstruction/state scratch, status, mapped status staging, 236-byte POD parameter records, and the 16-byte dispatch uniform (plus a native-F64 dummy when used) have exclusive exact-size/usage/alignment leases. One stream allocation is reused in queue order across every group/segment batch. The map callback owns those leases through completion; abandonment still unmaps staging before return. Output leases retain their reservation beyond session drop. Memory and bounded-poller saturation are explicit prefetch backpressure, with poll capacity reserved before source consumption and queue submission; the count limiter remains independent. | Explicit stream caps below 40 bytes are rejected. Every accepted stock Modular MA profile can span bounded windows; VarDCT AC and staged local-tree LF/HF packets do so through the separate engine while combined/global packets still bind whole ranges. Requested Modular window exposure above 64 MiB is rejected. Idle decoder retention is bounded independently at 32 MiB, 256 buffers total, and 32 per exact key by default; all limits can be reduced to zero. Clear invalidates outstanding generations without disrupting submitted work. Raw codestream and caller-owned output buffers are never pooled. Active logical bytes and idle physical bytes are reported separately rather than double-counted. Driver-private allocations excluded. |
| Generic image readback | `ImageReadbackStats` reports frame/output counts, logical bytes, exact aggregate staging bytes, and padding bytes. One `submit_frames` call uses one staging allocation, command buffer, queue submission, map callback, and completion future/wait across all supplied frames; `ImageReadbackLimits::max_transient_bytes` and device `max_buffer_size` are enforced on that aggregate. | `max_in_flight_bytes` is a hard byte-weighted budget shared by pipeline clones (or backend clones when created from a backend). The complete staging allocation is admitted atomically. A permit and every source lease remain attached through mapping/consumption; an abandoned future leaves them owned by the callback until GPU completion, and exhaustion is a typed non-blocking error. | Codec dispatches are not coalesced by this transport API. Driver-private mapping/command metadata excluded. |
| Display textures | Pitch-linear source buffers are fully range/usage bounded. RGB, numeric, and color-image dispatches use exact 32, 64, and 96-byte Pod uniforms respectively. | No texture-memory reservation API. | Portable `wgpu` cannot report driver-selected texture tiling/compression size; texture backing, short-lived uniform allocation internals, command metadata, and display-pipeline objects are intentionally excluded. |
| Video readback | Each frame pads and bounds its own staging copy. | Animation/session in-flight limits bound decode work. | It does not expose a separate aggregate staging-byte statistic. |

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
- byte-checked BT.709-primary conversion between Linear, sRGB, and BT.709 source/target transfer
  functions, plus pre-dispatch rejection of mismatched, undefined, wide-gamut, and HDR contracts;
- typed rejection of missing/mismatched numeric mappings and native-required F64 on devices without
  enabled `SHADER_F64`, plus an explicitly skipped native-F64 GPU test on unsupported adapters;
- the 1,536-byte DCT8 and 2,304-byte special-VarDCT workgroup-storage requirements, all 27 inverse
  transforms, mixed strategy buckets, and raster versus transposed coefficient placement;
- custom LF dequantization and LF/HF chroma-correlation parameters through Naga validation and
  actual-GPU numeric buffer readback, including the unchanged 64-byte LF uniform and expanded
  240-byte HF-lowering uniform;
- stream-defined inverse XYB, all six standard transfer curves, and all eight JPEG XL patch/frame
  blend modes, including straight and associated alpha; and
- the encoder's worst-case event capacity.

Any new WGSL host record must extend the ABI table and size/alignment tests. Any new buffer must be
added to both the per-job estimate and, where jobs can overlap, the corresponding reservation or
observable in-flight total before the shader is advertised as supported.
