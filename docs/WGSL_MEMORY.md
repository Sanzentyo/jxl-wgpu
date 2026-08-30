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
| `jxl_wgpu/gaborish_rgb.wgsl` | `GaborishRgbUniform` / `Params` | dimensions/6 strides, then four values for each of X, Y and B: `weight0, weight1, weight2, pad` | 80 | 4 | uniform |
| `jxl_wgpu/epf.wgsl` | `EpfUniform` / `Params` | dimensions/6 image strides, sigma dimensions/stride/kind, 6 filter floats, `_pad0, _pad1` | 80 | 4 | uniform |
| `jxl_wgpu/upsample.wgsl` | `UpsampleUniform` / `Params` | `input_width, input_height, output_width, output_height, input_stride, output_stride, factor, _pad0` | 32 | 4 | uniform |
| `jxl_wgpu/ycbcr_to_rgb.wgsl` | `YcbcrUniform` / `Params` | `width, height, cb_stride, y_stride, cr_stride, output_stride, component, _pad0` | 32 | 4 | uniform |
| `jxl_wgpu/premultiply_alpha.wgsl` | `PremultiplyUniform` / `Params` | `width, height, color_stride, alpha_stride, output_stride, _pad0, _pad1, _pad2` | 32 | 4 | uniform |
| `jxl_wgpu/save.wgsl` | `SaveUniform` / `Params` | `width, height, source_stride, channels, channel, layout (output_layout), orientation, _pad0` | 32 | 4 | uniform |
| `jxl_wgpu/rgb_to_image.wgsl` | `ImageOutputUniform` / `Params` | dimensions/3 source strides, format fields, 4 plane offset/stride pairs, `logical_size, dispatch_width, orientation`, 3 pads | 128 | 4 | uniform |
| `jxl_wgpu/display_rgb.wgsl` | `DisplayRgbParams` / `DisplayRgbParams` | `width, height, channels, sample_type, layout (storage_layout), logical_samples, _padding0, _padding1` | 32 | 4 | uniform |
| `jxl_wgpu/display_image.wgsl` | `DisplayImageParams` / `Params` | dimensions/format fields, 4 plane offset/stride pairs, `chroma_width, chroma_height, _padding0` | 96 | 4 | uniform |
| `jxl_wgpu/vardct_dct8.wgsl` | `GpuTask` / `Task` | `coefficient_offset, destination_x, destination_y, quant_index, matrix_index, correlation_index, lf_index` | 28 | 4 | storage element |
| `jxl_wgpu/vardct_dct8.wgsl` | `GpuResourceVector` / `vec4<f32>` | four `f32` lanes | 16 | 16 | storage element |
| `jxl_wgpu/vardct_dct8.wgsl` | `Dct8Uniform` / `Params` | `task_count`, output dimensions/3 strides, 4 resource offsets, 2 pads, `quant_biases[4]`/`vec4<f32>` | 64 | 16 | uniform |
| `jxl_wgpu_encode/lossless_gray8.wgsl` | `Gray8Params` / `Params` | `width, height, row_stride, byte_offset` | 16 | 4 | uniform |
| `jxl_wgpu_encode/lossless_gray8.wgsl` | `Gray8ArtifactHeader` / `output_words[0..53]` | `event_count, raw_counts[19], lz77_counts[33]` | 212 | 4 | storage/readback record |
| `jxl_wgpu_encode/lossless_gray8.wgsl` | `Gray8Event` / four-word event | `kind, token, extra_bit_count, extra_bits` | 16 | 4 | storage/readback element |
| `jxl_wgpu_decode/lossless_gray8.wgsl` | `ShaderParams` / `Params` | token range, dimensions/sample count, output mode/transfer/range, 3 plane offset/stride pairs, chroma dimensions | 64 | 4 | uniform |
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
  lookup remains `Vec<u32>` and is uploaded with `cast_slice`; the raw codestream remains bytes so
  its bit/byte offsets and explicit four-byte sentinel are preserved.

Manual endian-aware serialization is therefore retained only for bitstream/container/file
formats. Fixed GPU records do not use hand-written byte flattening.

## Bindings and dispatch bounds

`RO`, `RW`, `U`, and `T` mean read-only storage, read-write storage, uniform, and write-only storage
texture. All two-dimensional kernels return before accessing memory when `gid.x >= width` or
`gid.y >= height`. The planner checks the selected workgroup dimensions/invocation count against
the device and checks every dispatched dimension against `max_compute_workgroups_per_dimension`.

| Shader / entry points | Bindings in order | Workgroup | Dispatch and address bound |
|---|---|---:|---|
| `copy` | input RO, output RW, U | 16x16 | checked 2-D extent and strides |
| `modular_to_f32` | input RO, output RW, U | 16x16 | checked 2-D extent and strides |
| `chroma_upsample` | input RO, output RW, U | 16x16 | checked input/output extents, axis and strides |
| `chroma_2d` | input RO, output RW, U | 16x16 | checked input/output extents and strides |
| `gaborish` | input RO, output RW, U | 16x16 | checked extent/strides; clamped neighbor reads |
| `gaborish_rgb` | X/Y/B RO, X/Y/B RW, U | 16x16 | checked common extent and per-plane strides |
| `epf0`, `epf1`, `epf2` | X/Y/B/sigma RO, X/Y/B RW, U | 16x16 | checked common extent, sigma shape and all strides |
| `upsample` | input/weights RO, output RW, U | 16x16 | checked factor, output extent, weights and strides |
| `ycbcr_to_rgb` | Cb/Y/Cr RO, output RW, U | 16x16 | one checked dispatch per output component |
| `premultiply_alpha` | color/alpha RO, output RW, U | 16x16 | one checked dispatch per color component |
| `save` | source RO, packed output RW, U | 16x16 | checked orientation and exact packed allocation |
| `rgb_to_image` | R/G/B RO, packed output RW, U | 256x1 | checked linear word count is split into a legal 2-D dispatch; shader checks `logical_size` before stores |
| `display_rgb` | source RO, destination T, U | 16x16 | source must have `STORAGE`; logical samples and final source address fit the bound range/WGSL `u32` |
| `display_image` | source RO, destination T, U | 16x16 | source must have `STORAGE`; each pitch-linear plane and its final address is bounded |
| `vardct_dct8` | coefficients/tasks/resources RO, X/Y/B RW, U | 8x8 | exactly one workgroup per validated task; task count and all upload bindings are device-bounded |
| encoder `lossless_gray8` | source words RO, artifact RW, U | 1x1 | profile dimensions are 2..=256; source subrange/alignment/u32 address and artifact capacity are prevalidated |
| decoder `lossless_gray8` | codestream/prefix RO, reconstructed/output/status RW, U | 1x1 | bounded `jwgp` index, aligned token words plus sentinel, prefix table, sample/output ranges and status allocation are prevalidated |

Display source buffers are now also checked for the usage needed by the operation: `STORAGE` for
shader conversion and `COPY_SRC` for direct RGBA8 buffer-to-texture copies. A multi-row direct
buffer-to-texture copy requires `bytes_per_row` to be a multiple of 256.

## Workgroup-local storage

Only `vardct_dct8.wgsl` declares `var<workgroup>` memory:

| Shader | Declaration | Bytes per workgroup | Host validation |
|---|---|---:|---|
| `vardct_dct8` | two `array<f32, 192>` scratch arrays | 1,536 | reject when `max_compute_workgroup_storage_size < 1,536` |

All other shaders use zero explicit workgroup-local bytes. The largest invocation count is 256
(`16x16` kernels and `rgb_to_image`); VarDCT DCT8 uses 64 and the Gray8 encoder/decoder each use one.
Planner capability checks cover the core kernels. The fixed VarDCT and Gray8 sizes are within
WebGPU's portable baseline; VarDCT additionally checks its workgroup storage limit at submission.

## Four-byte copy and mapping invariant

- `aligned_buffer_size` rounds every buffer-copy/readback allocation up to a nonzero multiple of
  `wgpu::COPY_BUFFER_ALIGNMENT` (4). Logical output lengths remain separate, so padding is never
  returned as image data.
- Core packed output and CPU staging copies use the same padded byte count. VarDCT upload element
  sizes (4, 16, and 28) are already multiples of 4.
- Generic `ImageReadbackPipeline` independently pads each source copy and each aggregate staging
  offset to 4 bytes, validates `COPY_SRC`, validates the source allocation, and maps one bounded
  aggregate staging buffer.
- Encoder artifact storage and mapped readback have identical checked, 4-byte-aligned sizes.
  Its 212-byte header and 16-byte events are parsed as checked `Pod` records. Decoder codestream
  storage is padded to a word and includes a four-byte sentinel; its 16-byte status is parsed as a
  checked `DecodeStatus` record.
- Buffer-to-texture paths apply WebGPU's separate 256-byte multi-row pitch rule. Texture-to-buffer
  tests likewise use 256-byte row padding.

## Memory accounting and concurrency

| Path | Per-job accounting | Concurrent accounting / admission | Deliberate exclusions or remaining gap |
|---|---|---|---|
| Core render session | `WgpuSubmissionStats` reports physical resident-plane bytes and exact explicit transient bytes: uniforms, uploads, packed outputs and staging. `max_transient_bytes` is enforced per submission. | `WgpuFrameSession::pending_transient_bytes()` checked-adds submitted jobs and checked-subtracts them on all wait paths. | The aggregate is observable, not a second admission limit. Queue-ordered reusable resident allocations make it conservative. Caller-owned GPU outputs can outlive `wait`, so the session cannot track them afterward. |
| Core resident arena | Planner accounts physical slots once, respects simultaneous lifetimes, validates every slot against `max_buffer_size` and every bound plane against `max_storage_buffer_binding_size`. | Buffer pool has a configured hard byte limit and never leases one buffer concurrently. | Pipeline/driver memory excluded. |
| VarDCT DCT8 | Exact coefficient, task, resource, and uniform upload bytes are included in the core transient total. Every upload is checked against `min(max_buffer_size, max_storage_buffer_binding_size)`. | Included in core pending total. | Non-DCT8 transform buckets return a typed rejection. There is no extra global scratch buffer; 1,536-byte workgroup storage is not global buffer memory. |
| Gray8 encoder | `LosslessGray8MemoryPlan` reports source binding, 16-byte uniform, artifact storage, mapped readback, owned bytes/job and addressed bytes/job. | `for_in_flight(max_jobs)` checked-multiplies both totals and exposes the caller-selected ceiling. | API reports but does not internally semaphore caller concurrency; the application must enforce its selected ceiling. |
| Gray8 decoder | `WgpuDecodeMemoryStats` reports complete per-frame explicit allocation and `reserved_bytes = per_frame_bytes * max_in_flight`. | A session holds a checked reservation (default 64 MiB/session) in an engine-wide checked budget (default 256 MiB); the in-flight limiter enforces its count. | Driver-private allocations excluded. |
| Generic image readback | `ImageReadbackStats` reports logical bytes and exact aggregate staging bytes; `ImageReadbackLimits::max_transient_bytes` and device `max_buffer_size` are enforced per submission. | No aggregate admission counter across several independently live `ImageReadbackSubmission` values. | Applications requiring a global cap must limit live submissions; adding a shared reservation is future work. |
| Display textures | Pitch-linear source buffers are fully range/usage bounded. | No texture-memory reservation API. | Portable `wgpu` cannot report driver-selected texture tiling/compression size; texture backing and display-pipeline objects are intentionally excluded. |
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

## Regression coverage

The ABI tests pin every Rust size, natural alignment, and field order, including the 16-byte
VarDCT alignment and both Gray8 readback schemas.
GPU tests compile and execute every portable core shader, all display formats, VarDCT DCT8, the
deterministic encoder fixture, and the bounded decoder. Dedicated tests cover:

- storage-binding device-limit selection;
- checked resident aliasing and exact transient estimates;
- pending transient accumulation and release for multiple submissions;
- 4-byte and 256-byte copy-pitch rules;
- display final-address rejection above WGSL's `u32` space;
- the 1,536-byte VarDCT workgroup-storage requirement; and
- the encoder's worst-case event capacity.

Any new WGSL host record must extend the ABI table and size/alignment tests. Any new buffer must be
added to both the per-job estimate and, where jobs can overlap, the corresponding reservation or
observable in-flight total before the shader is advertised as supported.
