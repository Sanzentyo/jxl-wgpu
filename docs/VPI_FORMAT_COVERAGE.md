# NVIDIA VPI 4.1.3 portable-format coverage

This matrix is audited against NVIDIA's VPI 4.1.3 `ImageFormat.h`, generated
2026-06-03. “Supported” below means directly addressable pitch-linear bytes in
an ordinary `wgpu::Buffer`; it does not claim VPI ABI or NVIDIA external-memory
interop.

## Pitch-linear formats

All 30 non-invalid pitch-linear predefined formats have an exact
`jxl_gpu_formats::PixelFormat`, checked `ImageLayout`, GPU decode output path,
byte-preserving aggregate readback, and a same-queue display path. Numeric
display is never implicit: it requires `NumericDisplayContract`.

| Family | Exact VPI names | Count | GPU output | Readback | Display contract |
|---|---|---:|---|---|---|
| Unsigned numeric | `U8`, `U16`, `U32` | 3 | normalized Gray8 with explicit mapping | exact bytes/layout | explicit unsigned scale+bias |
| Signed numeric | `S8`, `S16`, `S32`, `2S16` | 4 | normalized nonnegative Gray8 with explicit mapping | exact bytes/layout | explicit signed scale+bias; X as luma, or XY as luma/alpha or red/green |
| Floating numeric | `F32`, `F64`, `2F32` | 3 | normalized Gray8; F64 precision policy required | exact IEEE storage bytes | explicit scale+bias, NaN/Inf policy, clamp, channel map, and transfer |
| Luma | `Y8`, `Y8_ER`, `Y16`, `Y16_ER` | 4 | limited/full BT.601 code output | exact bytes/layout | color specification drives range/matrix/transfer |
| Semi-planar YCbCr | `NV12`, `NV12_ER`, `NV24`, `NV24_ER` | 4 | native two-plane output | exact bytes/layout | chroma siting, range, matrix, and transfer |
| Packed YCbCr 4:2:2 | `UYVY`, `UYVY_ER`, `YUYV`, `YUYV_ER` | 4 | native packed output | exact bytes/layout | order, range, matrix, and transfer |
| Interleaved RGB | `RGB8`, `BGR8`, `RGBA8`, `BGRA8` | 4 | native packed output | exact bytes/layout | order, alpha, and transfer |
| Planar RGB | `RGB8p`, `BGR8p`, `RGBA8p`, `BGRA8p` | 4 | native 3/4-plane output | exact bytes/layout | order, alpha, and transfer |
| **Total portable VPI predefined formats** |  | **30** | **30** | **30** | **30** |

For numeric decode output, `NumericSampleMapping::NormalizedGray8` maps an
8-bit decoded Gray code across unsigned `[0, MAX]`, signed `[0, MAX]`, or
floating `[0, 1]`; two-component outputs duplicate the value. F64 requires an
`F64OutputPolicy`. Native shader f64 is used when enabled; the portable policy
stores the exact binary64 widening of the rounded f32 normalization.

Numeric visualization is a separate contract. For finite stored value `x`,
the shader evaluates `x * scale + bias`, applies the chosen NaN/infinity policy,
clamps to `[0, 1]`, maps X/XY to RGBA, and optionally decodes sRGB to the
linear-light BT.709 texture contract. On F64 input, `DisplayTexture` reports
`NativeF64` or `F64RoundedToF32`; there is no silent precision choice.
Native F64 additionally requires a naturally eight-byte-aligned plane offset
and row pitch; an otherwise valid caller-defined pitch-linear layout uses the
reported portable path.

## Explicitly excluded predefined layouts

The remaining 30 official names use NVIDIA-proprietary block-linear storage.
They are inventoried so imports fail with
`VpiPortabilityError::NonPortableLayout`; they are **not implemented** and are
not included in the supported count.

| Logical stems | VPI memory layout | Exact suffix | Count | Portable result |
|---|---|---|---:|---|
| `U8`, `S16`, `2S16`, `Y8`, `Y8_ER`, `Y16`, `Y16_ER`, `NV12`, `NV12_ER`, `NV24`, `NV24_ER`, `UYVY`, `UYVY_ER`, `YUYV`, `YUYV_ER` | default block-linear | `_BL` | 15 | typed `NonPortableLayout` |
| same 15 stems | block height 16 | `_BL16` | 15 | typed `NonPortableLayout` |
| **Total excluded predefined formats** |  |  | **30** | **30 typed rejections** |

NVIDIA states that block-linear memory is proprietary and is not intended for
direct user addressing. CUDA arrays, CUDA device pointers, EGLImage, NvBuffer,
and NvSciBuf are external allocation/interop contracts, not portable byte
layouts. They require platform APIs and synchronization which WebGPU does not
standardize, so this workspace does not expose fake `wgpu::Buffer` layouts for
them.

## Verification

- `jxl_gpu_formats` tests enumerate all 60 official non-invalid predefined
  names: 30 valid pitch-linear descriptors and 30 exact typed rejections.
- decoder GPU tests write and read back every one of the 10 numeric and 20
  color pitch-linear formats.
- display GPU tests convert all 10 numeric formats using explicit contracts,
  including signed endpoints, two-component mappings, NaN, both infinities,
  clamp, Linear/sRGB transfer, and F64 precision reporting.
- `DisplayNumericParams` is a 64-byte `#[repr(C)]`/`bytemuck::Pod` record; tests
  pin its four-byte alignment, 16-byte size multiple, Rust word order, and WGSL
  field order.

## Primary NVIDIA sources

- [VPI 4.1.3 release notes](https://docs.nvidia.com/vpi/release_notes.html)
- [`ImageFormat.h` source](https://docs.nvidia.com/vpi/ImageFormat_8h_source.html)
- [VPI data-layout definitions](https://docs.nvidia.com/vpi/group__VPI__DataLayout.html)
- [VPI image buffer types](https://docs.nvidia.com/vpi/group__VPI__Image.html)
