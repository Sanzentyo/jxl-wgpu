# jxl_gpu_formats

Portable logical pixel-format and pitch-linear image-layout contracts for GPU
buffers. The crate separates:

- `PixelFormat`: color meaning, subsampling, sample type, swizzle, planes, and
  bit/word packing;
- `ImageLayout`: image extent plus validated plane offsets, sample extents, row
  byte counts, and row strides in one byte allocation;
- `convert_rgb_f32` (feature `cpu-reference`, default off): a scalar
  correctness oracle. It is not a production codec fallback.

There is no CUDA or NVIDIA dependency.

## Portable storage boundary

Only directly addressable **pitch-linear** storage is represented. The layout
uses one byte allocation, per-plane offsets, and byte pitches. `ImageLayout`
requires non-overlapping planes and a pitch at least as large as the row's
packed data. `ImageLayout::packed` makes rows tight and aligns plane starts to
four bytes for convenient `wgpu` storage-buffer access. Callers can use
`ImageLayout::from_planes` for larger pitches or gaps.

This intentionally excludes VPI block-linear `_BL` and `_BL16` formats and
opaque CUDA array, EGLImage, NvBuffer, and NvSciBuf storage. Those need
vendor-specific allocation/import synchronization and must not be presented as
ordinary portable `wgpu` buffers. VPI permits up to six planes and can describe
separate plane base pointers; this crate's one-allocation/non-overlap rule is a
smaller, safer portable contract.

## NVIDIA VPI inventory

Audited against NVIDIA VPI **4.1.3** (the current 4.1 update documented on
2026-06-03). `vpi::VpiPitchLinearFormat::ALL` contains the complete set of 30
predefined pitch-linear formats from `ImageFormat.h`:

| Logical family | VPI 4.1 pitch-linear predefined formats | Representation | Scalar oracle |
|---|---|---|---|
| Non-color integer | `U8`, `S8`, `U16`, `U32`, `S32`, `S16`, `2S16` | unsigned/signed words; one or two channels | No |
| Non-color float | `F32`, `F64`, `2F32` | 32/64-bit float words | No |
| Luma | `Y8`, `Y8_ER`, `Y16`, `Y16_ER` | BT.601 limited/full, one plane | Yes |
| Semi-planar YCbCr | `NV12`, `NV12_ER`, `NV24`, `NV24_ER` | 4:2:0 or 4:4:4 Y + CbCr | Yes |
| Packed YCbCr | `UYVY`, `UYVY_ER`, `YUYV`, `YUYV_ER` | two pixels in four 8-bit words | Yes |
| Interleaved RGB | `RGB8`, `BGR8`, `RGBA8`, `BGRA8` | one plane, swizzle retains channel meaning | Yes |
| Planar RGB | `RGB8p`, `BGR8p`, `RGBA8p`, `BGRA8p` | three or four planes | Yes |

The related `_BL` and `_BL16` variants are explicitly **out of scope**. A VPI
predefined image format also does not imply that every VPI algorithm accepts
it; NVIDIA documents algorithm support separately.

### VPI 4.1 named color specifications

`vpi::VpiColorSpec::ALL` maps all 23 non-invalid predefined values without
collapsing BT.601, BT.709, BT.2020 NCL, BT.2020 constant-luminance, transfer,
range, or chroma siting.

| VPI name | space | YCbCr encoding | transfer | range | chroma H/V |
|---|---|---|---|---|---|
| `DEFAULT` | undefined | undefined | linear | full | both/both |
| `UNDEFINED` | BT.709 | undefined | linear | full | both/both |
| `BT601` / `BT601_ER` | BT.709 | BT.601 | BT.709 | limited / full | even/even |
| `BT709` / `BT709_ER` | BT.709 | BT.709 | BT.709 | limited / full | even/even |
| `BT709_LINEAR` | BT.709 | BT.709 | linear | limited | even/even |
| `BT2020` / `BT2020_ER` | BT.2020 | BT.2020 NCL | BT.2020 | limited / full | even/even |
| `BT2020_LINEAR` | BT.2020 | BT.2020 NCL | linear | limited | even/even |
| `BT2020_PQ` / `BT2020_PQ_ER` | BT.2020 | BT.2020 NCL | PQ | limited / full | even/even |
| `BT2020c` / `BT2020c_ER` | BT.2020 | BT.2020 constant-luminance | BT.2020 | limited / full | even/even |
| `MPEG2_BT601` | BT.709 | BT.601 | BT.709 | full | even/center |
| `MPEG2_BT709` | BT.709 | BT.709 | BT.709 | full | even/center |
| `MPEG2_SMPTE240M` | BT.709 | SMPTE 240M | SMPTE 240M | full | even/center |
| `sRGB` | BT.709 | undefined | sRGB | full | both/both |
| `sYCC` | BT.709 | BT.601 | sYCC | full | center/center |
| `SMPTE240M` | BT.709 | SMPTE 240M | SMPTE 240M | limited | even/even |
| `DISPLAYP3` / `DISPLAYP3_LINEAR` | DCI-P3 | undefined | sRGB / linear | full | both/both |
| `SENSOR` | sensor | undefined | linear | full | both/both |

VPI's available logical components are represented directly: color models
undefined/non-color, YCbCr, RGB, RAW, and XYZ; RAW mosaics RGGB, BGGR, GRBG,
GBRG, RCCB, BCCR, CRBC, CBRC, RCCC, CRCC, CCRC, CCCR, and CCCC; chroma
subsampling 4:4:4, 4:2:2, rotated 4:2:2, 4:1:1, rotated 4:1:1, and 4:2:0;
full/limited range; even/center/odd/both chroma locations; canonical swizzles;
and per-plane multi-word bit packing. Packing fields are MSB-to-LSB inside a
word, while the `words` vector describes consecutive independently endian-
addressed words.

## Common video constructors

The same generic descriptor covers formats beyond VPI's predefined set:

| Constructors | Meaning |
|---|---|
| `i444`, `i422`, `i420` | 8/10/12/16-bit planar Y/Cb/Cr (or another valid/storage width pair) |
| `nv12`, `nv21` | 8-bit 4:2:0 CbCr / CrCb |
| `nv16`, `nv61` | 8-bit 4:2:2 CbCr / CrCb |
| `nv24`, `nv42` | 8-bit 4:4:4 CbCr / CrCb |
| `p010`, `p012`, `p016` | 4:2:0, 10/12/16 valid bits in 16-bit words |
| `p210`, `p212`, `p216` | 4:2:2, 10/12/16 valid bits in 16-bit words |
| `p410`, `p412`, `p416` | 4:4:4, 10/12/16 valid bits in 16-bit words |

For P-family constructors, samples are MSB-aligned (`X10b6`, `X12b4`, or
`X16`). Low-bit-aligned and other custom arrangements can be described with
`PackingWord`/`PackingField` directly. Every constructor takes an explicit
`ColorSpecification`, so BT.601, BT.709, and BT.2020 non-constant-luminance
matrices remain distinct.

## Official sources

- [VPI 4.1 release notes](https://docs.nvidia.com/vpi/release_notes.html)
- [Image format API](https://docs.nvidia.com/vpi/group__VPI__ImageFormat.html)
- [`ImageFormat.h` source](https://docs.nvidia.com/vpi/ImageFormat_8h_source.html)
- [Color specification API](https://docs.nvidia.com/vpi/group__VPI__ColorSpec.html)
- [`ColorSpec.h` source](https://docs.nvidia.com/vpi/ColorSpec_8h_source.html)
- [Data layout and packing](https://docs.nvidia.com/vpi/group__VPI__DataLayout.html)
- [Image buffers and pitch-linear plane constraints](https://docs.nvidia.com/vpi/group__VPI__Image.html)

