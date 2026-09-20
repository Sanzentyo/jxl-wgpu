# JPEG XL container metadata

`jxl_gpu_bitstream::metadata` exposes Exif, XMP (`xml `), JUMBF (`jumb`) and unknown auxiliary
payloads. Retention and decompression are separate operations. `MetadataBox` retains the complete
original wire payload, including an unchanged `brob` representation; `decode` returns decoded
bytes and borrows uncompressed payloads. No TIFF, XML or JUMBF content is interpreted or repaired.
Callers supply and validate those opaque documents, including the four-byte TIFF offset in Exif.

This follows the box and metadata precedence rules in the
[ISO/IEC 18181-2 proof, clauses 9.5–9.7](https://standards.iteh.ai/catalog/standards/iso/db9e07be-4d7d-465a-8dc9-eba2215ce61d/iso-iec-prf-18181-2)
and the [libjxl box API](https://libjxl.readthedocs.io/en/latest/api_decoder.html).
The codestream's orientation, dimensions, color encoding and image intensity remain authoritative.
Opaque storage is not validation of TIFF/XML/JUMBF content, JPEG reconstruction, an animation index,
or HDR gain-map rendering.

## Retain, edit and write

`ParsedJxl::metadata` copies selected payloads from an already parsed container. Selection is
explicit: `None`, `All`, or `Types(Vec<[u8; 4]>)`. Types match the underlying type of compressed
boxes. Unknown types and duplicate occurrences preserve their relative order. `Metadata::replace`
replaces all occurrences at the first matching position, appends when absent, or removes all
matches when passed `None`. Size or type errors leave the original collection unchanged.

```rust
use jxl_gpu_bitstream::{parse, ParseLimits};
use jxl_gpu_bitstream::metadata::{
    BrotliOptions, MetadataBox, MetadataCompression, MetadataLimits, MetadataSelection, XMP,
};

# fn rewrite(input: &[u8]) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
let limits = MetadataLimits::default();
let parsed = parse(input, ParseLimits::default())?;
let mut metadata = parsed.metadata(&MetadataSelection::All, limits)?;
let replacement = MetadataBox::new(
    XMP, b"<xmp/>", MetadataCompression::Brotli(BrotliOptions::default()), limits,
)?;
metadata.replace(XMP, Some(replacement), limits)?;
let output = metadata.write_container(parsed.codestream())?;
# Ok(output) }
```

`write_container` emits canonical size headers and places retained boxes before one `jxlc` box.
It preserves payload bytes and logical codestream bytes, not original box positions, transport
fragmentation, size encodings or signatures over the original file. Use
`MetadataBox::as_container_box` with `FragmentedContainerWriter::push_box` when assembling `jxlp`
output from GPU-encoded packets. No image samples pass through a CPU codec in either operation.
Replacing metadata that JPEG reconstruction or an application signature depends on may invalidate
that separate relationship; this API does not reconstruct or re-sign it.

## Incremental ownership

`MetadataCollector` observes the same borrowed `ContainerStreamEvent` values as
`GpuDecodeStream::push_transport_event`. Feed each event to both consumers, including the final
`End`, then call each consumer's `finish`. The image decoder retains its own admitted codestream
slices; the metadata collector copies only selected payloads into one vector per box. It never
retains codestream bytes or caller `Arc` allocations. Unselected compressed boxes need only a
four-byte inline probe to resolve their underlying type.

Collection requires the transport scanner's authoritative `End`. An individual box end is not
proof that the rest of the container is valid. Mismatched types, offsets, lengths or ordering poison
the collector and immediately release its collected/partial payloads. Dropping the collector also
releases them independently of GPU sessions and held output images. The original exact box headers
remain available in the transport events for applications implementing byte-preserving relay.

## Host bounds

`MetadataLimits` has distinct limits for selected box count, per-box encoded bytes, total retained
encoded bytes, per-box decoded bytes, total decoded bytes, expansion ratio, and Brotli window bits.
Defaults are 256 boxes, 64 MiB per encoded/decoded box, 128 MiB total encoded/decoded payload,
1024× maximum expansion, and a maximum 24-bit Brotli window. `retained_bytes` includes the current
selected box; it excludes vector capacity, object/header overhead and the fixed type probe.
These host limits are separate from decoder input admission and GPU allocations.

`decode_all` caps each operation by the remaining total decoded allowance. The decoder checks the
[RFC 7932 stream header](https://www.rfc-editor.org/rfc/rfc7932.html#section-9.1) before initializing
codec state. Standard windows 10–24 are supported; large-window/shared-dictionary extensions are
not enabled. A 4096-byte output buffer is shortened near the limit, with at most one excess byte
decoded to distinguish exact completion from overflow. Neither that byte nor a partial decoded
collection is returned on failure. Codec scratch is bounded by the standard window and grammar;
the logical payload limits are not a total allocator/RSS guarantee.

Ratio uses the decoded size divided by compressed stream bytes, excluding the four-byte type.
Truncated streams, malformed syntax, concatenated streams and trailing bytes are errors. Reserved
transport types cannot be metadata; compressed types cannot begin with `jxl`, be `jbrd`, or be
another `brob`. Opaque retention validates the type prefix, but does not decompress or validate
an unrequested body. Compression accepts qualities 0–11 and window hints 10–24; the actual emitted
window and output size are also checked. Small/low-quality streams may use the codec's adjusted
window, which must still fit the caller's limit.

## Executed evidence

On Rust 1.98.1 / Apple M5 Metal, the focused tests passed:

- Every two-chunk split and byte-drip input with compact, extended and to-end metadata boxes,
  interleaved ordered/v1 out-of-order `jxlp`, selected/ignored compressed types and exact payloads.
- Atomic replace/remove, duplicate/unknown preservation, zero/exact/exceeded count and byte
  bounds, malformed events, incomplete transport, poison release, truncated/extended Brotli,
  empty output, expansion limits and a 256 KiB decoded payload.
- Google Brotli 1.2.0 interoperability in both directions for all 180 combinations of quality
  0–11 and window hint 10–24, including empty, one-byte and multi-output-buffer inputs.
- Six raw libjxl 0.12.0 box extractions: Exif/XMP/JUMBF, each plain and Brotli-compressed, with
  byte equality and unchanged image/frame inventories. The native helper uses the raw box API;
  `djxl` image/metadata export normalizes Exif orientation and is not an opaque-byte oracle.
- Seventeen decoder selections span both codecs, mixed animation, main/preview, eight ICC
  sources and four PQ/HLG sources. Apply/Keep orientation, plain/`brob`/edited metadata,
  `jxlc`/`jxlp`, whole and 43-byte input with 256-byte GPU windows produce 444 immutable
  presentations and 1,357,296 exactly equal F32 words, including alpha. Timing, progression,
  retained images, metadata independence and released GPU/input budgets are checked.

See the [native oracle recipe](../crates/jxl_gpu_bitstream/test-data/metadata_oracle/README.md).
No existing source image or reference asset changes. Full container ordering/compatibility,
`jxli`, JPEG byte reconstruction, full `jhgm` conformance, encoder quality and the remaining full JPEG XL gates remain
open. [Gain-map interpretation](GAIN_MAP.md) now has a separate bounded metadata and GPU
alternate-still API; opaque metadata retention alone does not invoke it.

## JPEG reconstruction metadata

`jxl_gpu_bitstream::jpeg_reconstruction` is a separate typed metadata boundary for `jbrd`.
`JpegReconstructionMetadata::parse` accepts one complete payload; the convenience method
`ParsedJxl::jpeg_reconstruction` first inventories every auxiliary box in transport-validated
input. It returns `None` when absent, and typed `DuplicateBox` or `WrappedBox` errors before
parsing an ambiguous or `brob`-wrapped record. Incremental transport events alone cannot establish
this complete-container condition. Opaque collection and this parser do not execute one another.

The immutable owned result exposes borrowed marker, component, quantization/Huffman table and
scan records; restart intervals, reset points and redundant zero-run records; packed preserved
entropy-padding bits; APP classifications and body ranges; exact COM/intermarker/tail contents.
Known ICC/Exif/XMP APP records declare required sizes but refer to separately supplied metadata.
Quantization values, sampling grids and coefficients belong to the image frame, not `jbrd`.
The initial gray hint and per-scan last-pass metadata are preserved without granting image
authority. Noncanonical byte-preservation records are not a JPEG syntax-conformance assertion.

Parsing validates field ranges, selectors, table groups/use, Huffman uniqueness/terminal/DC
alphabet/prefix space, marker lengths, native metadata block-index bounds, zero header alignment,
and exact decoded-body size. Strict standard Brotli rejects truncated, trailing and excess output.
Declared block indices and scan schedules still require actual frame-grid and progression checks
before use. Errors distinguish bit input, invalid metadata, allocation failure, resource limits
and the shared strict Brotli errors; an error returns no partial metadata or output.

`JpegReconstructionLimits` defaults to 64 MiB encoded input, 64 MiB logical owned storage,
16,384 markers, 1,048,576 total vector entries and 64 MiB decoded opaque body. The marker grammar
also has an intrinsic 16,384 ceiling. Entries include nested symbols, scan/reset/ZRL arrays and
packed padding bytes. Owned storage counts vector elements, including their inline record
fields, and the decoded body once. Borrowed input, the enclosing inline object, allocator capacity
and overhead, fixed scratch and Brotli state are separate. The 4096-byte decoder scratch,
standard window limit (default 24 bits) and expansion ratio (default 1024×) bound their respective
work; logical owned storage is not total allocator/RSS usage. Ratio excludes the `jbrd` header.

`encode(JpegReconstructionEncodeOptions)` emits canonical metadata field encodings and a new
Brotli body. It preserves the logical metadata, not the original compressed `jbrd` byte sequence.
The resulting `EncodedJpegReconstruction` reports header/body lengths and simultaneous logical
peak ownership. The default 128 MiB owned limit covers existing metadata, compressed temporary
body and final encoded payload together; encoded output defaults to 64 MiB. The header is counted
without allocating, and compression is bounded by the remaining output and ownership allowances
before final allocation. Failed admission leaves the input reusable. Individual emissions do not
reserve bytes against a shared session budget; callers retaining multiple models/results account
for those separately. Pass `as_bytes()` to a `ContainerBox` of type `JBRD` when assembling a container.

The [JPEG metadata corpus](../crates/jxl_gpu_bitstream/test-data/jpeg_reconstruction/README.md)
verifies original-JPEG byte identity through the independent libjxl JPEG-only API, including
multiple compression settings and malformed inputs. No CPU image/entropy/coefficient decoding
enters production. The separate [GPU coefficient API](../crates/jxl_wgpu_decode/README.md#gpu-jpeg-reconstruction-inputs)
now binds actual decoded quantizers and integer LF/AC to checked padded JPEG component grids.
It validates and leases those GPU integers without interpreting ICC or applying orientation.
The separate [original-byte API](../crates/jxl_wgpu_decode/README.md#gpu-original-jpeg-byte-output)
binds the qualified scan/progression and external-metadata profile, generates GPU JPEG entropy,
and validates budget-owned assembly before publishing bytes. Metadata parsing/emission alone
still grants no image or JPEG-byte authority; broader legal reconstruction variants remain open.
