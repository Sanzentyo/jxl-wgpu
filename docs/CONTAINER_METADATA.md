# Opaque JPEG XL container metadata

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
`jxli`, `jbrd`, `jhgm`, encoder quality and the remaining full JPEG XL gates remain open.
