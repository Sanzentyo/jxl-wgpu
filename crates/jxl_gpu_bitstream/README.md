# jxl_gpu_bitstream

Bounded JPEG XL transport and codestream inventory for GPU codec front ends.

`CodestreamInventory::select_image` selects Main or Preview from a complete inventory;
`ImageHeaderInventory::select_preview` lowers a fully inventoried preview prefix. The shared
`ImageSelection` and `ImageSelectionError` types are also re-exported by `jxl_wgpu_decode`.
Selection validates metadata and keeps physical frame IDs, noise seeds and byte ranges intact.
Preview gets its own canvas and still timing; Main excludes it. Decoder reconstruction and encoder
index generation use this same boundary. It does not validate entropy, pixels or transport end.

`ImageHeaderInventory::original_icc_profile` exports original color metadata under
`icc_profile::IccProfileLimits`: embedded ICC bytes are borrowed unchanged; enumerated RGB,
Gray and perceptual XYB generate owned ICCv4 bytes compatible with libjxl 0.12.0.
`ColourEncodingInventory::generate_icc_profile` serves standalone enumerated declarations.
Exact output size is admitted before allocation, and profile tables use fixed bounded metadata
scratch. Export does not authorize CMS conversion or decode pixels.
[Profile contract and evidence](../../docs/ICC_COLOR.md#original-profile-export).

`parse` validates transport framing for raw codestreams, `jxlc` containers, and ordered or indexed
`jxlp` fragment sequences. Raw and single-`jxlc` codestreams remain borrowed; only fragmented
streams are joined.

`FrameIndex` parses and emits bounded plain `jxli` payloads and rejects duplicate boxes or
unsupported compressed indexes. It retains logical offsets, rational tick units and displayed
frame intervals without authorizing a restart. `FrameSequencePlan` owns checked physical ordering,
reference versions, presentation timing and conservative transitive dependencies for a selected
image. `FrameIndex::from_sequence` generates independent anchors, and `bind_sequence` checks
actual boundaries, spans and exact rational durations under caller limits. Partial reconstruction
intervals cannot generate or bind indexes. GPU seeking belongs to `jxl_wgpu_decode`.
`FrameIndexCollector` observes borrowed transport events, retains only bounded
index metadata, and requires authoritative End before handoff. Known payload limits apply before
allocation; byte-drip growth is geometric. Failures drop encoded and parsed storage, and unrelated
boxes never retain their payload or caller allocation.
[Index contract and remaining scope](../../docs/FRAME_SEEKING.md).

The public `metadata` module retains Exif, XMP, JUMBF and unknown auxiliary payloads by explicit
selection. `MetadataCollector` consumes borrowed transport events without retaining their source
allocations; `ParsedJxl::metadata` serves contiguous input. Original compressed payloads stay exact
until explicit `decode`/`decode_all`; independent limits cover retention, expansion, ratio and
standard Brotli windows. Collections support atomic replacement/removal and canonical `jxlc`
writing, while individual boxes also feed the existing `jxlp` writer. Opaque contents never override
codestream rendering fields. [API, ownership and executed evidence](../../docs/CONTAINER_METADATA.md).

The `gain_map` module parses and emits version-zero `jhgm` bundles with exact ISO 21496-1
rational metadata, optional serialized color/ICC metadata and a borrowed auxiliary codestream.
ISO fractions always carry individual denominators; there is no direction flag. Compatible newer
writer versions and opaque extensions are retained within the 65,535-byte ISO record limit.
Separate payload, codestream and transformed/decoded ICC bounds apply. It validates metadata
without decoding image samples; alternate rendering belongs to `jxl_wgpu_decode`.
[Gain-map contract and interoperability](../../docs/GAIN_MAP.md).

The `jpeg_reconstruction` module parses bounded `jbrd` marker, table, scan and preservation
metadata into an immutable owned model. `ParsedJxl::jpeg_reconstruction` rejects duplicate and
forbidden `brob`-wrapped records after complete transport validation. Canonical emission preserves
decoded metadata contents under simultaneous input/temporary/output byte limits. This is a host
metadata boundary; it does not establish JPEG frame or coefficient validity or produce JPEG image
bytes. [API limits and reconstruction boundary](../../docs/CONTAINER_METADATA.md#jpeg-reconstruction-metadata).

`ContainerStreamScanner` is the non-accumulating transport path. It accepts owned `Arc<[u8]>`
chunks at arbitrary byte boundaries and emits raw/`jxlc`/`jxlp` codestream slices in logical order.
Except for the two-byte codestream signature reconstructed inline across arbitrary chunk
boundaries, ordered payload slices share the caller allocation. A file-type version 1 fragment
received ahead of a gap is the only copied payload; arbitrary input chunks are coalesced into one
retained payload buffer per future fragment under an independent logical-byte limit, then released
as soon as the gap closes. Auxiliary-box events preserve the exact 8/16-byte header encoding and
stream payload slices without assembling the box. Typed limits cover input chunks, total input,
box count/size, codestream size, and buffered future fragments. The
terminal `End` event is emitted only after end-of-input validates transport order and completeness;
earlier events are deliberately non-authoritative.

`CodestreamStreamScanner` observes those transport events by reference, so auxiliary metadata
remains available to the caller. It reconstructs only bounded image-header and current-frame
header/TOC probes, then emits an `Arc`-owned image inventory, an `Arc`-owned frame inventory, and
ordered `SectionChunk` ranges as soon as each TOC is known. Probe sizes grow geometrically rather
than reparsing every byte; logical live/peak and cumulative copied-prefix bytes are observable.
Section payload following the probe retains its `StreamSlice` backing, while a small probe
overshoot remains backed by the bounded prefix allocation. Byte-drip animation, entropy-permuted
TOC, version-1 fragment reorder, caller-`Arc` section identity, truncation, rollback, and poisoned
state tests match the contiguous inventory. `GpuDecoder::stream` consumes this event path directly
and hands its shared spans to the same stock engines used by contiguous input.

After transport validation, `ParsedJxl::codestream_inventory` extracts the standard image header,
enumerated/ICC color and tone-mapping metadata, typed extra channels, animation timing, complete
color and extra-channel blending contracts, per-channel upsampling, XYB quant-matrix scales,
resolved opsin inverse parameters and 2x/4x/8x upsampling weights, progressive-pass schedules,
exact Gaborish/EPF restoration parameters, frame headers, TOC sizes, and byte/bit ranges for every
physical frame section. Progressive-DC inventory records each LF frame's exact level and resolves
every `USE_LF_FRAME` read to the earlier producer frame in the corresponding one of four normative
slots; a missing producer is a typed error before GPU submission. The same resolver runs in the
contiguous and incremental scanners, with a libjxl `--progressive_dc=2` chain checked under
one-byte delivery. It never decodes image samples or frame-section entropy.

Pass schedules cover all 1–11 pass counts (`FramePassesInventory::MAX_PASSES`) and up to four strictly
decreasing downsampling factors with strictly increasing last-pass indices. The number of boundaries
may equal the number of passes for 2–4 passes. Header tests enumerate every representable boundary
schedule with exact bit consumption, and reject excess, duplicate, reversed or out-of-range boundaries.

Color and each extra channel independently determine the presence of their blend source field:
full-frame Replace omits its source, while other modes or partial coverage read it. The channel's
own mode controls this rule; it does not inherit the color mode. Actual libjxl RGBA animations
with different color/alpha modes and sources exercise this grammar through contiguous and
fragmented GPU decode, including the formerly misaligned full-frame MultiplyAdd/Replace header.

Image/frame-header and TOC-size grammar is parsed locally with explicit limits, reusing the
lightweight `jxl-image` primitive bundles for sample/color metadata. Entropy-coded TOC permutations use the published
`jxl-coding` metadata decoder, producing both physical bitstream indices and logical TOC indices.
Embedded ICC streams are reconstructed with bounded `jxl-color` primitives and retained alongside
their exact compressed bit range. Neither path decodes Modular, VarDCT, or pixel data. Returned
section ranges are relative to the contiguous standard codestream, so the same inventory applies
to raw, `jxlc`, and reconstructed `jxlp` input.

Image and gain-map color declarations share the local outer grammar. XYB omits white-point,
primary and transfer-function payloads and supplies their implicit values; the following intent
and metadata retain their exact bit positions. Serialized gamma must lie in `[1/8192, 1]`.

Preview width is read only when its aspect-ratio selector is zero; all eight ratios and both
dimension encodings preserve the following metadata bit position. Explicit and derived preview
axes are limited to 4096, with `InvalidPreviewDimensions` for larger values. There is exactly
one independent Regular preview frame, even when its `is_last` is false. A preview cannot crop,
blend or save a nonzero reference slot. Both scanners reset LF dependencies after that frame.
Noise counters follow physical frames: the preview increments the nonvisible count, which
continues into leading main LF/hidden frames until the first visible frame resets it.
`CodestreamInventory::frame_position` resolves physical IDs in projections that start after zero.

Unknown image, frame, and restoration extension selectors return the typed
`InventoryError::UnsupportedExtensions { scope, selector }` before an authoritative decode can
use the inventory. The single image-metadata walk checks the selector directly, avoiding an
opaque extension parser and a second grammar walk. Empty unknown payloads are also
rejected; frame/restoration payload lengths still obey the explicit extension-bit limit. Safe
auxiliary container boxes continue through the transport event path independently.

Retained reference colour is validated before the remaining frame header and TOC.
`InventoryError::XybIccReference` identifies a prohibited reference slot. Public inventories expose
`can_be_referenced` and `validate_color_reference` for callers that construct or edit fields.
See the [reference-validity audit](../../docs/ICC_COLOR.md#xyb-reference-validity) for the constraint,
negative corpus and preserved output/LF/preview cases.
