# jxl_gpu_bitstream

Bounded JPEG XL transport and codestream inventory for GPU codec front ends.

`parse` validates transport framing for raw codestreams, `jxlc` containers, and ordered or indexed
`jxlp` fragment sequences. Raw and single-`jxlc` codestreams remain borrowed; only fragmented
streams are joined.

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
