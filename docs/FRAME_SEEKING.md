# Frame indexes and bounded GPU seeking

`GpuDecoder::open_seek` and `open_seek_shared` select one zero-based **main-image presentation**.
They accept complete raw, `jxlc`, or ordered/indexed `jxlp` input under the decoder's existing
transport and inventory limits. The complete container, image/frame headers and TOCs are checked
before an index can choose GPU work. Embedded previews keep their physical IDs and noise counters
but are not counted as animation presentations.

The returned `GpuSeekSession` reconstructs the required reference span, discards preroll
presentations, and returns only the target. Blocking, polling and runtime-neutral async
`next_frame`/`next_update` forms use the ordinary validated GPU session. `with_progressive_output`
retains its normal meaning: only the requested presentation can publish validated intermediate
images. Preroll always completes validation before the target uses its references.

Target metadata keeps the original index, presentation ticks, duration, timecode, name and
`is_last`. A subsequent `None` ends the seek operation, even when the target was not the last
animation frame. Physical frame IDs, LF dependencies, entropy ranges and noise seeds are never
renumbered. The reconstruction view preserves original frame finality and reference-saving color
semantics; the stock engine also preserves the full animation's choice of surface renderer.
This keeps seek output bit-identical to sequential GPU output for the covered paths.

## Incremental input

`GpuDecoder::stream_seek(request, index_limits)` returns `GpuDecodeSeekStream`. Feed borrowed
events from `ContainerStreamScanner` under `decoder.container_stream_limits()`, including the
events returned by `finish_input()`. Then `finish(target, seek_limits)` transfers the retained
shared codestream spans into the same `GpuSeekSession` used by contiguous input. It never joins
the complete host codestream. Raw input, compact/extended/to-end index boxes, ordered `jxlp`,
and version-1 out-of-order fragments follow the existing scanner's transport contract.

`FrameIndexCollector` observes those events alongside header inventory. It copies only the bounded
plain index payload, drops its encoded storage after parsing, and keeps parsed entries until End.
Other boxes retain fixed-size state only; a four-byte `brob` probe rejects compressed `jxli`.
Duplicate, malformed, oversized or inconsistent indexes cannot fall back to a generated index.
`finish` on the collector requires authoritative End, and header/dependency binding still occurs
before the GPU engine opens. `stats()` and `index_stats()` report separate input and metadata
ownership. The index's host bounds are independent of `IncrementalInputBudget` and GPU memory.

Input byte/span admission precedes both frontends: capacity exhaustion leaves the borrowed event
unconsumed and retryable after another owner releases capacity. Other event errors poison the
stream and release its retained input spans and encoded/parsed index. A failed consuming `finish`
also releases its input ownership. An upstream scanner failure requires dropping the frontend;
events from a failed scanner cannot establish transport completion.

`is_preview_ready` and `take_preview` retain the ordinary incremental preview contract. A complete
preview can decode and validate on GPU before main input ends, sharing only intersecting input
tokens. Later index or main-stream failure cannot invalidate that independently validated output.
Main-image seeking waits for the complete transport and header inventory even if the target's
bytes arrived earlier. Byte-range fetching and seeking before main-input completion remain open.

## Index metadata and binding

`jxl_gpu_bitstream::FrameIndex` reads at most one plain `jxli` through `from_container`, or
parses/emits its payload directly. The wire format uses an entry count, two big-endian u32 tick
fields, and bounded unsigned little-endian base-128 integers for each offset delta, duration
and displayed-frame count. The model stores absolute logical-codestream offsets, excluding
container headers and `jxlp` indexes. Its nonzero tick denominator defines seconds per tick as
numerator/denominator. The final entry's interval includes its own presentation through stream end.
Emission is canonical; accepted nonminimal integer encodings are normalized.

`FrameSequencePlan` in the bitstream crate owns physical ordering, reference-slot versions,
LF producers, presentation timing and transitive dependencies. Decoder execution adds output
orientation and public presentation metadata to that checked plan; index generation and binding
use its immutable original offsets and clock. Partial reconstruction intervals cannot produce
or bind a full-image index.

`BoundFrameIndex::new` selects the main image, builds this plan, and calls
`FrameIndex::bind_sequence` to bind an optional index to the authoritative header inventory.
Every entry must identify the expected presentation's first physical header, including any
leading hidden or recursive-DC producers. Counts must cover the complete presentation sequence.
Durations are compared to original image ticks using exact u128 rational products. Wrong offsets,
dependent anchors, count/duration mismatches, duplicates, truncated records and limit violations
return typed errors before GPU submission. `brob`-wrapped indexes are explicitly unsupported.
Ordinary `open` does not request index semantics; seeking never silently ignores an invalid index.

When no index is supplied, binding generates all conservatively independent presentation anchors.
`index().encode(...)` can be placed in a `ContainerBox` with `FRAME_INDEX_BOX_TYPE`, then written
using the existing `jxlc` or `jxlp` container writer. Container edits do not change logical offsets,
but editing codestream headers or frame data requires rebuilding the index.

## Encoder index emission

`CodestreamAssembler`, `LosslessModularSequenceSession` and `VarDctSequenceSession` expose
`finish_indexed_container(inventory_limits, index_limits)`. They emit a plain `jxli` and ordinary
`jxlc` after all completed GPU frame artifacts are ordered. Existing `finish_raw` and
`finish_container` remain unindexed. Indexed assembly inventories the actual serialized headers,
constructs `FrameSequencePlan`, and generates entries with `FrameIndex::from_sequence`; it does
not derive offsets or independence from requested options or artifact labels.

Every independently restartable presentation receives an entry. Leading hidden frames are part
of its physical span, dependent presentations extend the prior entry, and a final zero-duration
frame still counts as a presentation. Durations retain the original tick unit, with no rounding
or subtraction of the first duration. Still images use a zero-duration interval with tick unit
1/1. Only container framing changes; the raw codestream is preserved byte for byte.

Inventory limits bound all retained header/ICC/TOC metadata and physical frames; index limits
independently bound displayed frames, entry count and payload bytes. Failure returns a typed
error with no container. This host operation does not check frame entropy or reconstruct pixels.
The encoder emits regular and reference-only main-image frames; preview emission, caller-selected sparse
index policies and compressed indexes are outside this API's current scope.

## Dependency and ownership bounds

An independent target can restart directly. A dependent target needs the earlier exact versions
of LF prediction, color and extra-channel reference slots, including the background chosen by a
blend's alpha channel. Patch selectors are GPU entropy, so the planner conservatively retains
every occupied reference slot when the patch flag is present. Transitive dependencies can force
the chosen anchor backward across a later independent frame that did not overwrite an older slot.

`FrameSeekPlan` reports original target metadata, restart presentation, physical frame range and
preroll count. `FrameSeekLimits` checks both preroll and total physical work before opening the GPU
engine. `SelectedImageInventory::reconstruction_is_complete` distinguishes the full image from a
bounded seek interval; custom engines can use `FrameExecutionPlan::negotiate_selected` to preserve
nonfinal terminal frame semantics.

| Limit | Default |
|---|---:|
| Encoded `jxli` payload | 1 MiB |
| Index entries | 16,384 |
| Displayed presentations | 16,384 |
| Preroll presentations per seek | 16,383 |
| Physical frames per seek, including hidden/LF frames | 16,384 |

These host bounds supplement the existing input/header limits, GPU byte admission and submission
poller limits. A seek carries the same shared source and GPU reservations as an ordinary decode.
Initial resource pressure is retryable. Dropping the seek cancels its work while completion
callbacks retain submitted resources. Completion of the target releases the internal session and
its reference caches; retained output buffers remain immutable and budget-owned until their final
tracked lease is dropped. The index stores metadata, never a CPU image or decoded reference cache.

## Evidence and remaining scope

The [frame-seek corpus](CONFORMANCE_CORPUS.md#frame-index-and-seek-checkpoint) compares whole and
256-byte-window GPU seeking with sequential GPU output and libjxl pixels/timing. It includes mixed
coding modes, recursive DC, composition, previews, physical noise seeds, progressive patch images,
selected extras, rejection, retry, cancellation and retained output.

The independent public libjxl **0.12.0** encoder supplies a still index and dense/sparse animated
indexes. Their offsets and displayed-frame spans agree with the actual headers. The animated
indexes encode a zero first time interval despite nonzero image duration and are rejection
witnesses: they do not pass temporal binding. Generated indexes retain the original image clock,
and native decoding of those containers preserves pixels and durations. The wire reference is
libjxl revision `a7a9c787341cf703dede03c2009fa460cae5e5df`,
[`encode_internal.h`](https://github.com/libjxl/libjxl/blob/a7a9c787341cf703dede03c2009fa460cae5e5df/lib/jxl/encode_internal.h)
and its `EncodeFrameIndexBox` implementation in `encode.cc`.

Seeking requires complete input, received contiguously or incrementally. Byte-range acquisition,
compressed-index policy, non-coalesced layer output, and broader container/feature
conformance remain open. Skipped frame entropy is deliberately not validated or reported as
validated; a valid target lease only attests to that target and the dependencies actually decoded.
`CONT-03` and `FRAME-04` remain **Partial**, and no CPU codec fallback is introduced.
