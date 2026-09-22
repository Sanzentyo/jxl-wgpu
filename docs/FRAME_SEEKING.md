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

## Index metadata and binding

`jxl_gpu_bitstream::FrameIndex` reads at most one plain `jxli` through `from_container`, or
parses/emits its payload directly. The wire format uses an entry count, two big-endian u32 tick
fields, and bounded unsigned little-endian base-128 integers for each offset delta, duration
and displayed-frame count. The model stores absolute logical-codestream offsets, excluding
container headers and `jxlp` indexes. Its nonzero tick denominator defines seconds per tick as
numerator/denominator. The final entry's interval includes its own presentation through stream end.
Emission is canonical; accepted nonminimal integer encodings are normalized.

`BoundFrameIndex::new` binds an optional index to the authoritative main-image header inventory.
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

This API requires complete input. Byte-range acquisition, incremental index collection and seek
handoff, compressed-index policy, non-coalesced layer output, and broader container/feature
conformance remain open. Skipped frame entropy is deliberately not validated or reported as
validated; a valid target lease only attests to that target and the dependencies actually decoded.
`CONT-03` and `FRAME-04` remain **Partial**, and no CPU codec fallback is introduced.
