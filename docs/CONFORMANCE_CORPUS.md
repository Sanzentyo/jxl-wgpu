# Encode/decode conformance corpus

`tools/jxl_gpu_harness/conformance-corpus.toml` is the source of truth for multi-aspect-ratio and
multi-resolution image coverage. It does not claim that every listed image can already be encoded
or decoded by the stock GPU profile. Instead, every case has one of two explicit expectations:

- `stock_gpu_round_trip`: executable today through the GPU encoder, GPU decoder, and exact CPU
  readback comparison. The current boundary is pitch-linear Gray U8 with width and height in
  `2..=256`.
- `future_gpu_profile`: deterministic generator and inventory coverage only. This currently
  includes RGB, RGBA, 10/12/16-bit samples, extents above 256, and HD/FHD/UHD 4K cases.

The checked-in inventory includes tiny, odd, square, portrait, landscape, panorama, tall,
255/256/257 boundary, HD 1280x720, FHD 1920x1080, and UHD 4K 3840x2160 metadata. Future entries
stay in reports when GPU round-trip mode is selected, but have no fabricated execution result.

## Deterministic source contract

Every case describes:

- `source.model`: `gray`, `rgb`, or `rgba`;
- `source.depth`: `u8`, `u10`, `u12`, or `u16`;
- `source.alpha`: `none` for non-alpha images, or an `opaque`, `checkerboard`, horizontal-ramp, or
  vertical-ramp contract for RGBA;
- `row_layout.alignment`, `extra_padding`, and `padding_byte`;
- a fixed `pattern.seed` interpreted by generator schema version 1.

Samples above eight bits use little-endian canonical in-memory storage. PGM/PPM/PAM writers swap
those samples to the network-order representation required by Netpbm. Reports distinguish two
BLAKE3 hashes:

- `input_hash` covers the complete pitch-linear storage, including deterministic padding;
- `pixel_hash` covers active, interleaved samples only, in the canonical little-endian order.

This makes stride or padding regressions visible without confusing them with decoded-pixel
equality. The stock round-trip path uploads the complete padded storage with an explicit
`ImageLayout` row stride, so the real GPU encoder consumes the declared pitch. The GPU decoder's
active rows must then equal the generator's `pixel_hash`.

## Bounded generation

`LazyImage` validates every multiplication with checked arithmetic and exposes an iterator that
allocates one row at a time. `--max-row-bytes` defaults to 64 MiB and rejects a descriptor before
allocation when its padded stride exceeds that limit. Total logical and storage sizes remain
64-bit metadata; a UHD case is never collected into one `Vec` by the corpus API. Hashing and file
generation are therefore O(row stride) in resident image memory.

## Commands

Inventory all cases and hashes:

```console
cargo run -p jxl_gpu_harness -- conformance --action inventory \
  --output /tmp/jxl-conformance-inventory.json
```

Run exact GPU encode/decode/readback for the checked-in stock cases:

```console
cargo run -p jxl_gpu_harness -- conformance --action gpu-round-trip \
  --output /tmp/jxl-conformance-gpu.json
```

Select cases with repeated or comma-separated `--case` values. Unknown or duplicate selections
are configuration errors.

## Development-only standard fixtures

The external path is intentionally outside all production codec crates. It adds no Rust CPU codec
dependency and cannot replace an unavailable GPU result. Without `--apply`, it only reports the
planned paths:

```console
cargo run -p jxl_gpu_harness -- conformance --action external-fixtures \
  --case tiny-gray8-2x2,hd-rgb8-1280x720 \
  --fixture-dir /tmp/jxl-reference-fixtures
```

To execute installed libjxl tools:

```console
cargo run -p jxl_gpu_harness -- conformance --action external-fixtures \
  --case tiny-gray8-2x2 \
  --cjxl /opt/homebrew/bin/cjxl --djxl /opt/homebrew/bin/djxl \
  --fixture-dir /tmp/jxl-reference-fixtures --apply
```

The harness streams a PGM or PPM source, calls lossless `cjxl`, decodes with `djxl`, parses the
binary PNM output row by row, and verifies extent, channels, maximum sample value, and exact pixel
hash. RGBA entries are kept inventory-only in this external path until a portable alpha-bearing
fixture transport is selected. Existing outputs are not overwritten unless `--force` is supplied.
