# GPU performance evidence

This document records measurements, not promised speedups. A result is accepted only after output
comparison and path verification pass. Production reports cover GPU-required encode/decode. Any
dev-only CPU-oracle or historical CPU-decode-plus-GPU-conversion measurement is labelled separately
and can never select a production codec path.

## Workloads

The harness measures the cases that matter for application policy instead of using one global
pixel threshold:

- a single small still image;
- a single medium/large still image;
- a warm sequential batch sharing one device, queue, pipelines, uploads, and readback pool;
- bounded concurrent decoders, reporting per-image latency and aggregate throughput;
- animation playback with bounded in-flight frames;
- GPU-resident display output without readback;
- direct-map and staged-copy CPU readback controls.

Each workload records warmup, sample count, dimensions, encoded/output bytes, format, decode path,
adapter fingerprint, submission count, readback mode, latency percentiles, and throughput. Device
creation and shader compilation are reported separately from steady-state measurements.

The runtime selector uses a workload key rather than assuming large images always favor a device:

```text
(adapter, codec profile, output format, size bucket, batch depth, concurrency, readback/display)
    -> GPU kernel/batch/readback policy or typed unsupported
```

An unknown key uses the conservative GPU variant or rejects; it never selects a CPU codec.

## Prototype evidence retained for context

The old source-tree prototype (`prototype/jxl-fork`) was measured on Apple M5 / Metal on
2026-08-30. It included CPU entropy decode, GPU render work, output packing, mapping, and readback.
It did not show a full-decode CPU-readback crossover in its checked corpus:

| Case | Dimensions | CPU mean | prototype GPU-assisted mean | CPU / GPU |
|---|---:|---:|---:|---:|
| `basic` | 1×1 | 50.98 µs | 897.8 µs | 0.0568× |
| `odd` | 257×257 | 1.4828 ms | 4.8771 ms | 0.3040× |
| `green_queen_vardct_e3` | 438×589 | 9.4604 ms | 22.4620 ms | 0.4212× |

Those numbers explain why the standalone API has no automatic claim that GPU readback wins. They
are not benchmarks of the current stable facade and are not used as current tuning entries.

The same experiment showed that bounded reuse matters: a repeated 512×512 readback workload on the
same adapter improved from 21.25 ms with pooling disabled to 18.55 ms with pooling enabled
(0.873× elapsed). The standalone backend retains bounded buffer and pipeline caches, direct mapping
when the adapter permits it, and map-on-submit so that later CPU work overlaps GPU execution.

## Acceptance policy

For a CPU-readback route to be selected automatically it must:

1. pass the format-specific accuracy contract;
2. execute the expected GPU submissions with no silent CPU fallback;
3. improve the target workload's median and not regress its p95 beyond the configured guardrail;
4. include upload, conversion, mapping, and row de-padding in the measured interval;
5. reproduce across enough runs to survive ordering and thermal bias.

Sequential and concurrent benchmarks use alternating CPU/GPU ordering. Small images are batched so
one command buffer and one completion wait can cover multiple conversions; animation batches remain
bounded by presentation latency and the in-flight limit. GPU display output is tuned independently
because eliminating readback changes the crossover substantially.

Checked-in tuning data should contain an adapter fingerprint and expire when shader, protocol,
format-layout, driver, or `wgpu` versions change. Until new standalone reports are checked in, the
default CPU-readback policy remains conservative while callers can explicitly force either path for
measurement.
