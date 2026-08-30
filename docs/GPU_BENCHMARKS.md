# GPU performance evidence

This document defines the evidence required for performance claims. A result is accepted only
after output comparison and path verification pass. Production reports cover GPU-required
encode/decode; development oracles are labelled separately and cannot select a production codec
path.

## Workloads

The codec harness separates the cases that matter for application policy instead of reducing a
workload to image dimensions alone:

- `single-latency`: one measured operation;
- `warm-sequential`: repeated operations on one host thread while reusing backend objects;
- `concurrent-burst`: barrier-synchronized independent host threads, repeated by iteration;
- `concurrent`: persistent independent host workers, each executing its own sequential stream;
- `animation`: an actual animated codestream session, never a repeated-still substitute;
- `gpu-resident`, `display-texture`, and explicit `cpu-readback` decoded-image targets.

The stock executable profile currently rejects animation. The harness preserves that result as
typed `unsupported`; it does not synthesize animation performance from repeated still images.

Neither `concurrent-burst` nor `concurrent` is codec GPU batching. Every logical image/frame still
has its own codec submission and completion path. A `concurrent-burst` CPU-readback run can retain
already-produced GPU frames and aggregate them into one staging buffer, submission, map, and wait
per burst. That is readback transport aggregation, not codec batching. Reports therefore keep
`coalesced_gpu_batching=false`, label the scheduling model, and record codec, display, and readback
submission/completion-wait counts for measured operations. A future true codec-batch result must
show fewer codec submissions than logical operations and identify the coalesced command buffer;
host thread fan-out or readback aggregation cannot be relabelled as codec batching.

Each workload records warmup configuration, measured sample count, encoded/result bytes, decoded
GPU-layout logical bytes, CPU-readback logical and staging bytes, format, adapter fingerprint,
latency percentiles, wall-clock throughput, and the selected output target. Staging bytes include
copy padding. Readback is reported as `direct_map` for a sole producer-marked output on an eligible
native UMA backend, `aggregate_staged_copy` for a multi-frame aggregate, and `staged_copy`
otherwise. Warmup activity is excluded from submission, wait, frame, and byte totals. Device
creation and initial pipeline compilation happen before the measured loop but do not yet have
separate timing fields.

Per-operation samples begin after the fan-out start barrier. Workload wall time includes host-thread
creation, synchronization, all measured operations, and joins; aggregate throughput uses that wall
time. This keeps scheduler overhead visible without adding thread-start skew to individual samples.

Policy data uses the complete workload key:

```text
(adapter, codec profile, output format, size bucket, burst depth, concurrency, output target)
    -> validated GPU kernel/scheduling/readback policy or typed unsupported
```

An unknown key uses a validated general GPU variant or returns a typed rejection.

## CPU-readback evidence

CPU readback is a transport after GPU codec and conversion work. A CPU-readback result is reported
as favorable only when it:

1. passes the format-specific accuracy contract;
2. reports the codec and readback submission/wait counts and verifies the expected GPU path;
3. improves the target workload's median without exceeding its p95 guardrail;
4. includes upload, conversion, mapping, row de-padding, and output allocation in the measured
   interval; and
5. reproduces across enough interleaved runs to survive ordering and thermal bias.

Small-image latency, warm sequential reuse, concurrent independent sessions, and animation are
separate results. GPU-resident, display-texture, and CPU-readback targets are also separate because
they terminate at different points. A display-texture run currently measures same-queue conversion
enqueue and reports zero display completion waits; it is not end-to-end presentation latency.
CPU-readback timing includes the explicit mapped wait and host byte copies.

A direct-map result must identify the adapter memory architecture and enabled features and must not
be combined with staged-copy results. The mapped buffer is exclusively reserved until mapping and
host consumption finish; concurrent aliases are rejected instead of being silently remapped. GPU
timestamp queries, peak driver allocation, and buffer-pool counters are not yet emitted by the
codec report.

An opt-in `--cpu-baseline-djxl PATH` comparator supplies a labelled external CPU-codec baseline for
the narrow Gray8 U8 CPU-readback contract. Its cold interval includes process spawn, decode,
temporary PGM I/O, parsing, allocation, and output hashing. Warm repetition preserves the GPU
workload's sequential, barrier-burst, or persistent-host-worker launch shape, while explicitly
reporting that every operation starts a new external process. It records executable/version
provenance, timeout, p50, p95, wall throughput, byte totals, and exact GPU/CPU pixel verification.
Unavailable executables and inapplicable formats are typed skipped/unsupported results.

The comparator intentionally emits measurements, not a winner field. A performance conclusion must
compare the recorded values for the same fixture and workload, account for the external-process
model, and satisfy the accuracy and reproducibility rules above. A `verified=true` result establishes
pixel equivalence only; it is not itself a speed claim.

## Current verified development snapshot

These measurements were taken on an Apple M5 and passed the exact output comparison for their
reported path. They are a development decision record, not portable release thresholds. The
summary does not pin the full driver and `wgpu` provenance required by the report lifecycle below,
so release claims require a fresh standalone report.

### Workgroup specialization

The 8K exact CPU-readback workload produced these median end-to-end times:

| `workgroup_size` | Median (ms) | Decision |
| ---: | ---: | --- |
| 32 | 134.031 | retained for comparison |
| 64 | 133.437 | selected |
| 128 | 145.302 | slower |
| 256 | 152.713 | slower |

The selected default is therefore 64. The 128- and 256-thread variants were measured rather than
excluded by assumption, but neither improved this workload.

### Direct-map CPU readback versus `djxl`

The integrated GPU measurement includes decode and direct-map CPU readback. The comparison is the
labelled external-process `djxl` path, whose interval also includes process startup, temporary PGM
I/O, parsing, allocation, and hashing.

| Extent | GPU + direct-map readback median (ms) | External-process `djxl` p50 (ms) | GPU relative to comparator |
| --- | ---: | ---: | ---: |
| 8K | 108.334542 | 70.692292 | 53.2% slower |
| 16K | 410.830 | 263.098917 | 56.1% slower |

For these large exact CPU-readback cases, the current GPU path is approximately 53–56% slower even
than the external-process comparator. There is no large-image CPU-readback win in this snapshot.

For the 17x13 small-image case, direct-map GPU + readback measured 0.731167 ms versus 6.638708 ms
for the external-process `djxl` comparator. This result is specific to a comparator that starts a
new process for each operation; it must not be presented as a 9x advantage over an in-process CPU
decoder. The persistent concurrent workload with four workers and 200 operations per worker
(800 operations total) measured 3307.52 images/s for the same small-image regime.

### Aggregate readback accounting

A four-worker, ten-operation-per-worker burst (40 logical decodes) recorded 40 codec submissions
and 10 readback submissions. Each readback submission aggregated at most four source frames; total
reported staging bytes were 8,960. The report correctly retained
`coalesced_gpu_batching=false`: readback work was aggregated, while codec work was not.

## Report lifecycle

Checked-in tuning data must contain an adapter fingerprint and the codec profile, shader, format
layout, driver, and `wgpu` revisions that produced it. A change to any of those inputs invalidates
the affected entry until it is measured again. Performance claims require a current standalone
report for the exact workload key and a labelled comparator measured under the same conditions.
