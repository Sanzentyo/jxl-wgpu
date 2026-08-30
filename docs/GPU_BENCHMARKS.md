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

Neither `concurrent-burst` nor `concurrent` is GPU batching. Every logical image/frame and every
explicit readback currently has its own submission and completion path. Reports set
`coalesced_gpu_batching=false`, label the scheduling model, and record codec, display, and readback
submission/completion-wait counts for measured operations. A future true-batch result must show
fewer codec submissions than logical operations and identify the coalesced command buffer; host
thread fan-out cannot be relabelled as batching.

Each workload records warmup configuration, measured sample count, encoded/result bytes, decoded
GPU-layout logical bytes, CPU-readback logical and staging bytes, format, adapter fingerprint,
latency percentiles, wall-clock throughput, and the selected output target. Staging bytes include
copy padding. The current generic image readback is reported as `staged_copy`; there is no
direct-map codec-harness result yet. Warmup activity is excluded from submission, wait, frame, and
byte totals. Device creation and initial pipeline compilation happen before the measured loop but
do not yet have separate timing fields.

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

Any future direct-map result must identify the adapter memory architecture and enabled features and
must not be combined with staged-copy results. GPU timestamp queries, peak driver allocation, and
buffer-pool counters are not yet emitted by the codec report.

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

## Report lifecycle

Checked-in tuning data must contain an adapter fingerprint and the codec profile, shader, format
layout, driver, and `wgpu` revisions that produced it. A change to any of those inputs invalidates
the affected entry until it is measured again. Performance claims require a current standalone
report for the exact workload key and a labelled comparator measured under the same conditions.
